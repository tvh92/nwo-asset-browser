"""Blender side of nwo.export.models: import one .glb, write FBX / USD / .blend. Run by Blender only.

    blender --background --factory-startup --python blender_convert.py -- --glb in.glb [--fbx out.fbx] [--usd out.usdc] [--blend out.blend]
        [--materials rcra_materials.json] [--scene]
    blender --background --factory-startup --python blender_convert.py -- --validate file.blend

``--materials`` (nwo.games.rcra.blender_export) replaces each named imported material with the game's translated
material (nwo.export.rcra_blender) before any file is written, and draws fur shells in the .blend only
(build_fur_shells, after FBX and USD are written; the .blend's Cycles transparent bounces are raised to cross them). ``--validate`` reopens a written .blend and checks
every translated material (its surface group, the BRDF group and its images) and, in a file that records the shared
review lighting, that lighting (scene_defaults.check_shared_lighting); it exits non-zero on a problem.

``--scene`` makes the .blend a ready-to-render scene, as the research pipeline's Blender Scene: the shared review
lighting with its HDRI world (assets/lighting/shared_lighting.json, packed), the Cycles OptiX GPU render setup, and
one camera at the reference angle dollied so the model fills the frame, with the Render Region cropped around it
(scene_defaults.frame_camera_zoom, fit_render_region). Without it the .blend gets only the plain defaults below.
With ``--atmosphere bake.npz`` as well, a Rift Apart atmosphere replaces the review lighting and the scene shows
through the game's display transform, metered from the camera (nwo.export.rcra_atmosphere_blender).

Factory settings, nothing saved to the user's preferences. The .blend gets the repository's
plain scene/render defaults (nwo.pipeline.scene_defaults.apply_new_scene_defaults, without the HDRI world or OptiX GPU setup) and packs its images.
"""
import argparse
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root, for nwo.pipeline.scene_defaults

args = argparse.ArgumentParser()
args.add_argument('--glb')
args.add_argument('--materials')
args.add_argument('--validate')
args.add_argument('--fbx')
args.add_argument('--usd')
args.add_argument('--blend')
args.add_argument('--scene', action='store_true')
args.add_argument('--atmosphere')   # with --scene: a Rift Apart atmosphere bake (.npz) replaces the review lighting
args = args.parse_args(sys.argv[sys.argv.index('--') + 1:])

if args.validate:
    from nwo.export.rcra_blender import translated as is_translated, validate_material
    bpy.ops.wm.open_mainfile(filepath=args.validate)
    translated = [m for m in bpy.data.materials if is_translated(m)]
    problems = [p for m in translated for p in validate_material(m)]
    if 'shared_lighting' in bpy.context.scene:
        from nwo.pipeline.scene_defaults import check_shared_lighting
        problems += check_shared_lighting(bpy.context.scene)
    if 'rcra_atmosphere' in bpy.context.scene:
        from nwo.export.rcra_atmosphere_blender import check as check_atmosphere
        problems += check_atmosphere(bpy.context.scene)
    for problem in problems:
        print('PROBLEM', problem, flush=True)
    if problems:
        sys.exit(1)
    print('VALIDATED', args.validate, len(translated), 'translated materials', flush=True)
    sys.exit(0)

bpy.ops.wm.read_factory_settings(use_empty=True)
# No bone-display Icosphere: FBX/USD would export it as a mesh.
bpy.ops.import_scene.gltf(filepath=args.glb, disable_bone_shape=True)
# One collection per model (named after it), so a combined file's models can be shown or hidden as a group.
from nwo.export.model_collections import group_by_model
group_by_model(bpy.context.scene)
actions = list(bpy.data.actions)
for action in actions:
    action.use_fake_user = True   # every clip stays in the file, not only the active one

entries = None
if args.materials:
    import json
    from nwo.export.rcra_blender import TANGENT_FRAME, apply_recipe, build_fur_shells, store_tangent_frame
    entries = json.loads(Path(args.materials).read_text(encoding='utf-8'))['materials']
    users = {}
    for mesh in bpy.data.meshes:   # the decoded tangent as coordinates in the UV0 frame, before any material reads it
        store_tangent_frame(mesh)
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for slot in obj.material_slots:
                if slot.material is not None:
                    users.setdefault(slot.material.name, obj.data)
    for name, entry in entries.items():
        material = bpy.data.materials.get(name)
        if material is None or not (entry.get('recipe') or entry.get('forward')):
            continue
        mesh = users.get(name)
        uv_maps = [layer.name for layer in mesh.uv_layers] if mesh is not None else []
        colour = mesh.color_attributes[0].name if mesh is not None and len(mesh.color_attributes) else None
        apply_recipe(material, entry, uv_maps, colour, tangent=mesh is not None and TANGENT_FRAME in mesh.attributes)
        print('TRANSLATED', name, flush=True)

if args.fbx:
    bpy.ops.export_scene.fbx(filepath=args.fbx, path_mode='COPY', embed_textures=True, add_leaf_bones=False,
                             bake_anim=bool(actions), bake_anim_use_all_actions=True, bake_anim_use_nla_strips=False,
                             bake_anim_force_startend_keying=True)
    print('EXPORTED', args.fbx, flush=True)
if args.usd:
    bpy.ops.wm.usd_export(filepath=args.usd, export_textures_mode='NEW', export_animation=bool(actions))
    print('EXPORTED', args.usd, flush=True)
if args.blend:
    from nwo.pipeline import scene_defaults
    scene = bpy.context.scene
    # Plain scene/render defaults only, unless --scene: then the shared-lighting world and the OptiX device setup too.
    scene_defaults.apply_new_scene_defaults(scene, configure_world=args.scene, configure_gpu=args.scene)
    # Fur shells go into the .blend only, after FBX and USD are written: those formats keep no coverage alpha.
    shell_layers = build_fur_shells(entries) if entries else 0
    cycles = getattr(bpy.context.scene, 'cycles', None)
    if shell_layers and cycles is not None:   # a ray crosses every fur shell twice (front and back of the strand layer)
        cycles.transparent_max_bounces = max(cycles.transparent_max_bounces, 2 * shell_layers + 8)
    for obj in bpy.data.objects:   # rigs stay in the file (they drive the clips) but start hidden, as the pipeline's skeletons do
        if obj.type == 'ARMATURE':
            obj.hide_set(True)
            obj.hide_render = True
    if args.scene:
        # One camera at the reference angle, dollied so the visible meshes (at the current frame) fill the frame,
        # then the Render Region around them, as the research pipeline frames its scenes.
        points = scene_defaults.visible_mesh_points(scene)
        if not len(points):
            raise RuntimeError('No visible mesh to frame the review camera on')
        lo, hi = points.min(axis=0), points.max(axis=0)
        radius = max(float(((hi - lo) ** 2).sum() ** 0.5) / 2, .05)
        camera_data = bpy.data.cameras.new('Review camera')
        camera_data.type, camera_data.lens = 'PERSP', 50
        camera_data.clip_start, camera_data.clip_end = max(.0001, radius / 10000), radius * 100
        camera = bpy.data.objects.new('Review camera', camera_data)
        scene.collection.objects.link(camera)
        scene.camera = camera
        scene_defaults.frame_camera_zoom(scene, points)
        scene_defaults.fit_render_region(scene)
        scene_defaults.fit_render_size(scene)
        for screen in bpy.data.screens:   # open through the camera
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    area.spaces.active.lock_camera = True
                    area.spaces.active.region_3d.view_perspective = 'CAMERA'
        if args.atmosphere:
            # A Rift Apart atmosphere instead of the review lighting, metered from this camera and shown through the
            # game's display transform (nwo.export.rcra_atmosphere_blender).
            from nwo.export import rcra_atmosphere_blender
            lighting = rcra_atmosphere_blender.apply_lighting(scene, args.atmosphere)
            bpy.ops.file.pack_all()
            display = rcra_atmosphere_blender.apply_display(scene, args.atmosphere)
            fill = rcra_atmosphere_blender.apply_fill(scene)   # a presentation light, after the metering
            print('SCENE', 'atmosphere %s, metered %.2f EV, exposure %.4f, fill %s, camera and render region'
                  % (lighting['path'], display['measured_ev'], display['exposure'],
                     '%.3f' % fill['energy'] if fill else 'none'), flush=True)
        else:
            scene_defaults.pack_scene_images(scene)   # packs the HDRI too, checked against the shared EXR
            print('SCENE', 'review lighting, camera and render region', flush=True)
    else:
        bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=args.blend, compress=True)
    print('EXPORTED', args.blend, flush=True)
