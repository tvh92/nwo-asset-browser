"""Blender side of the Rift Apart atmosphere lighting (nwo.games.rcra.atmosphere) in a Blender Scene export. Runs in
Blender only (blender_convert.py --scene --atmosphere bake.npz).

``apply_lighting`` turns the scene's review lighting into the atmosphere's: the world's environment image becomes the
atmosphere's probe (equirectangular, Blender directions, world strength 1, no sun cap) and the review HDRI sun lamp
becomes the key light (irradiance pi x KeyLightColor x KeyLightIntensity, visible in glossy rays: the game's key light
has a specular term). ``apply_display`` meters a small render from the scene camera the way the game meters its frame
(atmosphere.meter, adapted_log2), and shows the result through Rift Apart's display transform as compositor nodes: the
exposure 0.18 / adapted luminance, then the game's own 32^3 ACES lookup (nwo.games.rcra.display.build_lookup) sampled
trilinearly at its log2 coordinate, as CS_ToneMap does; the view transform is Raw, so Blender adds nothing after it.
The metered exposure stays fixed: moving the camera does not re-meter (the group's Exposure input can be changed).
Not reproduced in the .blend: bloom, vignette, film grain and the colour-correction LUT. ``check`` validates a reopened
file.
"""
import json
import os
import tempfile

import bpy
import numpy as np
from mathutils import Vector

from nwo.games.rcra import atmosphere as A
from nwo.games.rcra import display as D
from nwo.pipeline.game_view_compositor import _float_image, _map_uv, _math

KEY_NAME = 'Rift Apart atmosphere key light'
KEY_ROLE = 'rcra_atmosphere_key'
PROBE_IMAGE = 'Rift Apart atmosphere probe'
GROUP_NAME = 'Rift Apart display (compositor)'
TREE_NAME = 'Rift Apart display compositor'
LOOKUP_IMAGE = 'Rift Apart display 3D lookup (tiled)'
METER_EDGE = 512        # long edge of the metering render
METER_SAMPLES = 32


def _world_nodes(scene):
    nodes = scene.world.node_tree.nodes
    background = next(n for n in nodes if n.bl_idname == 'ShaderNodeBackground')
    environment = next(n for n in nodes if n.bl_idname == 'ShaderNodeTexEnvironment')
    return scene.world.node_tree, background, environment


def apply_lighting(scene, bake_path):
    """Replace the review world and sun with the baked atmosphere (nwo.export.models writes the bake); returns the record
    stored in scene['rcra_atmosphere']."""
    from nwo.pipeline import scene_defaults
    bake = np.load(bake_path)
    record = json.loads(str(bake['record']))
    tree, background, environment = _world_nodes(scene)
    cap = tree.nodes.get(scene_defaults.SUN_CAP_NODE)
    if cap is not None:
        tree.nodes.remove(cap)
    mapping = next((n for n in tree.nodes if n.bl_idname == 'ShaderNodeMapping'), None)
    if mapping is not None:
        mapping.inputs['Rotation'].default_value = (0.0, 0.0, 0.0)   # the bake is already in Blender world directions
    equirect = bake['equirect']
    h, w, _ = equirect.shape
    old = environment.image
    image = bpy.data.images.new(PROBE_IMAGE, w, h, alpha=True, float_buffer=True)
    image.colorspace_settings.name = 'Linear Rec.709'   # before the pixels: changing it resets a generated image
    rgba = np.concatenate([equirect[::-1], np.ones((h, w, 1), np.float32)], -1)   # Blender pixels: bottom row first
    image.pixels.foreach_set(rgba.ravel())
    image.file_format = 'OPEN_EXR'
    image.pack()
    environment.image = image
    environment.label = environment.name = 'Rift Apart atmosphere probe'
    if old is not None and old.users == 0:
        bpy.data.images.remove(old)
    tree.links.new(environment.outputs['Color'], background.inputs['Color'])
    background.inputs['Strength'].default_value = 1.0
    scene_defaults.remove_review_sun(scene)
    data = bpy.data.lights.new(KEY_NAME, 'SUN')
    key = bpy.data.objects.new(KEY_NAME, data)
    scene.collection.objects.link(key)
    irradiance = np.asarray(bake['sun_irradiance'], np.float64)
    data.energy = float(irradiance.max())
    data.color = tuple(float(c) for c in irradiance / irradiance.max())
    data.angle = 0.00925   # the review sun's 0.53 degrees: the game's sun diameter is not read
    key.rotation_mode = 'QUATERNION'
    key.rotation_quaternion = Vector(tuple(float(c) for c in bake['sun_direction'])).to_track_quat('Z', 'Y')
    key['lighting_role'] = KEY_ROLE
    if 'shared_lighting' in scene:
        del scene['shared_lighting']   # no longer the review lighting; check() validates this file instead
    record.update(probe_image=PROBE_IMAGE, key_light=KEY_NAME, sun_energy=data.energy, sun_colour=list(data.color))
    scene['rcra_atmosphere'] = json.dumps(record, sort_keys=True)
    return record


def _render_pixels(scene):
    """Render the scene camera's full frame (no Render Region, opaque film, no compositor) small; (h, w, 4) float32."""
    r = scene.render
    saved = dict(percentage=r.resolution_percentage, transparent=r.film_transparent, border=r.use_border,
                 crop=r.use_crop_to_border, compositing=r.use_compositing, path=r.filepath,
                 fmt=r.image_settings.file_format, depth=r.image_settings.color_depth, mode=r.image_settings.color_mode,
                 samples=scene.cycles.samples if scene.render.engine == 'CYCLES' else None)
    folder = tempfile.mkdtemp(prefix='rcra-meter-')
    path = os.path.join(folder, 'meter.exr')
    try:
        r.resolution_percentage = max(1, min(100, round(100 * METER_EDGE / max(r.resolution_x, r.resolution_y))))
        r.film_transparent = False
        r.use_border = r.use_crop_to_border = False
        r.use_compositing = False
        if saved['samples'] is not None:
            scene.cycles.samples = METER_SAMPLES
        r.image_settings.file_format, r.image_settings.color_depth, r.image_settings.color_mode = 'OPEN_EXR', '32', 'RGBA'
        r.filepath = path
        bpy.ops.render.render(write_still=True, scene=scene.name)
        image = bpy.data.images.load(path)
        try:
            pixels = np.array(image.pixels[:], np.float32).reshape(image.size[1], image.size[0], 4)
        finally:
            bpy.data.images.remove(image)
        return pixels
    finally:
        r.resolution_percentage, r.film_transparent = saved['percentage'], saved['transparent']
        r.use_border, r.use_crop_to_border, r.use_compositing = saved['border'], saved['crop'], saved['compositing']
        r.filepath = saved['path']
        r.image_settings.file_format, r.image_settings.color_depth = saved['fmt'], saved['depth']
        r.image_settings.color_mode = saved['mode']
        if saved['samples'] is not None:
            scene.cycles.samples = saved['samples']
        try:
            os.remove(path)
            os.rmdir(folder)
        except OSError:
            pass


def lookup_image():
    """The game's 32^3 display lookup as a packed float image: blue slice b in columns 32 b .. 32 b + 31, red along x,
    green along y (game_view_compositor's tiled layout)."""
    table = D.build_lookup()                      # [r, g, b]
    n = table.shape[0]
    cube = table.transpose(2, 1, 0, 3)            # [b, g, r]
    tiled = cube.transpose(1, 0, 2, 3).reshape(n, n * n, 3)   # [g, b * n + r]
    return _float_image(LOOKUP_IMAGE, n * n, n, tiled.astype(np.float32)), n


def build_group():
    """Compositor group: Image (scene-linear, premultiplied) x Exposure -> the game's lookup at
    (log2 x / 16 + 0.65625), trilinear; colour un-premultiplied for the lookup and premultiplied again."""
    image, n = lookup_image()
    group = bpy.data.node_groups.get(GROUP_NAME)
    if group is not None:
        bpy.data.node_groups.remove(group)
    group = bpy.data.node_groups.new(GROUP_NAME, 'CompositorNodeTree')
    group.interface.new_socket('Image', in_out='INPUT', socket_type='NodeSocketColor')
    group.interface.new_socket('Exposure', in_out='INPUT', socket_type='NodeSocketFloat')
    group.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    N, L = group.nodes, group.links
    gi, go = N.new('NodeGroupInput'), N.new('NodeGroupOutput')
    sep = N.new('CompositorNodeSeparateColor')
    L.new(gi.outputs['Image'], sep.inputs[0])
    alpha = sep.outputs['Alpha']
    safe_alpha = _math(N, L, 'MAXIMUM', alpha, 1e-6)
    coords = []
    for ch in range(3):
        x = _math(N, L, 'DIVIDE', sep.outputs[ch], safe_alpha)
        x = _math(N, L, 'MULTIPLY', x, gi.outputs['Exposure'])
        x = _math(N, L, 'MAXIMUM', x, 1e-30)
        c = _math(N, L, 'LOGARITHM', x, 2.0)
        c = _math(N, L, 'MULTIPLY_ADD', c, D.LOG2_SCALE)
        c.node.inputs[2].default_value = D.LOG2_OFFSET
        c.node.use_clamp = True
        coords.append(_math(N, L, 'MULTIPLY', c, float(n - 1)))
    pr, pg, pb = coords
    b0 = _math(N, L, 'MINIMUM', _math(N, L, 'FLOOR', pb), float(n - 2))
    f = _math(N, L, 'SUBTRACT', pb, b0)
    v = _math(N, L, 'MULTIPLY_ADD', pg, 1.0 / n)
    v.node.inputs[2].default_value = 0.5 / n
    samples = []
    for k in (0, 1):
        slab = _math(N, L, 'ADD', b0, float(k))
        u = _math(N, L, 'MULTIPLY_ADD', slab, float(n))      # (b n + r (n - 1) + 0.5) / n^2
        L.new(pr, u.node.inputs[2])
        u = _math(N, L, 'DIVIDE', _math(N, L, 'ADD', u, 0.5), float(n * n))
        samples.append(_map_uv(N, L, image, u, v))
    comb = N.new('CompositorNodeCombineColor')
    for ch in range(3):
        a0, a1 = samples[0][ch], samples[1][ch]
        mixed = _math(N, L, 'MULTIPLY_ADD', _math(N, L, 'SUBTRACT', a1, a0), f)
        L.new(a0, mixed.node.inputs[2])
        L.new(_math(N, L, 'MULTIPLY', mixed, alpha), comb.inputs[ch])
    L.new(alpha, comb.inputs['Alpha'])
    L.new(comb.outputs[0], go.inputs['Image'])
    return group


def apply_display(scene):
    """Meter the scene camera's frame, then install the display group with its exposure; returns the record stored in
    scene['rcra_display']."""
    record = json.loads(scene['rcra_atmosphere'])
    pixels = _render_pixels(scene)
    measured = A.meter(pixels[..., :3])
    adapted = A.adapted_log2(measured, record['anchor'], record['range'])
    exposure = A.exposure(adapted)
    group = build_group()
    tree = bpy.data.node_groups.get(TREE_NAME) or bpy.data.node_groups.new(TREE_NAME, 'CompositorNodeTree')
    tree.nodes.clear()
    tree.interface.clear()
    tree.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    layers = tree.nodes.new('CompositorNodeRLayers')
    node = tree.nodes.new('CompositorNodeGroup')
    node.node_tree, node.name = group, GROUP_NAME
    node.inputs['Exposure'].default_value = exposure
    out = tree.nodes.new('NodeGroupOutput')
    tree.links.new(layers.outputs['Image'], node.inputs['Image'])
    tree.links.new(node.outputs['Image'], out.inputs['Image'])
    scene.compositing_node_group = tree
    scene.render.use_compositing = True
    view = scene.view_settings
    view.view_transform, view.look, view.exposure, view.gamma = 'Raw', 'None', 0.0, 1.0
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.shading.use_compositor = 'ALWAYS'
    display = dict(measured_ev=measured + A.EV_OFFSET, adapted_ev=adapted + A.EV_OFFSET, exposure=exposure,
                   meter_size=list(pixels.shape[:2]), view='Rift Apart display (compositor)', view_transform='Raw')
    scene['rcra_display'] = json.dumps(display, sort_keys=True)
    return display


def check(scene):
    """Problems with a reopened file's atmosphere lighting and display, [] when it matches its records."""
    problems = []
    try:
        record = json.loads(scene['rcra_atmosphere'])
    except (KeyError, ValueError):
        return ['atmosphere.record']
    try:
        _tree, background, environment = _world_nodes(scene)
        image = environment.image
        if image is None or image.name != PROBE_IMAGE or not image.packed_file:
            problems.append('atmosphere.probe_image')
        elif abs(background.inputs['Strength'].default_value - 1.0) > 1e-6:
            problems.append('atmosphere.world_strength')
    except (StopIteration, AttributeError):
        problems.append('atmosphere.world')
    keys = [o for o in scene.objects if o.type == 'LIGHT' and o.get('lighting_role') == KEY_ROLE]
    if len(keys) != 1 or keys[0].data.type != 'SUN' or abs(keys[0].data.energy - record['sun_energy']) > 1e-4:
        problems.append('atmosphere.key_light')
    from nwo.pipeline import scene_defaults
    if any(scene_defaults.review_sun(o) for o in scene.objects):
        problems.append('atmosphere.review_sun')
    try:
        display = json.loads(scene['rcra_display'])
        tree = scene.compositing_node_group
        node = tree.nodes.get(GROUP_NAME) if tree else None
        if (node is None or abs(node.inputs['Exposure'].default_value - display['exposure']) > 1e-6 * display['exposure']
                or not scene.render.use_compositing or scene.view_settings.view_transform != 'Raw'):
            problems.append('atmosphere.display')
        lookup = bpy.data.images.get(LOOKUP_IMAGE)
        if lookup is None or not lookup.packed_file:
            problems.append('atmosphere.display_lookup')
    except (KeyError, ValueError):
        problems.append('atmosphere.display_record')
    return problems
