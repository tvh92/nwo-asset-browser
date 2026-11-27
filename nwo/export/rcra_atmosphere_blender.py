"""Blender side of the Rift Apart atmosphere lighting (nwo.games.rcra.atmosphere) in a Blender Scene export. Runs in
Blender only (blender_convert.py --scene --atmosphere bake.npz).

``apply_lighting`` turns the scene's review lighting into the atmosphere's: the world's environment image becomes the
atmosphere's probe (equirectangular, Blender directions, world strength 1, no sun cap) and the review HDRI sun lamp
becomes the key light (irradiance pi x KeyLightColor x KeyLightIntensity, visible in glossy rays: the game's key light
has a specular term) that casts no shadows (a presentation choice: the game's gallery viewer shows models without
self-shadowing; the probe still occludes). ``apply_display`` meters the environment alone the way the game meters its
frame (atmosphere.meter_environment, adapted_log2: a level camera turned through a full circle, no model), so every model
and camera in one atmosphere gets the same exposure, the viewer's, and shows the result through Rift Apart's display
transform as compositor nodes: the
exposure 0.18 / adapted luminance, then the game's own 32^3 ACES lookup (nwo.games.rcra.display) with the
atmosphere's colour-correction LUT mixed in at weight 1 (one fully active atmosphere), sampled trilinearly at its log2
coordinate, as CS_ToneMap does; the view transform is Raw, so Blender adds nothing after it.
The group's Exposure input can be changed by hand.
``apply_fill`` then adds a presentation light that is not the game's: a soft, neutral sun shining up from below at a
third of the key light's strength (it is added after the metering, so it brightens the shot; its strength can be
changed on the lamp).
Not reproduced in the .blend: bloom, vignette and film grain. ``check`` validates a reopened file.
"""
import json

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
FILL_NAME = 'Presentation fill from below (not game lighting)'
FILL_ROLE = 'presentation_fill_below'
FILL_ANGLE = 1.5708     # a 90-degree disc: soft shading and shadows
FILL_RATIO = 1 / 3      # fill strength / key light strength


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
    data.use_shadow = False   # presentation: the game's gallery viewer shows models without self-shadowing
    key.rotation_mode = 'QUATERNION'
    key.rotation_quaternion = Vector(tuple(float(c) for c in bake['sun_direction'])).to_track_quat('Z', 'Y')
    key['lighting_role'] = KEY_ROLE
    if 'shared_lighting' in scene:
        del scene['shared_lighting']   # no longer the review lighting; check() validates this file instead
    record.update(probe_image=PROBE_IMAGE, key_light=KEY_NAME, sun_energy=data.energy, sun_colour=list(data.color),
                  key_cast_shadow=False)
    scene['rcra_atmosphere'] = json.dumps(record, sort_keys=True)
    return record


def lookup_image(lut=None):
    """The game's 32^3 display lookup (with colour-correction ``lut`` at weight 1) as a packed float image: blue slice
    b in columns 32 b .. 32 b + 31, red along x, green along y (game_view_compositor's tiled layout)."""
    table = D.colour_corrected([(lut, 1.0)]) if lut is not None else D.build_lookup()   # [r, g, b]
    n = table.shape[0]
    cube = table.transpose(2, 1, 0, 3)            # [b, g, r]
    tiled = cube.transpose(1, 0, 2, 3).reshape(n, n * n, 3)   # [g, b * n + r]
    return _float_image(LOOKUP_IMAGE, n * n, n, tiled.astype(np.float32)), n


def build_group(lut=None):
    """Compositor group: Image (scene-linear, premultiplied) x Exposure -> the game's lookup (with colour-correction
    ``lut``) at (log2 x / 16 + 0.65625), trilinear; colour un-premultiplied for the lookup and premultiplied again."""
    image, n = lookup_image(lut)
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


def apply_display(scene, bake_path):
    """Meter the bake's environment (atmosphere.meter_environment), then install the display group with its exposure
    and the bake's colour-correction LUT; returns the record stored in scene['rcra_display']."""
    record = json.loads(scene['rcra_atmosphere'])
    with np.load(bake_path) as bake:
        lut = bake['colour_correction'] if bake['colour_correction'].size else None
        measured = A.meter_environment(bake['equirect'])
    adapted = A.adapted_log2(measured, record['anchor'], record['range'])
    exposure = A.exposure(adapted)
    group = build_group(lut)
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
                   meter='environment alone: level camera turned through a full circle', view='Rift Apart display (compositor)', view_transform='Raw',
                   colour_correction=record.get('colour_correction'), colour_correction_weight=1.0 if lut is not None else 0.0)
    scene['rcra_display'] = json.dumps(display, sort_keys=True)
    return display


def fill_irradiance(record):
    """Blender sun strength of the presentation fill: FILL_RATIO x the key light's (a 3:1 key-to-fill ratio, a
    photographic convention, not game data)."""
    return float(FILL_RATIO * record['sun_energy'])


def apply_fill(scene):
    """Add the presentation fill (see the module): a neutral sun lamp pointing straight up, invisible in glossy rays (no
    highlight from below); returns the record stored in scene['rcra_fill']."""
    record = json.loads(scene['rcra_atmosphere'])
    energy = fill_irradiance(record)
    data = bpy.data.lights.new(FILL_NAME, 'SUN')
    data.energy, data.color, data.angle = energy, (1.0, 1.0, 1.0), FILL_ANGLE
    fill = bpy.data.objects.new(FILL_NAME, data)
    fill.rotation_euler = (np.pi, 0.0, 0.0)          # a sun shines along its -Z: turned over, it shines up
    fill.visible_glossy = False
    fill['lighting_role'] = FILL_ROLE
    scene.collection.objects.link(fill)
    result = dict(name=FILL_NAME, energy=energy, angle=FILL_ANGLE, presentation=True,
                  rule='key light strength / 3, neutral (presentation, not game lighting)')
    scene['rcra_fill'] = json.dumps(result, sort_keys=True)
    return result


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
    if (len(keys) != 1 or keys[0].data.type != 'SUN' or abs(keys[0].data.energy - record['sun_energy']) > 1e-4 or
            bool(keys[0].data.use_shadow) != bool(record.get('key_cast_shadow', True))):
        problems.append('atmosphere.key_light')
    fills = [o for o in scene.objects if o.type == 'LIGHT' and o.get('lighting_role') == FILL_ROLE]
    if 'rcra_fill' in scene:
        fill = json.loads(scene['rcra_fill'])
        if len(fills) != 1 or fills[0].data.type != 'SUN' or abs(fills[0].data.energy - fill['energy']) > 1e-4:
            problems.append('atmosphere.fill')
    elif fills:
        problems.append('atmosphere.fill')
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
