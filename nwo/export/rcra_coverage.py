"""Additive light on a transparent film, for a Rift Apart Blender Scene export (blender_convert.py --scene). Runs in Blender;
the threshold helpers run anywhere.

An additive material (nwo.export.rcra_blender: Transparent(1 + alpha B) + its light) adds light without covering what is
behind it. Over the transparent film its pixels carry that light with alpha near 0, and a PNG, which stores colour
divided by alpha, loses it: the unicroid wasp's wings vanished from its renders. ``apply`` makes the compositor carry it as
coverage. The scene's display transform moves into the compositor, so the coverage is worked out in display values: the
review scene's view (Khronos PBR Neutral, its exposure) becomes Exposure then Convert Colorspace to "Khronos PBR Neutral
sRGB", which matches the view within 16-bit PNG steps; an atmosphere scene's display group
(nwo.export.rcra_atmosphere_blender) already is one.
The view transform is then Raw. Per pixel, with C its premultiplied scene-linear colour, a its alpha and disp the display
transform of an opaque colour:

    N  = a disp(C / a)                       what Blender writes (the usual straight-alpha encoding)
    E  = C max(0, 1 - a W / max(C))         the light past what the coverage carries (straight colour above W)
    O  = N + (1 - a) max(0, disp(E) - disp(0))
    a' = min(1, max(a, max(O)))              written as O / a'

O moves smoothly with a: a form that subtracts the display of the carried light, disp(C - E), turned noise in a into
grain, since the display is steepest near black. And a is noisy over an additive surface: each Cycles sample either
passes through it or continues from its lit closure, which ends the transparent path, so the film's alpha there is a
binomial estimate of a small mean (0.017 +- 0.018 per pixel at 128 samples over the wasp's wings), and Cycles' denoiser
leaves alpha alone. Where the alpha's local mean (a Gaussian of SOFT_PIXELS) is below SOFT_BELOW and the pixel is lit
past its coverage (straight colour max(C) / a at least W), that mean is used for a. It eases back to the pixel's own
alpha as the mean reaches SOFT_ABOVE or the straight colour falls to W / 2, so covered pixels, thin features included,
keep their alpha exactly.

W is the exposed grey whose display reaches COVERED_DISPLAY of display white: brighter straight colour is past what the
display shows anyway, so the rest is light no coverage carries. Opaque pixels (a = 1) and pixels whose straight colour
stays below W (antialiased edges, glass) keep Blender's own value. Pure additive light over the film (a = 0) is written
so that the PNG over black shows disp(C), the render's own display of that light; over another background it is a
presentation approximation (it covers the background by a' instead of adding to it), labelled in scene['rcra_coverage'].
"""
import json
import math

COVERED_DISPLAY = 0.98
SOFT_PIXELS = 6.0      # Gaussian size of the alpha's local mean
SOFT_BELOW, SOFT_ABOVE = 0.05, 0.15
NODE_NAME = 'Rift Apart additive coverage (compositor)'
REVIEW_GROUP = 'Rift Apart review display with additive coverage'
ATMOSPHERE_GROUP = 'Rift Apart display with additive coverage'
TREE_NAME = 'Rift Apart review compositor'
# Views whose compositor form is a colourspace of the same name (Blender 5.0's OCIO config).
VIEW_COLOURSPACES = {'Khronos PBR Neutral': 'Khronos PBR Neutral sRGB', 'Standard': 'sRGB'}
NOTE = ('Additive light over the transparent film is written as coverage (nwo.export.rcra_coverage): over black the PNG '
        'shows the render\'s light; over another background it covers it by alpha instead of adding to it. Opaque pixels '
        'and straight colours below W keep Blender\'s encoding. Presentation, not game data.')


def srgb_to_linear(v):
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def khronos_grey_limit(fraction=COVERED_DISPLAY):
    """The exposed grey x whose Khronos PBR Neutral display (sRGB) is ``fraction`` of white. For grey the curve subtracts
    0.04, then above 0.76 compresses the peak p to 1 - 0.24^2 / (p + 0.24 - 0.76); solved for the linear target."""
    target = srgb_to_linear(fraction)
    return 0.04 + 0.52 + 0.24 ** 2 / (1.0 - target)


def standard_grey_limit(fraction=COVERED_DISPLAY):
    """The sRGB view clips at 1: the linear value of ``fraction``."""
    return srgb_to_linear(fraction)


def game_grey_limit(table=None, fraction=COVERED_DISPLAY):
    """The exposed grey whose Rift Apart display (nwo.games.rcra.display, ``table`` a colour-corrected lookup) reaches
    ``fraction`` of the display at the top of the lookup, by bisection in log2."""
    import numpy as np
    from nwo.games.rcra import display as D
    top = float(np.mean(D.display(np.full((1, 3), 2.0 ** ((1.0 - D.LOG2_OFFSET) / D.LOG2_SCALE)), 1.0, table)))
    lo, hi = -10.0, (1.0 - D.LOG2_OFFSET) / D.LOG2_SCALE
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.mean(D.display(np.full((1, 3), 2.0 ** mid), 1.0, table))) >= fraction * top:
            hi = mid
        else:
            lo = mid
    return 2.0 ** hi


# ------------------------------------------------------------------------------------------------ Blender side

def _math(N, L, op, a=None, b=None, clamp=False):
    from nwo.pipeline.game_view_compositor import _math as math_node
    return math_node(N, L, op, a, b, clamp)


def _combine(N, L, rgb, alpha):
    node = N.new('CompositorNodeCombineColor')
    for ch, socket in enumerate(rgb):
        if isinstance(socket, (int, float)):
            node.inputs[ch].default_value = float(socket)
        else:
            L.new(socket, node.inputs[ch])
    if isinstance(alpha, (int, float)):
        node.inputs['Alpha'].default_value = float(alpha)
    else:
        L.new(alpha, node.inputs['Alpha'])
    return node.outputs[0]


def _separate(N, L, image):
    node = N.new('CompositorNodeSeparateColor')
    L.new(image, node.inputs[0])
    return node.outputs


def _coverage(N, L, colour, alpha, gain, limit, display):
    """The coverage nodes (module docstring) on premultiplied ``colour`` (three sockets) and ``alpha``; ``gain`` the
    exposure multiplier (socket or float) that brings C to the units of ``limit`` (W); ``display(rgb, alpha) -> rgb``
    builds the premultiplied display of a colour. Returns (rgb sockets O, alpha socket a')."""
    peak = _math(N, L, 'MAXIMUM', _math(N, L, 'MAXIMUM', colour[0], colour[1]), colour[2])
    peak = _math(N, L, 'MAXIMUM', _math(N, L, 'MULTIPLY', peak, gain), 1e-30)
    blur = N.new('CompositorNodeBlur')
    blur.inputs['Size'].default_value = (SOFT_PIXELS, SOFT_PIXELS)
    L.new(_combine(N, L, [alpha, alpha, alpha], 1.0), blur.inputs['Image'])
    mean = _separate(N, L, blur.outputs[0])[0]
    faint = _math(N, L, 'DIVIDE', _math(N, L, 'SUBTRACT', SOFT_ABOVE, mean), SOFT_ABOVE - SOFT_BELOW, clamp=True)
    # straight colour / W: only pixels lit past their coverage soften (2 r - 1 from r = 1/2 to 1); covered pixels,
    # thin features included, keep their alpha
    ratio = _math(N, L, 'DIVIDE', peak, _math(N, L, 'MAXIMUM', _math(N, L, 'MULTIPLY', alpha, limit), 1e-30))
    lit = _math(N, L, 'MULTIPLY_ADD', ratio, 2.0)
    lit.node.inputs[2].default_value = -1.0
    lit.node.use_clamp = True
    weight = _math(N, L, 'MULTIPLY', faint, lit)
    soft = _math(N, L, 'MULTIPLY_ADD', _math(N, L, 'SUBTRACT', mean, alpha), weight)   # a + (mean - a) weight
    L.new(alpha, soft.node.inputs[2])
    alpha = soft
    covered = display(colour, alpha)                                    # N
    past = _math(N, L, 'MAXIMUM', _math(N, L, 'SUBTRACT', 1.0, _math(N, L, 'DIVIDE', _math(N, L, 'MULTIPLY', alpha, limit),
                                                                     peak)), 0.0)
    shown = display([_math(N, L, 'MULTIPLY', c, past) for c in colour], 1.0)   # disp(E)
    black = display([0.0, 0.0, 0.0], 1.0)                                      # disp(0): Khronos lifts black by 3e-4
    open_ = _math(N, L, 'SUBTRACT', 1.0, alpha)
    out = []
    for ch in range(3):
        excess = _math(N, L, 'MAXIMUM', _math(N, L, 'SUBTRACT', shown[ch], black[ch]), 0.0)
        out.append(_math(N, L, 'MULTIPLY_ADD', excess, open_))
        L.new(covered[ch], out[-1].node.inputs[2])
    top = _math(N, L, 'MAXIMUM', _math(N, L, 'MAXIMUM', out[0], out[1]), out[2])
    return out, _math(N, L, 'MINIMUM', _math(N, L, 'MAXIMUM', alpha, top), 1.0)


def review_group(view_transform, limit):
    """Compositor group: Image (scene-linear, premultiplied), Exposure (stops) -> the view's display with coverage."""
    import bpy
    group = bpy.data.node_groups.get(REVIEW_GROUP)
    if group is not None:
        bpy.data.node_groups.remove(group)
    group = bpy.data.node_groups.new(REVIEW_GROUP, 'CompositorNodeTree')
    group.interface.new_socket('Image', in_out='INPUT', socket_type='NodeSocketColor')
    group.interface.new_socket('Exposure (stops)', in_out='INPUT', socket_type='NodeSocketFloat')
    group.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    N, L = group.nodes, group.links
    gi, go = N.new('NodeGroupInput'), N.new('NodeGroupOutput')
    sep = _separate(N, L, gi.outputs['Image'])
    gain = _math(N, L, 'POWER', 2.0, gi.outputs['Exposure (stops)'])
    space = VIEW_COLOURSPACES[view_transform]

    def display(rgb, alpha):
        safe = alpha if isinstance(alpha, (int, float)) else _math(N, L, 'MAXIMUM', alpha, 1e-6)
        straight = [_math(N, L, 'MULTIPLY', _math(N, L, 'DIVIDE', c, safe), gain) for c in rgb]
        conv = N.new('CompositorNodeConvertColorSpace')
        conv.from_color_space, conv.to_color_space = 'scene_linear', space
        L.new(_combine(N, L, straight, 1.0), conv.inputs[0])
        shown = _separate(N, L, conv.outputs[0])
        if isinstance(alpha, (int, float)):
            return [shown[ch] for ch in range(3)]
        return [_math(N, L, 'MULTIPLY', shown[ch], alpha) for ch in range(3)]
    rgb, alpha = _coverage(N, L, [sep[ch] for ch in range(3)], sep['Alpha'], gain, limit, display)
    L.new(_combine(N, L, rgb, alpha), go.inputs['Image'])
    group['rc_limit'] = float(limit)
    group['rc_view_transform'] = view_transform
    return group


def atmosphere_group(display_group, limit):
    """Compositor group: Image, Exposure (the display group's) -> the Rift Apart display with coverage; three
    instances of ``display_group`` (nwo.export.rcra_atmosphere_blender.build_group) share the Exposure."""
    import bpy
    group = bpy.data.node_groups.get(ATMOSPHERE_GROUP)
    if group is not None:
        bpy.data.node_groups.remove(group)
    group = bpy.data.node_groups.new(ATMOSPHERE_GROUP, 'CompositorNodeTree')
    group.interface.new_socket('Image', in_out='INPUT', socket_type='NodeSocketColor')
    group.interface.new_socket('Exposure', in_out='INPUT', socket_type='NodeSocketFloat')
    group.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    N, L = group.nodes, group.links
    gi, go = N.new('NodeGroupInput'), N.new('NodeGroupOutput')
    sep = _separate(N, L, gi.outputs['Image'])

    def display(rgb, alpha):   # the display group un-premultiplies and premultiplies again itself
        node = N.new('CompositorNodeGroup')
        node.node_tree = display_group
        L.new(gi.outputs['Exposure'], node.inputs['Exposure'])
        L.new(_combine(N, L, rgb, alpha), node.inputs['Image'])
        shown = _separate(N, L, node.outputs['Image'])
        return [shown[ch] for ch in range(3)]
    rgb, alpha = _coverage(N, L, [sep[ch] for ch in range(3)], sep['Alpha'], gi.outputs['Exposure'], limit, display)
    L.new(_combine(N, L, rgb, alpha), go.inputs['Image'])
    group['rc_limit'] = float(limit)
    return group


def _viewport_compositor():
    import bpy
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.shading.use_compositor = 'ALWAYS'


def apply_review(scene):
    """The review scene's view transform moved into the compositor with coverage; returns the record stored in
    scene['rcra_coverage'], or None (scene unchanged) for a view, look, gamma or compositor it does not reproduce."""
    import bpy
    view = scene.view_settings
    if (view.view_transform not in VIEW_COLOURSPACES or view.look != 'None' or abs(view.gamma - 1.0) > 1e-6
            or scene.render.use_compositing or scene.display_settings.display_device != 'sRGB'):
        return None
    limit = khronos_grey_limit() if view.view_transform == 'Khronos PBR Neutral' else standard_grey_limit()
    group = review_group(view.view_transform, limit)
    tree = bpy.data.node_groups.get(TREE_NAME) or bpy.data.node_groups.new(TREE_NAME, 'CompositorNodeTree')
    tree.nodes.clear()
    tree.interface.clear()
    tree.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    layers = tree.nodes.new('CompositorNodeRLayers')
    node = tree.nodes.new('CompositorNodeGroup')
    node.node_tree, node.name = group, NODE_NAME
    node.inputs['Exposure (stops)'].default_value = view.exposure
    out = tree.nodes.new('NodeGroupOutput')
    tree.links.new(layers.outputs['Image'], node.inputs['Image'])
    tree.links.new(node.outputs['Image'], out.inputs['Image'])
    record = dict(kind='review', view_transform=view.view_transform, exposure=float(view.exposure), limit=limit,
                  covered_display=COVERED_DISPLAY, note=NOTE)
    scene.compositing_node_group = tree
    scene.render.use_compositing = True
    view.view_transform, view.exposure = 'Raw', 0.0
    _viewport_compositor()
    scene['rcra_coverage'] = json.dumps(record, sort_keys=True)
    return record


def apply_atmosphere(scene, lut=None):
    """``lut`` the atmosphere's colour-correction lookup (as apply_display reads it from the bake), for W. Wrap the
    atmosphere's display node (rcra_atmosphere_blender.apply_display) in the coverage group, keeping its
    name and Exposure; returns the record stored in scene['rcra_coverage']."""
    from nwo.export import rcra_atmosphere_blender as R
    from nwo.games.rcra import display as D
    tree = scene.compositing_node_group
    node = tree.nodes[R.GROUP_NAME]
    display_group = node.node_tree
    limit = game_grey_limit(D.colour_corrected([(lut, 1.0)]) if lut is not None else None)
    node.node_tree = atmosphere_group(display_group, limit)   # same sockets: links and Exposure stay
    record = dict(kind='atmosphere', limit=limit, covered_display=COVERED_DISPLAY, note=NOTE)
    scene['rcra_coverage'] = json.dumps(record, sort_keys=True)
    return record


def check_review(scene, expected):
    """Problems with a reopened review scene whose view is in the coverage compositor, against the shared lighting's
    expected view ([] when it matches)."""
    try:
        record = json.loads(scene['rcra_coverage'])
    except (KeyError, ValueError):
        return ['coverage.record']
    view = scene.view_settings
    tree = scene.compositing_node_group
    node = tree.nodes.get(NODE_NAME) if tree is not None else None
    if (record.get('kind') != 'review' or record.get('view_transform') != expected['view_transform']
            or expected['look'] != 'None' or abs(float(expected['gamma']) - 1.0) > 1e-5
            or view.view_transform != 'Raw' or view.look != 'None' or abs(view.exposure) > 1e-6
            or abs(view.gamma - 1.0) > 1e-6 or not scene.render.use_compositing or node is None
            or node.node_tree is None or node.node_tree.get('rc_view_transform') != expected['view_transform']
            or abs(node.inputs['Exposure (stops)'].default_value - float(expected['exposure'])) > 1e-5
            or not math.isclose(node.node_tree.get('rc_limit', 0.0), record['limit'], rel_tol=1e-6)):
        return ['colour_management.coverage_view']
    return []


def check_atmosphere(scene):
    """Problems with a reopened atmosphere scene's coverage group ([] when it matches its record)."""
    from nwo.export import rcra_atmosphere_blender as R
    try:
        record = json.loads(scene['rcra_coverage'])
    except (KeyError, ValueError):
        return ['coverage.record']
    tree = scene.compositing_node_group
    node = tree.nodes.get(R.GROUP_NAME) if tree is not None else None
    if (record.get('kind') != 'atmosphere' or node is None or node.node_tree is None
            or node.node_tree.name != ATMOSPHERE_GROUP
            or not math.isclose(node.node_tree.get('rc_limit', 0.0), record['limit'], rel_tol=1e-6)):
        return ['coverage.atmosphere']
    return []
