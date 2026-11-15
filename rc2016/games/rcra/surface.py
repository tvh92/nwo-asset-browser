"""Rift Apart surface terms from a translated G-buffer writer (pass slot 4), and the forward colour of blended materials
(pass slot 1), for Blender and other renderers.

The writer ends in the engine's G-buffer packing, the same code in every graph. ``surface`` translates the writer with the
material bound (rc2016.games.rcra.shaders) and unpacks that code, using the decode of the engine's deferred apply program
(docs/rift-apart.md, "G-buffer and deferred lighting"):

    NormalSpec x, y   ftou(n.xy rsq(8 |n.z| + 8) 5791.205 + 2048), z sign in y bit 15  -> world normal n
    NormalSpec z      iadd(imul(Q(sqrt F0.r), 256), Q(sqrt F0.g)),  Q(x) = ftoi(255 x + 0.5)
    NormalSpec w      iadd(imul(Q(sqrt F0.b), 256), Q(gloss))
    DiffuseEmis rgb   albedo (an 8-bit UNORM target: clamped to [0, 1])
    DiffuseEmis a     occlusion when NormalSpec y bit 13 is set, else 0.2 log2(1 + min(E, 31)): emission E times albedo

F0 and gloss keep the game's 8-bit rounding (round(255 x) / 255, F0 squared after it); the 12-bit normal and the 8-bit
albedo, occlusion and emission are taken before their rounding. Everything the result still reads is a float expression
of interpolants, textures and named engine inputs (``PORTABLE_OPS``). Integer operations on constants and at most
MAX_CASE_CONDITIONS boolean conditions (a triplanar writer's dominant-axis index) are evaluated per case into a select of
constants, exactly; other integer operations that survive make the material unsupported, never approximated.

``forward`` translates the forward program of a blended material (rc2016.games.rcra.material.material_blend) under the
forward profile of rc2016.games.rcra.shaders: every light source zero, so its target is the material's own colour and
alpha. It reads the view through the inputs VIEW.xyz (camera-to-point offset) and CAMERA.xyz (camera position). A
program that samples the scene behind it (g_FrameBuffer) is kept when each colour channel is affine in that channel of the
background: colour = A + B x background, split into the terms colour (A) and background (B).
"""
import math
from collections import defaultdict

from rc2016.games.rcra import shaders as rshaders
from rc2016.shaders import sm5

# Float expression operations a node emitter must support (select takes a 'b' condition).
PORTABLE_OPS = {'const', 'input', 'channel', 'add', 'mul', 'div', 'min', 'max', 'neg', 'abs', 'sat', 'sqrt', 'rsq', 'rcp',
                'log2', 'exp2', 'frc', 'round_ne', 'round_ni', 'round_pi', 'round_z', 'sin', 'cos', 'select', 'eq', 'ne',
                'lt', 'ge', 'b_and', 'b_or', 'b_not', 'sample', 'ddx', 'ddy'}
# Interpolants a renderer supplies (engine vertex program layout, docs/rift-apart.md).
GEOMETRY_INPUTS = {'TEXCOORD.x': 'uv0.x', 'TEXCOORD.y': 'uv0.y', 'TEXCOORD1.x': 'uv1.x', 'TEXCOORD1.y': 'uv1.y',
                   'TEXCOORD2.x': 'normal.x', 'TEXCOORD2.y': 'normal.y', 'TEXCOORD2.z': 'normal.z',
                   'TEXCOORD3.x': 'tangent.x', 'TEXCOORD3.y': 'tangent.y', 'TEXCOORD3.z': 'tangent.z',
                   'TEXCOORD3.w': 'tangent.sign', 'TEXCOORD4.x': 'colour.r', 'TEXCOORD4.y': 'colour.g',
                   'TEXCOORD4.z': 'colour.b', 'TEXCOORD4.w': 'colour.a', 'SV_IsFrontFace': 'front_facing'}
# Per-draw and per-object values a material does not own, bound to the neutral draw: no dissolve, opaque, no render flags
# (class 0 lighting, occlusion mode decided by the material), instance 0, shader time 0. Viewer choices, labelled as such.
NEUTRAL_DRAW_FLOATS = {'TEXCOORD.z': 0.0, 'TEXCOORD.w': 1.0, 'TEXCOORD2.w': 0.25, 'GlobalWorldCBuffer.m_ShaderTimer': 0.0,
                       'GlobalWorldCBuffer.m_DeltaTime': 0.0, 'GlobalWorldCBuffer.m_TimeOfDay': 0.0,
                       'GlobalWorldCBuffer.m_MaterialLoDOffset': 0.0}
NEUTRAL_DRAW_DYNAMIC = {'MatDrawCB': 0}   # m_RenderFlags and the other per-draw words read through the instance index
NEUTRAL_DRAW_FLOATS.update(rshaders.NEUTRAL_OBJECT_INTERPOLANTS)   # the neutral object's custom values, as the VS passes them
SURFACE_TERMS = ('albedo.r', 'albedo.g', 'albedo.b', 'f0.r', 'f0.g', 'f0.b', 'gloss', 'normal.x', 'normal.y', 'normal.z',
                 'occlusion', 'emission', 'clip')
NORMAL_SCALE = 5791.205
# SM5 resource dimensions of a sample instruction (its extended opcode token).
DIMENSION_TEXTURE3D, DIMENSION_CUBE, DIMENSION_CUBE_ARRAY = 5, 6, 10
CUBE_DIMENSIONS = (DIMENSION_CUBE, DIMENSION_CUBE_ARRAY)
# Lookups of a cube map that a renderer can answer from the direction alone (the mip level or bias argument is not applied).
CUBE_OPCODES = ('sample', 'sample_l', 'sample_b', 'sample_d')
CUBE_NOTE = 'Cube maps (%s) are read at their top mip along the direction the program computes.'
CUBE_LOD_NOTE = " The explicit mip level, bias or gradients of its lookups (%s) are not applied."


class Unsupported(ValueError):
    pass


# Engine values a Blender material cannot supply, bound to review values (everything not listed is 0): adapted
# luminance 1 (as the viewer's RCRA_ADAPTED_LUMINANCE), the pixel centre of a 1920x1080 view for screen-space dither.
REVIEW_VALUES = {'GlobalWorldCBuffer.m_AdaptedLuminance': 1.0, 'SV_POSITION.x': 0.5, 'SV_POSITION.y': 0.5,
                 'SV_POSITION.z': 0.5, 'SV_POSITION.w': 1.0, 'GlobalViewportCBuffer.m_Dimensions': 1920.0,
                 'GlobalViewportCBuffer.m_Dimensions+4': 1080.0, 'GlobalViewportCBuffer.m_InvDimensions': 1 / 1920.0,
                 'GlobalViewportCBuffer.m_InvDimensions+4': 1 / 1080.0, 'GlobalViewportCBuffer.m_NearClip': 0.1,
                 'GlobalViewportCBuffer.m_FarClip': 1000.0}


class _Bindings(rshaders.NeutralEngine):
    """Material constants, the neutral engine and draw, and review values for every other non-geometry input."""

    def __init__(self, material, reflection=None, units=None, texture_sizes=None):
        super().__init__(material)
        self.review = {}
        self.object = rshaders.object_words(reflection)
        self.object_read = False
        self.units, self.texture_sizes = units or {}, texture_sizes
        self.sized = set()
        self.records = {}
        for buffer, members in rshaders.NEUTRAL_RECORDS.items():
            layout = reflection.constant_buffer(buffer) if reflection is not None else None
            if layout is not None and layout.variables and layout.variables[0].members:
                self.records[buffer] = {at: members[name] for name, at, _m in layout.variables[0].members if name in members}
        self.record_read = False
        self.engine_read = set()

    def texture_size(self, binding):
        """The bound material texture's size (``texture_sizes(path)``), for the programs that query it (resinfo)."""
        if binding in rshaders.SCENE_GBUFFER:
            self.engine_read.add(binding)
            return rshaders.SCENE_SIZE
        if binding not in self.units or self.texture_sizes is None:
            return None
        try:
            size = self.texture_sizes(self.units[binding])
        except (ValueError, KeyError, OSError):
            return None
        if size is not None:
            self.sized.add(binding)
        return size

    def resource(self, graph, op, binding, args):
        if binding == rshaders.VOLUME_FLAGS and op == 'ld':
            return [graph.const('i', 0)] * 4
        if binding in rshaders.SCENE_GBUFFER and op == 'ld':
            self.engine_read.add(binding)
            return [graph.const('i', rshaders.SCENE_GBUFFER[binding])] * 4
        if binding in rshaders.FORWARD_RESOURCES:   # an integer read (ld_structured) takes the same value's bits
            self.engine_read.add(binding)
            value = rshaders.FORWARD_RESOURCES[binding]
            kind = 'f' if op.startswith('sample') or binding == 'g_ViewDepthBuffer' else 'i'
            return [graph.const(kind, value if kind == 'f' else int(value))] * 4
        if binding in self.records and op == 'ld_structured':
            offset = graph.value(args[4]) if len(args) > 4 else None
            if offset is None or offset not in self.records[binding]:
                return None
            self.record_read = True
            words = self.records[binding]
            return [graph.const('i', sm5.float_bits(words[offset + 4 * c])) if offset + 4 * c in words
                    else graph.unknown('%s+%d' % (binding, offset + 4 * c), 'i') for c in range(4)]
        if binding == rshaders.OBJECT_BUFFER and op == 'ld_structured' and self.object is not None:
            offset = graph.value(args[4]) if len(args) > 4 else None
            if offset is None:
                return None
            self.object_read = True
            return [graph.const('i', self.object.get(offset + 4 * c, 0)) for c in range(4)]
        return None

    def _review(self, name):
        value = NEUTRAL_DRAW_FLOATS.get(name, REVIEW_VALUES.get(name, 0.0))
        if name not in NEUTRAL_DRAW_FLOATS:
            self.review[name] = value
        return self.graph.const('f', value)

    def input(self, name):
        if name == 'SV_IsFrontFace.x':   # read only as a truth value: a boolean input
            return self.graph.input('SV_IsFrontFace', 'b')
        if name in GEOMETRY_INPUTS:
            return None
        return self._review(name)

    def constant(self, name):
        bound = super().constant(name)
        return bound if bound is not None else self._review(name)

    def dynamic_constant(self, buffer, offset, index):
        if buffer in NEUTRAL_DRAW_DYNAMIC:
            return self.graph.const('i', NEUTRAL_DRAW_DYNAMIC[buffer])
        self.review['%s+%d[dynamic]' % (buffer, offset)] = 0.0
        return self.graph.const('i', 0)


SIZE_NOTE = ('Texture sizes the program queries (%s) are those of the full-resolution textures (every streamed mip resident); '
             'array sizes and depths do not depend on streaming.')
OBJECT_NOTE = (' Per-object values are a plain object: identity transform, alpha 1, white object colour, custom values, '
               'creation time and bias 0.')
ENGINE_NOTE = (' Engine textures read at the profile values: %s (stored scene depth at the far clip, so no soft depth fade; '
               'the scene G-buffer cleared to 0 at the review viewport size; light sources and probes 0).')
SCENE_GBUFFER_NOTE = ' The scene G-buffer reads 0 (cleared: the empty scene of the far-clip depth), at the review viewport size.'
RECORD_NOTE = ' No material LOD distance is set (material and subset records): full detail.'
FORWARD_FLAGS_NAME = 'MatDrawCB.m_RenderFlags'


class _ForwardBindings(_Bindings):
    """_Bindings with the forward profile: lit draw, light sources zero, engine textures and the view bound."""

    def input(self, name):
        if name in rshaders.FORWARD_VIEW_INPUTS:
            return self.graph.input(rshaders.FORWARD_VIEW_INPUTS[name], 'f')
        return super().input(name)

    def _view_constant(self, name):
        if name in rshaders.FORWARD_VIEW_INPUTS:
            return self.graph.input(rshaders.FORWARD_VIEW_INPUTS[name], 'f')
        if name in rshaders.FORWARD_VIEW_FLOATS:
            return self.graph.const('f', rshaders.FORWARD_VIEW_FLOATS[name])
        return super().constant(name)

    def __init__(self, material, draw_flags, reflection=None, units=None, texture_sizes=None):
        super().__init__(material, reflection, units, texture_sizes)
        self.background_read = False
        self.draw_flags = draw_flags

    def constant(self, name):
        if name == FORWARD_FLAGS_NAME:   # the draw index folds to 0 (TEXCOORD2.w bound), so the read is a plain constant
            return self.graph.const('f', sm5.bits_float(self.draw_flags))
        return self._view_constant(name)

    def resource(self, graph, op, binding, args):
        if binding == BACKGROUND_BUFFER and op.startswith('sample'):   # the scene behind: symbolic, split out afterwards
            self.background_read = True
            return [graph.input('BACKGROUND.' + c, 'f') for c in 'rgba']
        return super().resource(graph, op, binding, args)


class _EngineFreeBindings(_ForwardBindings):
    """_ForwardBindings with every value the forward profile zeroes left symbolic (inputs ENGINE.*): the engine constants
    that fall to 0 (key light, ambient, GI, cloud shadows and every other unbound one) and the samples of the lighting
    textures (probes, SSR, AO, BRDF lookup, shadows, gobos). The light-list counts stay 0. Used to prove a term does not
    depend on any of them."""

    def resource(self, graph, op, binding, args):
        if binding in rshaders.FORWARD_LIGHTING and op.startswith(('sample', 'gather')):
            return [graph.input('ENGINE.%s.%d' % (binding, c), 'f') for c in range(4)]
        return super().resource(graph, op, binding, args)

    def _review(self, name):
        if name in NEUTRAL_DRAW_FLOATS or name in REVIEW_VALUES:
            return super()._review(name)
        return self.graph.input('ENGINE.' + name, 'f')


class _Unpack:
    def __init__(self, graph):
        self.g = graph

    def node(self, i):
        return self.g.nodes[i]

    def value(self, i):
        return self.g.value(i)

    def strip_or(self, i, out):
        """Leaves of an or-tree (bit fields), constants dropped into ``out['bits']``."""
        n = self.node(i)
        if n['op'] == 'or':
            for a in n['args']:
                self.strip_or(a, out)
        elif n['op'] == 'const':
            out['bits'] |= n['value']
        else:
            out['parts'].append(i)
        return out

    def quantised(self, i):
        """x of ftoi(add(mul(x, 255), 0.5)) (or ftou), else None."""
        n = self.node(i)
        if n['op'] not in ('ftoi', 'ftou'):
            return None
        a = self.node(n['args'][0])
        if a['op'] != 'add' or self.value(a['args'][1]) != 0.5:
            return None
        m = self.node(a['args'][0])
        if m['op'] != 'mul':
            return None
        for x, k in ((0, 1), (1, 0)):
            if self.value(m['args'][k]) == 255.0:
                return m['args'][x]
        return None

    def byte_pair(self, i):
        """(high, low) float expressions of iadd(imul(Q(high), 256), Q(low)), pushing through selects."""
        n = self.node(i)
        if n['op'] == 'select':
            c, a, b = n['args']
            (ha, la), (hb, lb) = self.byte_pair(a), self.byte_pair(b)
            return self.g.select(c, ha, hb), self.g.select(c, la, lb)
        g = self.g
        if n['op'] == 'const' and n['type'] == 'i':   # a constant word: both bytes exact
            return g.const('f', (n['value'] >> 8 & 0xFF) / 255.0), g.const('f', (n['value'] & 0xFF) / 255.0)
        single = self.quantised(i)
        if single is not None:                       # only the low byte written
            return g.const('f', 0.0), single
        if n['op'] == 'iadd':
            for x, y in ((0, 1), (1, 0)):
                m = self.node(n['args'][x])
                if m['op'] == 'imul' and self.value(m['args'][1]) == 256:
                    high, low = self.quantised(m['args'][0]), self.byte(n['args'][y])
                    if high is not None and low is not None:
                        return high, low
                if m['op'] == 'const' and m['type'] == 'i' and m['value'] & 0xFF == 0:
                    low = self.quantised(n['args'][y])
                    if low is not None:
                        return g.const('f', (m['value'] >> 8 & 0xFF) / 255.0), low
        raise Unsupported('G-buffer byte pair is not the engine packing')

    def byte(self, i):
        """x of Q(x), or a constant byte as x = byte / 255."""
        n = self.node(i)
        if n['op'] == 'const' and n['type'] == 'i' and n['value'] < 256:
            return self.g.const('f', n['value'] / 255.0)
        return self.quantised(i)

    def normal_component(self, i):
        """(component, z) of ftou(add(mul(mul(rsq(add(mul(abs(z), 8), 8)), component), 5791.205), 2048))."""
        n = self.node(i)
        if n['op'] == 'select':
            c, a, b = n['args']
            (ca, za), (cb, zb) = self.normal_component(a), self.normal_component(b)
            return self.g.select(c, ca, cb), self.g.select(c, za, zb)
        if n['op'] != 'ftou':
            raise Unsupported('G-buffer normal is not the engine packing')
        a = self.node(n['args'][0])
        if a['op'] != 'add' or self.value(a['args'][1]) != 2048.0:
            raise Unsupported('G-buffer normal offset')
        m = self.node(a['args'][0])
        if m['op'] != 'mul' or abs((self.value(m['args'][1]) or 0) - NORMAL_SCALE) > 0.01:
            raise Unsupported('G-buffer normal scale')
        p = self.node(m['args'][0])
        if p['op'] != 'mul':
            raise Unsupported('G-buffer normal product')
        for r, c in ((0, 1), (1, 0)):
            q = self.node(p['args'][r])
            if q['op'] == 'rsq':
                s = self.node(q['args'][0])
                if s['op'] == 'add' and self.value(s['args'][1]) == 8.0:
                    t = self.node(s['args'][0])
                    if t['op'] == 'mul' and self.value(t['args'][1]) == 8.0 and self.node(t['args'][0])['op'] == 'abs':
                        return p['args'][c], self.node(t['args'][0])['args'][0]
        raise Unsupported('G-buffer normal weight')

    def normal_word(self, i):
        """(component, z, bit 13 condition) of a NormalSpec x or y word. A word chosen by a select whose cases are each a
        packed word (snow layers write another lighting class) is unpacked per case and joined by the same select."""
        fields = self.strip_or(i, dict(bits=0, parts=[]))
        packed, other = [], []
        for part in fields['parts']:
            (packed if self.node(part)['op'] in ('ftou', 'select') and self._is_normal(part) else other).append(part)
        if not packed and fields['bits'] == 0 and len(other) == 1 and self.node(other[0])['op'] == 'select':
            c, a, b = self.node(other[0])['args']
            (ca, za, ba), (cb, zb, bb) = self.normal_word(a), self.normal_word(b)
            g = self.g
            return g.select(c, ca, cb), g.select(c, za, zb), g.select(g.boolean(c), ba, bb)
        if len(packed) != 1:
            raise Unsupported('G-buffer normal word has %d packed normals' % len(packed))
        component, z = self.normal_component(packed[0])
        return component, z, self.bit13(other, fields['bits'])

    def _is_normal(self, i):
        try:
            self.normal_component(i)
            return True
        except Unsupported:
            return False

    def bit13(self, parts, bits):
        """The condition of NormalSpec y bit 13 (occlusion mode): constant, or bfi(1, 13, b2i(cond), 0)."""
        if bits & 0x2000:
            return self.g.const('b', True)
        for part in parts:
            n = self.node(part)
            if n['op'] == 'bfi' and [self.value(a) for a in n['args'][:2]] == [1, 13] and self.value(n['args'][3]) == 0:
                return self.g.boolean(n['args'][2])
        return self.g.const('b', False)

    def emission(self, i):
        """E of mul(log2(add(min(E, 31), 1)), 0.2), or the decoded clamp(Y, 1, 32) - 1 of another mul(log2(Y), 0.2), pushing
        through selects; None when not that form."""
        g = self.g
        n = self.node(i)
        if n['op'] == 'select':
            c, a, b = n['args']
            ea, eb = self.emission(a), self.emission(b)
            if ea is None or eb is None:
                return None
            return g.select(c, ea, eb)
        if n['op'] == 'const':   # a constant alpha decodes exactly: E = 2^(5 a) - 1
            return g.const('f', 2.0 ** (5.0 * n['value']) - 1.0)
        if n['op'] != 'mul' or abs((self.value(n['args'][1]) or 0) - 0.2) > 1e-6:
            return None
        lg = self.node(n['args'][0])
        if lg['op'] != 'log2':
            return None
        ad = self.node(lg['args'][0])
        if ad['op'] == 'add' and self.value(ad['args'][1]) == 1.0:
            mn = self.node(ad['args'][0])
            if mn['op'] == 'min' and self.value(mn['args'][1]) == 31.0:
                return mn['args'][0]
        # The compiler folds the encoding (min(E, 31) dropped when E is bounded, the 1 merged into other constants): the
        # deferred apply decodes 2^(5a) - 1 of the target's [0, 1] alpha, which is clamp(Y, 1, 32) - 1 of a = 0.2 log2 Y.
        g = self.g
        y = g.op('min', g.op('max', lg['args'][0], g.const('f', 1.0)), g.const('f', 32.0))
        return g.op('add', y, g.const('f', -1.0))


def _rounded(g, x):
    """round(255 x) / 255 as the G-buffer stores x (x >= 0 here)."""
    return g.op('mul', g.op('round_ni', g.op('add', g.op('mul', x, g.const('f', 255.0)), g.const('f', 0.5))),
                g.const('f', 1.0 / 255.0))


def _split_select(g, i, condition):
    """(when condition, otherwise) branches of a select on ``condition`` (or its negation), else (i, i)."""
    n = g.nodes[i]
    if n['op'] == 'select':
        c, a, b = n['args']
        if c == condition:
            return a, b
        if g.nodes[c]['op'] == 'b_not' and g.nodes[c]['args'][0] == condition:
            return b, a
    return i, i


def _run(translator):
    """The translation; a program too deep for Python's recursion (very long unrolled chains) is unsupported."""
    try:
        return translator.run()
    except RecursionError:
        raise Unsupported('Program is too deep to translate (recursion limit)') from None


class _ViewBindings(_Bindings):
    """_Bindings with the forward profile's view (VIEW and CAMERA inputs, FORWARD_VIEW_FLOATS), for writers that read the
    view direction."""

    def input(self, name):
        if name in rshaders.FORWARD_VIEW_INPUTS:
            return self.graph.input(rshaders.FORWARD_VIEW_INPUTS[name], 'f')
        return super().input(name)

    def constant(self, name):
        if name in rshaders.FORWARD_VIEW_INPUTS:
            return self.graph.input(rshaders.FORWARD_VIEW_INPUTS[name], 'f')
        if name in rshaders.FORWARD_VIEW_FLOATS:
            return self.graph.const('f', rshaders.FORWARD_VIEW_FLOATS[name])
        return super().constant(name)


VIEW_NOTE = ('The writer reads the view direction, which the review screen position leaves undefined (the camera at the '
             'point: a normalised zero vector, NaN). It is translated with the view of the forward profile instead: the camera-to-'
             'point offset and camera position are inputs (VIEW, CAMERA), and screen coordinates are that offset.')


NAN_NOTE = ('Under the neutral values some terms are undefined (NaN): the program divides by a per-object or per-draw value '
            'the game sets at run time (an animation frame count, a random seed), which the neutral object holds at 0. '
            'Renderers resolve the NaN by D3D rules (blender_recipe); a texture read at an undefined address is undefined.')


def _reads_nan(result):
    """True when a term of a translation depends on a NaN constant."""
    g = result['graph']
    return any(g.nodes[i]['op'] == 'const' and isinstance(g.nodes[i]['value'], float) and g.nodes[i]['value'] != g.nodes[i]['value']
               for i in g.reachable(list(result['terms'].values())))


def surface(graph_data, material_data, slot=rshaders.SLOT_GBUFFER, texture_sizes=None):
    """The material's surface terms as one expression DAG. ``texture_sizes(path)`` gives a texture's (width, height, depth or
    array size, mip count) for programs that query sizes (rc2016.games.rcra.texture.texture_size); without it they fail.

    A writer whose terms depend on NaN under the review screen position (it reads the view direction) is translated again
    with the forward profile's view (_ViewBindings, VIEW_NOTE), and that translation is used when it has no NaN.

    Returns dict(graph, terms {name: node}, textures {binding: path}, inputs [names], limitations [text]).
    Raises Unsupported when the writer is not the engine packing or keeps non-portable operations."""
    result = _surface(graph_data, material_data, slot, texture_sizes, _Bindings)
    if not _reads_nan(result):
        return result
    try:
        viewed = _surface(graph_data, material_data, slot, texture_sizes, _ViewBindings)
    except Unsupported:
        viewed = None
    if viewed is not None and not _reads_nan(viewed):
        viewed['limitations'].append(VIEW_NOTE)
        return viewed
    result['limitations'].append(NAN_NOTE)
    return result


def _surface(graph_data, material_data, slot, texture_sizes, binding_class):
    shader = rshaders.pass_shader(graph_data, slot)
    cb2 = rshaders.material_constants(graph_data, material_data)
    bindings = binding_class(cb2, shader.reflection, rshaders.material_units(graph_data, material_data), texture_sizes)
    translator = sm5.Translator(shader, bindings)
    bindings.graph = translator.g
    tr = _run(translator)
    g = tr.graph
    out = lambda r, c: tr.outputs.get(('o', r, c))   # noqa: E731
    if any(out(r, c) is None for r, c in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3))):
        raise Unsupported('Writer does not write the NormalSpec and DiffuseEmis targets')
    unknown = g.unknowns([out(r, c) for r in (0, 1) for c in range(4)] + [tr.discard])
    if unknown:
        raise Unsupported('Writer depends on unknown values: ' + ', '.join(unknown[:3]))
    u = _Unpack(g)
    for key in [k for k in tr.outputs if k[0] == 'o' and k[1] == 1]:
        tr.outputs[key] = g.cast(tr.outputs[key], 'f')   # an 8-bit UNORM target: integer-typed words hold float bits
    nx, nz, _x13 = u.normal_word(out(0, 0))
    ny, nz_y, occlusion_mode = u.normal_word(out(0, 1))
    if nz != nz_y:
        raise Unsupported('Normal words disagree on z')
    f0r, f0g = u.byte_pair(out(0, 2))
    f0b, gloss = u.byte_pair(out(0, 3))
    one, zero = g.const('f', 1.0), g.const('f', 0.0)
    occlusion_branch, emission_branch = _split_select(g, out(1, 3), occlusion_mode)
    emission = u.emission(emission_branch)
    mode = g.value(occlusion_mode)
    if mode is True:
        occlusion, emission = out(1, 3), zero
    elif mode is False:
        if emission is None:
            raise Unsupported('Emission alpha is not the engine encoding')
        occlusion = one
    else:
        if emission is None:
            raise Unsupported('Emission alpha is not the engine encoding')
        occlusion = g.select(occlusion_mode, occlusion_branch, one)
        emission = g.select(occlusion_mode, zero, emission)
    terms = {
        'albedo.r': g.op('sat', out(1, 0)), 'albedo.g': g.op('sat', out(1, 1)), 'albedo.b': g.op('sat', out(1, 2)),
        'normal.x': nx, 'normal.y': ny, 'normal.z': nz,
        'occlusion': g.op('sat', occlusion), 'emission': g.op('max', g.op('min', emission, g.const('f', 31.0)), zero),
        'clip': tr.discard,
    }
    for name, byte in (('f0.r', f0r), ('f0.g', f0g), ('f0.b', f0b)):
        stored = _rounded(g, g.op('sat', byte))
        terms[name] = g.op('mul', stored, stored)
    terms['gloss'] = _rounded(g, g.op('sat', gloss))
    limitations = ['Per-draw values are the neutral draw: no dissolve, opaque, no render flags (lighting class 0), '
                   'no flag volumes, shader time 0.' + OBJECT_NOTE * bindings.object_read + RECORD_NOTE * bindings.record_read
                   + (ENGINE_NOTE % ', '.join(sorted(bindings.engine_read)) if bindings.engine_read else ''), 'The 12-bit normal and the 8-bit albedo, occlusion and emission are used before the '
                   "G-buffer's rounding; F0 and gloss keep it."]
    return _portable(g, terms, bindings, graph_data, material_data, tr, limitations, 'Surface')


MAX_CASE_CONDITIONS = 4
_NOT_CASE_OPS = {'input', 'unknown', 'load', 'sample', 'channel', 'ddx', 'ddy'}


def _case_conditions(g, root):
    """The boolean conditions an integer expression depends on, when it reads only constants and those conditions
    through non-portable operations and selects; None when it reads anything else or more than MAX_CASE_CONDITIONS."""
    conditions, seen, stack = [], set(), [root]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        n = g.nodes[i]
        if n['op'] == 'const':
            continue
        if i != root and n['type'] == 'b' and n['op'] in PORTABLE_OPS:
            conditions.append(i)
            if len(conditions) > MAX_CASE_CONDITIONS:
                return None
        elif n['op'] == 'select' or (n['op'] not in PORTABLE_OPS and n['op'] not in _NOT_CASE_OPS):
            stack.extend(n['args'])
        else:
            return None
    return conditions


def _case_value(g, root, fixed):
    """The value of ``root`` with each condition fixed (as sm5.evaluate computes it)."""
    values = dict(fixed)

    def visit(i):
        if i in values:
            return values[i]
        n = g.nodes[i]
        if n['op'] == 'const':
            v = n['value']
        elif n['op'] == 'select':
            v = visit(n['args'][1]) if visit(n['args'][0]) else visit(n['args'][2])
        else:
            result, _types, fold = sm5.OPS[n['op']]
            v = fold([visit(a) for a in n['args']])
            if result == 'f':
                v = sm5.f32(v)
        values[i] = v
        return v
    return visit(root)


def _enumerate_cases(g, terms):
    """``terms`` with each topmost non-portable integer expression of a few conditions replaced by a select of its value
    in every case (exact); unchanged when there is none."""
    roots = list(terms.values())
    reach = g.reachable(roots)
    users = defaultdict(list)
    for i in reach:
        for a in g.nodes[i]['args']:
            users[a].append(i)
    replace = {}
    for i in sorted(reach):
        n = g.nodes[i]
        if n['op'] in PORTABLE_OPS or n['op'] in _NOT_CASE_OPS:
            continue
        if users[i] and all(g.nodes[u]['op'] not in PORTABLE_OPS for u in users[i]):
            continue   # inside a larger non-portable expression: the topmost one is replaced
        conditions = _case_conditions(g, i)
        if conditions is None:
            continue

        def build(k, fixed):
            if k == len(conditions):
                return g.const(n['type'], _case_value(g, i, fixed))
            c = conditions[k]
            return g.select(c, build(k + 1, {**fixed, c: True}), build(k + 1, {**fixed, c: False}))
        try:
            replace[i] = build(0, {})
        except (ValueError, OverflowError, ZeroDivisionError):
            continue
    return _substitute(g, terms, replace)


def _substitute(g, terms, replace):
    """``terms`` with the nodes in ``replace`` ({node: new node}) swapped in; consumers are rebuilt with their op and meta."""
    if not replace:
        return terms
    roots = list(terms.values())
    order, seen = [], set()
    for r in roots:   # post-order without recursion: unrolled loops leave deep chains
        stack = [(r, False)]
        while stack:
            i, ready = stack.pop()
            if ready:
                order.append(i)
            elif i not in seen:
                seen.add(i)
                stack.append((i, True))
                if i not in replace:
                    stack.extend((a, False) for a in g.nodes[i]['args'] if a not in seen)
    new = {}
    for i in order:
        if i in replace:
            new[i] = replace[i]
            continue
        n = g.nodes[i]
        args = [new.get(a, a) for a in n['args']]
        if args == n['args']:
            new[i] = i
        else:
            meta = {k: v for k, v in n.items() if k not in ('op', 'type', 'args')}
            new[i] = g._add(n['op'], n['type'], args, **meta)
    return {name: new[i] for name, i in terms.items()}


INT_FLOAT_LIMIT = 2.0 ** 24
INT_FLOAT_ROUNDS = 12   # each round rewrites one level of nesting (a case index tested inside another case's test)
INT_FLOAT_NOTE = ('Integer arithmetic on values derived from floats (cell indices, hashes) is carried in float: exact while '
                  'every intermediate value stays within +-2^24, as texture coordinates and positions do.')
PACKED_NOTE = ('Bit fields the program packs into a word and reads back are each taken as their own value: exact while '
               'every field fits below the next field of the word, as the packing scales are set for.')


def _integers_in_float(g, terms):
    """``terms`` with each topmost non-portable integer expression of float values, constants and conditions rewritten as
    float arithmetic on the same integers (exact below INT_FLOAT_LIMIT in magnitude); (terms, rewritten, packed), packed
    when a field of a word the program packed itself was read back (PACKED_NOTE)."""
    memo, folded = {}, {}
    packed = [False]

    def mod(a, n):   # floored modulo by a power of two, as two's-complement masking gives
        q = g.op('round_ni', g.op('mul', a, g.const('f', 1.0 / n)))
        return g.op('add', a, g.op('mul', q, g.const('f', -float(n))))

    def power(v):
        return v is not None and v > 0 and v & (v - 1) == 0

    def fold(i, fn):   # fn over each case of a select tree of integer constants, as a float select tree; else None
        key = (i, fn)
        if key not in folded:
            n = g.nodes[i]
            out = None
            if n['op'] == 'const' and n['type'] == 'i':
                out = g.const('f', float(fn(n['value'] & 0xFFFFFFFF)))
            elif n['op'] == 'select' and n['type'] == 'i':
                a, b = fold(n['args'][1], fn), fold(n['args'][2], fn)
                out = None if a is None or b is None else g.select(n['args'][0], a, b)
            folded[key] = out
        return folded[key]

    def fields(i, out):   # the parts of an or-tree word: (shift, source, width or None), constant bits apart
        n = g.nodes[i]
        k = [g.value(a) for a in n['args']]
        if n['op'] == 'or':
            for a in n['args']:
                fields(a, out)
        elif n['op'] == 'const' and n['type'] == 'i':
            out['bits'] |= n['value'] & 0xFFFFFFFF
        elif n['op'] == 'ishl' and k[1] is not None:
            out['parts'].append((k[1] & 31, n['args'][0], None))
        elif n['op'] == 'bfi' and None not in k[:2] and k[3] == 0:
            out['parts'].append((k[1] & 31, n['args'][2], k[0] & 31))
        else:
            out['parts'].append((0, i, None))
        return out

    def bits(i, offset, width):   # bits [offset, offset + width) of one integer: floor(x / 2^offset) mod 2^width, exact
        if offset + width > 32 or width > 24:
            return None
        a = f(i)
        if a is None:
            return None
        if offset:
            a = g.op('round_ni', g.op('mul', a, g.const('f', 1.0 / 2 ** offset)))
        return mod(a, 2 ** width)

    def extract(word, offset, width):   # bits [offset, offset + width) of a word the program packed (or-tree, bfi)
        n = g.nodes[word]
        k = [g.value(a) for a in n['args']]
        if n['op'] == 'bfi' and None not in k[:2]:
            w, o = k[0] & 31, k[1] & 31
            if offset + width <= o or offset >= o + w:   # outside the inserted field: the base word's bits
                return extract(n['args'][3], offset, width)
            if o <= offset and offset + width <= o + w:
                return bits(n['args'][2], offset - o, width)
            return None
        if n['op'] != 'or':
            return bits(word, offset, width)
        parts = fields(word, dict(bits=0, parts=[]))
        shifts = sorted(s for s, _x, _w in parts['parts'])
        if len(set(shifts)) != len(shifts):
            return None
        at = [p for p in parts['parts'] if p[0] == offset]
        if any(offset < s < offset + width for s in shifts) or len(at) > 1:
            return None
        constant = parts['bits'] >> offset & ((1 << width) - 1)
        if not at:
            return g.const('f', float(constant))
        shift, source, slot = at[0]
        above = [s for s in shifts if s > offset] + [32]
        size = min(width, above[0] - offset, slot or 32)
        if constant & ((1 << size) - 1):
            return None
        a = f(source)
        if a is None:
            return None
        packed[0] = True
        a = mod(a, 2 ** size)
        return a if not constant else g.op('add', a, g.const('f', float(constant)))

    def f(i):
        if i in memo:
            return memo[i]
        n = g.nodes[i]
        op, args = n['op'], n['args']
        k = [g.value(a) for a in args]
        out = None
        if op == 'const' and n['type'] == 'i':
            out = g.const('f', float(sm5._signed(n['value'])))
        elif op == 'select' and n['type'] == 'i':
            a, b = f(args[1]), f(args[2])
            out = None if a is None or b is None else g.select(args[0], a, b)
        elif op == 'ftoi':
            out = g.op('round_z', args[0])
        elif op == 'ftou':
            out = g.op('round_z', g.op('max', args[0], g.const('f', 0.0)))
        elif op == 'b2i':
            out = g.select(args[0], g.const('f', -1.0), g.const('f', 0.0))
        elif op in ('iadd', 'imul', 'imin', 'imax', 'ineg'):
            a = [f(x) for x in args]
            if None not in a:
                out = g.op({'iadd': 'add', 'imul': 'mul', 'imin': 'min', 'imax': 'max', 'ineg': 'neg'}[op], *a)
        elif op == 'urem' and k[1]:
            out = fold(args[0], lambda v, m=k[1]: v % m)
            if out is None and power(k[1]):
                a = f(args[0])
                out = None if a is None else mod(a, k[1])
        elif op == 'and' and k[1] is not None and power(k[1] + 1):
            out = extract(args[0], 0, (k[1] + 1).bit_length() - 1)
            if out is None:
                a = f(args[0])
                out = None if a is None else mod(a, k[1] + 1)
        elif op == 'ubfe' and None not in k[:2]:
            out = extract(args[2], k[1] & 31, k[0] & 31)
        elif op == 'ushr' and k[1] is not None:
            out = extract(args[0], k[1] & 31, 32 - (k[1] & 31))
        elif op in ('ishl', 'ishr') and k[1] is not None:
            a = f(args[0])
            scale = g.const('f', float(2 ** (k[1] & 31)))
            if a is not None:
                out = g.op('mul', a, scale) if op == 'ishl' else g.op('round_ni', g.op('div', a, scale))
        memo[i] = out
        return out

    def top(i):
        n = g.nodes[i]
        op, args = n['op'], n['args']
        if op in ('itof', 'utof'):
            return f(args[0])
        if op in ('ieq', 'ine', 'ilt', 'ige'):
            a, b = f(args[0]), f(args[1])
            if a is None or b is None:
                return None
            return g.op({'ieq': 'eq', 'ine': 'ne', 'ilt': 'lt', 'ige': 'ge'}[op], a, b)
        return None

    roots = list(terms.values())
    reach = g.reachable(roots)
    replace = {}
    for i in sorted(reach):
        if g.nodes[i]['op'] in PORTABLE_OPS:
            continue
        try:
            new = top(i)
        except RecursionError:
            new = None
        if new is not None:
            replace[i] = new
    return _substitute(g, terms, replace), bool(replace), packed[0]


def _portable(g, terms, bindings, graph_data, material_data, tr, limitations, what):
    """The result dict, after checking that every term is a portable float expression of geometry and material textures."""
    roots = list(terms.values())
    reach = g.reachable(roots)
    bad = sorted({g.nodes[i]['op'] for i in reach} - PORTABLE_OPS)
    in_float = in_packed = False
    if bad:
        terms = _enumerate_cases(g, terms)
        roots = list(terms.values())
        reach = g.reachable(roots)
        bad = sorted({g.nodes[i]['op'] for i in reach} - PORTABLE_OPS)
    if bad:
        candidate, in_float = terms, False
        for _round in range(INT_FLOAT_ROUNDS):   # replacements may hold other non-portable nodes (a condition, a cast)
            candidate, changed, packed = _integers_in_float(g, candidate)
            in_float = in_float or changed
            in_packed = in_packed or packed
            if not changed:
                break
        if in_float:
            roots = list(candidate.values())
            reach = g.reachable(roots)
            if not {g.nodes[i]['op'] for i in reach} - PORTABLE_OPS:
                terms, bad = candidate, []
            else:
                in_float = in_packed = False
                roots = list(terms.values())
                reach = g.reachable(roots)
    if bad:
        raise Unsupported('Non-portable operations remain: ' + ', '.join(bad))
    if g.unknowns(roots):
        raise Unsupported('%s depends on unknown values' % what)
    textures = {}
    units = rshaders.material_units(graph_data, material_data)
    if bindings.sized:
        limitations = list(limitations) + [SIZE_NOTE % ', '.join(sorted(bindings.sized))]
    cubes, cube_opcodes = set(), set()
    for i in reach:
        n = g.nodes[i]
        if n['op'] == 'sample':
            if n['binding'] not in units:
                raise Unsupported('%s samples engine texture %s' % (what, n['binding']))
            if n.get('dimension') in CUBE_DIMENSIONS:
                if n.get('opcode') not in CUBE_OPCODES:
                    raise Unsupported('%s reads cube map %s with %s' % (what, n['binding'], n.get('opcode')))
                if n['dimension'] == DIMENSION_CUBE_ARRAY and g.value(n['args'][3]) is None:
                    raise Unsupported('%s samples cube array %s at a run-time index' % (what, n['binding']))
                cubes.add(n['binding'])
                if n['opcode'] != 'sample':
                    cube_opcodes.add(n['opcode'])
            textures[n['binding']] = units[n['binding']]
        if n['op'] == 'load':
            raise Unsupported('%s loads from %s' % (what, n['binding']))
    inputs = sorted({g.nodes[i]['name'] for i in reach if g.nodes[i]['op'] == 'input'})
    engine = sorted(bindings.review)
    limitations = list(limitations) + [INT_FLOAT_NOTE] * in_float + [PACKED_NOTE] * in_packed
    if cubes:
        limitations.append(CUBE_NOTE % ', '.join(sorted(cubes)) + (CUBE_LOD_NOTE % ', '.join(sorted(cube_opcodes))
                                                                   if cube_opcodes else ''))
    if engine:
        limitations.append('Engine inputs at review values (translation-time, may not all reach the %s): ' % what.lower() +
                           ', '.join('%s = %g' % (name, bindings.review[name]) for name in engine))
    return dict(graph=g, terms=terms, textures=textures, inputs=inputs, limitations=limitations,
                program_status=tr.status(), outputs=dict(tr.outputs))


FORWARD_TERMS = ('colour.r', 'colour.g', 'colour.b', 'background.r', 'background.g', 'background.b', 'alpha', 'clip')
BACKGROUND_BUFFER = 'g_FrameBuffer'
BACKGROUND_INPUTS = ('BACKGROUND.r', 'BACKGROUND.g', 'BACKGROUND.b', 'BACKGROUND.a')
LIT_BACKGROUND_NOTE = (' The material is lit, but the tint reads no light or other engine value, so the scene '
                       'passes through it unlit, as in the game.')


def _affine_split(g, root, names):
    """(A, {name: B}) with root = A + sum(B x name) exactly, A and B free of ``names``; None when root is not affine in them
    (they pass through anything but add, neg, a multiply or divide by a free value, or a select on a free condition)."""
    depends = {}

    def free(i):
        if i not in depends:
            n = g.nodes[i]
            depends[i] = (n['op'] == 'input' and n['name'] in names) or any(not free(a) for a in n['args'])
        return not depends[i]

    memo = {}
    zero = g.const('f', 0.0)

    def split(i):
        if i in memo:
            return memo[i]
        n = g.nodes[i]
        op, args = n['op'], n['args']
        if free(i):
            result = (i, {})
        elif op == 'input':
            result = (zero, {n['name']: g.const('f', 1.0)})
        elif op == 'add':
            a, b = split(args[0]), split(args[1])
            if a is None or b is None:
                return None
            coeffs = dict(a[1])
            for k, c in b[1].items():
                coeffs[k] = g.op('add', coeffs[k], c) if k in coeffs else c
            result = (g.op('add', a[0], b[0]), coeffs)
        elif op == 'neg':
            a = split(args[0])
            if a is None:
                return None
            result = (g.op('neg', a[0]), {k: g.op('neg', c) for k, c in a[1].items()})
        elif op in ('mul', 'div') and free(args[1]):
            a = split(args[0])
            if a is None:
                return None
            result = (g.op(op, a[0], args[1]), {k: g.op(op, c, args[1]) for k, c in a[1].items()})
        elif op == 'mul' and free(args[0]):
            a = split(args[1])
            if a is None:
                return None
            result = (g.op('mul', args[0], a[0]), {k: g.op('mul', args[0], c) for k, c in a[1].items()})
        elif op == 'select' and free(args[0]):
            a, b = split(args[1]), split(args[2])
            if a is None or b is None:
                return None
            keys = set(a[1]) | set(b[1])
            result = (g.select(args[0], a[0], b[0]),
                      {k: g.select(args[0], a[1].get(k, zero), b[1].get(k, zero)) for k in keys})
        else:
            return None
        memo[i] = result
        return result
    return split(root)


def _check_unlit_background(shader, graph_data, material_data, flags, texture_sizes):
    """Raise Unsupported unless a lit forward program passes the scene behind it through unlit: translated again with
    every value the profile zeroes symbolic (_EngineFreeBindings), each colour channel must be affine in its own
    background channel with a coefficient that reads none of them. The profile's coefficient is then the game's, whatever
    the lights. A program that lights the scene behind it (uses it as albedo) gets the light in its coefficient."""
    bindings = _EngineFreeBindings(rshaders.material_constants(graph_data, material_data), flags, shader.reflection,
                                   rshaders.material_units(graph_data, material_data), texture_sizes)
    translator = sm5.Translator(shader, bindings)
    bindings.graph = translator.g
    tr = _run(translator)
    g = tr.graph
    engine = {i for i, n in enumerate(g.nodes) if n['op'] == 'input' and n['name'].startswith('ENGINE.')}
    coefficients = []
    for c in range(3):
        colour = tr.outputs.get(('o', 0, c))
        split = _affine_split(g, g.cast(colour, 'f'), BACKGROUND_INPUTS) if colour is not None else None
        if split is None or set(split[1]) - {BACKGROUND_INPUTS[c]}:
            raise Unsupported('Forward colour is not affine in the scene behind (%s)' % BACKGROUND_BUFFER)
        coefficients += split[1].values()
    if engine & g.reachable(coefficients):
        raise Unsupported('Forward colour lights the scene behind it (%s) in a lit material: its background '
                          'coefficient reads the forward lighting' % BACKGROUND_BUFFER)


def forward(graph_data, material_data, texture_sizes=None):
    """The forward program's target under the forward profile: colour (unclamped, HDR), alpha and the discard.

    Returns the same dict as ``surface`` with terms FORWARD_TERMS. Raises Unsupported when the target depends on anything
    but geometry, the view inputs and the material's own textures and constants."""
    shader = rshaders.pass_shader(graph_data, rshaders.SLOT_FORWARD)
    from rc2016.games.rcra.material import draw_flags
    flags = draw_flags(material_data)
    bindings = _ForwardBindings(rshaders.material_constants(graph_data, material_data), flags, shader.reflection,
                                rshaders.material_units(graph_data, material_data), texture_sizes)
    translator = sm5.Translator(shader, bindings)
    bindings.graph = translator.g
    tr = _run(translator)
    g = tr.graph
    target = [tr.outputs.get(('o', 0, c)) for c in range(4)]
    if None in target:
        raise Unsupported('Forward program does not write a colour and alpha target')
    target = [g.cast(x, 'f') for x in target]
    terms = {'colour.r': target[0], 'colour.g': target[1], 'colour.b': target[2], 'alpha': g.op('sat', target[3]),
             'clip': tr.discard}
    zero = g.const('f', 0.0)
    background = dict.fromkeys(('background.r', 'background.g', 'background.b'), zero)
    if bindings.background_read:
        reach = g.reachable(list(terms.values()))
        if any(g.nodes[i]['op'] == 'input' and g.nodes[i]['name'] in BACKGROUND_INPUTS for i in reach):
            if flags:
                _check_unlit_background(shader, graph_data, material_data, flags, texture_sizes)
            for name in ('alpha', 'clip'):
                if not _affine_split(g, terms[name], BACKGROUND_INPUTS) or _affine_split(g, terms[name],
                                                                                          BACKGROUND_INPUTS)[1]:
                    raise Unsupported('Forward %s depends on the scene behind (%s)' % (name, BACKGROUND_BUFFER))
            for c in 'rgb':
                split = _affine_split(g, terms['colour.' + c], BACKGROUND_INPUTS)
                if split is None or set(split[1]) - {'BACKGROUND.' + c}:
                    raise Unsupported('Forward colour is not affine in the scene behind (%s)' % BACKGROUND_BUFFER)
                terms['colour.' + c] = split[0]
                background['background.' + c] = split[1].get('BACKGROUND.' + c, zero)
    terms.update(background)
    limitations = [
        'Forward colour with every engine light source zero (key light, light grid, probes, GI, reflections): the '
        "material's own emission and effects. The renderer's lights shade its G-buffer surface instead of the forward "
        "program's lighting.",
        "Draw: render flags hold only the material's lit bit (%s), no dissolve, no particle values, no flag volumes, "
        'shader time 0; stored scene depth at the far clip (no soft depth fade); view space with the world axes.'
        % ('lit' if flags else 'unlit: lighting term 1') + OBJECT_NOTE * bindings.object_read
        + RECORD_NOTE * bindings.record_read
        + (SCENE_GBUFFER_NOTE if set(rshaders.SCENE_GBUFFER) & bindings.engine_read else '')]
    if bindings.background_read:
        limitations.append('The program samples the scene behind it (%s); the colour is affine in it, so it is split into '
                           'an emission and a background tint, and the sample\'s screen-space offset (distortion, '
                           'refraction) is dropped.' % BACKGROUND_BUFFER + LIT_BACKGROUND_NOTE * bool(flags))
    return _portable(g, terms, bindings, graph_data, material_data, tr, limitations, 'Forward colour')


def export_recipe(result):
    """A JSON-ready recipe: the reachable nodes renumbered, the terms, the textures and the limitations."""
    g, terms = result['graph'], result['terms']
    order = sorted(g.reachable(list(terms.values())))
    index = {old: new for new, old in enumerate(order)}
    nodes = []
    for old in order:
        n = dict(g.nodes[old])
        n['args'] = [index[a] for a in n['args']]
        for key in ('offsets',):
            if key in n:
                n[key] = list(n[key])
        nodes.append(n)
    return dict(version=1, nodes=nodes, terms={k: index[v] for k, v in terms.items()}, textures=result['textures'],
                inputs=result['inputs'], limitations=result['limitations'])


LOOKUP_MIN = 16      # shorter select chains stay selects
LOOKUP_SIZE = 4096   # at most this many table entries (key span)


def _lookup_link(nodes, i):
    """(X, key, value, rest) of select(eq(X, constant key), constant value, rest), else None."""
    n = nodes[i]
    if n['op'] != 'select' or n.get('type') == 'b':
        return None
    c = nodes[n['args'][0]]
    if c['op'] != 'eq':
        return None
    x, k = c['args']
    if nodes[k]['op'] != 'const':
        x, k = k, x
    value = nodes[n['args'][1]]
    if nodes[k]['op'] != 'const' or nodes[x]['op'] == 'const' or value['op'] != 'const':
        return None
    key = float(nodes[k]['value'])
    if not math.isfinite(key) or key != int(key):
        return None
    return x, int(key), float(value['value']), n['args'][2]


def _lookup(nodes, i):
    """A lookup node for the select chain at ``i`` (each link tests the same X against an integer constant), else None."""
    first = _lookup_link(nodes, i)
    if first is None:
        return None
    x, entries, rest = first[0], {}, i
    while True:
        link = _lookup_link(nodes, rest)
        if link is None or link[0] != x:
            break
        entries.setdefault(link[1], link[2])   # the outermost test of a key wins, as in the chain
        rest = link[3]
    if len(entries) < LOOKUP_MIN or max(entries) - min(entries) >= LOOKUP_SIZE:
        return None
    low = min(entries)
    table = [entries.get(low + k) for k in range(max(entries) - low + 1)]
    return dict(op='lookup', type='f', args=[x, rest], low=low, table=[0.0 if v is None else v for v in table],
                present=[v is not None for v in table])


def lookup_tables(recipe):
    """``recipe`` with each select chain that tests one value against integer constants (a table the compiler unrolled,
    such as a noise permutation) as one ``lookup`` node: args [X, rest], ``table[k]`` the value for X = low + k where
    ``present[k]``, else rest. Unreachable nodes are dropped. The terms are unchanged (evaluate_recipe expands lookups)."""
    nodes = recipe['nodes']
    out, seen, stack = {}, set(), list(recipe['terms'].values())
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        out[i] = _lookup(nodes, i) or dict(nodes[i])
        stack.extend(out[i]['args'])
    if all(n['op'] != 'lookup' for n in out.values()):
        return recipe
    order = sorted(out)
    index = {old: new for new, old in enumerate(order)}
    new = []
    for old in order:
        n = out[old]
        n['args'] = [index[a] for a in n['args']]
        new.append(n)
    return dict(recipe, nodes=new, terms={k: index[v] for k, v in recipe['terms'].items()})


NAN_PROPAGATES = {'add', 'mul', 'div', 'neg', 'abs', 'exp2', 'log2', 'rsq', 'sqrt', 'rcp', 'frc', 'round_ne', 'round_ni',
                  'round_pi', 'round_z', 'mad', 'dp2', 'dp3', 'dp4', 'sin', 'cos'}


def fold_nan(recipe):
    """``recipe`` with constant NaN resolved by D3D rules (a neutral profile value can leave one, such as a normalised
    zero vector): arithmetic on NaN is NaN, saturate of NaN is 0, min and max return the other operand, comparisons are
    false (ne true). Renderers whose arithmetic does not keep NaN (Cycles runs with fast math) then get the same values."""
    nodes = [dict(n) for n in recipe['nodes']]
    nan = [n['op'] == 'const' and isinstance(n.get('value'), float) and n['value'] != n['value'] for n in nodes]
    if not any(nan):
        return recipe
    changed = False
    for i, n in enumerate(nodes):   # arguments come before their users
        args, op = n['args'], n['op']
        hit = [nan[a] for a in args]
        if not any(hit) or op == 'select':
            continue
        if op in NAN_PROPAGATES:
            nodes[i] = dict(op='const', type='f', value=float('nan'), args=[])
        elif op == 'sat':
            nodes[i] = dict(op='const', type='f', value=0.0, args=[])
        elif op in ('min', 'max'):
            nodes[i] = dict(nodes[args[hit.index(False)]]) if not all(hit) else dict(nodes[args[0]])
        elif op in ('eq', 'lt', 'ge', 'ne'):
            nodes[i] = dict(op='const', type='b', value=op == 'ne', args=[])
        else:
            continue
        nan[i] = nodes[i]['op'] == 'const' and nodes[i]['type'] == 'f' and nodes[i]['value'] != nodes[i]['value']
        changed = True
    return dict(recipe, nodes=nodes) if changed else recipe


def blender_recipe(recipe):
    """The recipe as the Blender export builds it: constant NaN folded (fold_nan), unrolled tables as lookups."""
    return lookup_tables(fold_nan(recipe))


def _expand_lookups(nodes):
    """Lookup nodes back as the select chains they stand for (appended nodes; each lookup becomes its head select)."""
    for i in [i for i, n in enumerate(nodes) if n['op'] == 'lookup']:
        n = nodes[i]
        x, cur = n['args']
        links = [(n['low'] + k, v) for k, (v, p) in enumerate(zip(n['table'], n['present'])) if p]
        for key, value in reversed(links):
            nodes.append(dict(op='const', type='f', value=float(key), args=[]))
            nodes.append(dict(op='eq', type='b', args=[x, len(nodes) - 1]))
            nodes.append(dict(op='const', type='f', value=float(value), args=[]))
            nodes.append(dict(op='select', type='f', args=[len(nodes) - 2, len(nodes) - 1, cur]))
            cur = len(nodes) - 1
        nodes[i] = dict(nodes[cur])


def evaluate_recipe(recipe, inputs, sample):
    """Evaluate a recipe's terms numerically (reference for emitter checks). ``sample(node, args) -> 4 floats``."""
    graph = sm5.Graph()
    graph.nodes = [dict(n) for n in recipe['nodes']]
    _expand_lookups(graph.nodes)
    values = sm5.evaluate(graph, list(recipe['terms'].values()), defaultdict(float, inputs), sample)
    return {name: values[i] for name, i in recipe['terms'].items()}
