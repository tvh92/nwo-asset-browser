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
import contextlib
import hashlib
import math
import threading
from collections import defaultdict

from rc2016.games.rcra import material as rmaterial
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
# Per-draw and per-object values a material does not own, bound to the neutral draw: no dissolve, no render flags
# (class 0 lighting, occlusion mode decided by the material), instance 0, shader time 0. Viewer choices, labelled as such.
# TEXCOORD.w, the draw alpha, is replaced per material by the plain instance's (material.draw_interpolants: the material
# alpha, decoded); its 1 here is the neutral object and material records the vertex program reads when that is 0.
NEUTRAL_DRAW_FLOATS = {'TEXCOORD.z': 0.0, 'TEXCOORD.w': 1.0, 'TEXCOORD2.w': 0.25, 'GlobalWorldCBuffer.m_ShaderTimer': 0.0,
                       'GlobalWorldCBuffer.m_DeltaTime': 0.0, 'GlobalWorldCBuffer.m_TimeOfDay': 0.0,
                       'GlobalWorldCBuffer.m_MaterialLoDOffset': 0.0}


def _draw_alpha_text(material_data):
    """The draw alpha the translation binds, for the limitations."""
    return '%g' % rmaterial.draw_interpolants(material_data).get(rmaterial.DRAW_ALPHA_INPUT, 1.0)


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


class _NotAffine(Unsupported):
    """The forward target is not an affine tint of the scene behind it (forward keeps the scene as an input instead)."""


# Engine values a Blender material cannot supply, bound to review values (everything not listed is 0): adapted
# luminance 1 (as the viewer's RCRA_ADAPTED_LUMINANCE), the pixel centre of a 1920x1080 view for screen-space dither.
# The exposure a material's emission is scaled by (adapted and measured luminance, the tone-map anchor) is a review value of
# 1: the review render has no automatic exposure. Not game data.
REVIEW_VALUES = {'GlobalWorldCBuffer.m_AdaptedLuminance': 1.0, 'GlobalWorldCBuffer.m_ActualLuminance': 1.0,
                 'GlobalWorldCBuffer.m_ToneMapAnchor': 1.0, 'SV_POSITION.x': 0.5, 'SV_POSITION.y': 0.5,
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
        self.named = {}   # constants an engine program's caller binds by name (surface ``program``)
        self.interpolants = {}   # interpolants an engine program's vertex stage sets (surface ``program``: fur shells)
        self.symbolic = {}   # interpolants an engine program's vertex stage passes, as renderer inputs {name: input}

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
            kind = 'f' if op.startswith('sample') or binding in rshaders.FLOAT_RESOURCES else 'i'
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
        if name in self.interpolants:
            return self.graph.const('f', self.interpolants[name])
        if name in self.symbolic:
            return self.graph.input(self.symbolic[name])
        if name == 'SV_IsFrontFace.x':   # read only as a truth value: a boolean input
            return self.graph.input('SV_IsFrontFace', 'b')
        if name in GEOMETRY_INPUTS:
            return None
        return self._review(name)

    def constant(self, name):
        if name in self.named:
            return self.graph.const('f', self.named[name])
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


WORLD_BUFFER = 'GlobalWorldCBuffer.'   # the engine's lighting constants (key light, ambient, GI, light grid, cloud shadows)


class _EngineFreeBindings(_ForwardBindings):
    """_ForwardBindings with the light sources the forward profile zeroes left symbolic (inputs ENGINE.*): the unbound
    GlobalWorldCBuffer constants (key light, ambient fill, distant GI, light grid, cloud shadows, SSAO factors) and the
    samples of the lighting textures (probes, SSR, AO, BRDF lookup, shadows, gobos). The light-list counts stay 0, and
    the view and draw settings keep their profile values (view space with the world axes, no clip plane, depth of field
    or alpha override). Used to prove a term does not depend on the lighting."""

    def resource(self, graph, op, binding, args):
        if binding in rshaders.FORWARD_LIGHTING and op.startswith(('sample', 'gather')):
            return [graph.input('ENGINE.%s.%d' % (binding, c), 'f') for c in range(4)]
        return super().resource(graph, op, binding, args)

    def _review(self, name):
        if name.startswith(WORLD_BUFFER) and name not in NEUTRAL_DRAW_FLOATS and name not in REVIEW_VALUES:
            return self.graph.input('ENGINE.' + name, 'f')
        return super()._review(name)


def constant_normal(word_x, word_y):
    """The normal of constant NormalSpec x and y words: the inverse of the packing p = n.xy rsq(8 |n.z| + 8), stored as
    ftou(p 5791.205 + 2048) in bits 0 to 11, which for a unit normal gives |n.z| = 1 - 8 |p|^2; the z sign is y bit 15."""
    px, py = (((w & 0xFFF) - 2048.0) / NORMAL_SCALE for w in (word_x, word_y))
    z = min(max(1.0 - 8.0 * (px * px + py * py), 0.0), 1.0)
    weight = math.sqrt(8.0 * z + 8.0)
    return px * weight, py * weight, -z if word_y & 0x8000 else z


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
        """x of ftoi(add(mul(x, 255), 0.5)) (or ftou), else None. A writer whose factor the compiler folded into the scale,
        ftoi(add(mul(x, k), 0.5)), stores the byte of x k / 255."""
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
            scale = self.value(m['args'][k])
            if scale == 255.0:
                return m['args'][x]
            if isinstance(scale, float) and 0.0 < scale < 255.0:
                return self.g.op('mul', m['args'][x], self.g.const('f', scale / 255.0))
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


def surface(graph_data, material_data, slot=rshaders.SLOT_GBUFFER, texture_sizes=None, program=None):
    """The material's surface terms as one expression DAG. ``texture_sizes(path)`` gives a texture's (width, height, depth or
    array size, mip count) for programs that query sizes (rc2016.games.rcra.texture.texture_size); without it they fail.

    A writer whose terms depend on NaN under the review screen position (it reads the view direction) is translated again
    with the forward profile's view (_ViewBindings, VIEW_NOTE), and that translation is used when it has no NaN.

    ``program`` replaces the graph's writer with an engine program: dict(shader, cb2 [floats], units {binding: path},
    named {constant: value}, limitations [text], discard: False to drop the program's own discard, and optionally
    interpolants {input: value} its vertex stage sets, symbolic {input: renderer input name} it passes per vertex, and
    coverage: a builder (graph -> node) of a ``coverage`` term, the fraction of the surface the program keeps) (rc2016.games.rcra.fur.fur_program, shell_program); ``graph_data`` is
    then unused.

    Returns dict(graph, terms {name: node}, textures {binding: path}, inputs [names], limitations [text]).
    Raises Unsupported when the writer is not the engine packing or keeps non-portable operations."""
    result = _surface(graph_data, material_data, slot, texture_sizes, _Bindings, program)
    if not _reads_nan(result):
        return result
    try:
        viewed = _surface(graph_data, material_data, slot, texture_sizes, _ViewBindings, program)
    except Unsupported:
        viewed = None
    if viewed is not None and not _reads_nan(viewed):
        viewed['limitations'].append(VIEW_NOTE)
        return viewed
    result['limitations'].append(NAN_NOTE)
    return result


def _surface(graph_data, material_data, slot, texture_sizes, binding_class, program=None):
    if program is None:
        shader = rshaders.pass_shader(graph_data, slot)
        cb2, units = rshaders.material_constants(graph_data, material_data), rshaders.material_units(graph_data, material_data)
    else:
        shader, cb2, units = program['shader'], program['cb2'], program['units']
    bindings = binding_class(cb2, shader.reflection, units, texture_sizes)
    bindings.interpolants.update(rmaterial.draw_interpolants(material_data))   # the plain instance's draw alpha
    bindings.named.update(program['named'] if program else {})
    bindings.interpolants.update(program.get('interpolants', {}) if program else {})
    bindings.symbolic.update(program.get('symbolic', {}) if program else {})
    translator = sm5.Translator(shader, bindings)
    bindings.graph = translator.g
    tr = _run(translator)
    g = tr.graph
    if program and not program.get('discard', True):   # the program's own discard is not the material's (fur shells)
        tr.discard = g.const('b', False)
    out = lambda r, c: tr.outputs.get(('o', r, c))   # noqa: E731
    if any(out(r, c) is None for r, c in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3))):
        raise Unsupported('Writer does not write the NormalSpec and DiffuseEmis targets')
    unknown = g.unknowns([out(r, c) for r in (0, 1) for c in range(4)] + [tr.discard])
    if unknown:
        raise Unsupported('Writer depends on unknown values: ' + ', '.join(unknown[:3]))
    u = _Unpack(g)
    for key in [k for k in tr.outputs if k[0] == 'o' and k[1] == 1]:
        tr.outputs[key] = g.cast(tr.outputs[key], 'f')   # an 8-bit UNORM target: integer-typed words hold float bits
    words = g.value(out(0, 0)), g.value(out(0, 1))
    if None not in words:   # a writer that stores a fixed normal: decoded exactly from the words
        nx, ny, nz = (g.const('f', v) for v in constant_normal(*words))
        occlusion_mode = g.const('b', bool(words[1] & 0x2000))
    else:
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
    if program and program.get('coverage'):
        terms['coverage'] = program['coverage'](g)
    class_limitations = _class_terms(g, terms, out(0, 0), out(3, 0))
    for name, byte in (('f0.r', f0r), ('f0.g', f0g), ('f0.b', f0b)):
        stored = _rounded(g, g.op('sat', byte))
        terms[name] = g.op('mul', stored, stored)
    terms['gloss'] = _rounded(g, g.op('sat', gloss))
    limitations = ['Per-draw values are the neutral draw: draw alpha %s (the material alpha), no dissolve, no render '
                   'flags (lighting class 0), ' % _draw_alpha_text(material_data)
                   + 'no flag volumes, shader time 0.' + OBJECT_NOTE * bindings.object_read + RECORD_NOTE * bindings.record_read
                   + (ENGINE_NOTE % ', '.join(sorted(bindings.engine_read)) if bindings.engine_read else ''), 'The 12-bit normal and the 8-bit albedo, occlusion and emission are used before the '
                   "G-buffer's rounding; F0 and gloss keep it."]
    limitations += program['limitations'] if program else []
    if program is None and rmaterial.forward_technique(graph_data) and rmaterial.material_blend(material_data) == 'opaque':
        # an opaque material the game draws in the forward layer: its albedo from the forward program's diffuse part
        try:
            lit = forward_albedo(graph_data, material_data, texture_sizes)
        except (Unsupported, ValueError, KeyError) as error:
            limitations.append(rmaterial.FORWARD_TECHNIQUE_NOTE % error)
        else:
            names = [k for k in ('albedo.r', 'albedo.g', 'albedo.b', 'unlit.r', 'unlit.g', 'unlit.b') if k in lit['terms']]
            imported = dict(zip(names, _import(g, lit['graph'], [lit['terms'][k] for k in names])))
            terms.update({k: g.op('sat', x) if k.startswith('albedo.') else g.op('max', x, zero)
                          for k, x in imported.items()})
            limitations += [x for x in lit['limitations'] if x not in limitations]
    limitations += class_limitations
    return _portable(g, terms, bindings, graph_data, material_data, tr, limitations, 'Surface')


CLASS_RUNTIME_NOTE = ('The writer chooses its lighting class at run time (a select of packed words); it is lit as class 0.')


def _class_terms(g, terms, word_x, extra):
    """Add the lighting class (term ``class``, a constant) and, for classes 1 to 5, the fields of the g_GBufExtra word
    (SV_Target3) the class's apply program reads (terms ``extra.<k>``, their integer values; lighting.EXTRA_FIELDS).
    Returns limitations."""
    from rc2016.games.rcra import lighting
    cls = lighting.material_class(g, word_x)
    notes = []
    if cls is None:
        cls, notes = 0, [CLASS_RUNTIME_NOTE]
    if cls and extra is None:
        cls, notes = 0, ['The writer stores lighting class %d without its extra target; it is lit as class 0.' % cls]
    terms['class'] = g.const('f', float(cls))
    for k, (offset, width) in enumerate(lighting.EXTRA_FIELDS.get(cls, ())):
        word = g.cast(extra, 'i')
        terms['extra.%d' % k] = g.op('utof', g.op('ubfe', g.const('i', width), g.const('i', offset), word))
    return notes


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
        if n['op'] in ('or', 'iadd'):   # a sum of fields is their or when no constant bits fall in a field (checked)
            out['sum'] = out.get('sum', False) or n['op'] == 'iadd'
            for a in n['args']:
                fields(a, out)
        elif n['op'] == 'const' and n['type'] == 'i':
            if out.get('sum') and out['bits'] & n['value'] & 0xFFFFFFFF:
                out['overlap'] = True
            out['bits'] |= n['value'] & 0xFFFFFFFF
        elif n['op'] == 'ishl' and k[1] is not None:
            out['parts'].append((k[1] & 31, n['args'][0], None))
        elif n['op'] == 'imul' and k[1] is not None and power(k[1]):
            out['parts'].append((k[1].bit_length() - 1, n['args'][0], None))
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
        if n['op'] not in ('or', 'iadd'):
            return bits(word, offset, width)
        parts = fields(word, dict(bits=0, parts=[]))
        shifts = sorted(s for s, _x, _w in parts['parts'])
        if len(set(shifts)) != len(shifts) or parts.get('overlap'):
            return None if n['op'] == 'or' else bits(word, offset, width)
        # A sum of fields (iadd) is their or under the packing's assumption that each field fits its slot; the constant's
        # bits inside the field read back are checked below as for an or.
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
        elif op == 'and' and k[1] and power((k[1] >> ((k[1] & -k[1]).bit_length() - 1)) + 1):
            # a contiguous mask above bit 0 (a sign or mode bit): that field, scaled back to its place
            low = (k[1] & -k[1]).bit_length() - 1
            width = (k[1] >> low).bit_length()
            field = extract(args[0], low, width)
            if field is None:
                field = bits(args[0], low, width)
            out = None if field is None else g.op('mul', field, g.const('f', float(2 ** low)))
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

    def unsigned(i):   # an integer whose value is below 2^31: a bit field, a cast of a float to unsigned, a small constant
        n = g.nodes[i]
        if n['op'] == 'const':
            return n['type'] == 'i' and 0 <= n['value'] < 2 ** 31
        if n['op'] in ('ubfe', 'ftou'):
            return True
        k = [g.value(a) for a in n['args']]
        return (n['op'] == 'ushr' and k[1] is not None and k[1] & 31 > 0) or \
            (n['op'] == 'and' and any(v is not None and 0 <= v < 2 ** 31 for v in k))

    def top(i):
        n = g.nodes[i]
        op, args = n['op'], n['args']
        if op in ('itof', 'utof'):
            return f(args[0])
        if op == 'bits2f' and g.nodes[args[0]]['op'] == 'b2i':   # a boolean kept in a float register: all ones is NaN
            return g.select(g.nodes[args[0]]['args'][0], g.const('f', float('nan')), g.const('f', 0.0))
        if op in ('ieq', 'ine', 'ilt', 'ige', 'ult', 'uge'):
            a, b = f(args[0]), f(args[1])
            if a is None or b is None:
                return None
            if op in ('ult', 'uge') and not all(unsigned(x) for x in args):
                return None   # an unsigned test of values that may be negative: not the float test
            return g.op({'ieq': 'eq', 'ine': 'ne', 'ilt': 'lt', 'ige': 'ge', 'ult': 'lt', 'uge': 'ge'}[op], a, b)
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


NAN_PRUNE_BUDGET = 200000   # node rebuilds; past it the terms are kept as they are


class _Budget(Exception):
    pass


def _is_nan(n):
    return n['op'] == 'const' and isinstance(n.get('value'), float) and n['value'] != n['value']


def _learn(g, facts, cond, value):
    """``facts`` ({condition node: truth}) with cond = value and what follows from it through not, and, or."""
    out, pending = dict(facts), []

    def learn(node, v):
        if node in out or g.is_const(node):
            return
        out[node] = v
        n = g.nodes[node]
        if n['op'] == 'b_not':
            learn(n['args'][0], not v)
        elif n['op'] == ('b_and' if v else 'b_or'):
            for a in n['args']:
                learn(a, v)
        elif n['op'] in ('b_and', 'b_or'):
            pending.append(node)   # a false and, a true or: one side follows once the other is known
    learn(cond, value)
    for node, v in facts.items():   # earlier facts may now decide a pending side
        if g.nodes[node]['op'] in ('b_and', 'b_or') and v == (g.nodes[node]['op'] == 'b_or'):
            pending.append(node)
    changed = True
    while changed:
        changed = False
        for node in pending:
            x, y = g.nodes[node]['args']
            want = g.nodes[node]['op'] == 'b_or'
            for p, q in ((x, y), (y, x)):
                if out.get(p) is (not want) and q not in out:
                    learn(q, want)
                    changed = True
    return out


def _prune_nan(g, terms):
    """``terms`` with the NaN constants that only select branches no input reaches taken out: each select's branches are
    rebuilt knowing its condition (true in the first, false in the second), so a select inside them on a condition that
    follows takes its branch. Exact. An unrolled loop's exit flag kept in a float register (all ones, NaN) is carried into
    later iterations' registers only on paths the loop's own conditions rule out; an emitter whose select is arithmetic
    (Blender's b + c (a - b)) would otherwise spread the NaN of the branch not taken."""
    reach = g.reachable(list(terms.values()))
    if not any(_is_nan(g.nodes[i]) for i in reach):
        return terms
    tainted = set()
    for i in sorted(reach):   # arguments come before their users
        n = g.nodes[i]
        if _is_nan(n) or any(a in tainted for a in n['args']):
            tainted.add(i)
    memo, count = {}, [0]
    leaves = sm5.Translator.LEAVES

    def rebuild(n, args):
        meta = {m: v for m, v in n.items() if m not in ('op', 'type', 'args')}
        if not meta and n['op'] in sm5.OPS:
            return g.op(n['op'], *args)   # folds a comparison whose operands the facts decided
        return g._add(n['op'], n['type'], args, **meta)

    def under(i, facts, key):
        """i under facts: a select on a decided condition takes its branch; a select whose branches hold NaN is
        rebuilt with its condition known in each."""
        if i in facts:
            return g.const('b', facts[i])
        n = g.nodes[i]
        if i not in tainted and (n['op'] in leaves or not g.predicated[i]):
            return i   # a texture read at a NaN address is rebuilt like any other node
        k = (i, key)
        if k in memo:
            return memo[k]
        count[0] += 1
        if count[0] > NAN_PRUNE_BUDGET:
            raise _Budget
        if n['op'] == 'select':
            c = under(n['args'][0], facts, key)
            taken = g.value(c)
            if taken is not None:
                out = under(n['args'][1] if taken else n['args'][2], facts, key)
            elif i in tainted:
                yes, no = _learn(g, facts, c, True), _learn(g, facts, c, False)
                out = g.select(c, under(n['args'][1], yes, frozenset(yes.items())),
                               under(n['args'][2], no, frozenset(no.items())))
            else:
                out = g.select(c, under(n['args'][1], facts, key), under(n['args'][2], facts, key))
        else:
            args = [under(a, facts, key) for a in n['args']]
            out = i if args == n['args'] else rebuild(n, args)
        if out in facts:   # rebuilt into a node the facts decide (hash-consing makes it the same node)
            out = g.const('b', facts[out])
        memo[k] = out
        return out
    try:
        pruned = {name: under(i, {}, frozenset()) for name, i in terms.items()}
    except (_Budget, RecursionError):
        return terms
    return pruned


def _portable(g, terms, bindings, graph_data, material_data, tr, limitations, what, engine=()):
    """The result dict, after checking that every term is a portable float expression of geometry and material textures
    (and of the engine textures in ``engine``, kept as sample nodes for the renderer and not listed as material textures)."""
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
    pruned = _prune_nan(g, terms)
    if pruned is not terms:
        terms = pruned
        roots = list(terms.values())
        reach = g.reachable(roots)
    if g.unknowns(roots):
        raise Unsupported('%s depends on unknown values' % what)
    textures = {}
    units = bindings.units
    if bindings.sized:
        limitations = list(limitations) + [SIZE_NOTE % ', '.join(sorted(bindings.sized))]
    cubes, cube_opcodes = set(), set()
    for i in reach:
        n = g.nodes[i]
        if n['op'] == 'sample':
            if n['binding'] in engine:
                continue
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


FORWARD_TERMS = ('colour.r', 'colour.g', 'colour.b', 'background.r', 'background.g', 'background.b', 'lit_background.r',
                 'lit_background.g', 'lit_background.b', 'alpha', 'clip')
BACKGROUND_BUFFER = 'g_FrameBuffer'
BACKGROUND_INPUTS = ('BACKGROUND.r', 'BACKGROUND.g', 'BACKGROUND.b', 'BACKGROUND.a')
SCENE_NOTE = ('The program reads the scene behind it (%s) other than as an affine tint (blend curves, contrast, '
              'one channel into all three), so the target keeps the scene colour as inputs BACKGROUND.rgba, sampled '
              'at the pixel: the viewer reads a copy of its frame; Blender has no node for it. The background tint is '
              '0, and the sample\'s screen-space offset (distortion, refraction) is dropped.')
LIT_BACKGROUND_SPECULAR_NOTE = (' It also passes the scene behind through a specular factor (the BRDF lookup, gloss '
                                "reflections), which the forward profile's zero specular light leaves out.")
UNLIT_BACKGROUND_NOTE = (' The material is lit, but the tint reads no light or other engine value, so the scene '
                         'passes through it unlit, as in the game.')
LIT_BACKGROUND_NOTE = (' The material is lit and lights the scene behind it as albedo: the tint is background + '
                       'lit_background x the forward diffuse light (key light with its shadows, ambient fill or light '
                       'grid, distant GI, AO), for which the renderer substitutes its own diffuse light.')


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


# Engine inputs of the specular lighting (split-sum scale, reflections, probe cube maps, gloss reflection constants): a
# lighting factor that reads one is not the diffuse light.
SPECULAR_INPUTS = ('ENGINE.g_BrdfLookup.0', 'ENGINE.g_SSRFinalTex.', 'ENGINE.g_EnvProbeEnvs.', 'ENGINE.g_EnvProbeArray.',
                   'ENGINE.GlobalWorldCBuffer.m_GlossReflParams')


# Engine inputs that are a diffuse light colour (_EngineFreeBindings names): the key light's colour and the ambient fill.
DIFFUSE_LIGHT_INPUTS = {'ENGINE.GlobalWorldCBuffer.m_KeyLightColorLinear' + k for k in ('', '+4', '+8')} | \
    {'ENGINE.GlobalWorldCBuffer.m_AmbientFill' + k for k in ('', '+4', '+8')}


def _factor_split(g, root, variable, expand=False):
    """(F0, {L: Y}) with root = F0 + sum Y x L exactly, where the L are the maximal subexpressions ``variable(i)`` is true
    for, reached only through add, negate, a multiply or divide by a value it is false for, or a select on such a
    condition; F0 and every Y are free of them. None when a variable is reached through anything else.

    With ``expand`` (tried when a plain split does not give one diffuse coefficient: both rewrites can turn one factor
    into several), a select on a variable condition whose arms have one F0 splits per coefficient, F0 + sum Y
    select(c, La, Lb), and a product of two variable values is multiplied out, (A0 + sum Ya La)(B0 + sum Yb Lb), into
    factors La, Lb and La x Lb with coefficients Ya B0, A0 Yb and Ya Yb (exact up to float rounding: distributivity)."""
    zero, one = g.const('f', 0.0), g.const('f', 1.0)
    memo = {}

    def scaled(op, f0, x):   # F0 times or over a free value: an F0 of 0 is the split's own zero, not the program's, so 0
        return zero if g.value(f0) == 0.0 else g.op(op, f0, x)

    def product(a, b):
        coeffs = {}

        def put(factor, y):
            coeffs[factor] = g.op('add', coeffs[factor], y) if factor in coeffs else y
        for la, ya in a[1].items():
            put(la, g.op('mul', ya, b[0]))
            for lb, yb in b[1].items():
                put(g.op('mul', la, lb), g.op('mul', ya, yb))
        for lb, yb in b[1].items():
            put(lb, g.op('mul', a[0], yb))
        return g.op('mul', a[0], b[0]), {k: y for k, y in coeffs.items() if g.value(y) != 0.0}

    def split(i):
        if i in memo:
            return memo[i]
        n = g.nodes[i]
        op, args = n['op'], n['args']
        if not variable(i):
            result = (i, {})
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
            result = None if a is None else (g.op('neg', a[0]), {k: g.op('neg', c) for k, c in a[1].items()})
        elif op in ('mul', 'div') and not variable(args[1]):
            a = split(args[0])
            result = None if a is None else (scaled(op, a[0], args[1]),
                                             {k: g.op(op, c, args[1]) for k, c in a[1].items()})
        elif op == 'mul' and not variable(args[0]):
            a = split(args[1])
            result = None if a is None else (scaled('mul', a[0], args[0]),
                                             {k: g.op('mul', args[0], c) for k, c in a[1].items()})
        elif op == 'select' and not variable(args[0]):
            a, b = split(args[1]), split(args[2])
            if a is None or b is None:
                return None
            result = (g.select(args[0], a[0], b[0]), {k: g.select(args[0], a[1].get(k, zero), b[1].get(k, zero))
                                                      for k in set(a[1]) | set(b[1])})
        elif expand and op == 'select' and \
                _same_node(g, *(x[0] for x in (split(args[1]), split(args[2])) if x is not None)):
            # a select on a light-dependent condition (shadow cascades, N.L): select(c, F0 + sum Y La, F0 + sum Y Lb) is
            # F0 + sum Y select(c, La, Lb), the arms' factors grouped by coefficient (exact: every Y is free of them)
            arms = [split(args[1]), split(args[2])]
            by_coefficient = [{}, {}]
            for arm, grouped in zip(arms, by_coefficient):
                for factor, y in arm[1].items():
                    grouped[y] = g.op('add', grouped[y], factor) if y in grouped else factor
            coeffs = {}
            for y in sorted(set(by_coefficient[0]) | set(by_coefficient[1])):
                factor = g.select(args[0], by_coefficient[0].get(y, zero), by_coefficient[1].get(y, zero))
                coeffs[factor] = g.op('add', coeffs[factor], y) if factor in coeffs else y
            result = (arms[0][0], coeffs)
        elif expand and op == 'mul':
            a, b = split(args[0]), split(args[1])
            result = None if a is None or b is None else product(a, b)
        else:
            result = (zero, {i: one})
        memo[i] = result
        return result
    return split(root)


def _same_node(g, *nodes):
    """True for two nodes that are one value: the same node, or equal constants."""
    if len(nodes) != 2:
        return False
    a, b = nodes
    return a == b or (g.value(a) is not None and g.value(a) == g.value(b))


def _engine_translation(shader, graph_data, material_data, flags, texture_sizes):
    """(graph, translation, engine(i)) of the forward program with every value the profile zeroes symbolic
    (_EngineFreeBindings); engine(i) is true for nodes that read one."""
    bindings = _EngineFreeBindings(rshaders.material_constants(graph_data, material_data), flags, shader.reflection,
                                   rshaders.material_units(graph_data, material_data), texture_sizes)
    bindings.interpolants.update(rmaterial.draw_interpolants(material_data))
    translator = sm5.Translator(shader, bindings)
    bindings.graph = translator.g
    tr = _run(translator)
    g = tr.graph
    return g, tr, _reads(g, lambda name: name.startswith('ENGINE.'))


def _reads(g, wanted):
    """reads(i): true for the nodes of ``g`` that read an input whose name ``wanted(name)`` accepts."""
    memo = {}

    def reads(i):
        if i not in memo:
            n = g.nodes[i]
            memo[i] = (n['op'] == 'input' and wanted(n['name'])) or any(reads(a) for a in n['args'])
        return memo[i]
    return reads


def _separable(g, root, first, second):
    """True when root is a sum of parts that each read at most one of two input sets (``first(i)``, ``second(i)``): they
    meet only through add, negate, a multiply or divide by a value that reads neither, or a select on such a condition."""
    if not (first(root) and second(root)):
        return True
    n = g.nodes[root]
    op, args = n['op'], n['args']
    free = (lambda i: not first(i) and not second(i))
    if op == 'add':
        return _separable(g, args[0], first, second) and _separable(g, args[1], first, second)
    if op in ('neg', 'cast'):
        return _separable(g, args[0], first, second)
    if op in ('mul', 'div') and free(args[1]):
        return _separable(g, args[0], first, second)
    if op == 'mul' and free(args[0]):
        return _separable(g, args[1], first, second)
    if op == 'select' and free(args[0]):
        return _separable(g, args[1], first, second) and _separable(g, args[2], first, second)
    return False


def _check_scene_unlit(shader, graph_data, material_data, flags, texture_sizes):
    """Raise Unsupported unless a lit forward program's target reads the scene behind and the lighting only in separate
    summands (_separable over the symbolic translation): the scene then passes through unlit, and the profile's target
    with the scene colour kept as an input is the game's apart from the zeroed lighting."""
    g, tr, engine = _engine_translation(shader, graph_data, material_data, flags, texture_sizes)
    scene = _reads(g, lambda name: name in BACKGROUND_INPUTS)
    roots = [g.cast(tr.outputs[('o', 0, c)], 'f') for c in range(4)] + [tr.discard]
    if not all(_separable(g, r, scene, engine) for r in roots):
        raise Unsupported('Forward target lights the scene behind it (%s) non-linearly in a lit material'
                          % BACKGROUND_BUFFER)


def _lit_background(shader, graph_data, material_data, flags, texture_sizes):
    """How a lit forward program passes the scene behind it, translated again with every value the profile zeroes symbolic
    (_EngineFreeBindings). Each colour channel must be affine in its own background channel, colour = A + B x background.
    None when B reads none of the symbolic values: the profile's B is the game's, whatever the lights. Otherwise B must be
    B0 + Y x L with L one engine factor per channel that also lights the program's own colour (it reaches A) and reads no
    specular input: the forward diffuse light, so the program lights the scene behind as albedo. Factors that read a
    specular input (the BRDF lookup's scale, gloss reflections: the goo decals) are the specular light the profile has at
    0, so they are left out (LIT_BACKGROUND_SPECULAR_NOTE). Returns (graph, [(B0, Y)] per channel, whether a specular
    factor was left out), B0 and Y free of the symbolic values. Raises Unsupported otherwise."""
    g, tr, engine = _engine_translation(shader, graph_data, material_data, flags, texture_sizes)
    splits = []
    for c in range(3):
        colour = tr.outputs.get(('o', 0, c))
        split = _affine_split(g, g.cast(colour, 'f'), BACKGROUND_INPUTS) if colour is not None else None
        if split is None or set(split[1]) - {BACKGROUND_INPUTS[c]}:
            raise _NotAffine('Forward colour is not affine in the scene behind (%s)' % BACKGROUND_BUFFER)
        splits.append((split[0], split[1].get(BACKGROUND_INPUTS[c], g.const('f', 0.0))))
    if not any(engine(b) for _a, b in splits):
        return None
    parts, specular = [], False
    for a, b in splits:
        factors = _factor_split(g, b, engine)
        if factors is None:
            raise Unsupported('Forward colour lights the scene behind it (%s) in a lit material, not as a sum of light '
                              'factors' % BACKGROUND_BUFFER)
        diffuse = {}
        for light, y in factors[1].items():
            names = {g.nodes[i]['name'] for i in g.reachable([light]) if g.nodes[i]['op'] == 'input'}
            if any(name.startswith(SPECULAR_INPUTS) for name in names):
                specular = True   # the specular light the profile has at 0
            else:
                diffuse[light] = y
        if len(diffuse) != 1:
            raise Unsupported('Forward colour lights the scene behind it (%s) in a lit material, by %d diffuse light '
                              'factors' % (BACKGROUND_BUFFER, len(diffuse)))
        (light, y), = diffuse.items()
        names = {g.nodes[i]['name'] for i in g.reachable([light]) if g.nodes[i]['op'] == 'input'}
        if light not in g.reachable([a]) and not names & DIFFUSE_LIGHT_INPUTS:
            # the diffuse light: it also lights the program's own colour, or it reads a diffuse light colour
            raise Unsupported('Forward colour lights the scene behind it (%s) in a lit material, by a light factor that '
                              'is not its diffuse light' % BACKGROUND_BUFFER)
        parts.append((factors[0], y))
    return g, parts, specular


def _import(dst, src, roots):
    """The nodes ``roots`` of graph ``src`` added to graph ``dst``; returns their indices there."""
    new = {}
    for r in roots:
        stack = [(r, False)]
        while stack:   # post-order without recursion
            i, ready = stack.pop()
            if i in new:
                continue
            n = src.nodes[i]
            if ready:
                meta = {k: v for k, v in n.items() if k not in ('op', 'type', 'args')}
                new[i] = dst._add(n['op'], n['type'], [new[a] for a in n['args']], **meta)
            else:
                stack.append((i, True))
                stack.extend((a, False) for a in n['args'] if a not in new)
    return [new[r] for r in roots]


def _background_split(g, terms, shader, graph_data, material_data, flags, texture_sizes):
    """(lit, terms): the colour split into A + B x background (and the lit background term, from _lit_background, for a
    lit program that lights the scene behind). Raises _NotAffine when the target is not an affine tint of it."""
    zero = g.const('f', 0.0)
    lit = _lit_background(shader, graph_data, material_data, flags, texture_sizes) if flags else None
    out = {}
    for name in ('alpha', 'clip'):
        split = _affine_split(g, terms[name], BACKGROUND_INPUTS)
        if not split or split[1]:
            raise _NotAffine('Forward %s depends on the scene behind (%s)' % (name, BACKGROUND_BUFFER))
    for c in 'rgb':
        split = _affine_split(g, terms['colour.' + c], BACKGROUND_INPUTS)
        if split is None or set(split[1]) - {'BACKGROUND.' + c}:
            raise _NotAffine('Forward colour is not affine in the scene behind (%s)' % BACKGROUND_BUFFER)
        out['colour.' + c] = split[0]
        out['background.' + c] = split[1].get('BACKGROUND.' + c, zero)
    if lit is not None:   # B0 and Y from the symbolic translation (free of its symbolic values)
        imported = _import(g, lit[0], [x for part in lit[1] for x in part])
        for k, c in enumerate('rgb'):
            out['background.' + c], out['lit_background.' + c] = imported[2 * k:2 * k + 2]
    return lit, out


def reads_scene(recipe):
    """True when a forward recipe keeps the scene behind as inputs BACKGROUND.* (forward's scene mode)."""
    return any(n['op'] == 'input' and n.get('name') in BACKGROUND_INPUTS for n in recipe['nodes'])


def forward(graph_data, material_data, texture_sizes=None):
    """The forward program's target under the forward profile: colour (unclamped, HDR), alpha and the discard.

    Returns the same dict as ``surface`` with terms FORWARD_TERMS. Raises Unsupported when the target depends on anything
    but geometry, the view inputs and the material's own textures and constants."""
    shader = rshaders.pass_shader(graph_data, rshaders.SLOT_FORWARD)
    from rc2016.games.rcra.material import draw_flags
    flags = draw_flags(material_data)
    bindings = _ForwardBindings(rshaders.material_constants(graph_data, material_data), flags, shader.reflection,
                                rshaders.material_units(graph_data, material_data), texture_sizes)
    bindings.interpolants.update(rmaterial.draw_interpolants(material_data))   # the plain instance's draw alpha
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
    lit, scene = None, False
    if bindings.background_read:
        reach = g.reachable(list(terms.values()))
        if any(g.nodes[i]['op'] == 'input' and g.nodes[i]['name'] in BACKGROUND_INPUTS for i in reach):
            try:
                lit, split_terms = _background_split(g, terms, shader, graph_data, material_data, flags, texture_sizes)
                terms.update(split_terms)
            except _NotAffine:   # keep the scene colour as an input, for renderers that can sample it
                if flags:
                    _check_scene_unlit(shader, graph_data, material_data, flags, texture_sizes)
                scene = True
    terms.update({k: v for k, v in background.items() if k not in terms})
    limitations = [
        'Forward colour with every engine light source zero (key light, light grid, probes, GI, reflections): the '
        "material's own emission and effects. The renderer's lights shade its G-buffer surface instead of the forward "
        "program's lighting.",
        "Draw: render flags hold only the material's lit bit (%s), draw alpha %s (the material alpha), no dissolve, "
        'no particle values, no flag volumes, '
        'shader time 0; stored scene depth at the far clip (no soft depth fade); view space with the world axes.'
        % ('lit' if flags else 'unlit: lighting term 1', _draw_alpha_text(material_data)) + OBJECT_NOTE * bindings.object_read
        + RECORD_NOTE * bindings.record_read
        + (SCENE_GBUFFER_NOTE if set(rshaders.SCENE_GBUFFER) & bindings.engine_read else '')]
    if scene:
        limitations.append(SCENE_NOTE % BACKGROUND_BUFFER + UNLIT_BACKGROUND_NOTE * bool(flags))
    elif bindings.background_read:
        limitations.append('The program samples the scene behind it (%s); the colour is affine in it, so it is split into '
                           'an emission and a background tint, and the sample\'s screen-space offset (distortion, '
                           'refraction) is dropped.' % BACKGROUND_BUFFER
                           + (LIT_BACKGROUND_NOTE if lit is not None else UNLIT_BACKGROUND_NOTE * bool(flags))
                           + (LIT_BACKGROUND_SPECULAR_NOTE if lit is not None and lit[2] else ''))
    return _portable(g, terms, bindings, graph_data, material_data, tr, limitations, 'Forward colour')


# The forward program's own key-light term (forward_lighting): the key light's direction and colour as inputs L.xyz (to the
# light, game space) and LIGHT.rgb, unshadowed: past every shadow shell (as rc2016.games.rcra.lighting binds the apply
# programs), the shadow atlas, ray-traced shadows and cloud shadows reading 1. Every other light source keeps the forward
# profile's 0.
KEY_LIGHT_INPUTS = {'GlobalWorldCBuffer.m_KeyLightVec': 'L.x', 'GlobalWorldCBuffer.m_KeyLightVec+4': 'L.y',
                    'GlobalWorldCBuffer.m_KeyLightVec+8': 'L.z', 'GlobalWorldCBuffer.m_KeyLightColorLinear': 'LIGHT.r',
                    'GlobalWorldCBuffer.m_KeyLightColorLinear+4': 'LIGHT.g',
                    'GlobalWorldCBuffer.m_KeyLightColorLinear+8': 'LIGHT.b'}
UNSHADOWED = {'GlobalWorldCBuffer.m_KeyLightShellPlanesW': 1.0, 'GlobalWorldCBuffer.m_KeyLightShellPlanesW+4': 1.0,
              'GlobalWorldCBuffer.m_KeyLightShellPlanesW+8': 1.0, 'GlobalWorldCBuffer.m_KeyLightShellPlanesW+12': 1.0,
              'GlobalWorldCBuffer.m_KeyLightShellLodConsts+12': 10.0}
SHADOW_RESOURCES = ('g_ShadowAtlas', 'g_RayTracedShadows', 'g_CloudShadowMap')
LIGHT_INPUT_NAMES = tuple(KEY_LIGHT_INPUTS.values())
FORWARD_LIGHT_NOTE = ("Viewer: the forward program's own key-light term lights it, once per review light (direction L, "
                      'colour LIGHT, unshadowed); its ambient, probes, local lights and reflections stay 0, so ambient '
                      'light comes from the G-buffer surface, when it has one.')


class _ForwardLitBindings(_ForwardBindings):
    """_ForwardBindings with the key light symbolic (KEY_LIGHT_INPUTS) and unshadowed."""

    def constant(self, name):
        if name in KEY_LIGHT_INPUTS:
            return self.graph.input(KEY_LIGHT_INPUTS[name], 'f')
        if name in UNSHADOWED:
            return self.graph.const('f', UNSHADOWED[name])
        return super().constant(name)

    def resource(self, graph, op, binding, args):
        if binding in SHADOW_RESOURCES and op.startswith(('sample', 'gather')):
            return [graph.const('f', 1.0)] * 4
        return super().resource(graph, op, binding, args)


def forward_lighting(graph_data, material_data, texture_sizes=None):
    """The forward program's colour lit by one key light: terms colour.r/g/b of the inputs L.xyz and LIGHT.rgb besides
    those of ``forward`` (alpha and discard are forward's). With LIGHT = 0 it is forward's colour, and it is linear in
    LIGHT (checked at random inputs), so a renderer adds colour(L, LIGHT) - colour per light.

    Raises Unsupported for an unlit material, a program that does not read the key light, one whose lit colour reads
    the scene behind, or one that is not linear in the light colour."""
    from rc2016.games.rcra.material import draw_flags
    flags = draw_flags(material_data)
    if not flags:
        raise Unsupported('The material is unlit: its forward program reads no light')
    shader = rshaders.pass_shader(graph_data, rshaders.SLOT_FORWARD)
    bindings = _ForwardLitBindings(rshaders.material_constants(graph_data, material_data), flags, shader.reflection,
                                   rshaders.material_units(graph_data, material_data), texture_sizes)
    bindings.interpolants.update(rmaterial.draw_interpolants(material_data))
    translator = sm5.Translator(shader, bindings)
    bindings.graph = translator.g
    tr = _run(translator)
    g = tr.graph
    target = [tr.outputs.get(('o', 0, c)) for c in range(3)]
    if None in target:
        raise Unsupported('Forward program does not write a colour target')
    terms = {'colour.' + c: g.cast(x, 'f') for c, x in zip('rgb', target)}
    names = {g.nodes[i]['name'] for i in g.reachable(list(terms.values())) if g.nodes[i]['op'] == 'input'}
    if not names & set(LIGHT_INPUT_NAMES):
        raise Unsupported('The forward colour does not read the key light')
    if names & set(BACKGROUND_INPUTS):
        raise Unsupported('The lit forward colour reads the scene behind (%s)' % BACKGROUND_BUFFER)
    result = _portable(g, terms, bindings, graph_data, material_data, tr, [FORWARD_LIGHT_NOTE], 'Forward light')
    _check_linear_in_light(result)
    return result


# The forward programs' own ambient light and probe reflections (forward_ambient), for the viewer's review probe
# (rc2016.games.rcra.probe): the program translated with the key light off and the environment as its only light. The
# probe lists hold one review probe that covers everything (REVIEW_PROBE: identity transform, no extents or falloff, so
# its weight is 1, proxy planes FAR away each way, so no parallax correction, cube 0); the z bins and lookup word select it.
# The light grid hands every point to the probe (lighting.GRID_PROBE_WORD), and the probe cube, the probe array and the
# BRDF lookup stay sample nodes. The term is the colour minus the colour with every probe read 0, and must be linear in
# the probe.
PROBE_FAR = 1.0e6
REVIEW_PROBE = {'m_WorldToObjRot0': (1.0, 0.0, 0.0), 'm_FadeAtten': (1.0,), 'm_WorldToObjRot1': (0.0, 1.0, 0.0),
                'm_WorldToObjRotZFlip': (1.0,), 'm_WorldToObjTrans': (0.0, 0.0, 0.0), 'm_Flags': (0,),
                'm_RcpExtents': (0.0, 0.0, 0.0), 'm_CubeArrayIndex': (0.0,), 'm_RcpFalloffPos': (0.0, 0.0, 0.0),
                'm_ProxyDistNegX': (-PROBE_FAR,), 'm_RcpFalloffNeg': (0.0, 0.0, 0.0), 'm_ProxyDistNegY': (-PROBE_FAR,),
                'm_ProxyDistPos': (PROBE_FAR, PROBE_FAR, PROBE_FAR), 'm_ProxyDistNegZ': (-PROBE_FAR,),
                'm_Offset': (0.0, 0.0, 0.0), 'm_ZBinMinMax': (0,)}
PROBE_BUFFER, PROBE_LOOKUP, PROBE_ZBINS = 'g_EnvProbeEnvs', 'g_EnvProbeLookupView', 'g_EnvProbeZBinView'
PROBE_SAMPLED = ('g_EnvProbeDefault', 'g_EnvProbeArray')
AMBIENT_ENGINE = PROBE_SAMPLED + ('g_BrdfLookup',)
FORWARD_AMBIENT_NOTE = ("Viewer, Shared HDRI lighting: the forward program's own ambient light and probe reflections, with "
                        "the review environment as one probe that covers everything (no parallax correction) and as the "
                        "default probe, the engine's BRDF lookup, no light grid (every cell hands its points to the probe), "
                        "screen-space reflections or ambient fill, and m_LightGridIntensity 1 (review values).")


def probe_words(reflection):
    """{byte offset: u32 bits} of REVIEW_PROBE in the program's g_EnvProbeEnvs layout, or None without one."""
    layout = reflection.constant_buffer(PROBE_BUFFER) if reflection is not None else None
    if layout is None or not layout.variables or not layout.variables[0].members:
        return None
    words = {}
    for name, at, _member in layout.variables[0].members:
        for k, value in enumerate(REVIEW_PROBE.get(name, ())):
            words[at + 4 * k] = value if isinstance(value, int) else sm5.float_bits(value)
    return words


class _ForwardAmbientBindings(_ForwardBindings):
    """_ForwardBindings with one review probe in the lists and the ambient sources of forward_ambient."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe = probe_words(args[2] if len(args) > 2 else kwargs.get('reflection'))

    def constant(self, name):
        from rc2016.games.rcra import lighting
        if name == 'GlobalWorldCBuffer.m_EnvProbeLookupWordCount':
            return self.graph.const('i', 1)
        if name in lighting.AMBIENT_REVIEW:
            return self.graph.const('f', lighting.AMBIENT_REVIEW[name])
        return super().constant(name)

    def resource(self, graph, op, binding, args):
        from rc2016.games.rcra import lighting
        if binding in AMBIENT_ENGINE and op.startswith('sample'):
            return None
        if binding == PROBE_ZBINS:
            return [graph.const('i', 0xFFFFFFFF)] * 4
        if binding == PROBE_LOOKUP:
            return [graph.const('i', 1)] * 4
        if binding == lighting.GRID_LOOKUP:
            return [graph.const('i', lighting.GRID_PROBE_WORD)] * 4
        if binding == PROBE_BUFFER and op == 'ld_structured' and self.probe is not None:
            offset = graph.value(args[4]) if len(args) > 4 else None
            if offset is not None:
                return [graph.const('i', self.probe.get(offset + 4 * c, 0)) for c in range(4)]
        return super().resource(graph, op, binding, args)


def forward_ambient(graph_data, material_data, texture_sizes=None):
    """The forward program's ambient term: terms ambient.r/g/b (the colour minus the colour with every probe read 0), with
    sample nodes on AMBIENT_ENGINE. Raises Unsupported for an unlit material, one whose colour reads no probe or reads the
    scene behind, or one not linear in the probe."""
    from rc2016.games.rcra.material import draw_flags
    flags = draw_flags(material_data)
    if not flags:
        raise Unsupported('The material is unlit: its forward program reads no light')
    shader = rshaders.pass_shader(graph_data, rshaders.SLOT_FORWARD)
    bindings = _ForwardAmbientBindings(rshaders.material_constants(graph_data, material_data), flags, shader.reflection,
                                       rshaders.material_units(graph_data, material_data), texture_sizes)
    if bindings.probe is None:
        raise Unsupported('The forward program has no probe list')
    bindings.interpolants.update(rmaterial.draw_interpolants(material_data))
    translator = sm5.Translator(shader, bindings)
    bindings.graph = translator.g
    tr = _run(translator)
    g = tr.graph
    target = [tr.outputs.get(('o', 0, c)) for c in range(3)]
    if None in target:
        raise Unsupported('Forward program does not write a colour target')
    colour = [g.cast(x, 'f') for x in target]
    reach = g.reachable(colour)
    if not any(g.nodes[i]['op'] == 'sample' and g.nodes[i].get('binding') in PROBE_SAMPLED for i in reach):
        raise Unsupported('The forward colour reads no probe')
    if {g.nodes[i]['name'] for i in reach if g.nodes[i]['op'] == 'input'} & set(BACKGROUND_INPUTS):
        raise Unsupported('The forward colour reads the scene behind (%s)' % BACKGROUND_BUFFER)
    dark = _substitute(g, dict(enumerate(colour)), {i: g.const('f', 0.0) for i in reach if g.nodes[i]['op'] == 'channel'
                                                    and g.nodes[g.nodes[i]['args'][0]].get('binding') in PROBE_SAMPLED})
    terms = {'ambient.' + c: g.op('add', colour[k], g.op('neg', dark[k])) for k, c in enumerate('rgb')}
    result = _portable(g, terms, bindings, graph_data, material_data, tr, [FORWARD_AMBIENT_NOTE], 'Forward ambient',
                       engine=AMBIENT_ENGINE)
    _check_linear_in_probe(result)
    return result


def _check_linear_in_probe(result, trials=8):
    """Raise Unsupported unless the ambient term doubles with the probe's radiance, at random inputs and texels."""
    import random
    rng = random.Random(9)
    recipe = export_recipe(result)
    texels = {}
    for _trial in range(trials):
        texels.clear()
        values = {name: rng.uniform(-1.0, 1.0) for name in recipe['inputs']}
        values['SV_IsFrontFace'] = True
        out = []
        for scale in (1.0, 2.0):
            def sample(node, _args, scale=scale):
                texel = texels.setdefault(node.get('binding'), [rng.uniform(0.05, 0.95) for _ in range(4)])
                return [x * scale for x in texel] if node.get('binding') in PROBE_SAMPLED else texel
            out.append(evaluate_recipe(recipe, values, sample))
        for c in 'rgb':
            a, b = out[0]['ambient.' + c], out[1]['ambient.' + c]
            if math.isfinite(a) and math.isfinite(b) and abs(b - 2.0 * a) > 1e-3 * max(1.0, abs(b)):
                raise Unsupported('The forward ambient term is not linear in the probe')


def _check_linear_in_light(result, trials=8, through_zero=False):
    """Raise Unsupported unless colour(L, 2C) - colour(L, 0) = 2 (colour(L, C) - colour(L, 0)) at random inputs and texels,
    and with ``through_zero`` colour(L, 0) = 0."""
    import random
    rng = random.Random(5)
    recipe = export_recipe(result)
    texels = {}

    def sample(node, _args):
        return texels.setdefault(node.get('binding'), [rng.uniform(0.05, 0.95) for _ in range(4)])
    for _trial in range(trials):
        texels.clear()
        values = {name: rng.uniform(-1.0, 1.0) for name in recipe['inputs']}
        values['SV_IsFrontFace'] = True
        light = [rng.uniform(0.2, 2.0) for _ in range(3)]
        out = []
        for scale in (0.0, 1.0, 2.0):
            values.update({'LIGHT.' + c: scale * v for c, v in zip('rgb', light)})
            out.append(evaluate_recipe(recipe, values, sample))
        for c in 'rgb':
            a, b, d = (o['colour.' + c] for o in out)
            if not all(math.isfinite(x) for x in (a, b, d)):
                continue
            if abs((d - a) - 2.0 * (b - a)) > 1e-3 * max(1.0, abs(d - a)):
                raise Unsupported('The forward colour is not linear in the key light colour')
            if through_zero and abs(a) > 1e-6:
                raise Unsupported('The forward specular term is not 0 without the key light')


# The diffuse coefficient of a forward technique's program (forward_albedo): its colour is F0 + Y_d x L_d + Y_s x L_s per
# channel, with L_d the forward diffuse light (key light with its shadows, ambient fill or light grid, distant GI, AO) and L_s
# the specular light (it reads a specular input: BRDF lookup, reflections, probes). Y_d is the colour the program lights
# diffusely, the albedo a deferred renderer lights with its own diffuse light, as the scene behind of a lit blended material
# (LIT_BACKGROUND_NOTE).
FORWARD_ALBEDO_NOTE = ('The game draws this graph with its forward program (technique flag 0x200); its G-buffer writer is '
                       'a stub that stores black albedo. The albedo here is the forward program\'s diffuse coefficient: its '
                       'colour is split exactly into F0 + Y_d x L_d + Y_s x L_s per channel (L_d its diffuse light, L_s its '
                       'specular light), and Y_d is lit by the renderer\'s diffuse light. The G-buffer writer\'s normal, F0, '
                       'gloss and occlusion are kept.')
FORWARD_UNLIT_ZERO_NOTE = ' Its unlit part F0 is 0 at every tested input and texel.'
FORWARD_UNLIT_ONLY_NOTE = ('The game draws this graph with its forward program (technique flag 0x200); its G-buffer writer is '
                           'a stub that stores black albedo. The forward program has no diffuse light factor (its colour '
                           'is split exactly into F0 + Y_s x L_s per channel, L_s its specular light), so the albedo stays '
                           'black. The G-buffer writer\'s normal, F0, gloss and occlusion are kept.')
# F0 of the split reads no light source: an emissive colour, an effect, or the whole colour of an unlit program. The game adds
# it as the program computes it, so a renderer adds it unlit, in scene-linear units, after the lit terms.
FORWARD_UNLIT_NOTE = ("The forward program's unlit part F0 (its emission, or the whole colour of an unlit program) reads no "
                      'light source; it is drawn as an unlit colour (terms unlit.r/g/b, negative values clamped to 0) added '
                      'to the lit surface in scene-linear units.')
# A factor that reads the key light's direction but no light colour (a planet's terminator) is not light: it stays in the
# unlit part, with the direction at a review value (game space, y up: from above and in front). Not game data.
REVIEW_KEY_DIRECTION = (0.3 / 0.9899495, 0.8 / 0.9899495, 0.5 / 0.9899495)
FORWARD_DIRECTION_NOTE = ("The forward colour also reads the key light's direction without its colour (a terminator); that "
                          'part is in the unlit colour, with the direction at a review value, game (%.3f, %.3f, %.3f): from '
                          'above and in front.' % REVIEW_KEY_DIRECTION)
FORWARD_UNLIT_LIGHT_NOTE = ('The material is unlit, but its forward program still reads light sources for an effect (%s); '
                            'the forward profile has every light source at 0, so they read 0 here.')
FORWARD_BLACK_NOTE = ('The game draws this graph with its forward program (technique flag 0x200); its G-buffer writer is a '
                      'stub that stores black albedo. The forward colour has no diffuse light factor and is 0 at every '
                      'tested input and texel with every light source at the forward profile\'s 0, so it is drawn black.')
FORWARD_OWN_LIGHT_NOTE = ('The game draws this graph with its forward program (technique flag 0x200); its G-buffer writer is '
                          'a stub that stores black albedo. The forward colour has diffuse light factors with different '
                          'coefficients (%s), which one albedo cannot carry, so the albedo stays black. The colour with '
                          'every light source 0 (the forward profile) is drawn as an unlit colour (terms unlit.r/g/b), and '
                          "the viewer adds the program's own key-light term once per review light. Its ambient light, "
                          'local lights and reflections are not drawn; Blender has no per-light program, so it draws the '
                          'unlit colour alone.')
FORWARD_OWN_SPECULAR_NOTE = ("Viewer: the forward program's own key-light term (diffuse and specular) lights it, once per "
                             'review light (the key light as direction L and colour LIGHT, unshadowed; every other light '
                             'source 0).')
FORWARD_SPECULAR_NOTE = ("Viewer: the forward program's own specular term Y_s x L_s lights it, once per review light (the "
                         'key light as direction L and colour LIGHT, unshadowed; its screen-space and probe reflections and '
                         'every other light source 0). Blender has no per-light program, so there the stub\'s F0 of the '
                         'G-buffer surface gives the specular.')


class _ForwardLitFreeBindings(_EngineFreeBindings):
    """_EngineFreeBindings with the key light as the inputs L.xyz and LIGHT.rgb and unshadowed (as _ForwardLitBindings)."""

    def constant(self, name):
        if name in KEY_LIGHT_INPUTS:
            return self.graph.input(KEY_LIGHT_INPUTS[name], 'f')
        if name in UNSHADOWED:
            return self.graph.const('f', UNSHADOWED[name])
        return super().constant(name)

    def resource(self, graph, op, binding, args):
        if binding in SHADOW_RESOURCES and op.startswith(('sample', 'gather')):
            return [graph.const('f', 1.0)] * 4
        return super().resource(graph, op, binding, args)


def forward_split_drawn(limitations):
    """True when a surface's limitations say its albedo or unlit colour comes from its forward program's split
    (forward_albedo): the notes it starts its own limitations with."""
    heads = (FORWARD_ALBEDO_NOTE, FORWARD_UNLIT_ONLY_NOTE, FORWARD_BLACK_NOTE, FORWARD_OWN_LIGHT_NOTE.split('(%s)')[0])
    return any(line.startswith(heads) for line in limitations)


class _ManyDiffuse(Unsupported):
    """The forward colour has diffuse light factors with different coefficients; carries the translation (graph,
    translation, bindings) for the own-light fallback (forward_albedo, forward_specular)."""

    def __init__(self, message, translation):
        super().__init__(message)
        self.translation = translation


_SPLITS = threading.local()


@contextlib.contextmanager
def shared_forward_split():
    """Inside the block (on this thread), forward_albedo and forward_specular of the same graph and material share one
    _forward_split, the translation that dominates both, instead of translating the program twice. The shared graph only
    grows (hash-consed nodes), so each caller reads the same nodes it would have built. Nested blocks share the outer memo."""
    outer = getattr(_SPLITS, 'memo', None)
    if outer is None:
        _SPLITS.memo = {}
    try:
        yield
    finally:
        if outer is None:
            _SPLITS.memo = None


def _forward_split(graph_data, material_data, texture_sizes):
    """_split_forward, memoised inside shared_forward_split (its outcome, an exception included, by graph and material)."""
    memo = getattr(_SPLITS, 'memo', None)
    if memo is None:
        return _split_forward(graph_data, material_data, texture_sizes)
    key = (hashlib.sha256(bytes(graph_data)).digest(), hashlib.sha256(bytes(material_data)).digest())
    if key not in memo:
        try:
            memo[key] = (True, _split_forward(graph_data, material_data, texture_sizes))
        except (Unsupported, ValueError, KeyError) as error:   # _ManyDiffuse is an Unsupported
            memo[key] = (False, error)
    ok, outcome = memo[key]
    if not ok:
        raise outcome
    return outcome


def _split_forward(graph_data, material_data, texture_sizes):
    """The forward program of a forward technique split per colour channel into F0 + Y_d x L_d + sum Y_s x L_s, every
    light source symbolic (_ForwardLitFreeBindings: the key light L, LIGHT, the others ENGINE.*). A light factor is
    specular when it reads a specular input (SPECULAR_INPUTS), diffuse when it reads a light colour (LIGHT or another
    engine light), and part of F0 when it reads only the key light's direction (L.xyz stay inputs: FORWARD_DIRECTION_NOTE).
    Diffuse factors must share one coefficient (Y_d is 0 without one). An unlit material (no lit bit) has no diffuse or
    specular light: factors that read a light source are 0, as the forward profile has them (FORWARD_UNLIT_LIGHT_NOTE).
    When a channel does not split, the products of light terms are multiplied out and it is split again.
    Returns (graph, translation, bindings, {channel: (F0, Y_d, specular term)}, [notes]). Raises _ManyDiffuse when the
    diffuse coefficients differ, Unsupported otherwise."""
    flags = rmaterial.draw_flags(material_data)
    shader = rshaders.pass_shader(graph_data, rshaders.SLOT_FORWARD)
    bindings = _ForwardLitFreeBindings(rshaders.material_constants(graph_data, material_data), flags, shader.reflection,
                                       rshaders.material_units(graph_data, material_data), texture_sizes)
    bindings.interpolants.update(rmaterial.draw_interpolants(material_data))
    translator = sm5.Translator(shader, bindings)
    bindings.graph = translator.g
    tr = _run(translator)
    g = tr.graph
    light = _reads(g, lambda name: name.startswith(('ENGINE.', 'L.', 'LIGHT.')))
    parts, notes = {}, set()
    for c in 'rgb':
        colour = tr.outputs.get(('o', 0, 'rgb'.index(c)))
        if colour is None:
            raise Unsupported('Forward program does not write a colour target')
        args = (g, g.cast(colour, 'f'), light, flags, bindings, graph_data, material_data, tr)
        try:
            parts[c] = _channel_split(*args, notes=notes)
        except Unsupported:   # the albedo may sit inside a product of light terms (key light x AO x albedo): multiply out
            try:
                parts[c] = _channel_split(*args, expand=True, notes=notes)
            except _ManyDiffuse as error:
                raise _ManyDiffuse(str(error), (g, tr, bindings)) from None
    return g, tr, bindings, parts, sorted(notes)


def _channel_split(g, colour, light, flags, bindings, graph_data, material_data, tr, expand=False, notes=None):
    """(F0, Y_d, specular term) of one colour channel (_forward_split); adds the notes it needs to ``notes``."""
    zero = g.const('f', 0.0)
    split = _factor_split(g, colour, light, expand)
    if split is None:
        raise Unsupported('Forward colour is not a sum of light factors')
    unlit, diffuse, specular, dropped = split[0], [], zero, set()
    for factor, y in split[1].items():
        names = {g.nodes[i]['name'] for i in g.reachable([factor]) if g.nodes[i]['op'] == 'input'}
        sources = {n for n in names if n.startswith(('ENGINE.', 'LIGHT.'))}
        if not sources:   # the key light's direction alone (a terminator): not light, part of the unlit colour
            unlit = g.op('add', unlit, g.op('mul', y, factor))
            notes.add(FORWARD_DIRECTION_NOTE)
        elif not flags:   # an unlit program that reads a light source for an effect: 0 under the forward profile
            dropped.update(sources)
        elif any(name.startswith(SPECULAR_INPUTS) for name in names):
            specular = g.op('add', specular, g.op('mul', y, factor))
        else:
            diffuse.append(y)   # a light colour: the key light, ambient fill, GI, the light grid
    if dropped:
        notes.add(FORWARD_UNLIT_LIGHT_NOTE % ', '.join(sorted(n.replace('ENGINE.', '') for n in dropped)[:4]))
    diffuse = list(dict.fromkeys(diffuse))   # factors with one coefficient are one diffuse light, their sum
    if len(diffuse) > 1 and _unlit_zero(g, [g.op('add', y, g.op('neg', diffuse[0])) for y in diffuse[1:]], bindings,
                                        graph_data, material_data, tr):
        diffuse = diffuse[:1]   # the coefficients are one value at every tested input and texel
    if len(diffuse) > 1:
        raise _ManyDiffuse('Forward colour has %d diffuse light factors with different coefficients' % len(diffuse), None)
    return unlit, diffuse[0] if diffuse else zero, specular


def _review_direction(g, terms):
    """``terms`` with the key light's direction L.xyz at REVIEW_KEY_DIRECTION (FORWARD_DIRECTION_NOTE)."""
    direction = dict(zip(('L.x', 'L.y', 'L.z'), REVIEW_KEY_DIRECTION))
    return _set_inputs(g, terms, direction.get)


def _own_light(g, tr):
    """({channel: colour with every light source 0}, {channel: the key-light term}) of a forward program translated with
    _ForwardLitFreeBindings: the term is colour with the ENGINE.* inputs 0 minus that with LIGHT also 0, so it is 0
    without the key light."""
    colours = {c: g.cast(tr.outputs[('o', 0, 'rgb'.index(c))], 'f') for c in 'rgb'}
    keyed = _zero_inputs(g, colours, lambda name: name.startswith('ENGINE.'))
    unlit = _zero_inputs(g, keyed, lambda name: name.startswith('LIGHT.'))
    return unlit, {c: g.op('add', keyed[c], g.op('neg', unlit[c])) for c in 'rgb'}


def forward_albedo(graph_data, material_data, texture_sizes=None):
    """The diffuse coefficient Y_d of a forward technique's program (_forward_split, FORWARD_ALBEDO_NOTE): terms
    albedo.r/g/b, free of the light sources (0 when the program has no diffuse light). When its unlit part F0 is not 0
    (_unlit_zero: emission, or the whole colour of an unlit program), also terms unlit.r/g/b, F0 itself, which a renderer
    adds unlit (FORWARD_UNLIT_NOTE). F0 must not read the scene behind (g_FrameBuffer). A colour that is 0 is drawn black
    (FORWARD_BLACK_NOTE). When the diffuse coefficients differ, the albedo is 0 and the unlit colour is the colour with every
    light source 0 (FORWARD_OWN_LIGHT_NOTE); forward_specular then gives the whole key-light term.

    Returns the same dict as ``surface`` with terms albedo.r/g/b and optionally unlit.r/g/b. Raises Unsupported otherwise."""
    try:
        g, tr, bindings, parts, notes = _forward_split(graph_data, material_data, texture_sizes)
    except _ManyDiffuse as error:
        g, tr, bindings = error.translation
        unlit, _own = _own_light(g, tr)
        unlit = _review_direction(g, unlit)
        terms = {'albedo.' + c: g.const('f', 0.0) for c in 'rgb'}
        if not _unlit_zero(g, list(unlit.values()), bindings, graph_data, material_data, tr):
            terms.update(('unlit.' + c, unlit[c]) for c in 'rgb')
        return _portable(g, terms, bindings, graph_data, material_data, tr, [FORWARD_OWN_LIGHT_NOTE % error],
                         'Forward albedo')
    terms = {'albedo.' + c: parts[c][1] for c in 'rgb'}   # free of the light sources (_factor_split)
    lit = any(g.value(parts[c][1]) != 0.0 for c in 'rgb')
    notes = [FORWARD_ALBEDO_NOTE if lit else FORWARD_UNLIT_ONLY_NOTE] + notes
    unlit = _review_direction(g, {c: parts[c][0] for c in 'rgb'})
    if not _unlit_zero(g, list(unlit.values()), bindings, graph_data, material_data, tr):
        scene = _reads(g, lambda name: name in BACKGROUND_INPUTS)
        if any(scene(x) for x in unlit.values()):
            raise Unsupported('Forward colour\'s unlit part reads the scene behind (%s)' % BACKGROUND_BUFFER)
        terms.update(('unlit.' + c, unlit[c]) for c in 'rgb')
        notes.append(FORWARD_UNLIT_NOTE)
    elif not lit and not any(g.value(parts[c][2]) != 0.0 for c in 'rgb'):
        notes[0] = FORWARD_BLACK_NOTE   # no diffuse, specular or unlit part: the game draws it black
    else:
        notes.append(FORWARD_UNLIT_ZERO_NOTE)
    return _portable(g, terms, bindings, graph_data, material_data, tr, notes, 'Forward albedo')


def forward_specular(graph_data, material_data, texture_sizes=None):
    """The specular term of a lit forward technique's program (_forward_split) lit by the key light alone: terms
    colour.r/g/b of the inputs L.xyz and LIGHT.rgb, every other light source 0 (_zero_inputs) as the forward profile has
    them. Linear in LIGHT and 0 at LIGHT = 0 (checked at random inputs), so a renderer adds colour(L, LIGHT) per light
    (FORWARD_SPECULAR_NOTE).

    Raises Unsupported when the program has no specular term that reads the key light, or as forward_albedo."""
    if not rmaterial.draw_flags(material_data):
        raise Unsupported('The material is unlit: its forward program reads no light')
    try:
        g, tr, bindings, parts, _notes = _forward_split(graph_data, material_data, texture_sizes)
    except _ManyDiffuse as error:   # FORWARD_OWN_LIGHT_NOTE: the whole key-light term
        g, tr, bindings = error.translation
        terms, note = {'colour.' + c: x for c, x in _own_light(g, tr)[1].items()}, FORWARD_OWN_SPECULAR_NOTE
    else:
        terms = _zero_inputs(g, {'colour.' + c: parts[c][2] for c in 'rgb'}, lambda name: name.startswith('ENGINE.'))
        note = FORWARD_SPECULAR_NOTE
    names = {g.nodes[i]['name'] for i in g.reachable(list(terms.values())) if g.nodes[i]['op'] == 'input'}
    if not names & {'LIGHT.r', 'LIGHT.g', 'LIGHT.b'}:
        raise Unsupported('The forward specular term does not read the key light')
    result = _portable(g, terms, bindings, graph_data, material_data, tr, [note], 'Forward specular')
    _check_linear_in_light(result, through_zero=True)
    return result


def _zero_inputs(g, terms, wanted):
    """``terms`` with every input ``wanted(name)`` accepts set to 0 (_set_inputs). A product with 0 folds to 0 when the
    other factor has known bounds (Graph.bounds: never NaN or infinite), so it stays exact: a reflection blend weight times
    a zero reflection is 0."""
    return _set_inputs(g, terms, lambda name: 0.0 if wanted(name) else None)


def _set_inputs(g, terms, value):
    """``terms`` with every input whose ``value(name)`` is not None set to that constant, rebuilt through Graph.op and
    Graph.select so constants fold (a product with 0 as _zero_inputs says)."""
    zero = g.const('f', 0.0)
    replace = {i: g.const('f', value(n['name'])) for i, n in enumerate(g.nodes)
               if n['op'] == 'input' and value(n['name']) is not None}
    if not replace:
        return terms
    order, seen = [], set()
    for root in terms.values():   # post-order without recursion: unrolled loops leave deep chains
        stack = [(root, False)]
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
        elif n['op'] == 'mul' and any(g.value(a) == 0.0 for a in args) and \
                g.bounds(args[1] if g.value(args[0]) == 0.0 else args[0]) is not None:
            new[i] = zero
        elif n['op'] == 'select':
            new[i] = g.select(*args)
        elif n['op'] in sm5.OPS:
            new[i] = g.op(n['op'], *args)
        else:
            meta = {k: v for k, v in n.items() if k not in ('op', 'type', 'args')}
            new[i] = g._add(n['op'], n['type'], args, **meta)
    return {name: new[i] for name, i in terms.items()}


UNIT_INPUTS = ('TEXCOORD.x', 'TEXCOORD.y', 'TEXCOORD1.x', 'TEXCOORD1.y', 'TEXCOORD4.')   # UVs, and the vertex colour (squared)


def _unlit_zero(g, unlit, bindings, graph_data, material_data, tr, trials=24):
    """True when the unlit parts F0 (one per channel) are 0 at random inputs and texels (constant 0, or a portable
    expression that evaluates to 0: terms the program multiplies by a zero light source); False otherwise. UVs and the
    vertex colour are drawn from [0, 1] (UNIT_INPUTS), the other inputs from [-1, 1]: a colour masked by a texel or a UV
    range is not 0."""
    if all(g.value(x) == 0.0 for x in unlit):
        return True
    import random
    try:
        recipe = export_recipe(_portable(g, {'f0.' + c: x for c, x in zip('rgb', unlit)}, bindings, graph_data,
                                         material_data, tr, [], 'Forward unlit colour'))
    except Unsupported:
        return False
    rng = random.Random(7)
    for _trial in range(trials):
        texels = {}
        values = {name: rng.uniform(0.0 if name.startswith(UNIT_INPUTS) else -1.0, 1.0) for name in recipe['inputs']}
        values.update({'VIEW.z': rng.uniform(0.5, 10.0), 'SV_IsFrontFace': True})
        out = evaluate_recipe(recipe, values, lambda node, _a: texels.setdefault(
            node.get('binding'), [rng.uniform(0.0, 1.0) for _ in range(4)]))
        if any(not math.isfinite(v) or abs(v) > 1e-6 for v in out.values()):
            return False
    return True


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


UNIT_IRRADIANCE_NOTE = ('Blender: the scene behind is lit by a unit white diffuse light (background + lit_background), '
                        'not by the scene\'s lights: Cycles has no node for the diffuse light reaching a point. The '
                        'viewer uses its rig\'s diffuse light.')


def unit_irradiance(recipe):
    """A forward recipe with its lit background term folded into the background tint at a unit diffuse light:
    background + lit_background (the Blender stand-in for the renderer's diffuse light, UNIT_IRRADIANCE_NOTE)."""
    terms = recipe['terms']
    if 'lit_background.r' not in terms:
        return recipe
    nodes = [dict(n) for n in recipe['nodes']]
    terms = dict(terms)
    for c in 'rgb':
        nodes.append(dict(op='add', type='f', args=[terms['background.' + c], terms.pop('lit_background.' + c)]))
        terms['background.' + c] = len(nodes) - 1
    return dict(recipe, nodes=nodes, terms=terms)


def blender_recipe(recipe):
    """The recipe as the Blender export builds it: the lit background at a unit light (unit_irradiance), constant NaN
    folded (fold_nan), unrolled tables as lookups."""
    return lookup_tables(fold_nan(unit_irradiance(recipe)))


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
