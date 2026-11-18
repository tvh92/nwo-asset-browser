"""Rift Apart vertex programs of vertex-deforming material graphs (pass slots 3, 7, 10, 13, 17): the at-rest deformation.

A graph whose material moves its vertices (vertex-animation textures, wind sway, rotating debris, holograms) carries its own
vertex programs in place of the engine model vertex program. They pull the vertex themselves: ``SV_VertexID`` indexes the
model's vertex streams (``g_MainVBPosView`` 8-byte int16 x, y, z, w at element 2v, ``g_MainVBNrmTanView`` the packed
normal word at 4v + 2, ``g_MainVBUV0View`` int16 u, v at 4v + 3, ``g_MainVBUV1View`` and ``g_MainColorVBView`` at v), scale
the position by the subset's ``m_MetersPerUnit``, move it, and write the clip position.

``translate`` translates the G-buffer pass's vertex program (``SLOT``) with the SM5 translator, with the streams bound as
per-vertex inputs (``VB.*``), the material's cb2, the neutral object of rc2016.games.rcra.shaders, and a review view whose
world-to-clip matrix is the identity and whose camera sits at the origin, so ``SV_POSITION.xyz`` is the moved position in
model units and ``TEXCOORD2.xyz`` the world normal. Every other engine constant is bound to 0 (``REST_NOTE``): shader time 0,
no wind or ambient animation, no force fields, a plain object. A second translation under ``CHECK_VIEW`` tells a placement
that depends on the view, which is not applied. ``evaluate`` runs a translation on a model's vertices (numpy, float32 as
the GPU), reading the material's textures (``TextureReader``); ``deform_model`` applies it to a decoded model, so the viewer
and every export draw the vertices where the game places them at that state. Motion over time stays a limitation.

The draw's render flags (MatDrawCB) hold 0x2, 0x8000 and the material's lit bit. 0x2 is the base the engine gives a
model draw (RiftApart.exe 0x141038b60; other packet kinds take 0x1). The programs test the two low bits only together,
writing no geometry when both are clear, and every installed vertex program translates the same with 0x1. 0x8000 reads the
model colour stream, as the viewer does.
"""
from __future__ import annotations

import hashlib
import json
import struct

import numpy as np

from rc2016.games.rcra import shaders as rshaders
from rc2016.games.rcra.toc import sections
from rc2016.shaders import sm5

VERTEX_SLOTS = (3, 7, 10, 13, 17)
SLOT = 7                    # the vertex program of the G-buffer pass (the five agree at rest: Research note)
U32 = 0xFFFFFFFF
DRAW_FLAGS = 0x2 | 0x8000   # see the module docstring
VERTEX_ID = 'VERTEX_ID'
# Vertex streams: binding -> (input prefix, channel type, components, element index of vertex v).
STREAMS = {'g_MainVBPosView': ('VB.POS', 'i', 4, lambda v: 2 * v), 'g_MainVBNrmTanView': ('VB.NRM', 'f', 4, lambda v: 4 * v + 2),
           'g_MainVBUV0View': ('VB.UV0', 'i', 2, lambda v: 4 * v + 3), 'g_MainVBUV1View': ('VB.UV1', 'i', 2, lambda v: v),
           'g_MainColorVBView': ('VB.COLOUR', 'f', 4, lambda v: v)}
# Streams the neutral draw never reads (static subset, no paint): reading one is an error of the bindings.
UNREAD_STREAMS = ('g_MainPaintVBView', 'g_SkinVBNrmTanView', 'g_SkinVBPosView', 'g_SkinVBPosPrevView')
FORCE_FIELDS = 'g_ForceFieldsBitsetView'   # per-tile bitset of force fields: empty, no force field reaches the vertex
SUBSET_BUFFER, MATERIAL_BUFFER, DYNAMIC_BUFFER = 'g_ModelSubsetView', 'g_MaterialView', 'g_DynamicObjectView'
VIEW, DRAW_BUFFER = 'GlobalViewportCBuffer.', 'MatDrawCB.'
VIEW_MATRICES = ('m_CamWorldToClipMat', 'm_PrevCamWorldToClipMat')
# The review view: camera at the origin, world-to-clip the identity, so SV_POSITION.xyz is the world position. A second
# view (camera moved, an affine clip matrix; row-vector convention, rows at bytes 0, 16, 32, 48) tells a deformation that
# depends on the view (screen-space glitches, camera-facing quads) from one that does not.
REVIEW_VIEW = ((0.0, 0.0, 0.0), np.eye(4))
CHECK_VIEW = ((13.0, -7.0, 5.0), np.array([[2.0, 0.5, 0.0, 0.0], [0.0, 3.0, 0.25, 0.0], [0.5, 0.0, 0.5, 0.0],
                                            [0.25, -0.5, 0.1, 1.0]]))
CHECK_TIME = 0.75          # seconds of shader time: tells a deformation that moves over time
TIMER = 'GlobalWorldCBuffer.m_ShaderTimer'
REST_NOTE = ('Vertex program at rest: shader time 0, no wind, ambient animation or force fields, a plain object '
             '(identity transform, custom values, random seed and creation time 0); the game moves these vertices over '
             'time, which is not drawn.')
MOVE_TOLERANCE = 1e-5       # model units: a smaller change is the stored vertex


class Unsupported(ValueError):
    pass


def has_vertex_program(graph_data):
    try:
        return SLOT in rshaders.pass_table(graph_data)
    except ValueError:
        return False


def _record(reflection, buffer, values):
    """{byte offset: u32} of a structured-buffer element with ``values`` {member: tuple} (floats as bits, ints as words)."""
    layout = reflection.constant_buffer(buffer) if reflection is not None else None
    if layout is None or not layout.variables or not layout.variables[0].members:
        return None
    words = {at: 0 for at in range(0, layout.size, 4)}
    for name, at, _member in layout.variables[0].members:
        for i, value in enumerate(values.get(name, ())):
            words[at + 4 * i] = sm5.float_bits(value) if isinstance(value, float) else value & U32
    return words


def model_layout(data):
    """The model's stream facts the subset record carries: position scale, the UV-scale word, which streams exist."""
    from rc2016.games.rcra import model as rmodel
    dat1 = data[rmodel.HEADER:]
    table = sections(dat1)
    built = dat1[table[rmodel.BUILT][0]:][:0x34]
    return dict(scale=struct.unpack_from('<f', built, 0x2C)[0], uv_word=struct.unpack_from('<I', built, 0x30)[0],
                uv1=rmodel.UV1 in table, colour=rmodel.COLORS in table)


class _VertexBindings(sm5.Bindings):
    def __init__(self, material, reflection, layout, draw_flags, units=None, texture_sizes=None, view=REVIEW_VIEW,
                 values=None):
        self.graph = None
        self.view, self.values = view, values or {}
        self.material = {rshaders.constant_name(i): v for i, v in enumerate(material or ())}
        self.units, self.texture_sizes = units or {}, texture_sizes
        self.draw_flags = DRAW_FLAGS | draw_flags
        identity = rshaders.NEUTRAL_OBJECT['m_ObjToWorldMatTrans']
        self.records = {
            rshaders.OBJECT_BUFFER: rshaders.object_words(reflection),
            DYNAMIC_BUFFER: _record(reflection, DYNAMIC_BUFFER, {'m_PrevObjToWorldMatTrans': identity}),
            MATERIAL_BUFFER: _record(reflection, MATERIAL_BUFFER, {k: (v,) for k, v in
                                                                   rshaders.NEUTRAL_RECORDS[MATERIAL_BUFFER].items()}),
            SUBSET_BUFFER: _record(reflection, SUBSET_BUFFER, {
                'm_StdVertexIndex': (0,), 'm_VertexColorIndex': (0 if layout['colour'] else -1,),
                'm_VertexUV1Index': (0 if layout['uv1'] else -1,), 'm_PaintingIndex': (-1,), 'm_SkinningIndexPrev': (-1,),
                'm_GeomLodAndUVScales': (layout['uv_word'],), 'm_MetersPerUnit': (float(layout['scale']),)}),
        }
        self.engine = set()      # engine constants bound to 0
        self.errors = []         # stream reads not at the vertex's own element

    def input(self, name):
        if name == 'SV_VertexID.x':
            return self.graph.input(VERTEX_ID, 'i')
        if name == 'SV_InstanceID.x':
            return self.graph.const('i', 0)
        return None

    def constant(self, name):
        g = self.graph
        if name in rshaders.NEUTRAL_INTEGERS:
            return g.const('i', rshaders.NEUTRAL_INTEGERS[name])
        if name in rshaders.NEUTRAL_FLOATS:
            return g.const('f', rshaders.NEUTRAL_FLOATS[name])
        if name in self.material:
            return g.const('f', self.material[name])
        if name.startswith(rshaders.MATERIAL_BUFFER):
            return g.const('f', 0.0)
        if name.startswith(DRAW_BUFFER):   # the instance index is bound to 0, so the per-draw words read statically
            return g.const('f', sm5.bits_float(self.draw_flags) if name == DRAW_BUFFER + 'm_RenderFlags' else 0.0)
        if name.startswith(VIEW):
            member, _, offset = name[len(VIEW):].partition('+')
            offset = int(offset or 0)
            if member in VIEW_MATRICES and offset < 64:
                return g.const('f', float(self.view[1][offset // 16][offset % 16 // 4]))
            if member == 'm_ViewToWorldMat' and offset < 64:   # view axes = world axes, camera at the view's position
                row, column = offset // 16, offset % 16 // 4
                value = self.view[0][column] if row == 3 and column < 3 else float(row == column)
                return g.const('f', float(value))
        if name in self.values:
            return g.const('f', self.values[name])
        self.engine.add(name)
        return g.const('f', 0.0)

    def dynamic_constant(self, buffer, offset, index):
        if buffer == 'MatDrawCB':
            return self.graph.const('i', self.draw_flags if offset % 16 == 0 else 0)
        return None

    def texture_size(self, binding):
        if binding not in self.units or self.texture_sizes is None:
            return None
        try:
            return self.texture_sizes(self.units[binding])
        except (ValueError, KeyError, OSError):
            return None

    def resource(self, graph, op, binding, args):
        if binding in STREAMS and op == 'ld':
            prefix, kind, count, element = STREAMS[binding]
            try:
                at = [sm5.evaluate(graph, [args[0]], {VERTEX_ID: v})[args[0]] for v in (5, 6)]
            except (sm5.Unknown, KeyError) as error:
                self.errors.append('%s address: %s' % (binding, error))
                at = None
            if at != [element(5), element(6)]:
                self.errors.append('%s read at %s, not the vertex\'s own element' % (binding, at))
                return [graph.unknown(binding + ' address', kind)] * 4
            return [graph.input('%s.%s' % (prefix, 'xyzw'[c]), kind) if c < count else graph.const(kind, 0)
                    for c in range(4)]
        if binding in UNREAD_STREAMS:
            return [graph.unknown(binding, 'i')] * 4
        if binding == FORCE_FIELDS and op == 'ld':
            return [graph.const('i', 0)] * 4
        if binding in self.records and op == 'ld_structured':
            words = self.records[binding]
            offset = graph.value(args[4]) if len(args) > 4 else None
            if words is None or offset is None:
                return None
            return [graph.const('i', words.get(offset + 4 * c, 0)) for c in range(4)]
        return None


def _outputs(shader):
    registers = {e.semantic: e.register for e in shader.outputs}
    return {'position': registers.get('SV_POSITION'), 'normal': registers.get('TEXCOORD2'),
            'uv0': registers.get('TEXCOORD'), 'uv1': registers.get('TEXCOORD1'), 'colour': registers.get('TEXCOORD4'),
            'tangent': registers.get('TEXCOORD3')}


TERMS = (('position', 'xyzw'), ('normal', 'xyz'), ('uv0', 'xy'), ('uv1', 'xy'), ('colour', 'xyzw'), ('tangent', 'xyzw'))
TANGENT_TERMS = tuple('tangent.' + c for c in 'xyzw')


def translate(graph_data, material_data, layout, slot=SLOT, texture_sizes=None, view=REVIEW_VIEW, values=None):
    """The vertex program of ``slot`` translated for one material and model layout (``model_layout``) under ``view``, with
    ``values`` {engine constant name: float} in place of 0: {'graph', 'terms' {name: node}, 'units' {binding: texture path}
    the terms read, 'unbound' [bindings], 'engine' [constants bound to 0], 'shader'}. Raises Unsupported when the position
    or normal does not translate. The tangent terms (TEXCOORD3) are left out when they depend on an unknown value."""
    from rc2016.games.rcra.material import draw_flags
    shader = rshaders.pass_shader(graph_data, slot)
    units = rshaders.material_units(graph_data, material_data)
    bindings = _VertexBindings(rshaders.material_constants(graph_data, material_data), shader.reflection, layout,
                               draw_flags(material_data), units, texture_sizes, view, values)
    translator = sm5.Translator(shader, bindings)
    bindings.graph = translator.g
    tr = translator.run()
    g = tr.graph
    registers = _outputs(shader)
    terms = {}
    for name, lanes in TERMS:
        if registers[name] is None:
            continue
        for c, lane in enumerate(lanes):
            node = tr.output(registers[name], c)
            if node is not None:
                terms['%s.%s' % (name, lane)] = node
    needed = [terms.get('position.' + c) for c in 'xyzw'] + [terms.get('normal.' + c) for c in 'xyz']
    if None in needed:
        raise Unsupported('the vertex program writes no position or normal')
    unknown = g.unknowns(needed)
    if unknown or bindings.errors:
        raise Unsupported('; '.join(bindings.errors + unknown))
    tangent = [terms.get(name) for name in TANGENT_TERMS]
    if None in tangent or g.unknowns(tangent):
        for name in TANGENT_TERMS:
            terms.pop(name, None)
    else:
        needed += tangent
    reads = {g.nodes[i]['binding'] for i in g.reachable(needed) if g.nodes[i]['op'] in ('sample', 'load')}
    return dict(graph=g, terms=terms, units={b: units[b] for b in sorted(reads) if b in units},
                unbound=sorted(reads - set(units)), engine=sorted(bindings.engine), shader=shader)


def export(result):
    """A JSON-ready recipe of the translation: reachable nodes renumbered (arguments before their nodes), terms, units."""
    g, terms = result['graph'], result['terms']
    order = sorted(g.reachable(list(terms.values())))
    index = {old: new for new, old in enumerate(order)}
    nodes = []
    for old in order:
        n = {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.nodes[old].items()}
        n['args'] = [index[a] for a in n['args']]
        nodes.append(n)
    return dict(nodes=nodes, terms={k: index[v] for k, v in terms.items()}, units=result['units'],
                unbound=result['unbound'])


# ------------------------------------------------------------------------------------------- vectorised evaluation

F32, U32T, I32 = np.float32, np.uint32, np.int32


def _i(a):
    return np.asarray(a, U32T)


def _s(a):
    return _i(a).view(I32)


def _ftoi(a, low, high):
    x = np.asarray(a, np.float64)
    x = np.where(np.isnan(x), 0.0, np.clip(np.trunc(np.nan_to_num(x, nan=0.0, posinf=high, neginf=low)), low, high))
    return x.astype(np.int64).astype(U32T)


def _bfe(width, offset, value, signed):
    width, offset, value = _i(width) & 31, _i(offset) & 31, _i(value)
    short = (width + offset) < 32
    field = np.where(short, np.left_shift(value, (32 - width - offset) & 31), value)
    shift = (32 - width) & 31
    out = np.right_shift(field.view(I32), shift.astype(I32)).view(U32T) if signed else np.right_shift(field, shift)
    return np.where(width == 0, U32T(0), out).astype(U32T)


def _bfi(width, offset, src, dst):
    width, offset = _i(width).astype(np.uint64) & 31, _i(offset).astype(np.uint64) & 31
    mask = ((((np.uint64(1) << width) - np.uint64(1)) << offset) & np.uint64(U32)).astype(U32T)
    shifted = ((_i(src).astype(np.uint64) << offset) & np.uint64(U32)).astype(U32T)
    return (shifted & mask) | (_i(dst) & ~mask)


def _half(a):
    return (_i(a) & 0xFFFF).astype(np.uint16).view(np.float16).astype(F32)


def _udiv(a, b, rem):
    a, b = _i(a), _i(b)
    safe = np.where(b == 0, U32T(1), b)
    return np.where(b == 0, U32T(U32), a % safe if rem else a // safe).astype(U32T)


def _hi(a, b, signed):
    if signed:
        return ((_s(a).astype(np.int64) * _s(b).astype(np.int64)) >> 32).astype(np.int64).astype(U32T)
    return ((_i(a).astype(np.uint64) * _i(b).astype(np.uint64)) >> np.uint64(32)).astype(U32T)


VECTOR_OPS = {
    'add': lambda a, b: a + b, 'mul': lambda a, b: a * b, 'div': lambda a, b: a / b, 'min': np.fmin, 'max': np.fmax,
    'neg': np.negative, 'abs': np.abs, 'sat': lambda a: np.where(np.isnan(a), F32(0), np.clip(a, 0, 1)),
    'sqrt': np.sqrt, 'rsq': lambda a: F32(1) / np.sqrt(a), 'rcp': lambda a: F32(1) / a, 'log2': np.log2, 'exp2': np.exp2,
    'frc': lambda a: a - np.floor(a), 'round_ne': np.rint, 'round_ni': np.floor, 'round_pi': np.ceil, 'round_z': np.trunc,
    'sin': np.sin, 'cos': np.cos,
    'eq': lambda a, b: a == b, 'ne': lambda a, b: ~(a == b), 'lt': lambda a, b: a < b, 'ge': lambda a, b: a >= b,
    'b_and': np.logical_and, 'b_or': np.logical_or, 'b_not': np.logical_not,
    'iadd': lambda a, b: _i(a) + _i(b), 'imul': lambda a, b: _i(a) * _i(b),
    'umul_hi': lambda a, b: _hi(a, b, False), 'imul_hi': lambda a, b: _hi(a, b, True),
    'udiv': lambda a, b: _udiv(a, b, False), 'urem': lambda a, b: _udiv(a, b, True),
    'ineg': lambda a: U32T(0) - _i(a), 'iabs': lambda a: np.abs(_s(a)).view(U32T),
    'and': lambda a, b: _i(a) & _i(b), 'or': lambda a, b: _i(a) | _i(b), 'xor': lambda a, b: _i(a) ^ _i(b),
    'not': lambda a: ~_i(a), 'ishl': lambda a, b: np.left_shift(_i(a), _i(b) & 31),
    'ishr': lambda a, b: np.right_shift(_s(a), (_i(b) & 31).astype(I32)).view(U32T),
    'ushr': lambda a, b: np.right_shift(_i(a), _i(b) & 31),
    'imin': lambda a, b: np.minimum(_s(a), _s(b)).view(U32T), 'imax': lambda a, b: np.maximum(_s(a), _s(b)).view(U32T),
    'umin': lambda a, b: np.minimum(_i(a), _i(b)), 'umax': lambda a, b: np.maximum(_i(a), _i(b)),
    'ieq': lambda a, b: _i(a) == _i(b), 'ine': lambda a, b: _i(a) != _i(b), 'ilt': lambda a, b: _s(a) < _s(b),
    'ige': lambda a, b: _s(a) >= _s(b), 'ult': lambda a, b: _i(a) < _i(b), 'uge': lambda a, b: _i(a) >= _i(b),
    'itof': lambda a: _s(a).astype(F32), 'utof': lambda a: _i(a).astype(F32),
    'ftoi': lambda a: _ftoi(a, -2 ** 31, 2 ** 31 - 1), 'ftou': lambda a: _ftoi(a, 0, U32),
    'f32tof16': lambda a: np.asarray(a, F32).astype(np.float16).view(np.uint16).astype(U32T), 'f16tof32': _half,
    'ubfe': lambda w, o, v: _bfe(w, o, v, False), 'ibfe': lambda w, o, v: _bfe(w, o, v, True), 'bfi': _bfi,
    'f2bits': lambda a: np.asarray(a, F32).view(U32T), 'bits2f': lambda a: _i(a).view(F32),
    'b2i': lambda a: np.where(a, U32T(U32), U32T(0)),
}
DTYPES = {'f': F32, 'i': U32T, 'b': np.bool_}


def _typed(value, kind):
    value = np.asarray(value)
    if kind == 'i' and value.dtype.kind in 'iu':
        return value.astype(np.int64).astype(U32T)
    return value.astype(DTYPES[kind])


def evaluate(nodes, roots, inputs, resource):
    """Evaluate recipe ``nodes`` (arguments before their nodes) over arrays: ``inputs`` {name: array or scalar},
    ``resource(node, args) -> 4 arrays``. Returns {root: array}."""
    needed = set()
    stack = list(roots)
    while stack:
        i = stack.pop()
        if i not in needed:
            needed.add(i)
            stack.extend(nodes[i]['args'])
    values = {}
    with np.errstate(all='ignore'):
        for i in sorted(needed):
            n = nodes[i]
            op, kind = n['op'], n['type']
            args = [values[a] for a in n['args']]
            if op == 'const':
                v = n['value']
            elif op == 'input':
                if n['name'] not in inputs:
                    raise Unsupported('no value for ' + n['name'])
                v = inputs[n['name']]
            elif op == 'select':
                v = np.where(args[0], args[1], args[2])
            elif op == 'channel':
                v = args[0][n['index']]
            elif op in ('sample', 'load'):
                v = resource(n, args)
                values[i] = v
                continue
            elif op in VECTOR_OPS:
                v = VECTOR_OPS[op](*args)
            elif op in sm5.OPS and sm5.OPS[op][2] is not None:
                fn = sm5.OPS[op][2]
                v = np.frompyfunc(lambda *a, fn=fn: fn(list(a)), len(args), 1)(*[np.asarray(a).astype(object) for a in args])
            else:
                raise Unsupported('%s %s' % (op, n.get('reason', '')))
            values[i] = _typed(v, kind)
    return {r: values[r] for r in roots}


# ------------------------------------------------------------------------------------------------ streams, textures


def vertex_streams(data):
    """The model's raw vertex streams over the whole vertex buffer, as the engine's views read them: inputs VB.* and
    the decoder's position offset."""
    from rc2016.games.rcra import model as rmodel
    dat1 = data[rmodel.HEADER:]
    table = sections(dat1)

    def chunk(tag):
        if tag not in table:
            return b''
        offset, size = table[tag]
        return dat1[offset:offset + size]
    records = np.frombuffer(chunk(rmodel.VERTICES), rmodel.VERTEX)
    count = len(records)
    words = records['normal'].astype(U32T)
    inputs = {'VB.POS.' + c: records['xyzw'][:, k].astype(np.int64) for k, c in enumerate('xyzw')}
    # The normal word as four UNORM channels, 10, 10, 10 and 2 bits (R10G10B10A2): the programs rebuild the normal from
    # them as the decoder does (rc2016.games.rcra.model.unpack_normals), which a stored-normal pass-through checks.
    for k, (shift, bits) in enumerate(((0, 10), (10, 10), (20, 10), (30, 2))):
        inputs['VB.NRM.' + 'xyzw'[k]] = (((words >> shift) & ((1 << bits) - 1)).astype(F32) / F32((1 << bits) - 1))
    inputs['VB.UV0.x'], inputs['VB.UV0.y'] = (records['uv'][:, k].astype(np.int64) for k in (0, 1))
    uv1 = chunk(rmodel.UV1)
    if uv1:
        second = np.frombuffer(rmodel._per_vertex(uv1, count, 'UV1'), '<i2').reshape(-1, 2)
        inputs['VB.UV1.x'], inputs['VB.UV1.y'] = second[:, 0].astype(np.int64), second[:, 1].astype(np.int64)
    colours = chunk(rmodel.COLORS)
    if colours:
        rgba = np.frombuffer(rmodel._per_vertex(colours, count, 'Vertex colour'), np.uint8).reshape(-1, 4)
        for k, c in enumerate('xyzw'):
            inputs['VB.COLOUR.' + c] = rgba[:, k].astype(F32) / F32(255)
    built = chunk(rmodel.BUILT)
    return inputs, np.array(struct.unpack_from('<3f', built, 0x1C), np.float64), count


class TextureReader:
    """The material textures a vertex program reads, as stored (float32, top mip): ``ld`` returns the texel (0 outside
    the texture, as D3D), ``sample_l`` is bilinear with wrap at the top mip (the sampler state and mip level are not
    applied, labelled)."""

    def __init__(self, source, units):
        self.source, self.units, self.cache, self.notes = source, units, {}, set()

    def _slice(self, binding, index):
        from rc2016.games.rcra.blender_export import texel_array
        from rc2016.games.rcra.texture import load_texture_full
        path = self.units.get(binding)
        if path is None:
            raise Unsupported('vertex program reads %s, which the material does not bind' % binding)
        if path not in self.cache:
            self.cache[path] = load_texture_full(self.source, path)
        texture = self.cache[path]
        key = (path, index)
        if key not in self.cache:
            self.cache[key] = texel_array(texture, index)
        return self.cache[key]

    def __call__(self, node, args):
        opcode = node.get('opcode')
        n = max(np.size(a) for a in args)
        if opcode == 'ld':
            x, y = (_s(np.broadcast_to(a, (n,))) for a in args[:2])
            slices = np.broadcast_to(_s(args[2]), (n,))
            mips = np.unique(_s(args[3]))
            if mips.tolist() != [0]:
                raise Unsupported('vertex texture load below the top mip')
            out = np.zeros((n, 4), F32)
            for index in np.unique(slices):
                texels = self._slice(node['binding'], int(index))
                h, w = texels.shape[:2]
                pick = (slices == index) & (x >= 0) & (y >= 0) & (x < w) & (y < h)
                out[pick] = texels[y[pick], x[pick]]
            return [out[:, c] for c in range(4)]
        if opcode in ('sample_l', 'sample'):
            u, v = (np.broadcast_to(np.asarray(a, F32), (n,)).astype(np.float64) for a in args[:2])
            slices = np.broadcast_to(np.rint(np.asarray(args[2], np.float64)), (n,)) if node.get('dimension') in (7, 8) \
                else np.zeros(n)
            out = np.zeros((n, 4), F32)
            self.notes.add('bilinear')
            for index in np.unique(slices):
                texels = self._slice(node['binding'], int(index)).astype(np.float64)
                h, w = texels.shape[:2]
                pick = slices == index
                fx, fy = u[pick] * w - 0.5, v[pick] * h - 0.5
                x0, y0 = np.floor(fx), np.floor(fy)
                tx, ty = (fx - x0)[:, None], (fy - y0)[:, None]
                x0, y0 = x0.astype(np.int64) % w, y0.astype(np.int64) % h
                x1, y1 = (x0 + 1) % w, (y0 + 1) % h
                out[pick] = ((texels[y0, x0] * (1 - tx) + texels[y0, x1] * tx) * (1 - ty)
                             + (texels[y1, x0] * (1 - tx) + texels[y1, x1] * tx) * ty)
            return [out[:, c] for c in range(4)]
        raise Unsupported('vertex texture operation %s' % opcode)


# ------------------------------------------------------------------------------------------------------ application


def run(recipe, inputs, reader, count):
    """{term: array of ``count``} of a recipe on per-vertex ``inputs`` (a constant term repeats)."""
    values = evaluate(recipe['nodes'], list(recipe['terms'].values()), inputs, reader)
    return {name: np.broadcast_to(values[i], (count,)) for name, i in recipe['terms'].items()}


def _code_sha256():
    from pathlib import Path
    package = Path(__file__).resolve().parents[2]
    # The cached recipes also hold the slot binding (material.py) and the missing-texture stand-ins (blender_export.py,
    # texture.py).
    files = [Path(__file__), package / 'shaders' / 'sm5.py', package / 'shaders' / 'dxbc.py'] + [
        package / 'games' / 'rcra' / name for name in ('shaders.py', 'material.py', 'blender_export.py', 'texture.py')]
    try:
        return hashlib.sha256(b''.join(f.read_bytes() for f in files)).hexdigest()
    except OSError:
        return None


CODE_SHA256 = _code_sha256()


def recipes(graph_data, material_data, layout, texture_sizes=None):
    """{'rest': recipe under the review view, 'view': under CHECK_VIEW} of one material and model layout."""
    return {name: export(translate(graph_data, material_data, layout, texture_sizes=texture_sizes, view=view))
            for name, view in (('rest', REVIEW_VIEW), ('view', CHECK_VIEW))}


def material_recipes(source, graph_data, material_data, layout):
    """recipes() with each texture the install does not hold replaced as the pixel side does
    (rc2016.games.rcra.blender_export.replace_missing); each recipe gains 'limitations'."""
    from rc2016.games.rcra.blender_export import replace_missing
    from rc2016.games.rcra.texture import texture_size
    pair = recipes(graph_data, material_data, layout, texture_sizes=lambda p: texture_size(source, p))
    for recipe in pair.values():
        result = dict(textures=recipe['units'], limitations=[])
        replace_missing(source, graph_data, result)
        recipe['units'], recipe['limitations'] = result['textures'], result['limitations']
    return pair


def cached_recipes(source, graph_data, material_data, layout):
    """material_recipes() (or {'error'}), kept in the source's translation disk cache when it has one (non-finite
    constants encoded as the viewer's material receipts encode them)."""
    from rc2016.games.rcra.viewer_materials import _floats
    disk = getattr(source, 'translation_disk_cache', None)
    variant = 'rcra-vertex:%s:%s' % (CODE_SHA256, json.dumps(layout, sort_keys=True))
    key = disk.key(bytes(material_data), [bytes(graph_data)], variant) if disk is not None and CODE_SHA256 else None
    stored = _floats(disk.get(key), decode=True) if key else None
    if stored is not None:
        return stored
    try:
        out = material_recipes(source, graph_data, material_data, layout)
    except (Unsupported, sm5.Unknown, ValueError, KeyError) as error:
        out = dict(error=str(error))
    if key:
        disk.put(key, _floats(out))
    return out


def world(out, view):
    """World positions (model units) of a run's clip positions under ``view``; None where the clip w is not 1."""
    clip = np.stack([out['position.' + c] for c in 'xyzw'], 1).astype(np.float64)
    if not np.allclose(clip[:, 3], 1.0):
        return None
    camera, matrix = np.asarray(view[0], np.float64), np.asarray(view[1], np.float64)
    return (clip[:, :3] - matrix[3, :3]) @ np.linalg.inv(matrix[:3, :3]) + camera


def unit_normals(out, fallback):
    normal = np.stack([out['normal.' + c] for c in 'xyz'], 1).astype(np.float64)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    return np.where(length > 1e-12, normal / np.maximum(length, 1e-12), np.asarray(fallback, np.float64))


def at_rest(pair, inputs, reader, rows, offset, normals, with_tangents=False):
    """(positions, normals) of the vertices ``rows`` (indices into the whole vertex buffer) at rest, positions in the
    decoder's frame (plus its offset); None when where they go depends on the view (screen-space effects, camera-facing
    geometry: no at-rest shape). ``normals``: the stored normals, kept where the program writes a zero normal.
    ``with_tangents`` adds the program's tangent (TEXCOORD3.xyz) and TEXCOORD3.w, or (None, None) when the recipe has no
    tangent terms."""
    local = {k: (v[rows] if np.ndim(v) else v) for k, v in inputs.items()}
    local[VERTEX_ID] = rows
    rest, check = run(pair['rest'], local, reader, len(rows)), run(pair['view'], local, reader, len(rows))
    positions, other = world(rest, REVIEW_VIEW), world(check, CHECK_VIEW)
    if positions is None or other is None:
        return None
    normal = unit_normals(rest, normals)
    if not (np.all(np.isfinite(positions)) and np.all(np.isfinite(normal))):
        raise Unsupported('the at-rest position or normal is not finite')
    span = max(float(np.ptp(positions, axis=0).max(initial=0.0)), 1.0)
    if (np.max(np.abs(positions - other), initial=0.0) > 1e-4 * span
            or np.max(np.abs(normal - unit_normals(check, normals)), initial=0.0) > 1e-3):
        return None
    if not with_tangents:
        return positions + offset, normal
    tangent = w = None
    if all(name in rest for name in TANGENT_TERMS):
        tangent = np.stack([rest['tangent.' + c] for c in 'xyz'], 1).astype(np.float64)
        w = np.asarray(rest['tangent.w'], np.float64)
        if not (np.all(np.isfinite(tangent)) and not np.any(np.isnan(w))):
            tangent = w = None
        else:   # e = 0 is the engine's infinity: the largest float32 stands in, as in model.unpack_tangents
            limit = float(np.finfo(np.float32).max)
            w = np.clip(w, -limit, limit)
    return positions + offset, normal, tangent, w


def moved(positions, normals, stored_positions, stored_normals):
    """False when the at-rest vertices are the stored ones (within MOVE_TOLERANCE, normals within 1e-4)."""
    return not (np.max(np.abs(positions - np.asarray(stored_positions, np.float64)), initial=0.0) <= MOVE_TOLERANCE
                and np.max(np.abs(normals - np.asarray(stored_normals, np.float64)), initial=0.0) <= 1e-4)


MOVED_NOTE = ('Vertex program (pass slot %d) applied at rest: the drawn vertices and normals are where it places them. '
              + REST_NOTE + ' Exports keep the stored vertices as the shape key "%s" (weight 0).')
BILINEAR_NOTE = (' Textures it samples with filtering are read bilinearly at their top mip with wrap (the sampler state '
                 'and mip level are not applied).')
VIEW_NOTE = ('Vertex program (pass slot %d) not applied: where it places the vertices depends on the view (screen-space or '
             'camera-facing), so the stored vertices are drawn.')
ERROR_NOTE = 'Vertex program (pass slot %d) not applied (%s): the stored vertices are drawn.'
SKINNED_NOTE = (' The mesh is skinned: the at-rest offset is added to the bind pose, so joints carry it; the game adds it '
                'after skinning, in object space.')
STORED_SHAPE = 'stored vertices'   # rc2016.export.models.STORED_SHAPE
STORED_TANGENT_NOTE = (' Its tangent output does not translate, so the stored tangent is drawn with the moved normal.')


def _assign(mesh, key, values):
    array = np.asarray(values, np.float64)
    mesh[key] = array if isinstance(mesh[key], np.ndarray) else list(map(tuple, array.tolist()))


def _set_tangents(mesh, tangents, w):
    """The program's TEXCOORD3 as the mesh's tangent stream (rc2016.games.rcra.model fields tangents, tangent_signs,
    tangent_w)."""
    signs = np.where(w < 0, -1.0, 1.0)
    arrays = isinstance(mesh.get('positions'), np.ndarray)
    mesh['tangents'] = tangents if arrays else list(map(tuple, tangents.tolist()))
    mesh['tangent_signs'] = signs if arrays else signs.tolist()
    mesh['tangent_w'] = w if arrays else w.tolist()


def _material_meshes(source, decoded):
    """{material key: (graph, path, [meshes])} of the meshes whose material has a vertex program."""
    from rc2016.games.rcra.material import _material_key
    from rc2016.games.rcra.viewer_materials import NotTranslated, _graph
    groups, keys = {}, {}
    for mesh in decoded['meshes']:
        path = (mesh.get('material') or {}).get('path')
        if path not in keys:
            key = _material_key(source, path) if path else None
            graph = None
            if key is not None:
                try:
                    graph = _graph(source, key)
                except (NotTranslated, ValueError, KeyError, OSError):
                    graph = None
            keys[path] = key if graph is not None and has_vertex_program(graph) else None
            if keys[path] is not None:
                groups[key] = (graph, path, [])
        if keys[path] is not None:
            groups[keys[path]][2].append(mesh)
    return groups


def _group_states(source, key, graph, meshes, layout, inputs, offset):
    """{id(mesh): (state, note)} of one material's meshes, run as one batch; moved meshes are updated in place."""
    pair = cached_recipes(source, graph, source.asset_data(key), layout)
    if 'error' in pair:
        return {id(m): ('error', ERROR_NOTE % (SLOT, pair['error'])) for m in meshes}
    reader = TextureReader(source, pair['rest']['units'])
    spans = [np.arange(m['first_vertex'], m['first_vertex'] + m['vertex_count'], dtype=np.int64) for m in meshes]
    stored = np.concatenate([np.asarray(m['normals'], np.float64).reshape(-1, 3) for m in meshes])
    try:
        result = at_rest(pair, inputs, reader, np.concatenate(spans), offset, stored, with_tangents=True)
    except (Unsupported, sm5.Unknown, ValueError, KeyError, OSError) as error:
        return {id(m): ('error', ERROR_NOTE % (SLOT, error)) for m in meshes}
    if result is None:
        return {id(m): ('view', VIEW_NOTE % SLOT) for m in meshes}
    extra = (BILINEAR_NOTE if 'bilinear' in reader.notes else '') + ''.join(
        ' ' + n for n in pair['rest'].get('limitations', []))
    states, at = {}, 0
    for mesh, span in zip(meshes, spans):
        positions, normals, tangents, w = (r[at:at + len(span)] if r is not None else None for r in result)
        at += len(span)
        if not moved(positions, normals, mesh['positions'], mesh['normals']):
            states[id(mesh)] = ('still', None)
            continue
        old = np.asarray(mesh['positions'], np.float64)
        mesh['stored_positions'], mesh['stored_normals'] = mesh['positions'], mesh['normals']
        _assign(mesh, 'positions', positions)
        _assign(mesh, 'normals', normals)
        if tangents is not None:
            _set_tangents(mesh, tangents, w)
        mesh['vertex_offset'] = float(np.max(np.linalg.norm(positions - old, axis=1), initial=0.0))
        states[id(mesh)] = ('moved', MOVED_NOTE % (SLOT, STORED_SHAPE) + extra
                            + (SKINNED_NOTE if mesh.get('skinned') else '')
                            + (STORED_TANGENT_NOTE if tangents is None and mesh.get('tangents') is not None else ''))
    return states


def deform_model(source, decoded, data):
    """Apply the at-rest deformation of each mesh's vertex program to a decoded Rift Apart model, in place. A moved mesh
    keeps its stored vertices and normals as ``stored_positions`` / ``stored_normals`` and its largest offset as
    ``vertex_offset``. Sets ``decoded['vertex_programs']`` (one row per mesh whose material has a vertex program: mesh,
    material, state 'moved', 'still', 'view' or 'error', note) and ``decoded['material_notes']`` {material path: [note]}."""
    rows, notes = [], {}
    groups = _material_meshes(source, decoded)
    if groups:
        layout = model_layout(data)
        inputs, offset, _count = vertex_streams(data)
    for key, (graph, path, meshes) in groups.items():
        states = _group_states(source, key, graph, meshes, layout, inputs, offset)
        for mesh in meshes:
            state, note = states[id(mesh)]
            rows.append(dict(mesh=mesh['index'], material=path, state=state, note=note,
                             max_offset=mesh.get('vertex_offset', 0.0)))
            if note and note not in notes.setdefault(path, []):
                notes[path].append(note)
    decoded['vertex_programs'] = sorted(rows, key=lambda r: r['mesh'])
    decoded['material_notes'] = notes
    return decoded
