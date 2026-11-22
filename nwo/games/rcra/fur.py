"""Rift Apart fur materials: no material graph, a fur record and the engine's fur programs in the executable.

A fur material (19 in the install: Ratchet's and Rivet's head, ears and limbs, Daxter, the sheep, a carpet, a towel, a
cotton canopy and a cloud) has DAT1 section 0xD9B12454 instead of the serialized parameter section. The record is 13 u32:
the layer count (32 on Ratchet's head), eight floats (FUR_FIELDS) and four offsets into the material's DAT1 of the texture
paths, in the order base colour (_c), normal (_n), gloss (_g) and fur control (19 of 19 materials). RiftApart.exe names
them: the material loader (0x1410d196c) copies the nine values into its fur object, whose fields the per-instance override
pass (0x1410cec20) writes by name hash: Fur_LayerCount, Fur_LoDReduction, Fur_Length, Fur_Density, Fur_OffsetScale,
Fur_GlossScale, Fur_SpecularScale, Fur_TransmittanceScale and Fur_Wetness, then Fur_WindStrength and Fur_WindTurbulence
(0 from the file), Fur_BaseMap, Fur_NormalMap, Fur_GlossMap and Fur_ControlMap.

The executable holds the programs (found by their reflection names, nwo.shaders.dxbc): the fur G-buffer writer, a
ps_5_0 with constant buffer MatrialFurCBuffer (g_Fur: m_Density, m_Length, m_GlossScale, m_SpecularScale,
m_TransmittanceScale, m_Wetness), textures g_BaseMap (t5), g_FurLayerMap (t7, 3D), g_FurGlossMap (t8) and g_FurControlMap
(t9), and the engine's four G-buffer targets; a five-target variant; and the shell vertex programs (ModelFurCBuffer). The
writer binds no normal map: its normal is the vertex normal.

The fur draw (command 0x21 "ModelFurShellGBufferDeferred", 0x1410ee6b0) copies the fur object into g_Fur unchanged:
m_Density, m_Length, m_GlossScale, m_SpecularScale, m_TransmittanceScale and m_Wetness (fur_constants), and binds the base,
normal, gloss and control maps at t5, t6, t8 and t9. The writer's surface terms read m_GlossScale, m_SpecularScale (F0
and gloss) and m_Wetness (albedo). Every installed record holds wetness 0; gameplay can raise it at run time (the
override pass's Fur_Wetness, the FurWetness settings), which is not modelled.

Shells. The fur draw renders the fur meshes Fur_LayerCount times, instanced (ModelFurCBuffer: m_BaseOffsetMapScale =
Fur_OffsetScale, m_InverseLayerCount, m_Length, wind). The shell vertex program (VS_ModelFurShellGBufferDeferred) gives
instance i the layer k = L - 1 - i at height t = k / L, spread by the view: t' = sat(t / s), with
s = min((1 - sat(N.(-V) - 0.0065 q - 0.1))^2 + 0.1, 1) (q the stored TEXCOORD3.w magnitude), and culls a layer whose t / s
exceeds 1. Facing the camera a few layers span the strand; at the silhouette (s = 1) every layer is drawn at t' = t. It
moves the vertex along its object normal by control.b x t' x m_Length (the control map point-sampled at UV0, mip 0) and
passes the writer TEXCOORD.z = 32 t', TEXCOORD1.w = m_BaseOffsetMapScale (signed by the bitangent), TEXCOORD2.w =
(t'^2 + 0.4 t') m_Length m_BaseOffsetMapScale and TEXCOORD3.w = 1 on the base layer (k = 0), else 0 (shell_layer).

The writer cuts each layer: it reads g_FurLayerMap, the engine's "Default Fur Shells" 2D array (layer_map: 32 slices of
128 x 128 R8, slice round(TEXCOORD.z)), at uv0 x m_Density shifted along the control map's comb direction
((control.rg x 2 - 1) x |TEXCOORD1.w|) by TEXCOORD2.w, and keeps the pixel when
sat(layer x base.a + TEXCOORD3.w) x sat(1 - t' / 2) exceeds a screen-space dither threshold (0.5 far away, through
TEXCOORD.w). The game's temporal filter averages the dither, so a renderer draws that product as coverage
(shell_coverage, the surface's ``coverage`` term). The layer's occlusion is lerp(control.a, 1, t' / 2); its other terms
are the base surface's when dry. Layer 0 is the base surface itself (coverage 1).
"""
import os
import struct
import threading

import numpy as np

from nwo.games.rcra.toc import sections

FUR_RECORD = 0xD9B12454
EXECUTABLE = 'RiftApart.exe'
FUR_BUFFER = 'MatrialFurCBuffer'    # sic, the engine's name
FUR_UNITS = {0: 'g_BaseMap', 2: 'g_FurGlossMap', 3: 'g_FurControlMap'}   # record texture index -> writer binding
LAYER_BINDING, CONTROL_BINDING, BASE_BINDING = 'g_FurLayerMap', 'g_FurControlMap', 'g_BaseMap'
# The engine's "Default Fur Shells" texture (RiftApart.exe 0x141354a4b): a 2D array of 32 R8 slices (DXGI 61) with four
# mips each, built at start-up, so it is a constant texture of nwo.games.rcra.texture.
LAYER_MAP = 'engine/default_fur_shells.texture'
LAYER_SIZE, LAYER_SLICES, LAYER_MIPS, LAYER_FORMAT = 128, 32, 4, 61
XORSHIFT_SEED = (123456789, 362436069, 521288629, 88675123)   # 0x141598b60, the canonical xorshift128 state
# The eight floats of the record, in order (the engine's parameter names without "Fur_").
FUR_FIELDS = ('LoDReduction', 'Length', 'Density', 'OffsetScale', 'GlossScale', 'SpecularScale', 'TransmittanceScale',
              'Wetness')
# g_Fur member <- record field, as the fur draw copies them.
FUR_CONSTANTS = (('m_Density', 'Density'), ('m_Length', 'Length'), ('m_GlossScale', 'GlossScale'),
                 ('m_SpecularScale', 'SpecularScale'), ('m_TransmittanceScale', 'TransmittanceScale'),
                 ('m_Wetness', 'Wetness'))
FUR_NOTE = ('Fur material: the engine\'s fur G-buffer writer from the executable (constant buffer %s, filled from the '
            'material\'s fur record as the engine does), as its base layer (layer 0, coverage 1). Wetness is the '
            'record\'s (dry); gameplay can raise it at run time. The writer binds no normal map (vertex normal). The '
            'shell layers are drawn where the renderer builds them (shell_program); a plain mesh export keeps the base '
            'layer only.' % FUR_BUFFER)
SHELL_NOTE = ('Fur shell layer %d of %d: the fur writer with the shell vertex program\'s interpolants at height t = %.4g '
              '(every layer drawn, the spacing the game uses at the silhouette; facing the camera it draws fewer, wider-'
              'spaced layers), the engine layer map read at its top mip (the writer\'s view-dependent mip bias is not '
              'applied), and the writer\'s dithered cut drawn as coverage, as the game\'s temporal filter averages it.')
PIXEL = 0xFFFF
_LOCK = threading.Lock()
_WRITERS = {}


def fur_record(data):
    """(dword 0, [8 floats], [4 texture paths]) of a fur material, or None when ``data`` has no fur record."""
    dat1 = bytes(data[36:])
    try:
        table = sections(dat1)
    except ValueError:
        return None
    if FUR_RECORD not in table:
        return None
    offset, size = table[FUR_RECORD]
    if size < 52:
        raise ValueError('Fur record is too short')
    first = struct.unpack_from('<I', dat1, offset)[0]
    floats = list(struct.unpack_from('<8f', dat1, offset + 4))
    paths = []
    for at in struct.unpack_from('<4I', dat1, offset + 36):
        stop = dat1.find(b'\0', at)
        if at >= len(dat1) or stop < 0:
            raise ValueError('Fur texture string outside the material')
        paths.append(dat1[at:stop].decode('latin-1').replace('\\', '/').lower())
    return first, floats, paths


def fur_constants(record):
    """{'MatrialFurCBuffer.<member>': value} of a fur_record, as the engine's fur draw fills g_Fur."""
    fields = dict(zip(FUR_FIELDS, record[1]))
    return {FUR_BUFFER + '.' + member: float(fields[field]) for member, field in FUR_CONSTANTS}


def _writer(blob_source):
    """(offset, bytes) of the fur G-buffer writer in the executable bytes: the pixel program with FUR_BUFFER and four
    targets, found from each occurrence of the buffer name back to its DXBC container."""
    from nwo.shaders import dxbc
    data = blob_source
    found = {}
    at = data.find(FUR_BUFFER.encode())
    while at >= 0:
        start = data.rfind(dxbc.DXBC_MAGIC, max(0, at - (1 << 20)), at)
        if start >= 0:
            size = struct.unpack_from('<I', data, start + 24)[0]
            if start + size > at:
                blob = bytes(data[start:start + size])
                try:
                    shader = dxbc.shader(blob)
                except dxbc.DxbcError:
                    shader = None
                if (shader is not None and shader.reflection is not None and shader.reflection.program_type == PIXEL
                        and shader.reflection.constant_buffer(FUR_BUFFER) is not None
                        and sorted(e.index for e in shader.outputs if e.name.upper() == 'SV_TARGET') == [0, 1, 2, 3]):
                    found[start] = blob
        at = data.find(FUR_BUFFER.encode(), at + 1)
    if len({bytes(b) for b in found.values()}) != 1:
        raise ValueError('Expected one fur G-buffer writer in the executable, found %d' % len(found))
    start = min(found)
    return start, found[start]


def fur_writer(root):
    """The fur G-buffer writer's DXBC bytes from the install's executable (cached by path, size and time)."""
    path = os.path.join(str(root), EXECUTABLE)
    stat = os.stat(path)
    key = (path, stat.st_size, stat.st_mtime_ns)
    with _LOCK:
        if key not in _WRITERS:
            with open(path, 'rb') as f:
                data = f.read()
            _WRITERS[key] = _writer(data)[1]
        return _WRITERS[key]


def fur_program(source, data):
    """The surface program of a fur material (nwo.games.rcra.surface.surface ``program``): dict(shader, cb2, units,
    named, limitations, key); raises ValueError when ``data`` is not a fur material or the writer is not found."""
    from nwo.shaders import dxbc
    record = fur_record(data)
    if record is None:
        raise ValueError('Not a fur material')
    blob = fur_writer(source.root)
    return dict(shader=dxbc.shader(blob), cb2=(), units={binding: record[2][i] for i, binding in FUR_UNITS.items()},
                named=fur_constants(record), limitations=[FUR_NOTE], key=blob, discard=False)


def _xorshift128(state):
    """One step of the engine's generator (0x141598be0): returns (value, state)."""
    x, y, z, w = state
    t = (x ^ (x << 11)) & 0xFFFFFFFF
    w2 = (w ^ (w >> 19) ^ t ^ (t >> 8)) & 0xFFFFFFFF
    return w2, (y, z, w, w2)


def layer_map():
    """The 32 slices (128 x 128 bytes, rows y, x fastest) of the engine's Default Fur Shells array, as the start-up routine
    builds them in float32: per texel column, x outer and y inner, a strand of n = min(floor(32 (r / 2^24)^2) + 1, 32)
    layers (r the next random number >> 8), layer d < n holding floor(255 sat(1 - 0.8 max(2 (d + 1) / n - 1, 0)^2))."""
    with _LOCK:
        if 'layer_map' not in _WRITERS:
            f = np.float32
            scale, fall, full = f(2.0 ** -43), f(0.8), f(255.0)
            # A layer's byte depends only on (n, d): levels[n, d], computed in float32 as the routine does (0 for d >= n).
            levels = np.zeros((LAYER_SLICES + 1, LAYER_SLICES), np.uint8)
            for n in range(1, LAYER_SLICES + 1):
                inv = f(1.0) / f(n)
                for d in range(n):
                    h = max(f(f(f(f(d + 1) + f(d + 1)) * inv) - f(1.0)), f(0.0))
                    level = min(max(f(1.0) - f(f(h * h) * fall), f(0.0)), f(1.0))
                    levels[n, d] = int(f(level * full))
            values, state = [], XORSHIFT_SEED   # one number per texel column, x outer and y inner
            for _ in range(LAYER_SIZE * LAYER_SIZE):
                value, state = _xorshift128(state)
                values.append(value >> 8)
            r = np.array(values, np.uint32).astype(f).reshape(LAYER_SIZE, LAYER_SIZE).T   # [y, x]; exact (< 2^24)
            n = np.minimum(((r * r) * scale).astype(np.int64) + 1, LAYER_SLICES)        # float32 products, truncated
            data = np.ascontiguousarray(levels[n].transpose(2, 0, 1))                     # [d, y, x]
            _WRITERS['layer_map'] = [bytes(data[d]) for d in range(LAYER_SLICES)]
        return _WRITERS['layer_map']


def layer_count(record):
    """Fur_LayerCount: the layers the fur draw renders at full detail (Fur_LoDReduction removes some with distance)."""
    return int(record[0])


def shell_layer(record, k):
    """The shell vertex program's values for layer ``k`` at the silhouette spacing (s = 1, every layer drawn):
    dict(layer, layers, t, displace (metres along the normal, times control.b), interpolants {writer input: value})."""
    f = np.float32
    fields = dict(zip(FUR_FIELDS, record[1]))
    layers = layer_count(record)
    if not 0 <= k < layers:
        raise ValueError('Fur layer %d outside 0..%d' % (k, layers - 1))
    inverse = f(1.0) / f(layers)
    t = float(min(max(f(f(k) * inverse), f(0.0)), f(1.0)))
    length, offset = f(fields['Length']), f(fields['OffsetScale'])
    term = float(f(f(f(t) * f(t) + f(t) * f(0.4)) * length) * offset)
    return dict(layer=k, layers=layers, t=t, displace=float(f(t) * length),
                interpolants={'TEXCOORD.z': float(f(t) * f(32.0)), 'TEXCOORD.w': 1.0, 'TEXCOORD1.w': float(offset),
                              'TEXCOORD2.w': term, 'TEXCOORD3.w': 1.0 if k == 0 else 0.0})


def _sample_node(g, binding):
    found = [i for i, n in enumerate(g.nodes) if n['op'] == 'sample' and n['binding'] == binding]
    if len(found) != 1:
        raise ValueError('Expected one %s sample in the fur writer, found %d' % (binding, len(found)))
    return found[0]


def shell_coverage(t, base):
    """The surface's ``coverage`` builder for a layer at height ``t`` (base: layer 0): sat(layer x base.a + base) x
    sat(1 - t / 2), the quantity the writer compares with its dither threshold. The layer map is the writer's own lookup
    (its comb-shifted coordinate and slice) read without its mip bias. ``t`` and ``base`` are numbers, or None for the
    renderer inputs of shell_symbolic_program (t = SHELL_INPUTS z / 32)."""
    def build(g):
        layer = g.nodes[_sample_node(g, LAYER_BINDING)]
        meta = {k: layer[k] for k in ('binding', 'sampler', 'offsets', 'dimension', 'controls') if k in layer}
        plain = g._add('sample', 'v', tuple(layer['args'][:3]), opcode='sample', **meta)
        alpha = g.channel(_sample_node(g, BASE_BINDING), 3, 'f')
        flag = g.input(SHELL_INPUTS['TEXCOORD3.w']) if base is None else g.const('f', 1.0 if base else 0.0)
        density = g.op('sat', g.op('add', g.op('mul', g.channel(plain, 0, 'f'), alpha), flag))
        if t is None:
            height = g.op('sat', g.op('add', g.const('f', 1.0),
                                      g.op('mul', g.input(SHELL_INPUTS['TEXCOORD.z']), g.const('f', -1.0 / 64.0))))
        else:
            height = g.const('f', min(max(1.0 - t / 2.0, 0.0), 1.0))
        return g.op('mul', density, height)
    return build


# The shell vertex program's per-vertex interpolants, as renderer inputs (nwo.viewer.rcra_glsl draws them).
SHELL_INPUTS = {'TEXCOORD.z': 'FURSHELL.z', 'TEXCOORD2.w': 'FURSHELL.offset', 'TEXCOORD3.w': 'FURSHELL.base'}
SHELL_VIEW_NOTE = ('Fur shells: the shell vertex program\'s layers, view spacing and cull run per vertex (%d layers, the '
                   'control map read bilinear at its top mip where the game point-samples it), the engine layer map read '
                   'at its top mip (the writer\'s view-dependent mip bias is not applied), and the writer\'s dithered cut '
                   'drawn as coverage, blended inner layer first, as the game\'s temporal filter averages it.')


def shell_symbolic_program(source, data):
    """fur_program for every shell layer at once: the layer's interpolants are renderer inputs (SHELL_INPUTS), for a
    renderer that runs the shell vertex program itself; ``fur`` carries the ModelFurCBuffer values it needs."""
    program = fur_program(source, data)
    record = fur_record(data)
    fields = dict(zip(FUR_FIELDS, record[1]))
    program['units'] = dict(program['units'], **{LAYER_BINDING: LAYER_MAP})
    program['interpolants'] = {'TEXCOORD.w': 1.0, 'TEXCOORD1.w': float(np.float32(fields['OffsetScale']))}
    program['symbolic'] = dict(SHELL_INPUTS)
    program['coverage'] = shell_coverage(None, None)
    program['limitations'] = program['limitations'] + [SHELL_VIEW_NOTE % layer_count(record)]
    program['fur'] = dict(layers=layer_count(record), length=float(np.float32(fields['Length'])),
                          offset_scale=float(np.float32(fields['OffsetScale'])))
    return program


def shell_program(source, data, k):
    """fur_program for shell layer ``k``: the writer with the layer's interpolants, the layer map bound, and the coverage
    term; also carries the layer's ``shell`` values (shell_layer)."""
    program = fur_program(source, data)
    record = fur_record(data)
    layer = shell_layer(record, k)
    program['units'] = dict(program['units'], **{LAYER_BINDING: LAYER_MAP})
    program['interpolants'] = layer['interpolants']
    program['coverage'] = shell_coverage(layer['t'], k == 0)
    program['limitations'] = program['limitations'] + [SHELL_NOTE % (k, layer['layers'], layer['t'])]
    program['shell'] = layer
    return program
