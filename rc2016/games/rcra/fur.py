"""Rift Apart fur materials: no material graph, a fur record and the engine's fur programs in the executable.

A fur material (19 in the install: Ratchet's and Rivet's head, ears and limbs, Daxter, the sheep, a carpet, a towel, a
cotton canopy and a cloud) has DAT1 section 0xD9B12454 instead of the serialized parameter section. The record is 13 u32:
dword 0 is an integer (32 on Ratchet's head), eight floats follow, and dwords 9 to 12 are offsets into the material's DAT1
of four texture paths, in the order base colour (_c), normal (_n), gloss (_g) and fur control (19 of 19 materials).

The executable holds the programs (found by their reflection names, rc2016.shaders.dxbc): the fur G-buffer writer, a
ps_5_0 with constant buffer MatrialFurCBuffer (g_Fur: m_Density, m_Length, m_GlossScale, m_SpecularScale,
m_TransmittanceScale, m_Wetness), textures g_BaseMap (t5), g_FurLayerMap (t7, 3D), g_FurGlossMap (t8) and g_FurControlMap
(t9), and the engine's four G-buffer targets; a five-target variant; and the shell vertex programs (ModelFurCBuffer). The
writer binds no normal map: its normal is the vertex normal.

How the engine fills g_Fur from the record is not decoded. The writer's surface terms read m_GlossScale, m_SpecularScale
(F0 and gloss) and m_Wetness (albedo); they are bound to review values (FUR_REVIEW), labelled. The shells, layers extruded
along the normal and cut by the layer map, are not drawn: the material is its base surface. The writer's discard is that
cut, sat(layer term + TEXCOORD3.w) < 0.5, where the shell pass's interpolant and the engine's layer map (built at run
time: no texture of it ships) are not available, so it is not applied.
"""
import os
import struct
import threading

from rc2016.games.rcra.toc import sections

FUR_RECORD = 0xD9B12454
EXECUTABLE = 'RiftApart.exe'
FUR_BUFFER = 'MatrialFurCBuffer'    # sic, the engine's name
FUR_UNITS = {0: 'g_BaseMap', 2: 'g_FurGlossMap', 3: 'g_FurControlMap'}   # record texture index -> writer binding
FUR_REVIEW = {FUR_BUFFER + '.m_Density': 1.0, FUR_BUFFER + '.m_Length': 0.0, FUR_BUFFER + '.m_GlossScale': 1.0,
              FUR_BUFFER + '.m_SpecularScale': 1.0, FUR_BUFFER + '.m_TransmittanceScale': 1.0, FUR_BUFFER + '.m_Wetness': 0.0}
FUR_NOTE = ('Fur material: the engine\'s fur G-buffer writer from the executable (constant buffer %s), drawn as the base '
            'surface without its shells. How the engine fills the fur constants from the material\'s fur record is not '
            'decoded; they are review values: %s. The writer binds no normal map (vertex normal). Its discard, the shell '
            'cut by the engine layer map, is not applied.'
            % (FUR_BUFFER, ', '.join('%s %g' % (k.split('.')[1], v) for k, v in FUR_REVIEW.items())))
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


def _writer(blob_source):
    """(offset, bytes) of the fur G-buffer writer in the executable bytes: the pixel program with FUR_BUFFER and four
    targets, found from each occurrence of the buffer name back to its DXBC container."""
    from rc2016.shaders import dxbc
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
    """The surface program of a fur material (rc2016.games.rcra.surface.surface ``program``): dict(shader, cb2, units,
    named, limitations, key); raises ValueError when ``data`` is not a fur material or the writer is not found."""
    from rc2016.shaders import dxbc
    record = fur_record(data)
    if record is None:
        raise ValueError('Not a fur material')
    blob = fur_writer(source.root)
    return dict(shader=dxbc.shader(blob), cb2=(), units={binding: record[2][i] for i, binding in FUR_UNITS.items()},
                named=dict(FUR_REVIEW), limitations=[FUR_NOTE], key=blob, discard=False)
