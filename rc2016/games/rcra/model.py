"""Rift Apart (PC) ``.model`` geometry, in the dict schema of ``formats.native_model.decode_model``.

The asset bytes start with the 36-byte toc header (u32 magic 0x9D2C0FA9, u32 offset and u32 size
of the streamed sections, 24 zero bytes); the DAT1 (type 0x9D2C0FA9) follows at +36. String
pointers are offsets from the DAT1 start. Layouts follow ALERT ``dat1lib`` and RCRA-Forge, checked
on the installed files (docs/rift-apart.md):

  0x283D0383 built  f32 position offset at +0x1C, f32 position scale at +0x2C (1/4096 seen),
                    u32 at +0x30 whose low 4 bits give the UV scale (1 << n) / 16384
  0xA98BE69B        16-byte vertex: int16 x, y, z, w | u32 packed normal | int16 u, v
                    The normal is ALERT's two-angle form (10-bit x, 10-bit y, sign at bit 31). The
                    tangent is a second two-angle pair: x from the normal word's bits 20-29, y from
                    |w| bits 0-9, z's sign at bit 30. The bitangent sign is the sign of w; |w| bits
                    10-14 scale the interpolant (unpack_tangents, from the engine model vertex program)
  0x0859863D        u16 indices; mesh flag 0x10 makes them relative to the mesh's first vertex
  0x78D9CBDE        64-byte mesh record: +4 u64 id, +20 u32 first vertex, first index, index
                    count, vertex count, +36 u16 flags, material slot, first skin batch, batch count,
                    +56 u32 first weight index
  0x06EB7EFC        looks: 8 (u16 first mesh, u16 count) LOD slots per look
  0x3250BB80        materials: the first half holds (u64 path, u64 name) string pointers
  0x6B855EED        int16 UV1 per vertex (same scale); this and the colours may be zero-padded to 16 bytes
  0x5CBA9DE9        u8 RGBA per vertex
  0x15DF9D3B        joints: i16 parent, u16 index, u16, u16 flags, u32 name hash (game CRC-32 of
                    the name), u32 name pointer
  0xDCC88A19        joint rest S/Q/T (48 bytes each), then the inverse binds from the next 64-byte
                    boundary: the 2016 layout, read by formats.skeleton.inspect_model
  0xC61B1FF5        skin batches, 16 bytes: u32 offset into 0xDCA379A2, u32 remap, u16 palette size,
                    u16 (not read), u16 vertex count, u16 first vertex in the mesh. A mesh owns the low
                    byte of its batch count word, from its first skin batch. With a palette size, the
                    u8 joint ids index that many u16 skeleton joints at byte offset ``remap`` of
                    0x5240C82B; without one, ``remap`` is added to each id (the R&C 2016 rules, used by
                    rigs of more than 256 joints)
  0xDCA379A2        skin data: per batch, groups of 16 vertices (the last may be shorter). Each group
                    is a byte n, then for n = 0 one u8 skeleton joint index per vertex (weight 1), else
                    n + 1 (u8 joint, u8 weight) pairs per vertex; weights sum to 255 or 256
  0xCCBAFF15        the same weights as fixed 8-byte records (4 u8 joints, 4 u8 weights) for the
                    meshes with flag 0x100, from the mesh's first weight index. Not read: the batches
                    hold every skinned mesh, their raw ids equal these on every vertex both hold, and
                    these records keep the palette-local ids of a remapped rig
  0x811902D7        look built: one 80-byte record per look, the first record's first u32 being the records'
                    byte size; each ends with u32 count, u32 name hash, u32 lowercase-name hash and u32 name
                    pointer at +64 (the 16 u32 before them and the data after the records are not read). The
                    look names: Default, base, pristine/damaged, topiary, coll
  0x9F614FAB        locators, 64 bytes: u32 name hash, u32 name pointer, i32 joint (-1: none), u32 0, then a
                    3x4 transform stored column by column (three axis columns, then the translation),
                    relative to the joint (the model when there is none). 0x731CBC2E is the sorted
                    (name hash, locator) lookup and 0xEE31971C the sorted (name hash, joint) lookup of the
                    joints, closed by a (0xFFFFFFFF, 0xFFFFFFFF) row; both repeat the tables they index

SECTION_ROLES names every other section the corpus holds: identified but not interpreted, or unknown.
"""
import math
import struct

import numpy as np

from rc2016.common import read, sha
from rc2016.games.rcra.toc import sections as dat1_sections

MODEL_MAGIC = 0x9D2C0FA9
HEADER = 36
BUILT, VERTICES, INDEXES, MESHES = 0x283D0383, 0xA98BE69B, 0x0859863D, 0x78D9CBDE
LOOK, MATERIALS, UV1, COLORS, JOINTS = 0x06EB7EFC, 0x3250BB80, 0x6B855EED, 0x5CBA9DE9, 0x15DF9D3B
JOINT_TRANSFORMS, SKIN_BATCHES, SKIN_DATA, WEIGHTS = 0xDCC88A19, 0xC61B1FF5, 0xDCA379A2, 0xCCBAFF15
SKIN_REMAP = 0x5240C82B
LOOK_BUILT, LOCATORS, LOCATOR_LOOKUP, JOINT_LOOKUP = 0x811902D7, 0x9F614FAB, 0x731CBC2E, 0xEE31971C
MORPH_SECTIONS = (0x380A5744, 0x5E709570, 0xA600C108)   # read by attach_shapes (rcra.morph)
LOOK_RECORD = 80
LOCATOR_RECORD = 64
SKIN_SECTIONS = (SKIN_BATCHES, SKIN_DATA, WEIGHTS, SKIN_REMAP)
SKIN_GROUP = 16            # vertices per skin-data group
MAX_INFLUENCES = 8
LODS_PER_LOOK = 8
MESH_STRIDE = 64
VERTEX = np.dtype([('xyzw', '<i2', 4), ('normal', '<u4'), ('uv', '<i2', 2)])
DECODED = {BUILT, VERTICES, INDEXES, MESHES, LOOK, MATERIALS, UV1, COLORS, JOINTS, SKIN_BATCHES, SKIN_DATA, SKIN_REMAP,
           LOOK_BUILT, LOCATORS, *MORPH_SECTIONS,
           JOINT_TRANSFORMS}   # the rest pose and inverse binds: read by the skeleton (rcra.animation)
# Sections that only repeat a table that is read, as an index into it or in another layout (checked over the install).
DERIVED = {LOCATOR_LOOKUP: 'sorted (name hash, locator) lookup of 0x9F614FAB',
           JOINT_LOOKUP: 'sorted (name hash, joint) lookup of the joint table, closed by a 0xFFFFFFFF row',
           WEIGHTS: 'the skin weights again as fixed 8-byte records; the 0xC61B1FF5 batches are read instead'}
# Every other section in the installed models, with what is known of it (names in quotes are ALERT dat1lib's). Counts
# are models of 11,387. None of them is needed to draw the geometry.
SECTION_ROLES = {
    0xEFD92E68: 'Havok physics data: the collision shapes ("Model Physics Data"; 4,907 models)',
    0x707F1B58: 'unknown: u32 16, u32 data size, counts, then float matrices (9,216)',
    0x4CCEA4AD: 'unknown: records that hold look-name hashes and look indices (every model)',
    0xDF9FDF12: 'unknown: 16-byte (u32 x4) records ("0-70-0-2?"; 11,104)',
    0x90CDB60C: 'unknown: 80 bytes in every rigged model (1,123)',
    0xB7380E8C: 'unknown: u16 values, in rigged models (1,123)',
    0xC5354B60: 'joint mirror ids ("Model Mirror Ids"; 1,123); not read',
    0x0AD3A708: 'unknown: a count and 20-byte records, in skinned models (958)',
    0x665DA362: 'unknown: 8 or 16 bytes (371)',
    0x3C9DABDF: 'spline subsets ("Model Spline Subsets"): Ratchet and Rivet bangle models only (32), with 0x27CA5246, '
                '0xB25B3163 and 0xBB7303D5',
    0x27CA5246: 'unknown: with the spline subsets, Ratchet and Rivet bangle models only (32)',
    0xB25B3163: 'unknown: with the spline subsets, Ratchet and Rivet bangle models only (32)',
    0xBB7303D5: 'unknown: with the spline subsets, Ratchet and Rivet bangle models only (32)',
    0xBCE86B01: 'unknown (22 models)',
    0xADD1CBD3: 'unknown (20 models)',
    0x7CA37DA0: 'ambient shadow primitives ("Ambient Shadow Prims"; 11 models, 48 bytes); not read',
    0x9A434B29: 'unknown, related to the locators in ALERT (7 models)',
    0x5A39FAB7: 'unknown (7 models)',
    0x855275D7: 'unknown (3 models, 52 bytes)',
    0x8A84E4D6: 'unknown (3 models)',
    0xCD903318: 'unknown (3 models, 68 bytes)',
    0x00823787: 'unknown (1 model, 4 bytes)',
}
SCHEMA = 'rcra-pc-model-v1'
FLOAT32_MAX = float(np.finfo(np.float32).max)


class UnsupportedModelError(ValueError):
    pass


def _cstring(dat1, offset, bounds):
    start, end = bounds
    if not start <= offset < end:
        raise ValueError('String pointer outside the model string pool')
    stop = dat1.find(b'\0', offset, end)
    if stop < 0:
        raise ValueError('Unterminated model string')
    return dat1[offset:stop].decode('ascii')


def unpack_normals(words):
    """ALERT's normal decode: x, y from two 10-bit fields, z from the remainder, its sign in bit 31."""
    words = np.asarray(words, np.uint32).astype(np.int64)
    nx = (words & 0x3FF) * 0.00276483595 - math.sqrt(2)
    ny = ((words >> 10) & 0x3FF) * 0.00276483595 - math.sqrt(2)
    q = nx * nx + ny * ny
    w = np.sqrt(np.clip(1.0 - 0.25 * q, 0.0, None))
    nz = 1.0 - 0.5 * q
    nz = np.where((words >> 31) == 0, -nz, nz)
    vectors = np.stack((nx * w, ny * w, nz), axis=1)
    length = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.where(length > 1e-12, vectors / np.maximum(length, 1e-12), np.array([0.0, 0.0, 1.0]))


def unpack_tangents(w, words):
    """(tangents (n, 3), bitangent signs (n,) of +-1, TEXCOORD3.w (n,)) from the int16 position w and the packed normal word, as the
    engine model vertex program decodes TEXCOORD3 (RiftApart.exe, translated with rc2016.shaders.sm5):

    - a second two-angle pair, a = (|w| & 0x3FF) and b = the normal word's bits 20-29, each 0.0027648 field - 1.4142;
      q = a^2 + b^2, tangent = (b s, a s, +-|1 - q/2|) with s = sqrt(1 - q/4); z is positive when bit 30 is set;
    - TEXCOORD3.w = sign(w) / (e / 31)^2 with e = (|w| >> 10) & 31 (w = 0 counts as positive); the draw's render flag
      0x10000 would negate it. The pixel programs read its sign (w < 0); one forward program also reads |w|. e = 0
      (the engine's infinity) does not occur in the checked models; the largest float32 stands in for it.

    The program does not normalise; the pair is a unit vector while q <= 4 (s is clamped at 0 beyond)."""
    w = np.asarray(w, np.int64)
    words = np.asarray(words, np.uint32).astype(np.int64)
    magnitude = np.abs(w)
    a = (magnitude & 0x3FF) * 0.00276483595 - math.sqrt(2)
    b = ((words >> 20) & 0x3FF) * 0.00276483595 - math.sqrt(2)
    q = a * a + b * b
    s = np.sqrt(np.clip(1.0 - 0.25 * q, 0.0, None))
    z = np.abs(1.0 - 0.5 * q)
    z = np.where((words >> 30) & 1, z, -z)
    signs = np.where(w >= 0, 1.0, -1.0)
    e = ((magnitude >> 10) & 31) / 31.0
    with np.errstate(divide='ignore'):
        scale = np.where(e > 0, 1.0 / np.maximum(e * e, 1e-30), FLOAT32_MAX)
    return np.stack((b * s, a * s, z), axis=1), signs, signs * scale


def _looks(raw, mesh_count):
    if not raw:
        return [dict(index=0, source='implicit_all_meshes',
                     lods=[dict(index=0, start=0, count=mesh_count, meshes=list(range(mesh_count)), implicit=True)])]
    if len(raw) % (4 * LODS_PER_LOOK):
        raise ValueError('Look table is not a multiple of eight LOD records')
    looks = []
    for look_index, base in enumerate(range(0, len(raw), 4 * LODS_PER_LOOK)):
        lods = []
        for lod in range(LODS_PER_LOOK):
            start, count = struct.unpack_from('<HH', raw, base + 4 * lod)
            if start + count > mesh_count:
                raise ValueError('Look/LOD references mesh records outside the model')
            lods.append(dict(index=lod, start=start, count=count, meshes=list(range(start, start + count))))
        looks.append(dict(index=look_index, lods=lods))
    return looks


def _look_names(dat1, raw, bounds, looks):
    """Name each look from section 0x811902D7: u32 byte size of the 80-byte look records, one per look, each with
    u32 name hash, u32 lowercase-name hash and u32 name pointer from +68. Looks keep no name when the records do not
    match the look table."""
    if len(raw) < 4:
        return
    size = struct.unpack_from('<I', raw)[0]
    if size != LOOK_RECORD * len(looks) or size > len(raw):
        return
    for look in looks:
        name_hash, _lower, pointer = struct.unpack_from('<III', raw, LOOK_RECORD * look['index'] + 68)
        look['name'] = _cstring(dat1, pointer, bounds)
        look['name_hash'] = f'0x{name_hash:08X}'


def _locators(dat1, raw, bounds, joint_count):
    """The model's locators (section 0x9F614FAB): named attach points with a transform relative to a joint.

    ``matrix`` is the 4x4 local transform, rows of a column-vector matrix: the stored 3x4 is read column by column (three
    axis columns, then the translation), as the inverse binds are; ``joint`` is None for a locator on the model."""
    if len(raw) % LOCATOR_RECORD:
        raise ValueError('Locator table is not a multiple of 64 bytes')
    out = []
    for at in range(0, len(raw), LOCATOR_RECORD):
        name_hash, pointer, joint, _zero = struct.unpack_from('<IIiI', raw, at)
        if not -1 <= joint < joint_count:
            raise ValueError('Locator joint outside the joint table')
        columns = np.asarray(struct.unpack_from('<12f', raw, at + 16), np.float64).reshape(4, 3)
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = columns[:3].T, columns[3]
        out.append(dict(index=len(out), name=_cstring(dat1, pointer, bounds), name_hash=f'0x{name_hash:08X}',
                        joint=None if joint < 0 else joint, matrix=matrix.tolist()))
    return out


def section_notes(tags):
    """{'0xTAG': what is known of it} of the given section tags that are not read."""
    return {f'0x{t:08X}': DERIVED.get(t) or SECTION_ROLES.get(t, 'unknown') for t in sorted(tags) if t not in DECODED}


def geometry_free_note(table):
    """Why a model holds nothing to draw, naming what it does hold."""
    held = ['its bounds', 'an empty look table']
    if JOINTS in table:
        count = table[JOINTS][1] // 16
        held.append('%d joint%s' % (count, '' if count == 1 else 's'))
    if LOCATORS in table:
        held.append('locators')
    if 0xEFD92E68 in table:
        held.append('a Havok collision shape')
    return 'This model holds no geometry: it has no vertex or mesh section, only %s and %s.' % (', '.join(held[:-1]), held[-1])


def _materials(dat1, raw, bounds):
    if len(raw) % 32:
        raise ValueError('Material table is not two halves of 16-byte records')
    result = []
    for offset in range(0, len(raw) // 2, 16):
        path_ptr, name_ptr = struct.unpack_from('<QQ', raw, offset)
        path = _cstring(dat1, path_ptr, bounds)
        result.append(dict(index=len(result), path=path.replace('\\', '/'), name=_cstring(dat1, name_ptr, bounds),
                           raw_record_hex=raw[offset:offset + 16].hex()))
    return result


def _joints(dat1, raw, bounds):
    if len(raw) % 16:
        raise ValueError('Joint table is not a multiple of 16 bytes')
    joints = []
    for offset in range(0, len(raw), 16):
        parent, index, _a, _b, name_hash, name_ptr = struct.unpack_from('<hHHHII', raw, offset)
        if parent >= len(raw) // 16 or parent < -1:
            raise ValueError('Joint parent outside the joint table')
        joints.append(dict(index=index, parent=parent, name=_cstring(dat1, name_ptr, bounds), name_hash=f'0x{name_hash:08X}'))
    return joints


def _per_vertex(raw, vertex_count, label):
    """A 4-byte-per-vertex section; the stored section may be padded to a 16-byte multiple."""
    need = vertex_count * 4
    if not need <= len(raw) < need + 16 or any(raw[need:]):
        raise ValueError(f'{label} section size {len(raw)} does not match {vertex_count} vertices')
    return raw[:need]


def _batch_weights(skin, offset, count, limit):
    """(joints, weights) arrays (count, 8) of one skin batch, and the offset after it (at most ``limit``)."""
    joints = np.zeros((count, MAX_INFLUENCES), np.int64)
    weights = np.zeros((count, MAX_INFLUENCES), np.int64)
    position = offset
    for first in range(0, count, SKIN_GROUP):
        n = min(SKIN_GROUP, count - first)
        if position >= limit:
            raise ValueError('Skin batch runs past its data')
        mode = skin[position]
        position += 1
        if mode == 0:
            size = n
            if position + size > limit:
                raise ValueError('Skin batch runs past its data')
            joints[first:first + n, 0] = np.frombuffer(skin, np.uint8, n, position)
            weights[first:first + n, 0] = 255
        elif mode < MAX_INFLUENCES:
            k = mode + 1
            size = n * 2 * k
            if position + size > limit:
                raise ValueError('Skin batch runs past its data')
            block = np.frombuffer(skin, np.uint8, size, position).reshape(n, k, 2)
            joints[first:first + n, :k], weights[first:first + n, :k] = block[:, :, 0], block[:, :, 1]
        else:
            raise ValueError(f'Unsupported skin group with {mode + 1} influences')
        position += size
    return joints, weights, position


def _mesh_weights(skin, remap, batches, first_batch, batch_count, vertex_count, joint_count, mesh_index, as_arrays):
    """(per-vertex [(joint, weight)] lists, (n, 8) joint array, (n, 8) weight array) of one mesh's skin batches. With
    ``as_arrays`` the lists are a ``SkinLists`` view over the arrays, built per vertex only when read."""
    if first_batch + batch_count > len(batches):
        raise ValueError(f'Mesh {mesh_index} skin batches run past the batch table')
    joints = np.zeros((vertex_count, MAX_INFLUENCES), np.int64)
    raw = np.zeros((vertex_count, MAX_INFLUENCES), np.int64)
    covered = np.zeros(vertex_count, bool)
    for index in range(first_batch, first_batch + batch_count):
        offset, base, palette_size, _unknown, count, first = batches[index]
        limit = batches[index + 1][0] if index + 1 < len(batches) else len(skin)
        if first + count > vertex_count or covered[first:first + count].any():
            raise ValueError(f'Skin batch {index} overlaps or exceeds mesh {mesh_index}')
        ids, raw[first:first + count], _end = _batch_weights(skin, offset, count, limit)
        if palette_size:
            if base + 2 * palette_size > len(remap) or ids.max(initial=0) >= palette_size:
                raise ValueError(f'Skin batch {index} joint palette is out of range')
            ids = np.frombuffer(remap, '<u2', palette_size, base).astype(np.int64)[ids]
        else:
            ids = ids + base
        joints[first:first + count] = ids
        covered[first:first + count] = True
    if not covered.all():
        raise ValueError(f'Mesh {mesh_index} skin batches leave {int((~covered).sum())} vertices without weights')
    if joints.max(initial=0) >= joint_count:
        raise ValueError(f'Mesh {mesh_index} skin weights name a joint outside the skeleton')
    total = raw.sum(axis=1)
    if (total <= 0).any():
        raise ValueError(f'Mesh {mesh_index} has a vertex with zero skin weight')
    values = raw / total[:, None]
    joints = np.where(raw > 0, joints, 0)
    if as_arrays:
        return SkinLists(joints, values), joints, values
    return _influences(joints, values), None, None


def _influences(joints, values):
    return [[(j, w) for j, w in zip(jr, wr) if w > 0] for jr, wr in zip(joints.tolist(), values.tolist())]


class SkinLists:
    """The per-vertex [(joint, weight)] lists of an as_arrays mesh, read from its (n, 8) arrays on first use: the viewer
    reads the arrays only, and building ~200k Python lists took most of a large rig's decode."""

    def __init__(self, joints, values):
        self._joints, self._values, self._lists = joints, values, None

    def _built(self):
        if self._lists is None:
            self._lists = _influences(self._joints, self._values)
        return self._lists

    def __len__(self):
        return len(self._joints)

    def __bool__(self):
        return len(self._joints) > 0

    def __getitem__(self, index):
        return self._built()[index]

    def __iter__(self):
        return iter(self._built())

    def __eq__(self, other):
        return list(self) == list(other)


def _tuples(values):
    return list(map(tuple, values.tolist()))


def decode_model(path, lod='All', look=0, data=None, as_arrays=False):
    """Decode a Rift Apart model asset (toc header + DAT1). ``path`` is a label when ``data`` is given."""
    from rc2016.formats.native_model import _selected_meshes
    data = read(path) if data is None else bytes(data)
    source_sha256 = sha(data)
    if len(data) < HEADER + 16 or struct.unpack_from('<I', data, 0)[0] != MODEL_MAGIC:
        raise UnsupportedModelError('Not a Rift Apart model (expected magic 0x%08X)' % MODEL_MAGIC)
    dat1 = data[HEADER:]
    if dat1[:4] != b'1TAD' or struct.unpack_from('<I', dat1, 4)[0] != MODEL_MAGIC:
        raise UnsupportedModelError('Rift Apart model without its DAT1 at +36')
    declared = struct.unpack_from('<I', dat1, 8)[0]
    if declared != len(dat1):
        raise ValueError('Model DAT1 declares %d bytes, asset holds %d' % (declared, len(dat1)))
    table = dat1_sections(dat1)
    first = min((o for o, _s in table.values()), default=len(dat1))
    bounds = (16 + 12 * len(table), first)

    def chunk(tag, required=False):
        if tag not in table:
            if required:
                raise UnsupportedModelError('Missing model section 0x%08X' % tag)
            return b''
        offset, size = table[tag]
        return dat1[offset:offset + size]

    if VERTICES not in table and MESHES not in table:
        raise UnsupportedModelError(geometry_free_note(table))
    built = chunk(BUILT, True)
    if len(built) < 0x34:
        raise ValueError('Model built section is too short')
    offset = np.array(struct.unpack_from('<3f', built, 0x1C), np.float64)
    scale = struct.unpack_from('<f', built, 0x2C)[0]
    uv_bits = struct.unpack_from('<I', built, 0x30)[0]
    uv_scale = (1 << (uv_bits & 0xF)) / 16384.0
    if not (math.isfinite(scale) and scale > 0 and np.all(np.isfinite(offset))):
        raise ValueError('Invalid model position scale/offset')
    vertices_raw, indexes, meshes_raw = chunk(VERTICES, True), chunk(INDEXES, True), chunk(MESHES, True)
    if len(vertices_raw) % VERTEX.itemsize or len(indexes) % 2 or len(meshes_raw) % MESH_STRIDE:
        raise ValueError('Misaligned core geometry section')
    vertex_count = len(vertices_raw) // VERTEX.itemsize
    if not vertex_count:
        raise ValueError('Model has no vertices')
    records = np.frombuffer(vertices_raw, VERTEX)
    positions = records['xyzw'][:, :3].astype(np.float64) * scale + offset
    normals = unpack_normals(records['normal'])
    tangents, tangent_signs, tangent_w = unpack_tangents(records['xyzw'][:, 3], records['normal'])
    uv = records['uv'].astype(np.float64) * uv_scale
    uvs = np.stack((uv[:, 0], 1.0 - uv[:, 1]), axis=1)      # V flipped like the 2016 decoder (Blender convention)
    uv1 = None
    uv1_raw = chunk(UV1)
    if uv1_raw:
        second = np.frombuffer(_per_vertex(uv1_raw, vertex_count, 'UV1'), '<i2').reshape(-1, 2).astype(np.float64) * uv_scale
        uv1 = np.stack((second[:, 0], 1.0 - second[:, 1]), axis=1)
    colors = None
    colors_raw = chunk(COLORS)
    if colors_raw:
        colors = np.frombuffer(_per_vertex(colors_raw, vertex_count, 'Vertex colour'), np.uint8).reshape(-1, 4) / 255.0
    if not (np.all(np.isfinite(positions)) and np.all(np.isfinite(uvs))):
        raise ValueError('Non-finite vertex data')
    materials = _materials(dat1, chunk(MATERIALS), bounds)
    joints = _joints(dat1, chunk(JOINTS), bounds)
    batches_raw, skin, remap = chunk(SKIN_BATCHES), chunk(SKIN_DATA), chunk(SKIN_REMAP)
    if len(batches_raw) % 16:
        raise ValueError('Skin batch section is not a multiple of 16 bytes')
    batches = [struct.unpack_from('<IIHHHH', batches_raw, at) for at in range(0, len(batches_raw), 16)]
    mesh_count = len(meshes_raw) // MESH_STRIDE
    looks = _looks(chunk(LOOK), mesh_count)
    _look_names(dat1, chunk(LOOK_BUILT), bounds, looks)
    locators = _locators(dat1, chunk(LOCATORS), bounds, len(joints))
    membership = {i: [] for i in range(mesh_count)}
    for look_row in looks:
        for level in look_row['lods']:
            for index in level['meshes']:
                membership[index].append(dict(look=look_row['index'], lod=level['index']))
    selected = _selected_meshes(looks, mesh_count, lod, look)
    index_words = np.frombuffer(indexes, '<u2')
    # Lists mode keeps the 2016 schema's lists of tuples; as_arrays gives (n, k) arrays, colours and UV1 included
    # (consumers test those with ``is not None``, not for truth).
    if not as_arrays:
        positions_out, normals_out, uvs_out = _tuples(positions), _tuples(normals), _tuples(uvs)
        tangents_out, signs_out, w_out = _tuples(tangents), tangent_signs.tolist(), tangent_w.tolist()
        uv1_out = _tuples(uv1) if uv1 is not None else None
        colors_out = _tuples(colors) if colors is not None else None
    else:
        positions_out, normals_out, uvs_out = positions, normals, uvs
        tangents_out, signs_out, w_out = tangents, tangent_signs, tangent_w
        uv1_out, colors_out = uv1, colors
    result = []
    for mesh_index in selected:
        at = mesh_index * MESH_STRIDE
        unknown_u32_00, mesh_id = struct.unpack_from('<IQ', meshes_raw, at)
        first_vertex, first_index, index_count, local_count = struct.unpack_from('<4I', meshes_raw, at + 20)
        flags, material_index, first_skin_batch, skin_batch_count = struct.unpack_from('<4H', meshes_raw, at + 36)
        first_weight_index, unknown_u32_60 = struct.unpack_from('<II', meshes_raw, at + 56)
        if not local_count or index_count % 3:
            raise ValueError(f'Mesh {mesh_index} has invalid vertex/index counts')
        if first_vertex + local_count > vertex_count or first_index + index_count > len(index_words):
            raise ValueError(f'Mesh {mesh_index} ranges exceed the geometry buffers')
        if materials and material_index >= len(materials):
            raise ValueError(f'Mesh {mesh_index} uses material slot {material_index} outside the material table')
        base = 0 if flags & 0x10 else first_vertex
        local = index_words[first_index:first_index + index_count].astype(np.int64) - base
        if index_count and (local.min() < 0 or local.max() >= local_count):
            raise ValueError(f'Mesh {mesh_index} indices fall outside its vertices')
        triangles = local.reshape(-1, 3)
        window = slice(first_vertex, first_vertex + local_count)
        weights, skin_joints, skin_weights = None, None, None
        own_batches = skin_batch_count & 0xFF   # the high byte is not a count (it is not read)
        if own_batches:
            weights, skin_joints, skin_weights = _mesh_weights(skin, remap, batches, first_skin_batch, own_batches,
                                                               local_count, len(joints), mesh_index, as_arrays)
        extra = dict(skin_joint_array=skin_joints, skin_weight_array=skin_weights) if skin_joints is not None else {}
        result.append(dict(**extra,
            index=mesh_index, mesh_id=f'0x{mesh_id:016X}', first_vertex=first_vertex, first_index=first_index,
            index_count=index_count, vertex_count=local_count, flags=flags, material_index=material_index,
            first_skin_batch=first_skin_batch, skin_batch_count=skin_batch_count & 0xFF, skin_batch_count_word=skin_batch_count,
            first_weight_index=first_weight_index, unknown_u32_00=unknown_u32_00, unknown_u32_60=unknown_u32_60,
            raw_record_hex=meshes_raw[at:at + MESH_STRIDE].hex(),
            triangles=triangles if as_arrays else triangles.tolist(),
            positions=positions_out[window], normals=normals_out[window], uv0=uvs_out[window],
            uv1=uv1_out[window] if uv1_out is not None else None,
            colors=colors_out[window] if colors_out is not None else None,
            tangents=tangents_out[window], tangent_signs=signs_out[window], tangent_w=w_out[window], packed_normals=None, raw_vectors=None, uv0_secondary_packed=None,
            static_vector_eligible=False, weights=weights if weights is not None else [None] * local_count,
            unweighted_vertices=0 if weights is not None else local_count,
            skinned=weights is not None, material=materials[material_index] if material_index < len(materials) else None,
            index_base='local' if flags & 0x10 else 'global', memberships=membership[mesh_index]))
    section_sizes = {f'0x{k:08X}': size for k, (_o, size) in table.items()}
    unread = section_notes(table)
    unsupported = [tag for tag, note in unread.items() if note.startswith('unknown')]   # no interpretation at all
    stream_offset, stream_size = struct.unpack_from('<II', data, 4)
    return dict(
        schema=SCHEMA, game='rcra', source=str(path), source_sha256=source_sha256,
        dat1=dict(type_hash=f'0x{MODEL_MAGIC:08X}', endianness='little', header_bytes=HEADER,
                  streamed_sections_offset=stream_offset, streamed_sections_size=stream_size,
                  section_table_count=len(table)),
        source_units=dict(position=f'int16 * {scale!r} + {offset.tolist()}', position_scale=scale,
                          position_offset=offset.tolist(), uv_scale=uv_scale,
                          uv0_decode='int16 * uv scale; V flipped for the Blender convention like the 2016 decoder'),
        packed_normal='ALERT two-angle form: 10-bit x and y, z from the remainder, sign at bit 31 (face-normal checked)',
        tangent=('engine model vertex program: two-angle pair of |w| bits 0-9 and normal bits 20-29, z sign at bit 30; '
                 'bitangent sign = sign of w'),
        skin='0xC61B1FF5/0xDCA379A2 skin batches: 16-vertex groups of 1 to 8 (joint, weight) influences',
        vertex_count=vertex_count, index_count=len(index_words), mesh_record_count=mesh_count,
        selected_mesh_count=len(result), selected_mesh_indices=selected, import_lod=lod, import_look=look,
        looks=looks, materials=materials, joints=joints, skeleton_joint_count=len(joints), locators=locators, meshes=result,
        section_sizes=section_sizes, unsupported_sections=unsupported, unread_sections=unread,
        skin_sections_present=[f'0x{t:08X}' for t in SKIN_SECTIONS if t in table])


def attach_shapes(decoded, data):
    """The model's blend shapes (``rcra.morph.model_shapes``): ``decoded['shapes']`` lists their names and hashes in the
    model's order, and each mesh a shape moves gets ``morphs``: [(shape number, mesh vertices, position deltas, normal
    deltas)] in model units. A model whose shapes do not read keeps none, with the reason in ``shape_note``."""
    from rc2016.games.rcra.morph import model_shapes
    dat1 = data[HEADER:]
    meshes = {mesh['index']: mesh for mesh in decoded['meshes']}
    try:
        shapes = model_shapes(dat1, dat1_sections(dat1), {i: len(m['positions']) for i, m in meshes.items()})
    except (ValueError, struct.error) as error:
        decoded['shapes'], decoded['shape_note'] = [], f'Blend shapes not read: {error}'
        return
    decoded['shapes'] = [dict(name=shape['name'], hash=shape['hash']) for shape in shapes]
    for number, shape in enumerate(shapes):
        for target in shape['targets']:
            if target['mesh'] in meshes and len(target['vertices']):
                meshes[target['mesh']].setdefault('morphs', []).append(
                    (number, target['vertices'], target['positions'], target['normals']))


def decode_for_viewer(source, model_id, source_path, as_arrays):
    """The ``Game.decode_model`` hook: (decoded, data) with packed_sha256, codec and joint_count.

    joint_count is the skeleton size when the model stores its joint rest transforms (section
    0xDCC88A19), else 0; the clip library (``rcra.animation.RcraAnimationLibrary``) reads the
    skeleton from ``data``, the whole asset. Meshes whose material has a vertex program are drawn
    where it places them at rest (``rcra.vertex.deform_model``). ``undrawn_materials`` holds the
    materials the game's plain instance draw never draws (``material.draw_skip_note``), which the
    viewer and the model export leave out; their note also goes to ``material_notes``."""
    from rc2016.games.rcra.material import undrawn_materials
    from rc2016.games.rcra.vertex import deform_model
    data, digest = source.bytes(model_id)
    decoded = decode_model(source_path, lod='All', data=data, as_arrays=as_arrays)
    rigged = '0x%08X' % JOINT_TRANSFORMS in decoded['section_sizes']
    decoded.update(packed_sha256=digest, codec='rcra-dsar', joint_count=len(decoded['joints']) if rigged else 0)
    deform_model(source, decoded, data)
    attach_shapes(decoded, data)
    decoded['undrawn_materials'] = undrawn_materials(source, decoded['materials'])
    notes = decoded.setdefault('material_notes', {})
    for material_path, note in decoded['undrawn_materials'].items():
        notes.setdefault(material_path, []).append(note)
    return decoded, data
