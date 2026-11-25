"""Rift Apart blend shapes: a model's morph targets and a clip's morph weights.

**Model** (19 models carry them, Ratchet's 294 face shapes among them). Section names follow ALERT ``dat1lib``
(format evidence; this is an independent reader):

  0x380A5744 morph info   u32 0, u32 data + index bytes, u16 shapes, u16 mirror pairs, u32 shape table offset,
                          u32 mirror table offset, u32 2 (in all 19 models; the game's loader keeps its low byte
                          beside the counts, likely a format version); the shape table holds (u32 name hash, u32
                          offset) per shape.
                          A shape: u32 name hash, u32 name string offset, u32 data offset, u32 index offset; u8 2,
                          u8 3k, u8 k, u8 0 (k bits per field); f32 position scale, bias, normal scale, bias; u16
                          subsets, u16, u32 data bytes, u32 index bytes (packing u8 1: positions only); then per subset s8 mesh, (4-aligned) u32
                          data offset, u32 index offset, u16 vertex count, (4-aligned) u16 chunk-table position,
                          (4-aligned) the chunk table: (u16 vertices, u32 index words) rows until the subset's count
  0x5E709570 morph data   per vertex three (packing 1) or six (packing 2) k-bit unsigned fields, most significant
                          bit first: position xyz, then normal xyz; field * scale + bias. Each chunk's fields start
                          on a 4-byte boundary
  0xA600C108 morph index  per chunk, u32 words (u16 skip, u16 run) over the mesh's vertices; a run of 0 is 32.
                          Chunk c of a subset starts at mesh vertex c * 2560

A subset's mesh is the model's mesh record index. Left/right pairs mirror each other: on Ratchet the deltas of
``LF_x`` and ``RT_x`` at mirrored vertices differ by a median 0.003 of the shape's largest delta (8-bit fields).

**Clip** (2,821 clips: the face pose clips, cutscene shots and some animset clips):

  0x277563F5 morph info   u16 channels, u8 count bits c, u8 offset bits o, u8 first-channel bits a, u8 channel-step
                          bits d, u8 value bits v (10), u8 0, u32 frame-table bytes, u32 0; u32 shape name hash per
                          channel; u32 per channel, the offset of the shape's name in the DAT1 string pool (after
                          the section table; the game's clip loader adds the pool's address, and the names match
                          the hashes in all 2,821 clips); the frame table: per frame a
                          c-bit active count and an o-bit word offset into the frame data, most significant bit first
  0x69F64588 frame data   per frame, from its word offset: the active channels, each a channel field (the first
                          a bits, absolute; the next d bits each, a step from the previous channel) and a v-bit value

A channel a frame leaves out has weight 0. The weight is the value / 1023 (1022 and 1023 are the most common
values; reading 1023 as full weight is inferred, not traced in the runtime).
"""
from __future__ import annotations

import struct

import numpy as np

MORPH_INFO, MORPH_DATA, MORPH_INDICES = 0x380A5744, 0x5E709570, 0xA600C108
CLIP_MORPH_INFO, CLIP_MORPH_FRAMES = 0x277563F5, 0x69F64588
WINDOW = 2560
WEIGHT_SCALE = 1023.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _align(value, step):
    return (value + step - 1) // step * step


def _cstring(dat1, offset):
    end = dat1.find(b'\0', offset)
    try:
        return dat1[offset:end if end >= 0 else len(dat1)].decode('ascii')
    except UnicodeDecodeError:
        return ''


def _fields(raw, count, bits, per_vertex=6):
    """(count, per_vertex) unsigned ``bits``-bit fields, most significant bit first."""
    used = count * per_vertex * bits
    require(len(raw) * 8 >= used, 'morph data shorter than its vertices')
    stream = np.unpackbits(np.frombuffer(raw, np.uint8, (used + 7) // 8))[:used].reshape(count, per_vertex, bits)
    return stream.astype(np.int64) @ (1 << np.arange(bits - 1, -1, -1, dtype=np.int64))


def _runs(words, chunk):
    """Mesh vertex indices of one chunk's (skip, run) words."""
    words = np.asarray(words, np.int64)
    if not len(words):
        return np.zeros(0, np.int64)
    runs = words >> 16
    runs[runs == 0] = 32
    ends = chunk * WINDOW + np.cumsum((words & 0xFFFF) + runs)   # one past each run
    require(ends[-1] <= (chunk + 1) * WINDOW, 'morph runs leave their 2560-vertex window')
    # Each run's vertices: its first vertex (end - run) plus 0 .. run - 1.
    return np.repeat(ends - runs, runs) + (np.arange(runs.sum()) - np.repeat(np.cumsum(runs) - runs, runs))


def model_shapes(dat1, table, mesh_vertex_counts=None):
    """The blend shapes of a model DAT1, or [] without them.

    Each shape: dict(name, hash, mirror (name hash of its mirror shape or None), targets=[dict(mesh, vertices,
    positions, normals)]): ``vertices`` are indices into the mesh's own vertices, ``positions`` and ``normals`` the
    (n, 3) deltas in model units. ``mesh_vertex_counts`` ({mesh index: vertex count}) bounds-checks the indices."""
    if MORPH_INFO not in table:
        return []
    require(MORPH_DATA in table and MORPH_INDICES in table, 'morph info without its data and index sections')
    io, isize = table[MORPH_INFO]
    info = dat1[io:io + isize]
    do, dsize = table[MORPH_DATA]
    xo, xsize = table[MORPH_INDICES]
    _zero, buffers, count, mirrors, table_at, mirror_at, _flags = struct.unpack_from('<IIHHIII', info, 0)
    require(buffers == dsize + xsize, 'morph buffer size does not match its data and index sections')
    mirror = dict(struct.unpack_from(f'<{2 * mirrors}I', info, mirror_at)[i:i + 2] for i in range(0, 2 * mirrors, 2))
    mirror.update({b: a for a, b in list(mirror.items())})
    shapes = []
    for number in range(count):
        name_hash, at = struct.unpack_from('<II', info, table_at + 8 * number)
        own_hash, name_at, data_at, index_at = struct.unpack_from('<4I', info, at)
        fields, bits3, bits, zero = struct.unpack_from('<4B', info, at + 16)
        require(own_hash == name_hash and fields in (1, 2) and bits3 == 3 * bits and zero == 0 and 1 <= bits <= 16,
                'unknown morph packing')
        per_vertex = 3 * fields
        pscale, pbias, nscale, nbias = struct.unpack_from('<4f', info, at + 20)
        subsets, _info_len, data_len, index_len = struct.unpack_from('<HHII', info, at + 36)
        cur = at + 48
        meshes = struct.unpack_from(f'<{subsets}b', info, cur)
        cur = _align(cur + subsets, 4)
        data_offsets = struct.unpack_from(f'<{subsets}I', info, cur)
        index_offsets = struct.unpack_from(f'<{subsets}I', info, cur + 4 * subsets)
        cur += 8 * subsets
        counts = struct.unpack_from(f'<{subsets}H', info, cur)
        cur = _align(cur + 2 * subsets, 4)
        table_rows = struct.unpack_from(f'<{subsets}H', info, cur)
        cur = _align(cur + 2 * subsets, 4)
        require(data_at + data_len <= dsize and index_at + index_len <= xsize, 'morph shape outside its buffers')
        targets = []
        for k in range(subsets):
            chunks, got, row = [], 0, cur + 6 * table_rows[k]
            while got < counts[k]:
                chunks.append(struct.unpack_from('<HI', info, row))
                row += 6
                got += chunks[-1][0]
            require(got == counts[k], 'morph chunk table does not add up to the subset')
            word_at = xo + index_at + index_offsets[k]
            field_at = do + data_at + data_offsets[k]
            vertices, values = [], []
            for c, (vertex_count, word_count) in enumerate(chunks):
                words = np.frombuffer(dat1, '<u4', word_count, word_at)
                word_at += 4 * word_count
                index = _runs(words, c)
                require(len(index) == vertex_count, 'morph runs do not match the chunk vertex count')
                size = _align(vertex_count * per_vertex * bits, 32) // 8
                values.append(_fields(dat1[field_at:field_at + size], vertex_count, bits, per_vertex) if vertex_count
                              else np.zeros((0, per_vertex), np.int64))
                field_at += size
                vertices.append(index)
            index = np.concatenate(vertices)
            q = np.concatenate(values)
            mesh = meshes[k]
            if mesh_vertex_counts is not None:
                require(mesh in mesh_vertex_counts and (not len(index) or index.max() < mesh_vertex_counts[mesh]),
                        'morph vertices outside their mesh')
            normals = q[:, 3:] * nscale + nbias if fields == 2 else np.zeros((len(index), 3))
            targets.append(dict(mesh=mesh, vertices=index, positions=q[:, :3] * pscale + pbias, normals=normals))
        shapes.append(dict(name=_cstring(dat1, name_at), hash=name_hash, mirror=mirror.get(name_hash), targets=targets))
    return shapes


def shape_hashes(dat1, table):
    """Name hashes of a model's blend shapes in their order, without decoding them ([] without shapes)."""
    if MORPH_INFO not in table:
        return []
    io, _size = table[MORPH_INFO]
    count, table_at = struct.unpack_from('<H', dat1, io + 8)[0], struct.unpack_from('<I', dat1, io + 12)[0]
    return [struct.unpack_from('<I', dat1, io + table_at + 8 * i)[0] for i in range(count)]


def _number(bits, start, width):
    if not width:
        return 0
    return int(bits[start:start + width] @ (1 << np.arange(width - 1, -1, -1, dtype=np.int64)))


def clip_weights(dat1, table, frames):
    """A clip's morph weights: dict(hashes, names, weights (frames, channels) in 0..1), or None for a clip without
    them. ``names`` are the channels' shape names from the clip's string pool."""
    if CLIP_MORPH_INFO not in table:
        return None
    require(CLIP_MORPH_FRAMES in table, 'clip morph info without its frame data')
    io, isize = table[CLIP_MORPH_INFO]
    info = dat1[io:io + isize]
    channels, cbits, obits, first_bits, step_bits, value_bits, _zero, table_bytes = struct.unpack_from('<H6BI', info, 0)
    require(16 + 8 * channels + table_bytes == isize, 'clip morph info size mismatch')
    require(table_bytes == _align(frames * (cbits + obits), 32) // 8, 'clip morph frame table does not match the frames')
    hashes = list(struct.unpack_from(f'<{channels}I', info, 16))
    pool = 16 + 12 * len(table)
    names = [_cstring(dat1, pool + at) if pool + at < len(dat1) else ''
             for at in struct.unpack_from(f'<{channels}I', info, 16 + 4 * channels)]
    lookup = np.unpackbits(np.frombuffer(info, np.uint8, table_bytes, 16 + 8 * channels))
    fo, fsize = table[CLIP_MORPH_FRAMES]
    stream = np.unpackbits(np.frombuffer(dat1, np.uint8, fsize, fo))
    weights = np.zeros((frames, channels), np.float32)
    for frame in range(frames):
        at = frame * (cbits + obits)
        active, word = _number(lookup, at, cbits), _number(lookup, at + cbits, obits)
        pos, channel = 32 * word, 0
        need = (first_bits + value_bits + (active - 1) * (step_bits + value_bits)) if active else 0
        require(pos + need <= len(stream), 'clip morph frame past its data')
        for i in range(active):
            width = first_bits if i == 0 else step_bits
            step = _number(stream, pos, width)
            channel = step if i == 0 else channel + step
            require(channel < channels and (i == 0 or step > 0), 'clip morph channel out of order or range')
            weights[frame, channel] = _number(stream, pos + width, value_bits) / WEIGHT_SCALE
            pos += width + value_bits
    return dict(hashes=hashes, names=names, weights=weights)
