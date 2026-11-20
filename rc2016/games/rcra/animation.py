"""Rift Apart (PC) ``.animclip`` skeletal clips, in the dict schema of ``formats.animation_samples.decode_clip``.

A clip is its own asset: the 36-byte toc header, then a DAT1 of type 0x0F64FFE8 at +36 with the same
section ids as an R&C 2016 clip. Base records, the base scale tail, the 48-byte destination layout and
the frame-major bit packing are the 2016 version-15 layout, so ``animation_transforms.sample_transforms``
turns the result into scale, rotation and translation per joint (frame 0 of a clip matches the model's
rest SQT within quantization; docs/rift-apart.md). Four things differ from 2016:

  0x9DF23F77 built      +0 u32 is the clip name's string offset, not a format version; +4 name hash
                        (game CRC-32 of the name); +8 flags (bit 0x40: additive, bit 0x10000: frame
                        table); +12 f32 fps; +16 f32 seconds; +32 u16 joints, u16 frames, u16 elements,
                        u16 0; +40 u32 bytes per stored sample; +44 u16 base scale-tail blocks, u16
                        count of 4-byte records after the tail (u16 joint, u16 value; seen once, on an
                        additive clip, and not read); +48 u16 stored samples
  0xD070D358 elements   u32 destination per element (not u16), then u8 bit width - 1 each, zero-padded
                        to a 4-byte multiple
  0x3A7B4855 samples    with flag 0x10000: round(seconds * fps) + 1 u16 stored-sample indices (one per
                        frame, plus a closing one when the frame count leaves out the end pose),
                        zero-padded to 16 bytes, then the stored samples (a clip that repeats poses
                        stores each once); without it, frame i is stored sample i, and one clip
                        stores an uncounted closing pose after them
  0xA3B26640 joints     joint hashes (game CRC-32 of the joint names) with an 0xFFFFFFFF terminator

An additive clip may give a joint the packed scale 0x8000 on every axis: a joint the layer does not scale,
read as one by ``animation_transforms.packed_scales``.

Clips without joint tracks (only custom float tracks, sections 0x14014CB6/0x116EB684) are not skeletal and
raise ``UnsupportedClipError``.

``motion`` reads the root motion section 0xE08AA35F: a start transform, then one record per frame (plus,
in some clips, a closing one), each the root's rotation and translation relative to the start:

  +0   u8 record format (0: 16-byte records, 1: 24-byte records), u8 kind (0: no records, 2, 3), u16 records
  +4   f32 start rotation xyzw; +32 f32 1.0, then f32 start translation xyz
  +64  u32 bytes from +124 (records, zero-padded so the section is a multiple of 16), u32 frames; +76 f32 minimum xyz and
       +100 f32 maximum xyz of the root translation over the clip (start + start rotation * record)
  +124 records. Format 0: i16 rotation xyzw (unit at 32767), i16 translation xyz, u8 exponent e (the
       translation is the integer times 2^-e), u8 a value of kind 3 (locomotion) clips that is not read.
       Format 1: u64 smallest-three rotation (three signed 20-bit fields from bit 0, each times
       sqrt(1/2) / (2^19 - 1), in xyzw order without the largest; bits 60-61 its index; that component is
       positive), f32 translation xyz, u32 a value that is not read (0 in all but 171 cutscene clips).

The 0x3976E44C path section (two 4x4 float matrices) is not read; neither are triggers, morph and face
sections.

``driver_classes`` lists the clips of each driver class (animation graph node) of an ``.animset``: the 80-byte
class records of 0x66CA6C6F (counted, not read), the 16-byte 0xA40B51D2 lookup rows (+0 u32 byte offset of
the class's node data in 0x73CEE17F) and that node data. The node layout is not decoded: a class's bytes are
scanned for u32 values at 4-byte steps, and the caller keeps those that are clip name hashes it knows. A class
that holds an additive clip and absolute clips is a node that adds the layer over them (the thug shooter's aim
idle class holds ``shooter_aim_idle_fwd``, ``_l090`` and ``_r090`` with their ``_Pitch`` layers).
"""
from __future__ import annotations

import hashlib
import math
import struct

import numpy as np

from rc2016.games.rcra.toc import sections as dat1_sections

CLIP_MAGIC = 0x0F64FFE8
HEADER = 36
BUILT, BASE, ELEMENTS, SAMPLES, JOINT_HASHES = 0x9DF23F77, 0x2BB5BC8F, 0xD070D358, 0x3A7B4855, 0xA3B26640
CUSTOM_TRACKS = 0x14014CB6
FLAG_ADDITIVE, FLAG_FRAME_TABLE = 0x40, 0x10000
# Runtime S/Q/T byte offsets inside a 48-byte joint record (scale xyz, quaternion xyzw, translation xyz).
COMPONENTS = frozenset((0, 4, 8, 16, 20, 24, 28, 32, 36, 40))
SCHEMA = 'rcra-pc-animclip-v1'


class UnsupportedClipError(ValueError):
    """The asset is a clip this decoder does not read (no joint tracks, or not an animclip)."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _align(value, step):
    return (value + step - 1) // step * step


def clip_dat1(data):
    """(DAT1 bytes, {tag: (offset, size)}) of a clip asset (toc header + DAT1)."""
    if len(data) < HEADER + 16 or struct.unpack_from('<I', data, 0)[0] != CLIP_MAGIC:
        raise UnsupportedClipError('Not a Rift Apart animclip (expected magic 0x%08X)' % CLIP_MAGIC)
    dat1 = data[HEADER:]
    if dat1[:4] != b'1TAD' or struct.unpack_from('<I', dat1, 4)[0] != CLIP_MAGIC:
        raise UnsupportedClipError('Rift Apart animclip without its DAT1 at +36')
    return dat1, dat1_sections(dat1)


def _name(dat1, table, name_hash):
    """The clip's own name: the NUL-terminated string after 'animclip built' in the string pool."""
    from rc2016.formats.container import game_crc32
    first = min((offset for offset, _size in table.values()), default=len(dat1))
    pool = dat1[16 + 12 * len(table):first]
    marker = pool.find(b'animclip built\0')
    if marker < 0:
        return None
    start = marker + len(b'animclip built\0')
    end = pool.find(b'\0', start)
    try:
        text = pool[start:end if end >= 0 else len(pool)].decode('ascii')
    except UnicodeDecodeError:
        return None
    return text if text and game_crc32(text) == name_hash else None


def header(data):
    """The built-header facts of a clip asset, without decoding samples (cheap: one section read)."""
    dat1, table = clip_dat1(data)
    require(BUILT in table, 'animclip without its built section 0x9DF23F77')
    offset, size = table[BUILT]
    require(size >= 52, 'truncated animclip built section')
    name_hash, flags, fps, seconds = struct.unpack_from('<IIff', dat1, offset + 4)
    joints, frames, elements, _zero = struct.unpack_from('<4H', dat1, offset + 32)
    stride, tail_blocks, tail_records, stored = struct.unpack_from('<IHHH', dat1, offset + 40)
    return dict(name=_name(dat1, table, name_hash), name_hash=f'0x{name_hash:08X}', flags=flags,
                additive_flag=bool(flags & FLAG_ADDITIVE), sample_rate_fps=fps, duration_seconds=seconds,
                bone_count=joints, original_frame_count=frames, sample_element_count=elements,
                sample_stride_bytes=stride, unknown_u32_28=tail_blocks, base_tail_blocks=tail_blocks,
                base_tail_records=tail_records, stored_sample_count=stored, frame_table=bool(flags & FLAG_FRAME_TABLE),
                skeletal=BASE in table and JOINT_HASHES in table)


def _unpack(payload, stride, widths, rows):
    """Unsigned values (len(widths), rows) of frame-major LSB-first bit-packed samples; checks zero padding."""
    if not widths or not rows:
        return np.zeros((len(widths), rows), np.int64)
    raw = np.frombuffer(payload, np.uint8, rows * stride).reshape(rows, stride)
    bits = np.unpackbits(raw, axis=1, bitorder='little')
    used = int(sum(widths))
    require(not bits[:, used:].any(), 'nonzero padding bits in a stored sample')
    out = np.empty((len(widths), rows), np.int64)
    position = 0
    for channel, width in enumerate(widths):
        weights = (1 << np.arange(width, dtype=np.int64))
        out[channel] = bits[:, position:position + width].astype(np.int64) @ weights
        position += width
    return out


def decode_clip(data, names=None, first_frames=None, label=''):
    """One Rift Apart clip asset as ``animation_samples.decode_clip`` returns a 2016 clip.

    ``names`` maps joint hash (int) to joint name. ``first_frames`` decodes only that many leading frames
    (reported as ``original_frame_count``) with the same checks. The output feeds
    ``animation_transforms.sample_transforms`` unchanged."""
    names = names or {}
    dat1, table = clip_dat1(data)
    meta = header(data)
    if not meta['skeletal']:
        kind = 'custom float tracks only' if CUSTOM_TRACKS in table else 'no joint base state'
        raise UnsupportedClipError(f'Not a skeletal clip ({kind})')

    def chunk(tag):
        if tag not in table:
            return b''
        offset, size = table[tag]
        return dat1[offset:offset + size]

    count = meta['bone_count']
    fps, seconds, frames = meta['sample_rate_fps'], meta['duration_seconds'], meta['original_frame_count']
    require(math.isfinite(fps) and fps > 0 and math.isfinite(seconds) and seconds >= 0, 'invalid clip timing')
    require(frames > 0, 'clip has no frames')
    span = round(fps * seconds)
    require(frames in (span, span + 1), 'frame count does not match the clip timing')
    hashes = chunk(JOINT_HASHES)
    require(len(hashes) == 4 * (count + 1) and hashes[-4:] == b'\xff' * 4, 'joint count/terminator mismatch')
    joint_hashes = struct.unpack(f'<{count}I', hashes[:4 * count])
    require(len(set(joint_hashes)) == count, 'duplicate joint hashes')
    base = chunk(BASE)
    tail_blocks = meta['base_tail_blocks']
    require(len(base) == 16 * count + 32 * tail_blocks + 4 * meta['base_tail_records'],
            'base state size does not match joints and scale tail')
    n = meta['sample_element_count']
    stride = meta['sample_stride_bytes']
    stored = meta['stored_sample_count']
    elements, payload = chunk(ELEMENTS), chunk(SAMPLES)
    if n:
        require(len(elements) == _align(5 * n, 4) and not any(elements[5 * n:]), 'sample element table size mismatch')
        destinations = struct.unpack_from(f'<{n}I', elements, 0)
        widths = [code + 1 for code in elements[4 * n:5 * n]]
    else:
        require(not elements and not stride, 'malformed empty element table')
        destinations, widths = (), []
    require(all(1 <= w <= 16 for w in widths), 'unsupported integer bit width')
    require(len(set(destinations)) == len(destinations), 'duplicate component destinations')
    require(all(d % 48 in COMPONENTS and d < count * 48 for d in destinations), 'destination outside the joint records')
    require(stride == (sum(widths) + 31) // 32 * 4, 'stored sample stride does not match bit widths')
    if meta['frame_table']:
        entries = span + 1
        table_bytes = _align(2 * entries, 16)
        require(stored >= 1 and len(payload) == table_bytes + stored * stride, 'sample payload size mismatch')
        order = np.frombuffer(payload, '<u2', entries).astype(np.intp)
        require(not any(payload[2 * entries:table_bytes]), 'nonzero frame table padding')
        require(int(order.max()) < stored, 'frame table points past the stored samples')
        order = order[:frames]   # a closing entry repeats a pose (the first, on a loop); the header counts frames
        samples = payload[table_bytes:]
    else:
        # A clip without a frame table stores one sample per frame, sometimes one more (the closing pose). One
        # clip (rivet_phonemes) stores that closing pose without counting it: frames == span, stored == frames,
        # one sample more in the payload, equal to the first (a loop). It is not played, as with a frame table.
        require(stored in (frames, frames + 1) or (not n and not payload), 'stored sample count does not match frames')
        uncounted_close = bool(stride) and frames == span == stored and len(payload) == (stored + 1) * stride
        require(len(payload) == stored * stride or uncounted_close, 'sample payload size mismatch')
        order = np.arange(frames, dtype=np.intp)
        samples = payload
    if first_frames is not None:
        frames = max(1, min(frames, first_frames))
        order = order[:frames]
    values = _unpack(samples, stride, widths, stored if n else 0)[:, order] if n else np.zeros((0, frames), np.int64)
    channels = []
    for index, destination in enumerate(destinations):
        joint, component = divmod(destination, 48)
        channels.append(dict(joint_index=joint, hash=f'0x{joint_hashes[joint]:08X}', name=names.get(joint_hashes[joint]),
                             destination_byte_offset=destination, component_byte_offset=component,
                             bit_width=widths[index], unsigned_deltas=values[index].tolist()))
    metadata = dict(meta, format=SCHEMA, original_frame_count=frames)
    return {
        'name': meta['name'] or label, 'source_offset': 0, 'source_size': len(data), 'metadata': metadata,
        'used_bits_per_frame': int(sum(widths)), 'sample_times_seconds': [i / fps for i in range(frames)],
        'padding_bits_per_frame': stride * 8 - int(sum(widths)),
        'sample_sha256': hashlib.sha256(payload).hexdigest(), 'sample_roundtrip_exact': True,
        'joints': [dict(joint_index=i, hash=f'0x{h:08X}', name=names.get(h),
                        packed_base_i16=list(struct.unpack_from('<7h', base, 16 * i)),
                        packed_base_control_bytes=list(base[16 * i + 14:16 * i + 16]))
                   for i, h in enumerate(joint_hashes)],
        'base_tail_hex': base[16 * count:16 * count + 32 * tail_blocks].hex(),
        'channels': channels,
    }


MOTION = 0xE08AA35F
MOTION_HEADER = 124
SMALLEST_THREE = math.sqrt(0.5) / ((1 << 19) - 1)


def _smallest_three(words):
    """Unit quaternions xyzw (N, 4) of u64 smallest-three words (see the module notes)."""
    fields = np.stack([(words >> np.uint64(20 * k)) & np.uint64(0xFFFFF) for k in range(3)], 1).astype(np.int64)
    small = np.where(fields >= 1 << 19, fields - (1 << 20), fields) * SMALLEST_THREE
    largest = ((words >> np.uint64(60)) & np.uint64(3)).astype(np.intp)
    require(not (words >> np.uint64(62)).any(), 'nonzero top bits in a smallest-three rotation')
    rotation = np.empty((len(words), 4))
    rows = np.arange(len(words))
    rotation[rows, largest] = np.sqrt(np.clip(1.0 - (small * small).sum(1), 0.0, None))
    others = (np.arange(4)[None, :] != largest[:, None])
    rotation[others] = small.ravel()
    return rotation


def motion(data):
    """The root motion of a clip asset (section 0xE08AA35F), or None when the clip has none.

    Returns dict(start_rotation_xyzw, start_translation, rotation_xyzw (R, 4), translation (R, 3), bounds),
    where record r places the root at start * record (rotation start_q * q, translation start_t +
    start_q * t). R is the frame count or one more (a closing record, as with the frame table). A clip with
    no records returns empty arrays; its root stays at the start."""
    dat1, table = clip_dat1(data)
    if MOTION not in table:
        return None
    offset, size = table[MOTION]
    return motion_section(dat1[offset:offset + size])


def motion_section(section):
    """``motion`` of the bytes of one 0xE08AA35F section."""
    size = len(section)
    require(size >= 128, 'truncated root motion section')
    layout, kind, count = struct.unpack_from('<BBH', section, 0)
    require(layout in (0, 1) and kind in (0, 2, 3), 'unknown root motion record format')
    start_q = np.array(struct.unpack_from('<4f', section, 4))
    start_t = np.array(struct.unpack_from('<3f', section, 36))
    body_bytes, _frames = struct.unpack_from('<II', section, 64)
    bounds = (struct.unpack_from('<3f', section, 76), struct.unpack_from('<3f', section, 100))
    require(MOTION_HEADER + body_bytes == size, 'root motion record size mismatch')
    if kind == 0:
        count = 0
    stride = 24 if layout else 16
    require(stride * count <= body_bytes and not any(section[MOTION_HEADER + stride * count:]),
            'root motion records overrun or nonzero padding')
    records = np.frombuffer(section, np.uint8, stride * count, MOTION_HEADER).reshape(count, stride)
    if layout:
        rotation = _smallest_three(records[:, :8].copy().view('<u8').ravel())
        translation = records[:, 8:20].copy().view('<f4').reshape(count, 3).astype(float)
    else:
        values = records.copy().view('<i2').reshape(count, 8).astype(float)
        rotation = values[:, :4] / 32767.0
        norms = np.linalg.norm(rotation, axis=1)
        require(bool(np.all(np.abs(norms - 1.0) < 1e-3)), 'root motion rotation is not a unit quaternion')
        rotation /= norms[:, None]
        translation = values[:, 4:7] * np.exp2(-records[:, 14].astype(float))[:, None]
    require(np.isfinite(start_q).all() and np.isfinite(start_t).all() and np.isfinite(translation).all(),
            'non-finite root motion')
    return dict(start_rotation_xyzw=start_q, start_translation=start_t, rotation_xyzw=rotation,
                translation=translation, bounds=bounds, layout=layout, kind=kind)


PHONEME_TABLE = 0x1D4BD9FA
PERFORMANCE_SLOTS = 8


def pose_table(data):
    """The phoneme table of a face pose clip (section 0x1D4BD9FA): the pose frame of each performance index, or
    None for a clip without one. Every table in the game has 40 entries."""
    dat1, table = clip_dat1(data)
    if PHONEME_TABLE not in table:
        return None
    offset, size = table[PHONEME_TABLE]
    require(size and size % 2 == 0, 'malformed phoneme table')
    return list(struct.unpack_from(f'<{size // 2}H', dat1, offset))


DRIVER_BUILT, DRIVER_LOOKUP, DRIVER_DATA = 0x66CA6C6F, 0xA40B51D2, 0x73CEE17F


def driver_classes(data):
    """Per driver class of an animset asset (toc header + DAT1), in record order: the u32 words of its node data.
    [] for an animset without driver classes. The words are candidates for clip name hashes; the node layout
    is not decoded (module docstring)."""
    dat1 = data[HEADER:]
    table = dat1_sections(dat1)
    if not all(tag in table for tag in (DRIVER_BUILT, DRIVER_LOOKUP, DRIVER_DATA)):
        return []
    _built_at, built_size = table[DRIVER_BUILT]
    lookup_at, lookup_size = table[DRIVER_LOOKUP]
    data_at, data_size = table[DRIVER_DATA]
    count = built_size // 80
    require(lookup_size // 16 == count, 'driver class lookup does not match its records')
    starts = [struct.unpack_from('<I', dat1, lookup_at + 16 * i)[0] for i in range(count)]
    require(all(s <= data_size for s in starts), 'driver class data offset outside its section')
    ends = sorted(set(starts) | {data_size})
    classes = []
    for start in starts:
        end = ends[ends.index(start) + 1] if start < data_size else start
        classes.append(struct.unpack_from(f'<{(end - start) // 4}I', dat1, data_at + start))
    return classes


def performance_clip(data):
    """A ``.performanceclip`` (face performance of one voice line): dict(name, name_hash, sample_rate_fps,
    duration_seconds, frames, index (F, 8) and weight (F, 8)).

    The built header is a clip's (no joints); the samples section holds one 16-byte row per frame, the rows
    zero-padded to a multiple of 8: u8 performance index x 8, then u8 weight x 8 (out of 255; a frame's weights
    sum to 255 within rounding). An index becomes a pose frame through its pose clip's ``pose_table``. A few
    clips (31 parade and crowd lines) keep weighted rows past the header's frame count; they are not played."""
    dat1, table = clip_dat1(data)
    meta = header(data)
    require(SAMPLES in table, 'performance clip without samples')
    frames, fps = meta['original_frame_count'], meta['sample_rate_fps']
    require(frames > 0 and math.isfinite(fps) and fps > 0, 'invalid performance clip timing')
    offset, size = table[SAMPLES]
    require(size % (16 * 8) == 0 and size // 16 >= frames, 'performance rows do not cover the frames')
    rows = np.frombuffer(dat1, np.uint8, 16 * frames, offset).reshape(frames, 16)
    return dict(name=meta['name'], name_hash=meta['name_hash'], sample_rate_fps=fps,
                duration_seconds=meta['duration_seconds'], frames=frames,
                index=rows[:, :PERFORMANCE_SLOTS].astype(np.intp), weight=rows[:, PERFORMANCE_SLOTS:] / 255.0)


def performance_frames(pose_frames, table, performance):
    """Additive frames of a performance: per frame, the weighted blend of its pose frames.

    ``pose_frames`` are the pose clip's ``sample_transforms`` frames (additive deltas), ``table`` its
    ``pose_table``. Per joint, translation and scale are the weighted sums of the poses' (a weight the frame
    leaves unused, or a skipped joint, counts as the identity delta), and the rotation is the normalised
    weighted sum of the poses' quaternions in the identity's hemisphere. Weights above 1 in total (rounding:
    256/255) are scaled to 1. This blend is a reading of the data, not traced in the game's runtime."""
    index, weight = performance['index'], performance['weight']
    require(int(index[weight > 0].max(initial=0)) < len(table), 'performance index past the phoneme table')
    poses = sorted({table[i] for i in np.unique(index[weight > 0])})
    require(not poses or poses[-1] < len(pose_frames), 'phoneme table points past the pose clip')
    hashes = [row['hash'] for row in pose_frames[0]] if pose_frames else []
    names = [row.get('name') for row in pose_frames[0]] if pose_frames else []
    count = len(hashes)
    T = np.zeros((len(pose_frames), count, 3))
    Q = np.tile([0.0, 0.0, 0.0, 1.0], (len(pose_frames), count, 1))
    S = np.ones((len(pose_frames), count, 3))
    for f, frame in enumerate(pose_frames):
        for j, row in enumerate(frame):
            T[f, j], S[f, j] = row['translation'], row['scale']
            if row['rotation_xyzw']:
                q = np.asarray(row['rotation_xyzw'], float)
                Q[f, j] = -q if q[3] < 0 else q
    frame_of = np.asarray(table, np.intp)[np.minimum(index, len(table) - 1)]
    w = weight * (weight > 0)
    w = w / np.maximum(1.0, w.sum(1, keepdims=True))
    rest = 1.0 - w.sum(1)                                             # the identity delta's share
    t = np.einsum('fs,fsjc->fjc', w, T[frame_of])
    s = np.einsum('fs,fsjc->fjc', w, S[frame_of]) + rest[:, None, None]
    q = np.einsum('fs,fsjc->fjc', w, Q[frame_of])
    q[..., 3] += rest[:, None]
    q /= np.linalg.norm(q, axis=2, keepdims=True)
    return [[dict(name=names[j], hash=hashes[j], scale=s[f, j].tolist(), rotation_xyzw=q[f, j].tolist(),
                  translation=t[f, j].tolist()) for j in range(count)] for f in range(len(w))]
