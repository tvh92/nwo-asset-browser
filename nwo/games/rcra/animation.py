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
  +60  u32 record offset from +124 (0); +64 u32 bytes from +124 (records, zero-padded so the section is a multiple
       of 16), u16 frames, u8 count of 80-byte entries at that offset (0 in every clip); +76 f32 minimum xyz and
       +100 f32 maximum xyz of the root translation over the clip (start + start rotation * record)
  +124 records. Format 0: i16 rotation xyzw (unit at 32767), i16 translation xyz, u8 exponent e (the
       translation is the integer times 2^-e), i8 foot phase. Format 1: u64 smallest-three rotation (three
       signed 20-bit fields from bit 0, each times sqrt(1/2) / (2^19 - 1), in xyzw order without the largest;
       bits 60-61 its index; that component is positive), f32 translation xyz, u32 whose low byte is the i8 foot
       phase (0 in all but 171 cutscene clips; 164 of them kind 3).

Kind 3 (locomotion) clips set the foot phase: (2 * i8 + 1) / 255, so -128 is -1 where the left foot lands and 127
is +1 where the right foot lands, linear between (Ratchet's strafe walk and run). The game's root-motion sampler
(RiftApart.exe 0x140F29C00 and 0x140F2A100) computes it this way and reads the byte only when the kind byte's bit 0
is set (kind 3); it reads nothing else of the format-1 u32. 7 kind-2 clips (the spitter's distortion projectile and
six fixer-scaffold props of one cutscene) hold a smooth per-frame curve in that u32's high 16 bits, which the game
does not read.

The 0x3976E44C path section (two 4x4 float matrices) is not read: RiftApart.exe never names it (its id occurs
nowhere in the executable), so the game does not use it. Triggers, morph and face sections are read elsewhere or
not at all.

``driver_graph`` decodes the driver classes (animation graphs) of an ``.animset``. Each class is an 80-byte
record in 0x66CA6C6F, a 16-byte row in 0xA40B51D2 (+0 u32 byte offset of its data in 0x73CEE17F) and that data:
a run of nodes, then 12-byte variable bindings. Record (u32 words): +0 class name hash, +4 slot count, +8 flags,
+12 data bytes, +16 root node offset, +20 node count, +24 binding count, +28 binding offset, +56 four f32
transition durations in seconds (default, from none, to none, self). Names are game CRC-32 hashes
(``container.game_crc32``) of strings the animset holds.

The names of node types, fields and enums come from RiftApart.exe's reflection data (``AnimDriverDef``,
``AnimDriverBiBlendNodeDef``, ``kSelectCurrentIndex`` ...); which built kind is which type follows from the fields
and from the binding targets below. Class flags: bits 0-1 the layer (0 base, 1 partial, 3 partial additive: the
classes so marked are named ``*_Partial``, ``React_Twitch``, ``*_Additive``), bit 2 set on looping classes (by
name: 809 loop or idle classes have it, 220 do not; 1,228 intros, outros, reacts and deaths lack it, 93 have it).
The remaining bits are not read.

  node header (40 bytes)  +0 u32 node name hash, +4 u16 type (low byte the kind, high byte flags), +6 u16 id,
                          +8 f32 time, +16 f32 rate, +24 and +28 f32 -1, +36 u32 0xFFFFFFFF
  kind 1 clip            flags 1: additive (``kClipNodeFlagsIsAdditive``); +40 u32 0 or 2 (not read), +44 u32
                          reference (``clip_references``)
  kind 2 bi-blend        +40 u16 child 0, u16 child 1, +44 u8 type (1 additive, 2 combine additive, 3
                          difference), +52 f32 factor
  kind 5 blend           +40 u16 variable slot, u16 children, +44 u16 child list, u16 span list: per child a
                          (low, high) f32 span on +48, the current value
  kind 7 random          +40 u16, u16 children, +44 u16 child list, u16 weight list (0xFFFF: equal odds): per
                          child a u16 weight out of 65535
  kind 9 select          +40 u16 children, u16 child list, +44 u32 current index (the child played)
  kind 10, 12 wrap       +40 u32 child (10 has a variable hash at +44; neither type is named)

Child references and lists are u16 word offsets into the class data (byte offset / 4); a list holds u16 word
offsets of the children. Nodes are found by their header signature (+24, +28 and +36), which yields the record's
node count in every class of the install. A binding is u32 variable name hash, u16 slot, u16 word offset, u32
flags: with flags low byte nonzero the word offset holds a zero-terminated u16 list of the bound fields' word
offsets, else it is the bound field; flag 0x100 marks a float field (the select index is an integer). The bound
fields are the reflection's connection targets: a clip's time +8 (``kClipTime``: the head and aim layers pick
their pose by time), a bi-blend's factor +52 (``kBiBlendFactor``), a blend's current value +48
(``kBlendCurrentValue``) and a select's current index +44 (``kSelectCurrentIndex``). The value a variable has at a
moment is game state; without it a node plays its stored value (``driver_graph`` returns them).

The animset section 0xDF74DA06 lists (u32 variable hash, u32 class hash, u32 word) rows: which variables each class
uses; the word's bit 16 is set for float variables. Its low 16 bits are not read.

``clip_references`` resolves a clip node's reference through the animset's clip lookup 0xB79CF1D7 (24-byte rows:
u32 hash, u32 kind, i32 group offset, u64 asset id, u32 0). Kind 0 and 0x8000 name a clip asset by its path
CRC-64; kind 1 names a clip group: an inner DAT1 at the group offset in 0x9FD19C20, whose 0x5025990D section
(16-byte header, then rows of u32 list offset, u32, u32 clip count, u32 0) indexes tag lists in 0x51A11F85 (tag
hashes, two 0xFFFFFFFF words, then the entry's clip hashes, which the lookup resolves). The game picks a group's
entry by its tags (hit direction, height).

A bi-blend node whose child 1 holds additive clips adds them over child 0 (``RcraAnimationLibrary.layer_partners``):
the thug shooter's aim idle class blends, by its ``Aim_Horizontal`` variable, three bi-blends such as
(``shooter_aim_idle_fwd``, ``Shooter_Aim_Idle_Fwd_Pitch``). 1,183 of the 1,260 bi-blends have the type
combine additive; that it adds child 1 scaled by the factor is a reading of that name and the data, not traced
in the runtime.

Face pose clips name their pose frames in section 0x495BA079 ((u32 name hash, u32 frame) pairs sorted by hash;
``pose_names``): the 13 shared visemes ``vis_xxx`` (silence), ``vis_aaa``, ``vis_eee``, ``vis_iii``, ``vis_ooo``,
``vis_uuu``, ``vis_ow``, ``vis_www``, ``vis_lll``, ``vis_rrr``, ``vis_fv``, ``vis_bmp``, ``vis_nnn``, and for a few
speakers eye, brow and lid poses. The 40 performance indices of the phoneme table are the reflection's
``kPhoneme`` enum (``PHONEMES``: x, IY, IH, EH ... SH, h), which the visemes match: index 0 ``x`` plays
``vis_xxx``, ``v`` and ``f`` play ``vis_fv``, ``m``, ``b`` and ``p`` play ``vis_bmp``.
"""
from __future__ import annotations

import bisect
import hashlib
import math
import struct

import numpy as np

from nwo.games.rcra.toc import sections as dat1_sections

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
    from nwo.formats.container import game_crc32
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

    Returns dict(start_rotation_xyzw, start_translation, rotation_xyzw (R, 4), translation (R, 3), bounds, phase),
    where record r places the root at start * record (rotation start_q * q, translation start_t +
    start_q * t). R is the frame count or one more (a closing record, as with the frame table). ``phase`` is
    the foot phase per record (R,) in [-1, 1] for a kind-3 clip, else None. A clip with no records returns empty
    arrays; its root stays at the start."""
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
        phase_bytes = records[:, 20]
    else:
        values = records.copy().view('<i2').reshape(count, 8).astype(float)
        rotation = values[:, :4] / 32767.0
        norms = np.linalg.norm(rotation, axis=1)
        require(bool(np.all(np.abs(norms - 1.0) < 1e-3)), 'root motion rotation is not a unit quaternion')
        rotation /= norms[:, None]
        translation = values[:, 4:7] * np.exp2(-records[:, 14].astype(float))[:, None]
        phase_bytes = records[:, 15]
    phase = (2.0 * phase_bytes.astype(np.int8) + 1.0) / 255.0 if kind == 3 else None
    require(np.isfinite(start_q).all() and np.isfinite(start_t).all() and np.isfinite(translation).all(),
            'non-finite root motion')
    return dict(start_rotation_xyzw=start_q, start_translation=start_t, rotation_xyzw=rotation,
                translation=translation, bounds=bounds, layout=layout, kind=kind, phase=phase)


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


# The performance indices: RiftApart.exe's kPhoneme enum, in order (index 0 is silence).
PHONEMES = ('x IY IH EH AE AH UW UH AA AO EY AY OY AW OW l r y w ER m n NG CH j DH b d g p t k z ZH v f TH s SH '
            'h').split()
POSE_NAMES_SECTION = 0x495BA079
# Every pose name of the install's 22 face pose clips (game CRC-32 verified; the clips hold only the hashes).
POSE_NAMES = ('vis_xxx vis_aaa vis_eee vis_iii vis_ooo vis_uuu vis_ow vis_www vis_lll vis_rrr vis_fv vis_bmp vis_nnn '
              'Mouth_Open Mouth_Closed Eyes_Open Eyes_Closed Left_Eye_Closed Right_Eye_Closed mouth_open '
              'mouth_closed Smile Idle eyes_up eyes_down Eyes_left Eyes_right brows_up brows_down lids_closed').split()


def pose_names(data):
    """{pose frame: name} of a face pose clip (section 0x495BA079), or None for a clip without names. A name the
    game's list does not hold is given as its hash ('0x...'); a frame with two names keeps the first by hash."""
    from nwo.formats.container import game_crc32
    dat1, table = clip_dat1(data)
    if POSE_NAMES_SECTION not in table:
        return None
    offset, size = table[POSE_NAMES_SECTION]
    require(size % 8 == 0, 'malformed pose name table')
    known = {game_crc32(name): name for name in POSE_NAMES}
    found = {}
    for name_hash, frame in struct.iter_unpack('<II', dat1[offset:offset + size]):
        found.setdefault(frame, known.get(name_hash, f'0x{name_hash:08X}'))
    return found


DRIVER_BUILT, DRIVER_LOOKUP, DRIVER_DATA = 0x66CA6C6F, 0xA40B51D2, 0x73CEE17F
CLIP_LOOKUP, CLIP_GROUPS, GROUP_ENTRIES, GROUP_LISTS = 0xB79CF1D7, 0x9FD19C20, 0x5025990D, 0x51A11F85
NODE_HEADER = 40
NODE_CLIP, NODE_PAIR, NODE_BLEND_SPACE, NODE_RANDOM, NODE_SELECT = 1, 2, 5, 7, 9
NODE_WRAPS = (10, 12)
PAIR_WEIGHT = 52   # a bi-blend node's factor field, which a binding may drive
PAIR_TYPES = {1: 'additive', 2: 'combine additive', 3: 'difference'}   # kBiBlendNodeType
LAYERS = {0: 'base', 1: 'partial', 3: 'partial additive'}   # class flags bits 0-1
CLASS_LOOPING = 0x4
BINDING_FLOAT = 0x100
LOOKUP_GROUP = 1
_NODE_SIGNATURE = struct.pack('<2f', -1.0, -1.0)


def _word_list(blob, word, count):
    return [4 * w for w in struct.unpack_from(f'<{count}H', blob, 4 * word)]


def _graph_node(blob, at):
    """One driver class node at byte ``at`` of its class data (module docstring)."""
    name, kind_word, ident, phase, _zero, rate = struct.unpack_from('<IHHfIf', blob, at)
    kind = kind_word & 0xFF
    first, second = struct.unpack_from('<HH', blob, at + NODE_HEADER)
    node = dict(at=at, name=name, kind=kind, flags=kind_word >> 8, id=ident, phase=phase, rate=rate, children=[],
                bound={})
    if kind == NODE_CLIP:
        node.update(mode=struct.unpack_from('<I', blob, at + 40)[0], ref=struct.unpack_from('<I', blob, at + 44)[0])
    elif kind == NODE_PAIR:
        node.update(children=[4 * first, 4 * second], mode=blob[at + 44],
                    weight=struct.unpack_from('<f', blob, at + 52)[0])
    elif kind == NODE_BLEND_SPACE:
        children, spans = struct.unpack_from('<HH', blob, at + 44)
        values = struct.unpack_from(f'<{2 * second}f', blob, 4 * spans)
        node.update(slot=first, children=_word_list(blob, children, second),
                    spans=[values[2 * k:2 * k + 2] for k in range(second)],
                    value=struct.unpack_from('<f', blob, at + 48)[0])
    elif kind == NODE_RANDOM:
        children, weights = struct.unpack_from('<HH', blob, at + 44)
        node.update(children=_word_list(blob, children, second),
                    weights=(None if weights == 0xFFFF else
                             [w / 65535.0 for w in struct.unpack_from(f'<{second}H', blob, 4 * weights)]))
    elif kind == NODE_SELECT:
        node.update(children=_word_list(blob, second, first), index=struct.unpack_from('<I', blob, at + 44)[0])
    elif kind in NODE_WRAPS:
        node.update(children=[4 * struct.unpack_from('<I', blob, at + 40)[0]])
    return node


def _bound_fields(blob, word, flags):
    if not flags & 0xFF:
        return [4 * word]
    fields, at = [], 4 * word
    while at + 2 <= len(blob):
        value = struct.unpack_from('<H', blob, at)[0]
        if not value:
            break
        fields.append(4 * value)
        at += 2
    return fields


def driver_graph(data):
    """The driver classes (animation graphs) of an animset asset (toc header + DAT1), in record order; [] for an
    animset without them. Each is dict(name: class name hash, root: root node offset or None, nodes: {byte offset:
    node}, bindings: [dict(variable: name hash, slot, fields: [bound field byte offsets], float)], flags, layer
    ('base', 'partial', 'partial additive' or None), looping, transitions: (default, from none, to none, self)
    seconds). A node is dict(at, name, kind, flags, id, phase (its stored time), rate, children: [node offsets],
    bound: {field offset in the node: variable name hash}) plus its kind's fields: clip mode, ref; bi-blend mode
    (type), weight (factor); blend slot, spans, value; random weights (None: equal odds); select index (module
    docstring)."""
    dat1 = data[HEADER:]
    table = dat1_sections(dat1)
    if not all(tag in table for tag in (DRIVER_BUILT, DRIVER_LOOKUP, DRIVER_DATA)):
        return []
    built_at, built_size = table[DRIVER_BUILT]
    lookup_at, lookup_size = table[DRIVER_LOOKUP]
    data_at, data_size = table[DRIVER_DATA]
    count = built_size // 80
    require(lookup_size // 16 == count, 'driver class lookup does not match its records')
    classes = []
    for number in range(count):
        record = struct.unpack_from('<8I', dat1, built_at + 80 * number)
        start = struct.unpack_from('<I', dat1, lookup_at + 16 * number)[0]
        size, root, node_count, bound_count, bound_at = record[3:8]
        flags = record[2]
        about = dict(name=record[0], flags=flags, layer=LAYERS.get(flags & 3), looping=bool(flags & CLASS_LOOPING),
                     transitions=struct.unpack_from('<4f', dat1, built_at + 80 * number + 56))
        if not size:   # an empty class: no nodes, lookup offset 0xFFFFFFFF
            classes.append(dict(about, root=None, nodes={}, bindings=[]))
            continue
        require(start + size <= data_size, 'driver class data outside its section')
        blob = dat1[data_at + start:data_at + start + size]
        end = bound_at if bound_count else size
        require(end <= size and bound_at + 12 * bound_count <= size, 'driver class bindings outside its data')
        nodes = {at: _graph_node(blob, at) for at in range(0, end - NODE_HEADER + 1, 4)
                 if blob[at + 24:at + 32] == _NODE_SIGNATURE and blob[at + 36:at + 40] == b'\xff' * 4}
        require(len(nodes) == node_count, 'driver class node count mismatch')
        require(not nodes or root in nodes, 'driver class root is not a node')
        require(all(child in nodes for node in nodes.values() for child in node['children']),
                'driver class node links outside its nodes')
        bindings, starts = [], sorted(nodes)
        for k in range(bound_count):
            variable, slot, word, flags = struct.unpack_from('<IHHI', blob, bound_at + 12 * k)
            bindings.append(dict(variable=variable, slot=slot, fields=_bound_fields(blob, word, flags),
                                 float=bool(flags & BINDING_FLOAT)))
            for field in bindings[-1]['fields']:
                owner = starts[bisect.bisect_right(starts, field) - 1] if starts and field >= starts[0] else None
                if owner is not None:
                    nodes[owner]['bound'][field - owner] = variable
        classes.append(dict(about, root=root if nodes else None, nodes=nodes, bindings=bindings))
    return classes


def graph_reach(graph_class, start=None):
    """Node offsets reachable from ``start`` (default the class root), depth first in child order. Nodes no path
    reaches are not played by the class."""
    nodes = graph_class['nodes']
    start = graph_class['root'] if start is None else start
    order, stack, seen = [], [start] if start in nodes else [], set()
    while stack:
        at = stack.pop()
        if at in seen:
            continue
        seen.add(at)
        order.append(at)
        stack.extend(reversed(nodes[at]['children']))
    return order


def clip_references(data):
    """{clip node reference hash: [clip asset ids (path CRC-64)]} of an animset asset: one asset for a clip, a
    group's clips in entry order (each once) for a group (module docstring)."""
    dat1 = data[HEADER:]
    table = dat1_sections(dat1)
    if CLIP_LOOKUP not in table:
        return {}
    at, size = table[CLIP_LOOKUP]
    rows = [struct.unpack_from('<IIiQI', dat1, at + 24 * k) for k in range(size // 24)]
    assets = {row[0]: row[3] for row in rows if row[1] != LOOKUP_GROUP}
    found = {name: [asset] for name, asset in assets.items()}
    groups_at, groups_size = table.get(CLIP_GROUPS, (0, 0))
    for name, _kind, offset, _asset, _zero in (row for row in rows if row[1] == LOOKUP_GROUP):
        require(0 <= offset < groups_size, 'clip group outside its section')
        inner = dat1[groups_at + offset:groups_at + groups_size]
        sections = dat1_sections(inner)
        members = []
        if GROUP_ENTRIES in sections and GROUP_LISTS in sections:
            entries_at, entries_size = sections[GROUP_ENTRIES]
            lists_at, lists_size = sections[GROUP_LISTS]
            entries = [struct.unpack_from('<4I', inner, entries_at + 16 * k) for k in range(1, entries_size // 16)]
            ends = [entry[0] for entry in entries[1:]] + [lists_size]
            for (start, _flags, clips, _zero), end in zip(entries, ends):
                require(start <= end <= lists_size and 4 * clips <= end - start, 'malformed clip group')
                hashes = struct.unpack_from(f'<{clips}I', inner, lists_at + end - 4 * clips)
                members.extend(assets[h] for h in hashes if h in assets)
        found[name] = list(dict.fromkeys(members))
    return found


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
