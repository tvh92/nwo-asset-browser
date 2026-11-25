"""Unsigned BC6H (DXGI 95) decoder to linear float32, every block mode.

Pillow decodes BC6H to 8 bits and clamps it to [0, 1], which loses every value above 1 (sun,
bright cloud). This decoder keeps the full half-float range. It implements all fourteen modes of
the Direct3D 11 functional specification: the ten two-subset modes (1 to 10) and the four
one-subset modes (11 to 14). A block with one of the four reserved mode fields decodes to black, as
the specification requires, and is reported in the returned mask.

The mode field is 2 bits for modes 1 and 2, otherwise 5 bits, read little-endian from the low end of
the 128-bit block. ``LAYOUTS`` lists each mode's header bits in order after the mode field, as the
specification's bit-layout table does (``rw`` the first endpoint's red, ``rx``/``ry``/``rz`` the
other three, ``d`` the partition). Every mode is checked against the GPU's own BC6H decoder on the
game's probes (tests/test_bc6h.py, ``RCRA_GAME``).
"""
from __future__ import annotations

WEIGHTS_3 = (0, 9, 18, 27, 37, 46, 55, 64)
WEIGHTS_4 = (0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64)
# Two-subset partitions 0-31: row-major subset id of each pixel, with 128/129 marking the anchors
# (pixel 0 for subset 0, the "fix-up" pixel for subset 1), whose index has one bit fewer.
PARTITIONS = (
    (128, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 129), (128, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 129),
    (128, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 129), (128, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 129),
    (128, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 129), (128, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 129),
    (128, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 129), (128, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 129),
    (128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 129), (128, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 129),
    (128, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 129), (128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 129),
    (128, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 129), (128, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 129),
    (128, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 129), (128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 129),
    (128, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 129), (128, 1, 129, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    (128, 0, 0, 0, 0, 0, 0, 0, 129, 0, 0, 0, 1, 1, 1, 0), (128, 1, 129, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0),
    (128, 0, 129, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0), (128, 0, 0, 0, 1, 0, 0, 0, 129, 1, 0, 0, 1, 1, 1, 0),
    (128, 0, 0, 0, 0, 0, 0, 0, 129, 0, 0, 0, 1, 1, 0, 0), (128, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 129),
    (128, 0, 129, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0), (128, 0, 0, 0, 1, 0, 0, 0, 129, 0, 0, 0, 1, 1, 0, 0),
    (128, 1, 129, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0), (128, 0, 129, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0),
    (128, 0, 0, 1, 0, 1, 1, 1, 129, 1, 1, 0, 1, 0, 0, 0), (128, 0, 0, 0, 1, 1, 1, 1, 129, 1, 1, 1, 0, 0, 0, 0),
    (128, 1, 129, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0), (128, 0, 129, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0),
)
BLOCK_BYTES = 16

# mode field -> (mode number, endpoint bits, (red, green, blue) delta bits or None when untransformed, header layout).
# A layout token is a field name and either one bit ("gy4") or a range, high bit first ("rw9:0"); "rw10:15" is the
# specification's reversed range rw[10:15] (bit 10 stored first).
LAYOUTS = {
    0b00: (1, 10, (5, 5, 5), 'gy4 by4 bz4 rw9:0 gw9:0 bw9:0 rx4:0 gz4 gy3:0 gx4:0 bz0 gz3:0 bx4:0 bz1 by3:0 ry4:0 bz2 '
                             'rz4:0 bz3 d4:0'),
    0b01: (2, 7, (6, 6, 6), 'gy5 gz4 gz5 rw6:0 bz0 bz1 by4 gw6:0 by5 bz2 gy4 bw6:0 bz3 bz5 bz4 rx5:0 gy3:0 gx5:0 gz3:0 '
                            'bx5:0 by3:0 ry5:0 rz5:0 d4:0'),
    0b00010: (3, 11, (5, 4, 4), 'rw9:0 gw9:0 bw9:0 rx4:0 rw10 gy3:0 gx3:0 gw10 bz0 gz3:0 bx3:0 bw10 bz1 by3:0 ry4:0 bz2 '
                                'rz4:0 bz3 d4:0'),
    0b00110: (4, 11, (4, 5, 4), 'rw9:0 gw9:0 bw9:0 rx3:0 rw10 gz4 gy3:0 gx4:0 gw10 gz3:0 bx3:0 bw10 bz1 by3:0 ry3:0 bz0 '
                                'bz2 rz3:0 gy4 bz3 d4:0'),
    0b01010: (5, 11, (4, 4, 5), 'rw9:0 gw9:0 bw9:0 rx3:0 rw10 by4 gy3:0 gx3:0 gw10 bz0 gz3:0 bx4:0 bw10 by3:0 ry3:0 bz1 '
                                'bz2 rz3:0 bz4 bz3 d4:0'),
    0b01110: (6, 9, (5, 5, 5), 'rw8:0 by4 gw8:0 gy4 bw8:0 bz4 rx4:0 gz4 gy3:0 gx4:0 bz0 gz3:0 bx4:0 bz1 by3:0 ry4:0 bz2 '
                               'rz4:0 bz3 d4:0'),
    0b10010: (7, 8, (6, 5, 5), 'rw7:0 gz4 by4 gw7:0 bz2 gy4 bw7:0 bz3 bz4 rx5:0 gy3:0 gx4:0 bz0 gz3:0 bx4:0 bz1 by3:0 '
                               'ry5:0 rz5:0 d4:0'),
    0b10110: (8, 8, (5, 6, 5), 'rw7:0 bz0 by4 gw7:0 gy5 gy4 bw7:0 gz5 bz4 rx4:0 gz4 gy3:0 gx5:0 gz3:0 bx4:0 bz1 by3:0 '
                               'ry4:0 bz2 rz4:0 bz3 d4:0'),
    0b11010: (9, 8, (5, 5, 6), 'rw7:0 bz1 by4 gw7:0 by5 gy4 bw7:0 bz5 bz4 rx4:0 gz4 gy3:0 gx4:0 bz0 gz3:0 bx5:0 by3:0 '
                               'ry4:0 bz2 rz4:0 bz3 d4:0'),
    0b11110: (10, 6, None, 'rw5:0 gz4 bz0 bz1 by4 gw5:0 gy5 by5 bz2 gy4 bw5:0 gz5 bz3 bz5 bz4 rx5:0 gy3:0 gx5:0 gz3:0 '
                           'bx5:0 by3:0 ry5:0 rz5:0 d4:0'),
    0b00011: (11, 10, None, 'rw9:0 gw9:0 bw9:0 rx9:0 gx9:0 bx9:0'),
    0b00111: (12, 11, (9, 9, 9), 'rw9:0 gw9:0 bw9:0 rx8:0 rw10 gx8:0 gw10 bx8:0 bw10'),
    0b01011: (13, 12, (8, 8, 8), 'rw9:0 gw9:0 bw9:0 rx7:0 rw10:11 gx7:0 gw10:11 bx7:0 bw10:11'),
    0b01111: (14, 16, (4, 4, 4), 'rw9:0 gw9:0 bw9:0 rx3:0 rw10:15 gx3:0 gw10:15 bx3:0 bw10:15'),
}


def _parse_layout(text):
    """[(field, bit)] in stored order. In the specification's notation x[a:b] the first stored bit is bit b: x[9:0] is
    stored from bit 0 up, the reversed x[10:15] from bit 15 down."""
    out = []
    for token in text.split():
        split = 1 if token[0] == 'd' else 2
        name, bits = token[:split], token[split:]
        if ':' in bits:
            a, b = (int(v) for v in bits.split(':'))
            step = 1 if a >= b else -1
            out += [(name, k) for k in range(b, a + step, step)]
        else:
            out.append((name, int(bits)))
    return out


_PARSED = {field: (mode, width, deltas, _parse_layout(layout)) for field, (mode, width, deltas, layout) in LAYOUTS.items()}


def _unquantize(value, bits):
    if bits >= 15:
        return value
    if value == 0:
        return 0
    if value == (1 << bits) - 1:
        return 0xFFFF
    return ((value << 15) + 0x4000) >> (bits - 1)


def _sign_extend(value, bits):
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def _header(bits, field):
    """(mode, endpoints, partition): endpoints[subset] = ((r0, g0, b0), (r1, g1, b1)) unquantized to 16 bits; None for a
    reserved mode field."""
    entry = _PARSED.get(field)
    if entry is None:
        return None
    mode, width, deltas, layout = entry
    values = dict.fromkeys(('rw', 'gw', 'bw', 'rx', 'gx', 'bx', 'ry', 'gy', 'by', 'rz', 'gz', 'bz', 'd'), 0)
    position = 2 if mode <= 2 else 5
    for name, bit in layout:
        values[name] |= ((bits >> position) & 1) << bit
        position += 1
    mask = (1 << width) - 1
    names = (('rw', 'gw', 'bw'), ('rx', 'gx', 'bx'), ('ry', 'gy', 'by'), ('rz', 'gz', 'bz'))
    points = [[values[n] for n in names[0]]]
    for row in names[1:] if mode <= 10 else names[1:2]:
        if deltas is None:
            points.append([values[n] for n in row])
        else:
            points.append([(base + _sign_extend(values[n], d)) & mask for base, n, d in zip(points[0], row, deltas)])
    points = [[_unquantize(v, width) for v in p] for p in points]
    if mode <= 10:
        return mode, [(points[0], points[1]), (points[2], points[3])], values['d']
    return mode, [(points[0], points[1])], 0


def decode_bc6h(data, width, height):
    """(image, unsupported): float32 (height, width, 3) linear RGB and a bool (height/4, width/4) mask of
    blocks with a reserved mode field (their pixels are zero). ``data`` is the linear (not PS4-tiled) block
    stream, 16 bytes per 4x4 block, rows of blocks top first."""
    import numpy as np
    blocks_x, blocks_y = (width + 3) // 4, (height + 3) // 4
    count = blocks_x * blocks_y
    if len(data) < count * BLOCK_BYTES:
        raise ValueError(f'BC6H data holds {len(data) // BLOCK_BYTES} blocks, {count} needed')
    ends = np.zeros((count, 2, 2, 3), np.int64)       # [block, subset, endpoint, channel]
    subset = np.zeros((count, 16), np.int64)
    weight = np.zeros((count, 16), np.int64)
    unsupported = np.zeros(count, bool)
    cache = {}
    for n in range(count):
        raw = bytes(data[n * BLOCK_BYTES:(n + 1) * BLOCK_BYTES])
        hit = cache.get(raw)
        if hit is not None:
            ends[n], subset[n], weight[n], unsupported[n] = hit
            continue
        bits = int.from_bytes(raw, 'little')
        field = bits & 3 if bits & 3 < 2 else bits & 31
        header = _header(bits, field)
        if header is None:
            unsupported[n] = True
            cache[raw] = (ends[n], subset[n], weight[n], True)
            continue
        mode, pairs, partition = header
        for k, (e0, e1) in enumerate(pairs):
            ends[n, k, 0], ends[n, k, 1] = e0, e1
        if mode <= 10:
            layout = PARTITIONS[partition]
            stream = bits >> 82
            for pixel in range(16):
                anchor = layout[pixel] >= 128
                subset[n, pixel] = layout[pixel] & 1
                weight[n, pixel] = WEIGHTS_3[stream & (3 if anchor else 7)]
                stream >>= 2 if anchor else 3
        else:
            stream = bits >> 65           # one subset: 3 bits for pixel 0 (implicit top bit), 4 for the rest
            weight[n, 0] = WEIGHTS_4[stream & 7]
            stream >>= 3
            for pixel in range(1, 16):
                weight[n, pixel] = WEIGHTS_4[stream & 15]
                stream >>= 4
        cache[raw] = (ends[n].copy(), subset[n].copy(), weight[n].copy(), False)
    chosen = ends[np.arange(count)[:, None], subset]           # (blocks, 16, endpoint, channel)
    weight = weight[:, :, None]
    value = (chosen[:, :, 0] * (64 - weight) + chosen[:, :, 1] * weight + 32) >> 6
    half = ((value * 31) >> 6).astype(np.uint16)
    pixels = half.view(np.float16).astype(np.float32)          # (blocks, 16, 3)
    pixels[unsupported] = 0.0
    image = pixels.reshape(blocks_y, blocks_x, 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(blocks_y * 4, blocks_x * 4, 3)
    return image[:height, :width], unsupported.reshape(blocks_y, blocks_x)


__all__ = ['decode_bc6h']
