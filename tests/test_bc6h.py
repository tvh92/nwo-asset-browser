"""Unsigned BC6H decoding (nwo/formats/bc6h.py) on hand-packed blocks, and against the GPU on a Rift Apart probe."""
import os
import unittest

import numpy as np

from nwo.formats.bc6h import PARTITIONS, WEIGHTS_3, WEIGHTS_4, decode_bc6h


def pack(fields):
    """128-bit little-endian block from (start bit, width, value) triples."""
    bits = 0
    for start, width, value in fields:
        if value < 0 or value >= 1 << width:
            raise ValueError('field value does not fit')
        bits |= value << start
    return bits.to_bytes(16, 'little')


def indices_field(indices):
    """Index stream of a single-subset block starting at bit 65: 3 bits for pixel 0, 4 for the rest."""
    value, offset = indices[0], 3
    for pixel in indices[1:]:
        value |= pixel << offset
        offset += 4
    return 65, 63, value


def half(bits):
    return float(np.array([bits], np.uint16).view(np.float16)[0])


def finish(value16):
    return half((value16 * 31) >> 6)


def unquantize(value, bits):
    if value == 0:
        return 0
    if value == (1 << bits) - 1:
        return 0xFFFF
    return ((value << 15) + 0x4000) >> (bits - 1)


class Mode11Tests(unittest.TestCase):
    def block(self, e0, e1, indices):
        return pack([(0, 5, 0b00011), (5, 10, e0[0]), (15, 10, e0[1]), (25, 10, e0[2]),
                     (35, 10, e1[0]), (45, 10, e1[1]), (55, 10, e1[2]), indices_field(indices)])

    def test_endpoints_and_index_weights(self):
        e0, e1 = (0, 512, 1023), (1023, 512, 0)
        indices = list(range(8)) + list(range(8, 16))
        image, unsupported = decode_bc6h(self.block(e0, e1, indices), 4, 4)
        self.assertFalse(unsupported.any())
        self.assertEqual(image.shape, (4, 4, 3))
        for pixel, index in enumerate(indices):
            weight = WEIGHTS_4[index]
            for channel in range(3):
                a, b = unquantize(e0[channel], 10), unquantize(e1[channel], 10)
                expected = finish((a * (64 - weight) + b * weight + 32) >> 6)
                self.assertEqual(image[pixel // 4, pixel % 4, channel], np.float32(expected))

    def test_extremes_are_zero_and_half_float_maximum(self):
        image, _ = decode_bc6h(self.block((0, 0, 0), (1023, 1023, 1023), [0] * 15 + [15]), 4, 4)
        self.assertEqual(image[0, 0].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(image[3, 3].tolist(), [65504.0] * 3)

    def test_pixel_zero_index_has_three_bits(self):
        # With the implicit high bit, pixel 0 can reach index 7 but never 8 or more.
        image, _ = decode_bc6h(self.block((0, 0, 0), (1023, 1023, 1023), [7] + [0] * 15), 4, 4)
        weight = WEIGHTS_4[7]
        expected = finish((0xFFFF * weight + 32) >> 6)
        self.assertEqual(image[0, 0, 0], np.float32(expected))


class Mode12Tests(unittest.TestCase):
    def test_delta_endpoint_and_eleventh_bit(self):
        base = (1030, 1500, 700)             # 11-bit endpoint 0
        delta = (-5, 7, 0)                   # 9-bit signed deltas, stored two's complement
        stored = [d & 0x1FF for d in delta]
        block = pack([(0, 5, 0b00111),
                      (5, 10, base[0] & 0x3FF), (15, 10, base[1] & 0x3FF), (25, 10, base[2] & 0x3FF),
                      (35, 9, stored[0]), (44, 1, base[0] >> 10), (45, 9, stored[1]), (54, 1, base[1] >> 10),
                      (55, 9, stored[2]), (64, 1, base[2] >> 10), indices_field([0] * 15 + [15])])
        image, unsupported = decode_bc6h(block, 4, 4)
        self.assertFalse(unsupported.any())
        for channel in range(3):
            first = finish(unquantize(base[channel], 11))
            last = finish(unquantize((base[channel] + delta[channel]) & 0x7FF, 11))
            self.assertEqual(image[0, 0, channel], np.float32(first))
            self.assertEqual(image[3, 3, channel], np.float32(last))


class Mode13Tests(unittest.TestCase):
    def test_top_base_bits_are_stored_reversed(self):
        base = (0b1000_0000_0001, 0b0100_0000_0010, 0b1100_0000_0011)   # 12-bit endpoint 0
        delta = (3, -4, 0)
        stored = [d & 0xFF for d in delta]

        def top(value):                      # bits 11 and 10 of the base, bit 11 stored first
            return (value >> 11 & 1) | (value >> 10 & 1) << 1

        block = pack([(0, 5, 0b01011),
                      (5, 10, base[0] & 0x3FF), (15, 10, base[1] & 0x3FF), (25, 10, base[2] & 0x3FF),
                      (35, 8, stored[0]), (43, 2, top(base[0])), (45, 8, stored[1]), (53, 2, top(base[1])),
                      (55, 8, stored[2]), (63, 2, top(base[2])), indices_field([0] * 15 + [15])])
        image, _ = decode_bc6h(block, 4, 4)
        for channel in range(3):
            self.assertEqual(image[0, 0, channel], np.float32(finish(unquantize(base[channel], 12))))
            self.assertEqual(image[3, 3, channel],
                             np.float32(finish(unquantize((base[channel] + delta[channel]) & 0xFFF, 12))))


class Mode14Tests(unittest.TestCase):
    def test_sixteen_bit_base_with_reversed_top_bits(self):
        base = (0xA001, 0x4002, 0xC003)
        delta = (3, -4, 0)

        def top(value):                      # bits 15..10 of the base, bit 15 stored first
            return int(format(value >> 10, '06b')[::-1], 2)

        block = pack([(0, 5, 0b01111),
                      (5, 10, base[0] & 0x3FF), (15, 10, base[1] & 0x3FF), (25, 10, base[2] & 0x3FF),
                      (35, 4, delta[0] & 15), (39, 6, top(base[0])), (45, 4, delta[1] & 15), (49, 6, top(base[1])),
                      (55, 4, delta[2] & 15), (59, 6, top(base[2])), indices_field([0] * 15 + [15])])
        image, _ = decode_bc6h(block, 4, 4)
        for channel in range(3):
            self.assertEqual(image[0, 0, channel], np.float32(finish(base[channel])))
            self.assertEqual(image[3, 3, channel], np.float32(finish((base[channel] + delta[channel]) & 0xFFFF)))


class TwoSubsetTests(unittest.TestCase):
    def block(self, partition, base, deltas, indices):
        """Field ``01`` block. ``deltas`` = (x, y, z), each an (r, g, b) triple of signed 6-bit values;
        ``indices`` holds one value per pixel (anchors only 2 bits)."""
        def s6(v):
            return v & 0x3F
        (rx, gx, bx), (ry, gy, by), (rz, gz, bz) = deltas
        fields = [(0, 2, 0b01), (2, 1, gy >> 5 & 1), (3, 1, gz >> 4 & 1), (4, 1, gz >> 5 & 1),
                  (5, 7, base[0]), (12, 1, bz & 1), (13, 1, bz >> 1 & 1), (14, 1, by >> 4 & 1),
                  (15, 7, base[1]), (22, 1, by >> 5 & 1), (23, 1, bz >> 2 & 1), (24, 1, gy >> 4 & 1),
                  (25, 7, base[2]), (32, 1, bz >> 3 & 1), (33, 1, bz >> 5 & 1), (34, 1, bz >> 4 & 1),
                  (35, 6, s6(rx)), (41, 4, gy & 15), (45, 6, s6(gx)), (51, 4, gz & 15), (55, 6, s6(bx)),
                  (61, 4, by & 15), (65, 6, s6(ry)), (71, 6, s6(rz)), (77, 5, partition)]
        offset, stream = 82, 0
        for pixel, index in enumerate(indices):
            width = 2 if PARTITIONS[partition][pixel] >= 128 else 3
            stream |= index << (offset - 82)
            offset += width
        fields.append((82, offset - 82, stream))
        return pack(fields)

    def test_partition_table_shape(self):
        self.assertEqual(len(PARTITIONS), 32)
        for layout in PARTITIONS:
            self.assertEqual(len(layout), 16)
            self.assertEqual(layout[0], 128)
            self.assertEqual(sum(1 for v in layout if v >= 128), 2)

    def test_each_subset_uses_its_own_endpoints_and_weights(self):
        partition = 0
        base = (10, 60, 120)
        deltas = ((20, -10, 31), (-30, 5, 0), (7, -32, 12))      # x, y, z of (r, g, b)
        indices = [0] * 16
        anchor = [i for i, v in enumerate(PARTITIONS[partition]) if v == 129][0]
        indices[1] = 7                      # a subset 0 pixel at the top index
        indices[2] = 5                      # a subset 1 pixel
        indices[anchor] = 3
        image, unsupported = decode_bc6h(self.block(partition, base, deltas, indices), 4, 4)
        self.assertFalse(unsupported.any())

        def sample(pixel, subset):
            points = [[unquantize(b, 7) for b in base]]
            for k in range(3):
                points.append([unquantize((b + deltas[k][c]) & 0x7F, 7) for c, b in enumerate(base)])
            e0, e1 = (points[0], points[1]) if subset == 0 else (points[2], points[3])
            weight = WEIGHTS_3[indices[pixel]]
            return [finish((e0[c] * (64 - weight) + e1[c] * weight + 32) >> 6) for c in range(3)]

        for pixel in range(16):
            subset = PARTITIONS[partition][pixel] & 1
            self.assertEqual(image[pixel // 4, pixel % 4].tolist(), [float(np.float32(x)) for x in sample(pixel, subset)],
                             f'pixel {pixel}')


class LayoutTests(unittest.TestCase):
    def test_reserved_modes_are_zero_and_reported(self):
        blocks = b''.join(pack([(0, 5, field)]) for field in (0b10011, 0b10111, 0b11011, 0b11111))
        image, unsupported = decode_bc6h(blocks, 16, 4)
        self.assertEqual(unsupported.tolist(), [[True, True, True, True]])
        self.assertFalse(image.any())

    def test_every_mode_layout_fills_the_header(self):
        # Two-subset modes hold 82 header bits (mode field, four endpoints, partition), one-subset modes 65.
        from nwo.formats.bc6h import _PARSED
        self.assertEqual(sorted(mode for mode, *_ in _PARSED.values()), list(range(1, 15)))
        for field, (mode, _width, _deltas, layout) in _PARSED.items():
            self.assertEqual(len(layout) + (2 if mode <= 2 else 5), 82 if mode <= 10 else 65, f'mode {mode}')
            self.assertEqual(len(set(layout)), len(layout), f'mode {mode} repeats a bit')


    def test_blocks_fill_the_image_row_by_row(self):
        def constant(value):
            return pack([(0, 5, 0b00011), (5, 10, value), (15, 10, value), (25, 10, value),
                         (35, 10, value), (45, 10, value), (55, 10, value)])
        data = constant(1023) + constant(0) + constant(0) + constant(1023)
        image, unsupported = decode_bc6h(data, 8, 8)
        self.assertFalse(unsupported.any())
        self.assertEqual(image[0, 0, 0], 65504.0)
        self.assertEqual(image[0, 4, 0], 0.0)
        self.assertEqual(image[4, 0, 0], 0.0)
        self.assertEqual(image[7, 7, 0], 65504.0)

    def test_crops_to_the_requested_size_and_rejects_short_data(self):
        block = pack([(0, 5, 0b00011)])
        image, _ = decode_bc6h(block, 3, 2)
        self.assertEqual(image.shape, (2, 3, 3))
        with self.assertRaises(ValueError):
            decode_bc6h(block, 8, 4)


@unittest.skipUnless(os.environ.get('RCRA_GAME'), 'RCRA_GAME is not set')
class GpuMatchTests(unittest.TestCase):
    """Every block of a Rift Apart probe decodes as the GPU's own BC6H decoder does (read back as float)."""

    def test_probe_matches_the_gpu(self):
        from nwo.games.rcra import atmosphere
        from nwo.viewer.loader import SourcePreviewLoader
        from nwo.viewer.viewport import ModelViewport
        import ctypes
        import tkinter as tk
        source = SourcePreviewLoader()._ensure_archive(os.environ['RCRA_GAME'])
        faces = atmosphere.probe_blocks(source, atmosphere.read(source, atmosphere.DEFAULT)['probe'], 2)
        root = tk.Tk()
        try:
            view = ModelViewport(root)
            view.pack()
            root.update()
            if view._context_error or not view._make_current():
                self.skipTest(view._context_error or 'no OpenGL context')
            gl, upload = view._gl, view._compressed_uploader()
            gl.glGetTexImage.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
            name = ctypes.c_uint(0)
            gl.glGenTextures(1, ctypes.byref(name))
            gl.glBindTexture(0x0DE1, name.value)
            gl.glPixelStorei(0x0D05, 1)
            for blocks, size in faces:
                buffer = ctypes.create_string_buffer(bytes(blocks), len(blocks))
                upload(0x0DE1, 0, 0x8E8F, size, size, 0, len(blocks), buffer)   # GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT
                gpu = np.zeros((size, size, 3), np.float32)
                gl.glGetTexImage(0x0DE1, 0, 0x1907, 0x1406, gpu.ctypes.data)
                cpu, reserved = decode_bc6h(blocks, size, size)
                self.assertFalse(reserved.any())
                np.testing.assert_array_equal(cpu, gpu)
        finally:
            root.destroy()


if __name__ == '__main__':
    unittest.main()
