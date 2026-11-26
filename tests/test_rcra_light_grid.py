"""Rift Apart light grids (nwo/games/rcra/light_grid.py): the brick unpack, the apply programs' decode, the grids of a
probe's level, the world fit, and the real install's Megalopolis day grid (RCRA_GAME)."""
import math
import os
import struct
import unittest

import numpy as np

from nwo.games.rcra import light_grid as G

GAME = os.environ.get('RCRA_GAME')


def pack_brick(codes, planes, scale_bytes, bleed):
    """A brick blob as the game stores it: codes (4096,) 0..3, planes (6, N) u16 of the code-2/3 cells, N bytes, and
    three bytes per code-3 cell."""
    words = []
    for q in range(128):
        word = 0
        for k in range(32):
            word |= int(codes[q * 32 + k]) << (62 - 2 * k)
        words.append(word)
    out = struct.pack('<128Q', *words)
    out += np.asarray(planes, '<u2').tobytes() + bytes(scale_bytes) + bytes(np.asarray(bleed, np.uint8).ravel())
    return out


def field(y, b, c):
    return y | b << 6 | c << 11


class UnpackTests(unittest.TestCase):
    def test_codes_planes_scale_and_bleed_make_the_programs_records(self):
        codes = np.zeros(4096, np.uint8)
        codes[[5, 6, 7, 4095]] = [1, 2, 3, 2]
        planes = np.array([[11, 21, 31], [12, 22, 32], [13, 23, 33], [14, 24, 34], [15, 25, 35], [16, 26, 36]])
        blob = pack_brick(codes, planes, [0xA0, 0xB4, 0xAE], [[0x4D, 0xC0, 0xEC]])
        cells, code = G.unpack_brick(blob)
        np.testing.assert_array_equal(code, codes)
        self.assertEqual(cells[0, 3], G.BLOCKED)
        self.assertEqual(cells[5, 3], G.FILL)
        self.assertEqual(tuple(cells[6, :3]), (11 | 12 << 16, 13 | 14 << 16, 15 | 16 << 16))
        self.assertEqual(cells[6, 3], G.STORED | 0xA0 << 6)
        self.assertEqual(cells[7, 3], (0x4DC0ECEC & G.BLEED_MASK) | 0xB4 << 6)
        self.assertEqual(cells[4095, 3], G.STORED | 0xAE << 6)
        self.assertEqual(G.scale(cells[[6, 7]]).tolist(), [1.0, 2.0])          # exponent nibble 10 -> 2^0, 11 -> 2^1
        np.testing.assert_allclose(G.probe_weight(cells[[6, 7, 4095]]), [0.0, 4 / 15, 14 / 15])

    def test_a_wrong_size_is_refused(self):
        codes = np.zeros(4096, np.uint8)
        codes[0] = 2
        blob = pack_brick(codes, np.zeros((6, 1)), [0xA0], np.zeros((0, 3)))
        with self.assertRaises(ValueError):
            G.unpack_brick(blob + b'\0')


class DecodeTests(unittest.TestCase):
    def setUp(self):
        values = [field(40, 16, 16), field(10, 16, 16), field(63, 10, 22), field(20, 20, 12), field(30, 16, 16),
                  field(30, 16, 16)]
        cells = np.zeros((1, 4), np.uint32)
        cells[0, 0] = values[0] | values[1] << 16
        cells[0, 1] = values[2] | values[3] << 16
        cells[0, 2] = values[4] | values[5] << 16
        cells[0, 3] = G.STORED | 0xB0 << 6                    # exponent 11: scale 2
        self.cell = dict(fields=G.fields(cells)[0].tolist(), scale=float(G.scale(cells)[0]))
        self.values = values

    def test_an_axis_normal_reads_its_own_direction(self):
        y, b, c = 63, 10 << 6, 22 << 11
        u, v = b / 992 - 1, c / 31744 - 1
        u, v = abs(u) * u, abs(v) * v
        expect = np.array([y * (1 - u + v), y * (1 + u), y * (1 - u - v)]) * 2 / 63
        np.testing.assert_allclose(G.irradiance(self.cell, np.array([0.0, 1.0, 0.0])), expect, rtol=1e-12)
        np.testing.assert_allclose(G.irradiance(self.cell, np.array([-1.0, 0.0, 0.0])),
                                   G.decode(10, 16 << 6, 16 << 11) * 2 / 63, rtol=1e-12)

    def test_the_fields_are_summed_before_the_colour_decode(self):
        n = np.array([1.0, -1.0, 0.0]) / math.sqrt(2)
        summed = 0.5 * np.array([40, 16 << 6, 16 << 11]) + 0.5 * np.array([20, 20 << 6, 12 << 11])
        np.testing.assert_allclose(G.irradiance(self.cell, n), G.decode(*summed) * 2 / 63, rtol=1e-12)


class FakeSource:
    def __init__(self, paths):
        self.entries = {str(i): dict(path=p, type='zonelightbin') for i, p in enumerate(paths)}

    def ids_of_type(self, kind):
        return [k for k, e in self.entries.items() if e['type'] == kind]


class LevelTests(unittest.TestCase):
    def test_grids_of_the_probes_level_and_time_of_day(self):
        level = 'levels/i29/instance/a/a/'
        source = FakeSource([level + 'a_light_grid_day.zonelightbin', level + 'b_light_grid_day.zonelightbin',
                             level + 'a_light_grid_night.zonelightbin', level + 'sub/c_light_grid_day.zonelightbin',
                             'levels/other/x_light_grid_day.zonelightbin'])
        self.assertEqual(G.grids_for_probe(source, level + 'envprobes_day/0x1.texture'),
                         [level + 'a_light_grid_day.zonelightbin', level + 'b_light_grid_day.zonelightbin'])
        self.assertEqual(G.grids_for_probe(source, level + 'probes/0x1.texture'), [])


class FitTests(unittest.TestCase):
    def test_a_constant_target_under_a_brighter_constant_probe_keeps_its_share(self):
        from nwo.viewer.environment import equirect_directions
        w, h = 32, 16
        d = equirect_directions(w, h).reshape(-1, 3).astype(np.float64)
        lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
        omega = np.repeat(np.cos(lat) * (np.pi / h) * (2 * np.pi / w), w)
        probe = np.full((w * h, 3), 2.0)
        diffuse = (np.maximum(d @ d.T, 0) * omega[None, :] / np.pi) @ probe   # what the probe alone gives
        target = 0.25 * diffuse
        fit = G.fit_world(target, probe, d, omega)
        self.assertAlmostEqual(fit['k'], 0.25, delta=0.01)
        self.assertLess(fit['rms'], 0.01)


@unittest.skipUnless(GAME, 'RCRA_GAME is not set')
class InstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nwo.viewer.loader import SourcePreviewLoader
        cls.source = SourcePreviewLoader()._ensure_archive(GAME)

    def test_megalopolis_street_cell(self):
        probe = 'levels/i29/instance/malinon_intro/malinon_intro/envprobes_day/0xe54a425db9195aba.texture'
        grids = G.grids_for_probe(self.source, probe)
        self.assertEqual(grids, ['levels/i29/instance/malinon_intro/malinon_intro/malinon_intro_light_grid_day.zonelightbin',
                                 'levels/i29/instance/malinon_intro/malinon_intro/malinon_intro_rivet_light_grid_day.zonelightbin'])
        centres, blobs = G.bricks(self.source, grids[0])
        self.assertEqual(len(centres), 1796)
        np.testing.assert_allclose(centres[0], [744, -136, 1848])
        for blob in blobs:   # every brick's size is the unpack's
            G.unpack_brick(blob)
        cell = G.street_cell(self.source, probe)
        self.assertEqual(cell['cells'], 65717)
        self.assertEqual(cell['position'], [861.5, -117.5, 1891.5])
        self.assertEqual(cell['weight'], 0.0)
        lum = np.asarray(cell['axis_rgb']) @ G.LUMINANCE
        self.assertGreater(lum[2], 2 * lum[3])   # more light from above than from below at street level


if __name__ == '__main__':
    unittest.main()
