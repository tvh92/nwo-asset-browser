"""Rift Apart's decoded display transform (nwo.games.rcra.display) and the viewer's "Game (Rift Apart)" tonemap."""
import unittest

import numpy as np

from nwo.games import GAMES, RC2016, RCRA
from nwo.games.rcra import display
from nwo.levels.viewport import LevelViewport
from nwo.viewer import shading
from nwo.viewer.viewport import ModelViewport


class AcesTests(unittest.TestCase):
    def test_entries(self):
        # Values of the port, which matches the exe's CS_ToneMapCreateLookup (translated by nwo.shaders.sm5) to ~1e-5
        # with the game's SDR constants (m_A = 0: no D60 -> D65 adaptation).
        for rgb, expected in (((0.18, 0.18, 0.18), (0.352849, 0.356313, 0.361674)),
                              ((1.0, 1.0, 1.0), (0.821328, 0.810882, 0.793976)),
                              ((8.0, 0.5, 0.1), (1.0, 0.809933, 0.515107)),
                              ((0.02, 0.05, 0.4), (0.0, 0.125987, 0.566904))):
            np.testing.assert_allclose(display.aces(np.array(rgb)), expected, atol=2e-6)

    def test_the_game_display_skips_the_d60_to_d65_step(self):
        grey = np.array([0.18, 0.18, 0.18])
        self.assertLess(np.ptp(display.aces(grey)), np.ptp(display.aces(grey, cat=True)))   # neutral, not cool
        np.testing.assert_allclose(display.build_lookup()[5, 9, 2], display.aces(display.lookup_grid()[[5, 9, 2]]), atol=1e-12)

    def test_grey_is_monotonic_and_bounded(self):
        grey = display.aces(np.repeat(display.lookup_grid()[:, None], 3, 1))
        self.assertTrue(np.all(np.diff(grey, axis=0) >= -1e-9))   # flat at black and white, never falling
        self.assertGreaterEqual(grey.min(), 0.0)
        self.assertLessEqual(grey.max(), 1.0)

    def test_lookup_axes(self):
        table = display.build_lookup()
        self.assertEqual(table.shape, (32, 32, 32, 3))
        g = display.lookup_grid()
        np.testing.assert_allclose(table[31, 4, 9], display.aces(np.array([g[31], g[4], g[9]])))
        # The display coordinate puts the lookup's entries at the centres: c = 0 is entry 0, c = 1 is entry 31.
        for i in (0, 13, 31):
            c = np.log2(g[i]) * display.LOG2_SCALE + display.LOG2_OFFSET
            self.assertAlmostEqual(c * 31, i, places=9)

    def test_texture_layout_is_what_game_grade_samples(self):
        """Emulate shading's gameGrade (bilinear in red and green inside a blue slab, linear across slabs) on the texture
        and compare with the trilinear reference."""
        texture = display.lookup_texture().astype(np.float64)
        self.assertEqual(texture.shape, (33, 1024, 3))

        def bilinear(x, y):   # texel coordinates, clamped to the edge, as GL_LINEAR with CLAMP_TO_EDGE
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            fx, fy = x - x0, y - y0
            px = lambda i: min(max(i, 0), texture.shape[1] - 1)  # noqa: E731
            py = lambda i: min(max(i, 0), texture.shape[0] - 1)  # noqa: E731
            top = texture[py(y0), px(x0)] * (1 - fx) + texture[py(y0), px(x0 + 1)] * fx
            bottom = texture[py(y0 + 1), px(x0)] * (1 - fx) + texture[py(y0 + 1), px(x0 + 1)] * fx
            return top * (1 - fy) + bottom * fy

        def game_grade(c):
            p = np.clip(c, 0, 1) * 31
            slab = min(np.floor(p[2]), 30)
            f = p[2] - slab
            a = bilinear(slab * 32 + p[0], p[1])
            b = bilinear(slab * 32 + 32 + p[0], p[1])
            return a * (1 - f) + b * f

        rng = np.random.default_rng(3)
        for c in rng.uniform(0, 1, (40, 3)):
            np.testing.assert_allclose(game_grade(c), display.sample(display.build_lookup(), c), atol=1e-6)

    def test_colour_correction_mixes_luts_by_weight(self):
        n = 32
        axis = (np.arange(n) + 0.0) / (n - 1)
        identity = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1)
        # An identity LUT sampled at enc * 31/32 + 1/64 moves values slightly towards the middle, as the game's does.
        table = display.build_lookup()
        np.testing.assert_allclose(display.colour_corrected([(identity, 1.0)]), display.sample_lut(identity, table))
        inverted = 1.0 - identity
        mixed = display.colour_corrected([(inverted, 0.25), (identity, 0.25)])
        np.testing.assert_allclose(mixed, 0.5 * table + 0.25 * display.sample_lut(inverted, table)
                                   + 0.25 * display.sample_lut(identity, table))
        self.assertIs(display.colour_corrected([]), table)
        with self.assertRaises(ValueError):
            display.colour_corrected([(identity, 0.1)] * 5)
        np.testing.assert_allclose(display.sample_lut(identity, np.array([0.0, 0.5, 1.0])),
                                   [1 / 64 * 32 / 31 - 0.5 / 31, 0.5, 1 - (1 / 64 * 32 / 31 - 0.5 / 31)], atol=1e-12)

    def test_display_exposure(self):
        np.testing.assert_allclose(display.display(np.full(3, 0.09), 2.0), display.display(np.full(3, 0.18)))


class ViewerTests(unittest.TestCase):
    def test_tonemap_codes(self):
        self.assertEqual(ModelViewport.TONEMAPS[ModelViewport.RCRA_TONEMAP], 'Game (Rift Apart)')
        self.assertEqual(ModelViewport.LINEAR_TONEMAP, len(ModelViewport.TONEMAPS))
        self.assertIn('if (uTonemap == 3) return riftApartDisplay(color);', shading.LIGHTING)
        self.assertIn('if (uTonemap == 4) return color;', shading.LIGHTING)
        self.assertNotIn('Game (Rift Apart)', LevelViewport.TONEMAPS)

    def test_display_table_follows_the_tonemap(self):
        view = ModelViewport.__new__(ModelViewport)
        view.tonemap = 'Game (Rift Apart)'
        np.testing.assert_array_equal(view._display_table_build(), display.lookup_texture())
        key = view._display_table_key()
        view.tonemap = 'Game (captured)'
        self.assertNotEqual(view._display_table_key(), key)

    def test_game_atmosphere_adds_its_colour_correction(self):
        view = ModelViewport.__new__(ModelViewport)
        view.tonemap, view.lighting = 'Game (Rift Apart)', 'Game atmosphere'
        lut = np.full((32, 32, 32, 3), 0.5)
        view._environment_lighting = 'Game atmosphere'
        view._environment = dict(colour_correction=lut, atmosphere=dict(colour_correction='textures/lut/x.texture'))
        self.assertEqual(view._display_table_key(), ('rcra aces', 'textures/lut/x.texture'))
        np.testing.assert_allclose(view._display_table_build()[:32], 0.5, atol=1e-6)   # weight 1: the LUT alone
        view.lighting = 'Shared HDRI'
        self.assertEqual(view._display_table_key(), 'rcra aces')

    def test_each_game_names_a_model_viewer_tonemap(self):
        for game in GAMES:
            self.assertIn(game.display_tonemap, ModelViewport.TONEMAPS)
        self.assertEqual(RCRA.display_tonemap, 'Game (Rift Apart)')
        self.assertEqual(RC2016.display_tonemap, 'Game (captured)')


if __name__ == '__main__':
    unittest.main()
