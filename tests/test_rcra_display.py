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
        # Values of the port, which matches the exe's CS_ToneMapCreateLookup (translated by nwo.shaders.sm5) to ~1e-5.
        for rgb, expected in (((0.18, 0.18, 0.18), (0.344652, 0.357206, 0.377115)),
                              ((1.0, 1.0, 1.0), (0.804388, 0.812828, 0.825721)),
                              ((8.0, 0.5, 0.1), (1.0, 0.812559, 0.540973)),
                              ((0.02, 0.05, 0.4), (0.0, 0.124004, 0.58779))):
            np.testing.assert_allclose(display.aces(np.array(rgb)), expected, atol=2e-6)

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

    def test_each_game_names_a_model_viewer_tonemap(self):
        for game in GAMES:
            self.assertIn(game.display_tonemap, ModelViewport.TONEMAPS)
        self.assertEqual(RCRA.display_tonemap, 'Game (Rift Apart)')
        self.assertEqual(RC2016.display_tonemap, 'Game (captured)')


if __name__ == '__main__':
    unittest.main()
