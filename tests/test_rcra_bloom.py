"""Rift Apart's decoded bloom chain (nwo.games.rcra.bloom) and its use in the viewer."""
import os
import unittest

import numpy as np

from nwo.games.rcra import bloom
from nwo.viewer import shading

SETTINGS = dict(intensity=0.2, threshold=3.0, strengths=(0.4, 0.5, 0.6, 0.8, 1.0))


class ChainTests(unittest.TestCase):
    def test_up_constants_follow_the_pass_function(self):
        g = SETTINGS['strengths']
        self.assertEqual(bloom.up_constants(6, SETTINGS), (g[4], g[3], 'one_src_alpha'))
        self.assertEqual(bloom.up_constants(5, SETTINGS), (1.0, g[2], 'one_src_alpha'))
        self.assertEqual(bloom.up_constants(4, SETTINGS), (1.0, g[1], 'one_src_alpha'))
        self.assertEqual(bloom.up_constants(3, SETTINGS), (1.0, g[0], 'one_src_alpha'))
        self.assertEqual(bloom.up_constants(2, SETTINGS), (1.0, 0.0, 'one_src_alpha'))
        self.assertEqual(bloom.up_constants(1, SETTINGS), (0.2, 0.0, None))

    def test_pass_list(self):
        p = bloom.passes(1920, 1080, 1.0, SETTINGS)
        self.assertEqual([x['kind'] for x in p], ['rcra_half'] + ['rcra_down'] * 6 + ['rcra_up'] * 6)
        self.assertEqual(bloom.level_sizes(1920, 1080)[0], (960, 540))
        self.assertEqual(bloom.level_sizes(1920, 1080)[6], (15, 8))
        self.assertEqual(p[-1]['dst'], 'B0')
        gate = p[1]['gate']
        self.assertAlmostEqual(gate['threshold_scale'], 1 / 3.0)
        self.assertAlmostEqual(gate['normalise'], 1 / 3.3)
        self.assertTrue(all(x['gate'] is None for x in p[2:7]))
        self.assertEqual(bloom.passes(64, 64, 1.0, dict(SETTINGS, threshold=0.01))[1]['gate']['threshold_scale'], 16.0)

    def test_filters_preserve_a_constant(self):
        self.assertAlmostEqual(sum(w for _o, w in bloom.DOWN_TAPS), 1.0)
        self.assertAlmostEqual(sum(w for _o, w in bloom.UP_TAPS), 1.0)

    def test_uniform_scene(self):
        """On a uniform scene every level is uniform: the strengths sum back to the normalisation, so the bloom is
        intensity x gate x colour."""
        colour = np.array([2.0, 1.5, 1.0])
        scene = np.broadcast_to(colour, (256, 256, 3))
        inverse_luminance = 0.5 / bloom.MIDDLE_GRAY
        out = bloom.reference_chain(scene, inverse_luminance, SETTINGS)
        luma = colour @ np.array(bloom.LUMA)
        gate = np.clip(luma * inverse_luminance / 3.0 - 1.0, 0, 1)
        self.assertGreater(gate, 0.0)
        np.testing.assert_allclose(out, np.broadcast_to(0.2 * gate * colour, out.shape), rtol=1e-9)
        self.assertEqual(out.shape, (128, 128, 3))

    def test_below_threshold_gives_nothing(self):
        scene = np.full((128, 128, 3), 0.5)
        self.assertEqual(float(np.abs(bloom.reference_chain(scene, 1.0, SETTINGS)).max()), 0.0)

    def test_a_bright_spot_spreads_and_peaks_at_it(self):
        scene = np.zeros((256, 256, 3))
        scene[120:136, 120:136] = 50.0
        out = bloom.reference_chain(scene, 1.0 / bloom.MIDDLE_GRAY, SETTINGS)[..., 0]
        peak = np.unravel_index(np.argmax(out), out.shape)
        self.assertLessEqual(max(abs(peak[0] - 63.5), abs(peak[1] - 63.5)), 1.0)
        self.assertGreater(out[64, 10], 0.0)   # the widest level reaches far
        self.assertGreater(out[64, 64], out[64, 40])

    def test_viewer_composite(self):
        self.assertIn('if (uTonemap == 3) return riftApartDisplay((color + bloom) * uExposure);', shading.HDR_RESOLVE_FRAGMENT)
        self.assertEqual(set(bloom.PROGRAMS), {'rcra_half', 'rcra_down', 'rcra_up'})


@unittest.skipUnless(os.environ.get('RCRA_GAME'), 'RCRA_GAME not set')
class GameFileTests(unittest.TestCase):
    def test_default_atmosphere_settings(self):
        import tempfile
        from nwo.formats.container import read_container
        from nwo.games.rcra.source import RcraSource
        with tempfile.TemporaryDirectory() as out:
            s = RcraSource(os.environ['RCRA_GAME'], out, cache_dir='.cache')
            data = s.asset_data(s.by_path['required/atmospheres/default_atmosphere.atmosphere'][0])[36:]
            root = read_container(data, 0, {})
            sec = next(x for x in root['sections'] if int(x['label_hash'], 16) == 0x02F06D4E)
            got = bloom.settings_from_block(data[sec['offset']:sec['offset'] + sec['size']])
        self.assertAlmostEqual(got['intensity'], bloom.DEFAULT_SETTINGS['intensity'], places=6)
        self.assertAlmostEqual(got['threshold'], bloom.DEFAULT_SETTINGS['threshold'], places=6)
        np.testing.assert_allclose(got['strengths'], bloom.DEFAULT_SETTINGS['strengths'], rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
