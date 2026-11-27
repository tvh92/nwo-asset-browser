"""The portable .blend's clip layout (nwo.export.blend_animation): the parts that run without Blender."""
import unittest

import numpy as np

from nwo.export import blend_animation as B


class BlendAnimationTests(unittest.TestCase):
    def test_retime_puts_sample_k_on_frame_k_plus_one(self):
        # The glTF importer keys sample k of a 30 fps clip at (k / 30) * 24 in a 24 fps scene, from float32 times.
        times = np.arange(100, dtype=np.float32) / np.float32(30)
        imported = times.astype(np.float64) * 24
        np.testing.assert_array_equal(B.retime(imported, 30.0, 24.0), np.arange(1, 101))

    def test_retime_keeps_off_sample_handles(self):
        out = B.retime([0.4 * 24 / 30], 30.0, 24.0)
        self.assertAlmostEqual(float(out[0]), 1.4)

    def test_track_clips_maps_animation_order(self):
        names = ['a: idle', 'a: idle_roar', 'vertex programs']
        self.assertEqual(B.track_clips(names, 2), {'a: idle': 0, 'a: idle_roar': 1})

    def test_track_clips_leaves_out_names_that_collide_when_cut(self):
        long = 'x' * 70
        self.assertEqual(B.track_clips([long + '1', long + '2', 'b'], 3), {'b': 2})

    def test_blender_name_cuts_on_a_character(self):
        name = 'é' * 40   # two bytes each
        self.assertEqual(len(B.blender_name(name).encode('utf-8')), 62)

    def test_strip_labels_fit_blender(self):
        self.assertLessEqual(len(B.strip_label(1234, 'clip ' * 30).encode('utf-8')), 63)


if __name__ == '__main__':
    unittest.main()
