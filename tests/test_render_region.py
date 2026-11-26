"""Render Region padding math (nwo/pipeline/scene_defaults.padded_region); pure, no Blender."""
import unittest

from nwo.pipeline.scene_defaults import REGION_PADDING_LONG, REGION_PADDING_SHORT, REGION_PADDING_SIDE, padded_region


class PaddedRegionTests(unittest.TestCase):
    def assert_region(self, actual, expected):
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got, want, places=6)

    def test_taller_model_gets_long_padding_vertically_and_short_horizontally(self):
        # Box 0.2 wide x 0.6 high in frame fractions, square frame: taller on screen.
        region = padded_region((0.4, 0.2, 0.6, 0.8), (1000, 1000), side_padding=0.0)
        self.assert_region(region, (0.5 - 0.2 * 1.07 / 2, 0.5 - 0.6 * 1.10 / 2,
                                    0.5 + 0.2 * 1.07 / 2, 0.5 + 0.6 * 1.10 / 2))

    def test_wider_model_gets_long_padding_horizontally_and_short_vertically(self):
        region = padded_region((0.1, 0.4, 0.9, 0.5), (1000, 1000), side_padding=0.0)
        self.assert_region(region, (0.5 - 0.8 * 1.10 / 2, 0.45 - 0.1 * 1.07 / 2,
                                    0.5 + 0.8 * 1.10 / 2, 0.45 + 0.1 * 1.07 / 2))

    def test_aspect_is_judged_in_pixels_not_frame_fractions(self):
        # 0.2 x 0.3 of a 2000 x 1000 frame is 400 x 300 px: wider on screen, so horizontal gets 10%.
        region = padded_region((0.4, 0.2, 0.6, 0.5), (2000, 1000), side_padding=0.0)
        self.assert_region(region, (0.5 - 0.2 * 1.10 / 2, 0.35 - 0.3 * 1.07 / 2,
                                    0.5 + 0.2 * 1.10 / 2, 0.35 + 0.3 * 1.07 / 2))

    def test_horizontal_padding_is_at_least_the_side_padding(self):
        # Default: a tall model gets 25% beside it (12.5% each side) instead of the short-axis 7%.
        region = padded_region((0.4, 0.2, 0.6, 0.8), (1000, 1000))
        self.assert_region(region, (0.5 - 0.2 * 1.25 / 2, 0.5 - 0.6 * 1.10 / 2,
                                    0.5 + 0.2 * 1.25 / 2, 0.5 + 0.6 * 1.10 / 2))
        # A wider padding than the side minimum is kept.
        region = padded_region((0.3, 0.4, 0.7, 0.5), (1000, 1000), long_padding=0.4)
        self.assert_region(region, (0.5 - 0.4 * 1.40 / 2, 0.45 - 0.1 * 1.07 / 2,
                                    0.5 + 0.4 * 1.40 / 2, 0.45 + 0.1 * 1.07 / 2))

    def test_region_is_clamped_to_the_frame(self):
        self.assert_region(padded_region((0.0, 0.0, 1.0, 1.0), (100, 100)), (0.0, 0.0, 1.0, 1.0))
        region = padded_region((0.9, 0.9, 1.0, 1.0), (100, 100))
        self.assertLessEqual(region[2], 1.0)
        self.assertLessEqual(region[3], 1.0)

    def test_camera_distance_makes_points_span_the_requested_fraction(self):
        import numpy as np
        from nwo.pipeline.scene_defaults import camera_distance_for_fill

        # Camera looks down -Z (identity rotation); a 2 x 2 square at the origin, focal length 1.
        # At distance d the square's half-size 1 maps to +-1/d about the frame centre, so it spans 1 / d
        # of the normalised 0..1 frame: fill 0.5 -> d = 2, fill 0.25 -> d = 4.
        projection = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, -1, 0]], dtype=float)
        points = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]], dtype=float)
        distance = camera_distance_for_fill(points, (0, 0, 0), np.eye(3), projection, 0, 0.5)
        self.assertAlmostEqual(distance, 2.0, places=6)
        vertical = camera_distance_for_fill(points, (0, 0, 0), np.eye(3), projection, 1, 0.25)
        self.assertAlmostEqual(vertical, 4.0, places=6)

    def test_hull_keeps_only_outline_corners(self):
        from nwo.pipeline.scene_defaults import hull_2d

        square = [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.5, 0], [0.2, 0.7]]
        self.assertEqual(sorted(hull_2d(square).tolist()), [0, 1, 2, 3])
        self.assertEqual(sorted(hull_2d([[0, 0], [1, 1], [2, 2], [3, 3], [1.5, 1.5]]).tolist()), [0, 3])

    def test_framing_points_give_the_same_fit_as_every_point(self):
        import math
        import numpy as np
        from nwo.pipeline.scene_defaults import camera_distance_for_fill, framing_points

        rng = np.random.default_rng(7)
        points = np.r_[rng.normal(size=(20000, 3)) * [0.3, 0.9, 0.25],
                       rng.uniform(-1, 1, (10000, 3)) * [0.5, 0.2, 0.4] + [0, 1, 0]]
        a, b = math.radians(70), math.radians(-35)
        rotation = (np.array([[math.cos(b), -math.sin(b), 0], [math.sin(b), math.cos(b), 0], [0, 0, 1]])
                    @ np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]))
        # Perspective with a lens shift, as calc_matrix_camera gives for a shifted camera.
        projection = np.array([[2.7, 0, 0.05, 0], [0, 4.05, -0.02, 0], [0, 0, -1.0002, -0.02], [0, 0, -1, 0]])
        subset = framing_points(points, rotation)
        self.assertLess(len(subset), 200)
        centre = (points.min(axis=0) + points.max(axis=0)) / 2
        for axis, fill in ((0, 0.8), (1, 0.9)):
            for shift in ((0, 0, 0), (0.1, -0.2, 0.05)):
                at = centre + np.array(shift)
                self.assertEqual(camera_distance_for_fill(subset, at, rotation, projection, axis, fill),
                                 camera_distance_for_fill(points, at, rotation, projection, axis, fill))

    def test_meshless_diagnostic_rig_keeps_camera_and_resets_crop_to_full_frame(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from nwo.pipeline.scene_defaults import fit_render_region

        camera = SimpleNamespace(type="CAMERA", location=(1, 2, 3),
                                 matrix_world=SimpleNamespace(inverted=lambda: np.eye(4)),
                                 calc_matrix_camera=lambda *args, **kwargs: np.eye(4))
        render = SimpleNamespace(resolution_x=3840, resolution_y=2560,
                                 use_border=False, use_crop_to_border=False,
                                 border_min_x=0.2, border_min_y=0.3,
                                 border_max_x=0.8, border_max_y=0.9)
        scene = SimpleNamespace(camera=camera, render=render,
                                objects=[SimpleNamespace(type="ARMATURE", hide_render=False)])
        bpy = SimpleNamespace(context=SimpleNamespace(
            view_layer=SimpleNamespace(update=lambda: None), evaluated_depsgraph_get=lambda: None))
        with patch.dict("sys.modules", {"bpy": bpy}):
            receipt = fit_render_region(scene)
        self.assertEqual(camera.location, (1, 2, 3))
        self.assertEqual(receipt["status"], "full_frame_no_visible_mesh")
        self.assertEqual(receipt["region"], [0.0, 0.0, 1.0, 1.0])
        self.assertTrue(render.use_border)
        self.assertTrue(render.use_crop_to_border)
        self.assertEqual((render.border_min_x, render.border_min_y,
                          render.border_max_x, render.border_max_y), (0.0, 0.0, 1.0, 1.0))

    def test_default_padding_is_ten_long_seven_short_and_twenty_five_side_percent(self):
        self.assertEqual((REGION_PADDING_LONG, REGION_PADDING_SHORT, REGION_PADDING_SIDE), (0.10, 0.07, 0.25))


if __name__ == "__main__":
    unittest.main()
