"""Additive light on a transparent film (nwo.export.rcra_coverage): opaque pixels and straight colours below W keep the
usual encoding; pure additive light is written so that it shows over black as the render's display of it."""
import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from nwo.export import rcra_coverage

ROOT = Path(__file__).resolve().parents[1]
BLENDER = os.environ.get('RC2016_BLENDER', r'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe')
# (premultiplied colour, alpha): opaque, an antialiased edge below W, pure additive light, faint additive light
CASES = [((0.5, 0.3, 0.1), 1.0), ((0.2, 0.1, 0.05), 0.5), ((0.4, 0.6, 0.9), 0.0), ((0.3, 0.3, 0.3), 0.02),
         ((3.0, 2.0, 1.0), 1.0)]


class LimitTests(unittest.TestCase):
    def test_limits_are_where_the_display_reaches_the_covered_fraction(self):
        from nwo.games.rcra import display as D
        w = rcra_coverage.game_grey_limit()
        top = D.display(np.full((1, 3), 2.0 ** ((1.0 - D.LOG2_OFFSET) / D.LOG2_SCALE))).mean()
        self.assertAlmostEqual(D.display(np.full((1, 3), w)).mean(), rcra_coverage.COVERED_DISPLAY * top, places=4)
        self.assertAlmostEqual(rcra_coverage.khronos_grey_limit(), 1.843, places=3)
        self.assertAlmostEqual(rcra_coverage.standard_grey_limit(), 0.9551, places=4)


@unittest.skipUnless(os.path.isfile(BLENDER), 'Blender is not installed')
class BlenderCoverageTests(unittest.TestCase):
    def run_groups(self):
        script = textwrap.dedent('''
            import json, os, sys, tempfile, bpy, numpy as np
            sys.path.insert(0, ROOT)
            from nwo.export import rcra_coverage as C
            from nwo.export import rcra_atmosphere_blender as R
            out = sys.argv[sys.argv.index('--') + 1]
            scene = bpy.context.scene
            scene.render.engine = 'CYCLES'; scene.cycles.samples = 1
            scene.render.resolution_x = scene.render.resolution_y = 4; scene.render.resolution_percentage = 100
            scene.view_settings.view_transform = 'Raw'
            tree = bpy.data.node_groups.new('t', 'CompositorNodeTree')
            tree.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
            rgb = tree.nodes.new('CompositorNodeRGB')
            review = tree.nodes.new('CompositorNodeGroup')
            review.node_tree = C.review_group('Khronos PBR Neutral', C.khronos_grey_limit())
            review.inputs['Exposure (stops)'].default_value = -1.0
            plain = tree.nodes.new('CompositorNodeConvertColorSpace')   # the view of an opaque colour, for reference
            plain.from_color_space, plain.to_color_space = 'scene_linear', 'Khronos PBR Neutral sRGB'
            game = tree.nodes.new('CompositorNodeGroup')
            game.node_tree = C.atmosphere_group(R.build_group(), C.game_grey_limit())
            game.inputs['Exposure'].default_value = 0.6
            o = tree.nodes.new('NodeGroupOutput')
            for node in (review, game):
                tree.links.new(rgb.outputs[0], node.inputs['Image'])
            scene.compositing_node_group = tree; scene.render.use_compositing = True
            path = os.path.join(tempfile.gettempdir(), 'rcra_coverage_test.exr')
            scene.render.image_settings.file_format = 'OPEN_EXR'; scene.render.image_settings.color_depth = '32'

            def run(node):
                for link in list(tree.links):
                    if link.to_node is o:
                        tree.links.remove(link)
                tree.links.new(node.outputs[0], o.inputs[0])
                bpy.ops.render.render()
                bpy.data.images['Render Result'].save_render(path, scene=scene)
                image = bpy.data.images.load(path, check_existing=False)
                value = [float(v) for v in np.array(image.pixels[:4])]
                bpy.data.images.remove(image)
                return value
            results = []
            for colour, alpha in CASES:
                rgb.outputs[0].default_value = (*colour, alpha)
                row = dict(review=run(review), game=run(game))
                straight = [c / max(alpha, 1e-6) * 0.5 for c in colour]
                rgb.outputs[0].default_value = (*straight, 1.0)
                tree.links.new(rgb.outputs[0], plain.inputs[0])
                row['view_straight'] = run(plain)
                rgb.outputs[0].default_value = (*[c * 0.5 for c in colour], 1.0)
                row['view_whole'] = run(plain)
                results.append(row)
            open(out, 'w').write(json.dumps(results))
        ''').replace('ROOT', repr(str(ROOT))).replace('CASES', repr(CASES))
        with tempfile.TemporaryDirectory() as tmp:
            path, out = Path(tmp) / 'coverage_test.py', Path(tmp) / 'out.json'
            path.write_text(script)
            env = dict(os.environ, PYTHONPATH=str(ROOT), BLENDER_USER_RESOURCES=str(Path(tmp) / 'user'))
            run = subprocess.run([BLENDER, '--background', '--factory-startup', '--python-exit-code', '1', '--python',
                                  str(path), '--', str(out)], capture_output=True, text=True, env=env, timeout=600)
            self.assertEqual(run.returncode, 0, run.stdout[-2000:] + run.stderr[-2000:])
            return json.loads(out.read_text())

    def test_coverage_keeps_covered_pixels_and_carries_additive_light(self):
        from nwo.games.rcra import display as D
        rows = self.run_groups()
        for (colour, alpha), row in zip(CASES, rows):
            review, game = np.array(row['review']), np.array(row['game'])
            colour = np.array(colour)
            straight = max(colour) / alpha * 0.5 if alpha else float('inf')   # exposed straight peak (review, -1 stop)
            if alpha == 1.0 or straight < rcra_coverage.khronos_grey_limit():
                # Blender's own encoding: premultiplied a disp(C / a), alpha a
                np.testing.assert_allclose(review[:3], alpha * np.array(row['view_straight'][:3]), atol=1e-4)
                self.assertAlmostEqual(review[3], alpha, places=5)
                np.testing.assert_allclose(game[:3], alpha * D.display(colour / alpha, 0.6), atol=2e-4)
                self.assertAlmostEqual(game[3], alpha, places=5)
            if alpha == 0.0:
                # pure additive light: written as its whole display (premultiplied over black), covered by its peak; less
                # Khronos's display of black (3e-4), which the carried light's display subtracts
                np.testing.assert_allclose(review[:3], row['view_whole'][:3], atol=5e-4)
                self.assertAlmostEqual(review[3], max(row['view_whole'][:3]), delta=5e-4)
                np.testing.assert_allclose(game[:3], D.display(colour, 0.6), atol=2e-4)
                self.assertAlmostEqual(game[3], max(D.display(colour, 0.6)), places=4)
            if 0.0 < alpha < 1.0 and straight >= rcra_coverage.khronos_grey_limit():
                # faint additive light: between the two, at least what Blender writes and at most its whole display
                self.assertGreater(review[:3].max(), alpha * max(row['view_straight'][:3]) + 0.1)
                self.assertTrue(np.all(review[:3] <= np.array(row['view_whole'][:3]) + 1e-4))
                self.assertGreaterEqual(review[3], review[:3].max() - 1e-6)


if __name__ == '__main__':
    unittest.main()
