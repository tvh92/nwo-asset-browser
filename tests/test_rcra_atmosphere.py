"""Rift Apart atmospheres (nwo/games/rcra/atmosphere.py): the built block's fields, the key-light direction, the
game's metering and adaptation, the cube lookup, the Blender display group (Blender installed) and the real install's
Megalopolis day atmosphere (RCRA_GAME)."""
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from nwo.games.rcra import atmosphere as A

GAME = os.environ.get('RCRA_GAME')
BLENDER = os.environ.get('RC2016_BLENDER') or r'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe'
ROOT = Path(__file__).resolve().parents[1]


class BlockTests(unittest.TestCase):
    def test_parse_reads_the_documented_offsets(self):
        data = bytearray(800)
        for name, value in (('key_intensity', 6.0), ('key_azimuth', 220.0), ('key_elevation', 45.0), ('anchor', 12.0),
                            ('range', 10.0), ('bias', -0.5), ('adjust_time', 1.0)):
            struct.pack_into('<f', data, A.FIELDS[name], value)
        struct.pack_into('<3f', data, A.FIELDS['key_colour'], 1.0, 1.0, 0.83)
        struct.pack_into('<Q', data, A.FIELDS['probe'], 0x1234)
        names = {A.FIELDS['probe']: 'probe.texture'}
        fields = A.parse(bytes(data), lambda offset: names.get(offset) if struct.unpack_from('<Q', data, offset)[0] else None)
        self.assertEqual((fields['key_intensity'], fields['key_azimuth'], fields['key_elevation']), (6.0, 220.0, 45.0))
        self.assertEqual((fields['anchor'], fields['range'], fields['bias']), (12.0, 10.0, -0.5))
        self.assertAlmostEqual(fields['key_colour'][2], 0.83, places=6)
        self.assertEqual(fields['probe'], 'probe.texture')
        self.assertIsNone(fields['colour_correction'])
        np.testing.assert_allclose(A.key_light(fields), [6.0, 6.0, 4.98], atol=1e-5)

    def test_key_direction_turns_from_plus_z_towards_plus_x(self):
        np.testing.assert_allclose(A.key_direction(0, 0), [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(A.key_direction(90, 0), [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(A.key_direction(0, 90), [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(A.game_to_blender([0, 1, 0]), [0, 0, 1])
        np.testing.assert_allclose(A.blender_to_game(A.game_to_blender([0.2, 0.3, 0.4])), [0.2, 0.3, 0.4])


class ExposureTests(unittest.TestCase):
    def test_adaptation_follows_the_scene_near_the_anchor_and_clamps_far_from_it(self):
        anchor, rng = 12.0, 3.0
        a = anchor - A.EV_OFFSET
        self.assertAlmostEqual(A.adapted_log2(a, anchor, rng), a)
        slope = (A.adapted_log2(a + 1e-3, anchor, rng) - A.adapted_log2(a - 1e-3, anchor, rng)) / 2e-3
        self.assertAlmostEqual(slope, 1.0, places=2)   # the finalize program's 3.749
        self.assertAlmostEqual(A.adapted_log2(a + 40, anchor, rng), a + rng)
        self.assertAlmostEqual(A.adapted_log2(a - 40, anchor, rng), a - rng)
        self.assertAlmostEqual(A.adapted_log2(a + 5, anchor, 0.0), a)   # range 0: fixed at the anchor

    def test_meter_and_exposure(self):
        scene = np.full((9, 16, 3), 2.0)
        self.assertAlmostEqual(A.meter(scene), 1.0)
        self.assertAlmostEqual(A.exposure(math.log2(0.18)), 1.0)
        self.assertAlmostEqual(A.exposure(math.log2(0.36)), 0.5)


class CubeTests(unittest.TestCase):
    def test_each_axis_reads_its_d3d_face(self):
        faces = np.stack([np.full((8, 8, 3), k, np.float32) for k in range(6)])
        axes = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float64)
        np.testing.assert_array_equal(A.cube_sample(faces, axes)[:, 0], np.arange(6))

    def test_face_layout_puts_up_at_the_top_row(self):
        face = np.zeros((6, 4, 4, 3), np.float32)
        face[0, 0] = 1.0   # +X face, top row
        up = A.cube_sample(face, np.array([[1.0, 0.9, 0.0]]))[0, 0]
        down = A.cube_sample(face, np.array([[1.0, -0.9, 0.0]]))[0, 0]
        self.assertGreater(up, 0.5)
        self.assertEqual(down, 0.0)


class ExportKeyTests(unittest.TestCase):
    def test_the_atmosphere_is_in_the_export_key_only_when_chosen(self):
        from nwo.export.models import ExportOptions
        self.assertNotIn('atmosphere', ExportOptions(scene=True).key())
        self.assertEqual(ExportOptions(scene=True, atmosphere=A.DEFAULT).key()['atmosphere'], A.DEFAULT)


@unittest.skipUnless(os.path.isfile(BLENDER), 'Blender is not installed')
class BlenderDisplayTests(unittest.TestCase):
    """The compositor group (nwo.export.rcra_atmosphere_blender) shows a uniform world as the game's display does."""

    def test_group_matches_the_display_port(self):
        colours = [(0.02, 0.02, 0.02), (0.18, 0.18, 0.18), (1.5, 0.4, 0.1), (0.05, 0.3, 2.0), (12.0, 9.0, 7.0)]
        exposure = 0.6
        script = textwrap.dedent('''
            import json, sys, os, tempfile, bpy, numpy as np
            sys.path.insert(0, ROOT)
            from nwo.export import rcra_atmosphere_blender as R
            out = sys.argv[sys.argv.index('--') + 1]
            scene = bpy.context.scene
            scene.render.engine = 'CYCLES'; scene.cycles.samples = 1; scene.cycles.use_denoising = False
            scene.render.resolution_x = scene.render.resolution_y = 4
            scene.render.film_transparent = False
            camera = bpy.data.objects.new('c', bpy.data.cameras.new('c')); scene.collection.objects.link(camera)
            scene.camera = camera
            for o in [o for o in scene.objects if o.type == 'MESH']: bpy.data.objects.remove(o)
            group = R.build_group()
            tree = bpy.data.node_groups.new('t', 'CompositorNodeTree')
            tree.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
            layers = tree.nodes.new('CompositorNodeRLayers'); node = tree.nodes.new('CompositorNodeGroup'); node.node_tree = group
            node.inputs['Exposure'].default_value = EXPOSURE
            o = tree.nodes.new('NodeGroupOutput')
            tree.links.new(layers.outputs['Image'], node.inputs['Image']); tree.links.new(node.outputs['Image'], o.inputs['Image'])
            scene.compositing_node_group = tree; scene.render.use_compositing = True
            scene.view_settings.view_transform = 'Raw'
            world = bpy.data.worlds.new('w'); world.use_nodes = True; scene.world = world
            bg = world.node_tree.nodes['Background']; bg.inputs['Strength'].default_value = 1.0
            r = scene.render; r.image_settings.file_format = 'OPEN_EXR'; r.image_settings.color_depth = '32'
            results = []
            for colour in COLOURS:
                bg.inputs['Color'].default_value = (*colour, 1.0)
                r.filepath = os.path.join(tempfile.gettempdir(), 'rcra_display_test.exr')
                bpy.ops.render.render(write_still=True)
                image = bpy.data.images.load(r.filepath)
                results.append([float(v) for v in np.array(image.pixels[:], np.float32).reshape(-1, 4)[:, :3].mean(0)])
                bpy.data.images.remove(image)
            open(out, 'w').write(json.dumps(results))
        ''').replace('ROOT', repr(str(ROOT))).replace('EXPOSURE', repr(exposure)).replace('COLOURS', repr(colours))
        with tempfile.TemporaryDirectory() as tmp:
            path, out = Path(tmp) / 'display_test.py', Path(tmp) / 'out.json'
            path.write_text(script)
            env = dict(os.environ, PYTHONPATH=str(ROOT), BLENDER_USER_RESOURCES=str(Path(tmp) / 'user'))
            run = subprocess.run([BLENDER, '--background', '--factory-startup', '--python-exit-code', '1', '--python',
                                  str(path), '--', str(out)], capture_output=True, text=True, env=env, timeout=600)
            self.assertEqual(run.returncode, 0, run.stdout[-2000:] + run.stderr[-2000:])
            got = np.array(json.loads(out.read_text()))
        from nwo.games.rcra import display
        np.testing.assert_allclose(got, display.display(np.array(colours), exposure), atol=2e-4)


@unittest.skipUnless(GAME, 'RCRA_GAME is not set')
class InstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nwo.viewer.loader import SourcePreviewLoader
        cls.source = SourcePreviewLoader()._ensure_archive(GAME)

    def test_megalopolis_day(self):
        atm = A.read(self.source, A.DEFAULT)
        np.testing.assert_allclose(atm['key_colour'], [1.0, 1.0, 0.83077], atol=1e-5)
        self.assertEqual((atm['key_intensity'], atm['key_azimuth'], atm['key_elevation']), (6.0, 220.0, 45.0))
        self.assertEqual((atm['anchor'], atm['range'], atm['bias']), (12.0, 10.0, 0.0))
        self.assertEqual(atm['probe'], 'levels/i29/instance/malinon_intro/malinon_intro/envprobes_day/0xe54a425db9195aba.texture')
        self.assertEqual(atm['colour_correction'], 'textures/lut/lut_megalopolis_warm9.texture')
        self.assertIn(A.DEFAULT, A.list_paths(self.source))

    def test_environment_turns_the_key_to_the_review_sun(self):
        sun = np.array([0.11468, -0.93279, 0.34168])
        env = A.environment(self.source, A.DEFAULT, sun, width=256)
        self.assertEqual(env['equirect'].shape, (128, 256, 3))
        key = env['sun_direction']
        self.assertAlmostEqual(math.atan2(key[1], key[0]), math.atan2(sun[1], sun[0]), places=5)
        self.assertAlmostEqual(float(key[2]), math.sin(math.radians(45.0)), places=5)   # the elevation is kept
        np.testing.assert_allclose(env['sun_irradiance'], math.pi * np.array([6.0, 6.0, 4.98462]), rtol=1e-5)
        mean = env['equirect'].reshape(-1, 3).mean(0)
        self.assertTrue(0.5 < mean[0] < 1.2 and mean[2] > mean[0], mean)   # the blue Megalopolis day sky


if __name__ == '__main__':
    unittest.main()
