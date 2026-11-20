"""Rift Apart vertex programs over shader time: the timer as an input (rcra.vertex), the viewer payload and viewport,
the export's baked frames, and real-install checks (RCRA_GAME, skipped without it)."""
import ctypes
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rc2016.export import models
from rc2016.games.rcra import vertex as rv
from rc2016.shaders import sm5
from rc2016.viewer.loader import _motion_payload

GAME = os.environ.get('RCRA_GAME')


def wave_recipe():
    """A recipe whose position is (x + time, y, z, 1) and normal (0, 0, 1)."""
    g = sm5.Graph()
    x, t = g.input('X', 'f'), g.input(rv.TIMER, 'f')
    one, zero = g.const('f', 1.0), g.const('f', 0.0)
    terms = {'position.x': g.op('add', x, t), 'position.y': zero, 'position.z': zero, 'position.w': one,
             'normal.x': zero, 'normal.y': zero, 'normal.z': one}
    return rv.export(dict(graph=g, terms=terms, units={}, unbound=[]))


class FakeMotion:
    """VertexMotion stand-in: mesh 4 moves along x by the time."""
    meshes = [4]

    def __init__(self, count=3):
        self.count = count

    def at(self, seconds):
        positions = np.zeros((self.count, 3))
        positions[:, 0] = seconds
        return {4: (positions, np.tile([0.0, 0.0, 1.0], (self.count, 1)))}


class TimerTests(unittest.TestCase):
    def test_timer_is_an_input_of_the_evaluation(self):
        recipe = wave_recipe()
        local = {'X': np.array([1.0, 2.0], np.float32), rv.VERTEX_ID: np.array([0, 1])}
        positions, normals = rv.at_time(recipe, local, None, np.zeros(3), np.zeros((2, 3)), 0.5)
        np.testing.assert_allclose(positions[:, 0], [1.5, 2.5])
        np.testing.assert_allclose(normals, [[0, 0, 1], [0, 0, 1]])

    def test_motion_groups_slice_their_meshes(self):
        motion = rv.VertexMotion()
        local = {'X': np.array([0.0, 1.0, 2.0], np.float32), rv.VERTEX_ID: np.arange(3)}
        motion.add(wave_recipe(), local, None, np.array([0.0, 10.0, 0.0]), np.zeros((3, 3)), [(7, 0, 1), (9, 1, 2)])
        self.assertEqual(motion.meshes, [7, 9])
        out = motion.at(2.0)
        np.testing.assert_allclose(out[7][0], [[2, 10, 0]])
        np.testing.assert_allclose(out[9][0][:, 0], [3, 4])

    def test_translation_binds_the_timer_as_an_input(self):
        bindings = rv._VertexBindings(None, None, dict(colour=False, uv1=False, uv_word=0, scale=1.0), 0)
        bindings.records = {}
        bindings.graph = sm5.Graph()
        node = bindings.constant(rv.TIMER)
        self.assertEqual(bindings.graph.nodes[node]['op'], 'input')
        self.assertNotIn(rv.TIMER, bindings.engine)


class ViewerTests(unittest.TestCase):
    def test_payload_maps_meshes_into_the_payload_vertices(self):
        decoded = dict(vertex_motion=FakeMotion(2))
        meshes = [dict(index=1, positions=np.zeros((5, 3))), dict(index=4, positions=np.zeros((2, 3)))]
        at = _motion_payload(decoded, meshes)['vertex_motion']
        index, positions, normals = at(0.25)
        self.assertEqual(index.tolist(), [5, 6])
        self.assertEqual(positions.shape, (2, 3))
        self.assertAlmostEqual(float(np.abs(positions).max()), 0.25)
        self.assertEqual(_motion_payload(dict(vertex_motion=None), meshes), {})
        self.assertEqual(_motion_payload(decoded, meshes[:1]), {})   # the moving mesh is not shown

    def test_viewport_replaces_then_restores(self):
        from rc2016.viewer.viewport import ModelViewport
        vertices, normals = np.zeros((3, 3)), np.tile([0.0, 0.0, 1.0], (3, 1))
        arrays = ((ctypes.c_float * 9)(*vertices.ravel()), (ctypes.c_float * 9)(*normals.ravel()))
        drawn = []
        view = SimpleNamespace(_morphs=None, _mesh=(vertices, None, normals), _arrays=arrays, _stale=set(), _morphed=False,
                               _morph_weights=None, _motion=None, _draw=lambda: drawn.append(1))
        view._refresh_vertices = lambda: ModelViewport._refresh_vertices(view)
        frame = (np.array([2]), np.array([[1.0, 2.0, 3.0]], np.float32), np.array([[1.0, 0.0, 0.0]], np.float32))
        ModelViewport.set_vertex_motion(view, frame)
        np.testing.assert_allclose(np.frombuffer(arrays[0], np.float32).reshape(3, 3)[2], [1, 2, 3])
        np.testing.assert_allclose(np.frombuffer(arrays[1], np.float32).reshape(3, 3)[2], [1, 0, 0])
        self.assertTrue(view._morphed and drawn)
        ModelViewport.set_vertex_motion(view, None)
        np.testing.assert_allclose(np.frombuffer(arrays[0], np.float32), 0)
        self.assertFalse(view._morphed)


class ExportTests(unittest.TestCase):
    def test_frames_bake_deltas_and_a_stepping_animation(self):
        mesh = dict(index=4, positions=np.zeros((3, 3)), normals=np.tile([0.0, 0.0, 1.0], (3, 1)),
                    triangles=[(0, 1, 2)], uv0=None)
        parts = [dict(name='m4', material=None, arrays=models.mesh_arrays(mesh, 0), mesh_index=4)]
        times = models.vertex_frames(FakeMotion(), parts, seconds=1.0, fps=2)
        self.assertEqual(times, [0.0, 0.5, 1.0])
        frames = parts[0]['arrays']['frames']
        np.testing.assert_allclose(frames[2][0][:, 0], 1.0)
        exporter = models.ModelExporter.__new__(models.ModelExporter)
        gltf = exporter._build_gltf('m', parts, {}, None, [], (), times)
        doc, blob = gltf.finish(embed=True)
        names = doc['meshes'][0]['extras']['targetNames']
        self.assertEqual(names, ['time 0.000', 'time 0.500', 'time 1.000'])
        animation = doc['animations'][0]
        self.assertEqual(animation['name'], models.VERTEX_ANIMATION)
        self.assertEqual(animation['channels'][0]['target']['path'], 'weights')
        self.assertIn('vertex_animation', json.dumps(models.ExportOptions(vertex_animation=True).key()))
        self.assertNotIn('vertex_animation', json.dumps(models.ExportOptions().key()))

    def test_no_moving_part_bakes_nothing(self):
        parts = [dict(arrays={}, mesh_index=1)]
        self.assertEqual(models.vertex_frames(FakeMotion(), parts), [])


@unittest.skipUnless(GAME, 'RCRA_GAME is not set')
class RealMotionTests(unittest.TestCase):
    SWARMER = ('characters/enemy/enm_neo_nefarious_swarmer/animations/vertex_animations/'
               'enm_neo_nefarious_swarmer_passive_move_vertexanim/models/'
               'enm_neo_nefarious_swarmer_passive_move_vertexanim.model')
    FLAG = 'environment/megalopolis/prop/meg_prop_meg_flag_01/ja_meg_prop_meg_flag_01.model'

    @classmethod
    def setUpClass(cls):
        from rc2016.games.rcra.source import RcraSource
        cls.temp = tempfile.TemporaryDirectory()
        cls.source = RcraSource(GAME, str(Path(cls.temp.name) / 'out'))

    @classmethod
    def tearDownClass(cls):
        cls.source.close()
        cls.temp.cleanup()

    def decode(self, path):
        from rc2016.games.rcra.model import decode_for_viewer
        return decode_for_viewer(self.source, self.source.resolve(path)[0], path, True)[0]

    def test_time_zero_is_the_rest_pose_and_later_times_move(self):
        for path, meshes in ((self.SWARMER, 27), (self.FLAG, 1)):
            decoded = self.decode(path)
            motion = decoded['vertex_motion']
            self.assertEqual(len(motion.meshes), meshes)
            by_index = {m['index']: m for m in decoded['meshes']}
            rest, later = motion.at(0.0), motion.at(0.5)
            for index, (positions, _normals) in rest.items():
                np.testing.assert_allclose(positions, by_index[index]['positions'], atol=1e-5)
            moved = max(float(np.abs(later[i][0] - rest[i][0]).max()) for i in rest)
            self.assertGreater(moved, 0.01)
            notes = [r['note'] for r in decoded['vertex_programs']]
            self.assertTrue(all(rv.TIME_NOTE in n for n in notes))

    def test_still_programs_have_no_motion(self):
        decoded = self.decode('environment/megalopolis/shrub/os_meg_plant_d106a/os_meg_plant_d106a.model')
        self.assertIsNone(decoded.get('vertex_motion'))


if __name__ == '__main__':
    unittest.main()
