"""Rift Apart vertex programs at rest (nwo.games.rcra.vertex): the vectorised evaluator, the view check, the stored
vertices in the glTF export, plus real-install checks (RCRA_GAME, skipped without it)."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nwo.export import models
from nwo.games.rcra import vertex as rv
from nwo.shaders import sm5

GAME = os.environ.get('RCRA_GAME')


class EvaluatorTests(unittest.TestCase):
    def test_vector_ops_match_the_translator_bit_for_bit(self):
        rng = np.random.default_rng(7)
        n = 48
        floats = np.concatenate([rng.normal(0, 100, n - 8), [0.0, -0.0, 1.5, -2.5, np.inf, -np.inf, np.nan, 3e9]])
        floats = floats.astype(np.float32)
        ints = np.concatenate([rng.integers(0, 2 ** 32, n - 8, dtype=np.uint64),
                               [0, 1, 31, 32, 0xFFFF, 0x80000000, 0xFFFFFFFF, 0x3C00]]).astype(np.uint32)
        small = (ints % 40).astype(np.uint32)
        bools = rng.integers(0, 2, n).astype(bool)
        for op, fn in rv.VECTOR_OPS.items():
            kind, arg_types, scalar = sm5.OPS[op]
            if scalar is None:
                continue
            args = []
            for k, t in enumerate(arg_types):
                if t == 'f':
                    args.append(np.roll(floats, k))
                elif t == 'b':
                    args.append(np.roll(bools, k))
                else:
                    args.append(np.roll(small if op in ('ubfe', 'ibfe', 'bfi') and k < 2 else ints, 5 * k))
            got = rv._typed(fn(*args), kind)
            for i in range(n):
                want = scalar([float(a[i]) if a.dtype.kind == 'f' else (bool(a[i]) if a.dtype == bool else int(a[i]))
                               for a in args])
                if kind == 'f':   # bit-identical, except that min and max may pick either signed zero
                    w = np.float32(want)
                    same = (np.isnan(w) and np.isnan(got[i])) or w.view(np.uint32) == got[i].view(np.uint32) or (
                        op in ('min', 'max') and w == 0 and got[i] == 0)
                else:
                    same = int(got[i]) == (int(want) & 0xFFFFFFFF if kind == 'i' else bool(want))
                self.assertTrue(same, '%s %s: %r != %r' % (op, [a[i] for a in args], got[i], want))

    def test_recipe_evaluation_with_select_channel_and_texture(self):
        g = sm5.Graph()
        x = g.input('VB.POS.x', 'i')
        scaled = g.op('mul', g.op('itof', x), g.const('f', 0.5))
        load = g._add('load', 'v', (g.const('i', 0),), opcode='ld', binding='textureUnit5')
        texel = g.channel(load, 1, 'f')
        positive = g.op('lt', g.const('f', 0.0), scaled)
        out = g.select(positive, g.op('add', scaled, texel), scaled)
        values = rv.evaluate(g.nodes, [out], {'VB.POS.x': np.array([-4, 0, 6])},
                             lambda node, args: [np.full(3, v, np.float32) for v in (9.0, 2.0, 0.0, 1.0)])
        np.testing.assert_array_equal(values[out], np.float32([-2.0, 0.0, 5.0]))

    def test_check_view_inverts_back_to_the_world(self):
        world = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 9.0]])
        camera, matrix = rv.CHECK_VIEW
        clip = np.concatenate([world - camera, np.ones((2, 1))], 1) @ matrix
        out = {'position.' + c: clip[:, k] for k, c in enumerate('xyzw')}
        np.testing.assert_allclose(rv.world(out, rv.CHECK_VIEW), world)
        self.assertIsNone(rv.world(dict(out, **{'position.w': np.array([1.0, 2.0])}), rv.CHECK_VIEW))


def _recipe(offset_x, view_offset=None):
    """A vertex 'program' writing clip = (pos * 1/4096 + (offset_x, 0, 0) - camera) @ matrix, normal +z, under a view."""
    def make(view, extra):
        g = sm5.Graph()
        camera, matrix = view
        world = [g.op('add', g.op('mul', g.op('itof', g.input('VB.POS.' + c, 'i')), g.const('f', 1 / 4096)),
                      g.const('f', (extra if c == 'x' else 0.0) - camera[k])) for k, c in enumerate('xyz')]
        terms = {}
        for col, c in enumerate('xyzw'):
            node = g.const('f', float(matrix[3][col]))
            for row in range(3):
                node = g.op('add', node, g.op('mul', world[row], g.const('f', float(matrix[row][col]))))
            terms['position.' + c] = node
        for k, c in enumerate('xyz'):
            terms['normal.' + c] = g.const('f', float(k == 2))
        return dict(nodes=g.nodes, terms=terms, units={})
    return {'rest': make(rv.REVIEW_VIEW, offset_x),
            'view': make(rv.CHECK_VIEW, offset_x if view_offset is None else view_offset)}


class AtRestTests(unittest.TestCase):
    inputs = {'VB.POS.x': np.array([0, 4096]), 'VB.POS.y': np.array([0, 0]), 'VB.POS.z': np.array([4096, 0])}
    normals = np.array([[0.0, 0.0, 1.0]] * 2)

    def run_pair(self, pair, offset=(0.0, 0.0, 0.0), with_tangents=False):
        return rv.at_rest(pair, self.inputs, None, np.arange(2), np.asarray(offset), self.normals, with_tangents)

    def test_offset_lands_in_the_decoder_frame(self):
        positions, normals = self.run_pair(_recipe(0.25), offset=(10.0, 0.0, 0.0))
        np.testing.assert_allclose(positions, [[10.25, 0.0, 1.0], [11.25, 0.0, 0.0]], atol=1e-6)
        self.assertTrue(rv.moved(positions, normals, [[10.0, 0, 1], [11.0, 0, 0]], self.normals))
        self.assertFalse(rv.moved(positions, normals, positions, normals))

    def test_rest_tangent_comes_from_the_program(self):
        pair = _recipe(0.25)
        positions, normals, tangents, w = self.run_pair(pair, with_tangents=True)
        self.assertIsNone(tangents)   # the recipe writes no TEXCOORD3: the stored tangent stays
        for name in ('rest', 'view'):
            g = sm5.Graph()
            g.nodes = [dict(n) for n in pair[name]['nodes']]
            for k, c in enumerate('xyzw'):
                g.nodes.append(dict(op='const', type='f', value=(0.0, 1.0, 0.0, -4.0)[k], args=[]))
                pair[name]['terms']['tangent.' + c] = len(g.nodes) - 1
            pair[name]['nodes'] = g.nodes
        positions, normals, tangents, w = self.run_pair(pair, with_tangents=True)
        np.testing.assert_allclose(tangents, [[0.0, 1.0, 0.0]] * 2)
        np.testing.assert_allclose(w, [-4.0, -4.0])
        mesh = dict(positions=positions)
        rv._set_tangents(mesh, tangents, w)
        np.testing.assert_array_equal(mesh['tangent_signs'], [-1.0, -1.0])
        np.testing.assert_allclose(mesh['tangent_w'], w)

    def test_view_dependent_program_has_no_rest_shape(self):
        self.assertIsNone(self.run_pair(_recipe(0.25, view_offset=0.5)))


class ExportTests(unittest.TestCase):
    def test_stored_shape_names_agree(self):
        self.assertEqual(rv.STORED_SHAPE, models.STORED_SHAPE)

    def test_moved_mesh_keeps_its_stored_vertices_as_a_morph_target(self):
        base = dict(positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], normals=[(0.0, 0.0, 1.0)] * 3,
                    uv0=[(0.0, 0.0)] * 3, uv1=None, triangles=[[0, 1, 2]])
        moved = dict(base, positions=[(0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (0.0, 1.0, 0.5)],
                     stored_positions=base['positions'], stored_normals=base['normals'])
        parts = [dict(name='a', material=None, arrays=models.mesh_arrays(moved, 0)),
                 dict(name='b', material=None, arrays=models.mesh_arrays(base, 0))]
        exporter = models.ModelExporter.__new__(models.ModelExporter)
        doc, blob = exporter._build_gltf('m', parts, {}, None, []).finish(embed=True)
        mesh = doc['meshes'][0]
        self.assertEqual((mesh['weights'], mesh['extras']['targetNames']), ([0.0], [models.STORED_SHAPE]))
        offsets = []
        for prim in mesh['primitives']:
            accessor = doc['accessors'][prim['targets'][0]['POSITION']]
            view = doc['bufferViews'][accessor['bufferView']]
            data = np.frombuffer(bytes(blob), np.float32, accessor['count'] * 3, view['byteOffset'] + accessor.get('byteOffset', 0))
            offsets.append(data.reshape(-1, 3))
            self.assertIn('min', accessor)
        np.testing.assert_allclose(offsets[0][:, 2], -0.5)
        np.testing.assert_array_equal(offsets[1], 0.0)


class LoaderNoteTests(unittest.TestCase):
    def test_decoder_material_notes_join_the_material_rule(self):
        from nwo.viewer.loader import SourcePreviewLoader

        class Game:
            def decode_model(self, archive, model_id, path, view):
                return dict(meshes=[], materials=[dict(index=0, path='m.material')],
                            material_notes={'m.material': ['moved at rest']}), b''

            def material_textures(self, *a, **k):
                return {}

            def model_texture_list(self, *a):
                return []

            def material_rules(self, archive, materials, translations=None):
                return {'m.material': dict(native_limitations=['preview'])}

            def effect_preview(self, *a):
                return None
        loader = SourcePreviewLoader.__new__(SourcePreviewLoader)
        entry = loader._decode(None, 'id', 'x.model', 'view', {}, game=Game())
        self.assertEqual(entry['decoded']['material_rules']['m.material']['native_limitations'], ['preview', 'moved at rest'])


@unittest.skipUnless(GAME, 'RCRA_GAME is not set')
class RealVertexProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nwo.games.rcra.source import RcraSource
        cls.temp = tempfile.TemporaryDirectory()
        cls.source = RcraSource(GAME, Path(cls.temp.name) / 'out')

    @classmethod
    def tearDownClass(cls):
        cls.source.close()
        cls.temp.cleanup()

    def material(self, path):
        from nwo.games.rcra.material import _material_key
        from nwo.games.rcra.viewer_materials import _graph
        key = _material_key(self.source, path)
        return _graph(self.source, key), self.source.asset_data(key)

    def test_vertex_animation_texture_moves_the_swarmer(self):
        from nwo.games.rcra.model import decode_for_viewer
        path = ('characters/enemy/enm_neo_nefarious_swarmer/animations/vertex_animations/'
                'enm_neo_nefarious_swarmer_passive_move_vertexanim/models/'
                'enm_neo_nefarious_swarmer_passive_move_vertexanim.model')
        key = self.source.resolve(path)[0]
        decoded, data = decode_for_viewer(self.source, key, path, True)
        rows = decoded['vertex_programs']
        self.assertEqual({r['state'] for r in rows}, {'moved'})
        self.assertEqual(len(rows), len(decoded['meshes']))
        self.assertGreater(max(r['max_offset'] for r in rows), 0.5)
        mesh = decoded['meshes'][0]
        self.assertEqual(np.asarray(mesh['stored_positions']).shape, np.asarray(mesh['positions']).shape)
        # The vectorised run agrees with the translator's own evaluator on single vertices.
        graph, material = self.material(mesh['material']['path'])
        layout = rv.model_layout(data)
        from nwo.games.rcra.texture import texture_size
        tr = rv.translate(graph, material, layout, texture_sizes=lambda p: texture_size(self.source, p))
        recipe = rv.export(tr)
        inputs, offset, _count = rv.vertex_streams(data)
        reader = rv.TextureReader(self.source, recipe['units'])
        for v in (mesh['first_vertex'], mesh['first_vertex'] + mesh['vertex_count'] - 1):
            scalar = {k: (float(a[v]) if a.dtype.kind == 'f' else int(a[v])) for k, a in inputs.items()}
            scalar[rv.VERTEX_ID] = v
            scalar[rv.TIMER] = 0.0   # the shader timer is an input; the at-rest pose is time 0

            def resource(node, args):
                out = reader(node, [np.asarray([a]) for a in args])
                return [float(o[0]) for o in out]
            ref = sm5.evaluate(tr['graph'], [tr['terms']['position.' + c] for c in 'xyz'], scalar, resource)
            want = np.array([ref[tr['terms']['position.' + c]] for c in 'xyz']) + offset
            np.testing.assert_allclose(mesh['positions'][v - mesh['first_vertex']], want, atol=1e-6)

    def random_inputs(self, n=64):
        rng = np.random.default_rng(3)
        inputs = {'VB.POS.' + c: rng.integers(-8000, 8000, n) for c in 'xyzw'}
        for c in 'xyz':   # normals inside the encoding's range (|p|^2 <= 2): 10-bit fields near the middle
            inputs['VB.NRM.' + c] = rng.integers(300, 724, n).astype(np.float32) / np.float32(1023)
        inputs['VB.NRM.w'] = np.ones(n, np.float32)
        for name in ('VB.UV0', 'VB.UV1'):
            inputs[name + '.x'], inputs[name + '.y'] = rng.integers(0, 16384, n), rng.integers(0, 16384, n)
        for c in 'xyzw':   # varied: vertex colour gates some programs' offsets
            inputs['VB.COLOUR.' + c] = rng.integers(0, 256, n).astype(np.float32) / np.float32(255)
        return inputs

    def at_rest(self, path):
        graph, material = self.material(path)
        layout = dict(scale=1 / 4096, uv_word=0, uv1=True, colour=True)
        pair = rv.material_recipes(self.source, graph, material, layout)
        inputs = self.random_inputs()
        normals = np.tile([0.0, 0.0, 1.0], (64, 1))
        result = rv.at_rest(pair, inputs, rv.TextureReader(self.source, pair['rest']['units']), np.arange(64),
                            np.zeros(3), normals)
        base = np.stack([inputs['VB.POS.' + c] for c in 'xyz'], 1) / 4096
        return result, base

    def test_screen_space_glitch_is_view_dependent(self):
        result, _base = self.at_rest('characters/hero/hero_clank/materials/hero_clank_body_dimensional.material')
        self.assertIsNone(result)

    def test_program_without_rest_offset_keeps_the_stored_vertices(self):
        result, base = self.at_rest('material/environment/megalopolis/gameplay/meg_gp_bounceballoon/'
                                    'meg_gp_bounceballoon_top.material')
        np.testing.assert_allclose(result[0], base, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
