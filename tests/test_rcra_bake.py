"""Rift Apart material terms baked over UV0 for the portable export (rc2016.games.rcra.bake), plus a real-install check
(RCRA_GAME, skipped without it): the vectorised bake equals the scalar reference evaluator at random texels."""
import os
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rc2016.games.rcra import bake

GAME = os.environ.get('RCRA_GAME')


def node(op, *args, kind='f', **extra):
    return dict(op=op, type=kind, args=list(args), **extra)


def recipe(nodes, terms):
    return dict(nodes=nodes, terms=terms, inputs=sorted({n['name'] for n in nodes if n['op'] == 'input'}))


def uv_nodes():
    return [node('input', name='TEXCOORD.x'), node('input', name='TEXCOORD.y')]


def tinted(u=0, extra=()):
    """albedo = _c.rgb x (2, 1, 0.5) sampled at (nodes[u], v); gloss = _c.a; emission = 3 x _c.r."""
    nodes = uv_nodes() + list(extra)
    s = len(nodes)
    nodes += [node('sample', u, 1, kind='v', opcode='sample', binding='textureUnit5', dimension=3, offsets=[0, 0, 0])]
    nodes += [node('channel', s, index=i) for i in range(4)]
    nodes += [node('const', value=v) for v in (2.0, 1.0, 0.5, 3.0)]
    c, k = s + 1, s + 5
    nodes += [node('mul', c + i, k + i) for i in range(3)] + [node('mul', c, k + 3)]
    m = s + 9
    return recipe(nodes, {'albedo.r': m, 'albedo.g': m + 1, 'albedo.b': m + 2, 'gloss': c + 3, 'emission': m + 3})


def texture(texels, srgb=False):
    """A 2D RGBA8 texture dict of (h, w, 4) bytes."""
    texels = np.asarray(texels, np.uint8)
    return dict(format=29 if srgb else 28, width=texels.shape[1], height=texels.shape[0], pixels=texels.tobytes(),
                srgb=srgb)


def sampler(textures):
    s = bake.Sampler(None, {'textureUnit5': 'c.texture'})
    s.cache.update(textures)
    return s


class BlockerTests(unittest.TestCase):
    def test_uv0_terms_bake_and_other_inputs_say_why_not(self):
        r = tinted()
        self.assertIsNone(bake.blocker(r, ('albedo.r', 'gloss')))
        normal = recipe(uv_nodes() + [node('input', name='TEXCOORD2.y'), node('mul', 0, 2)], {'albedo.r': 3, 'gloss': 0})
        self.assertEqual(bake.blocker(normal, ('albedo.r',)), 'it reads the world normal')
        self.assertIsNone(bake.blocker(normal, ('gloss',)))   # only the terms asked for count
        uv1 = recipe([node('input', name='TEXCOORD1.x')], {'occlusion': 0})
        self.assertEqual(bake.blocker(uv1, ('occlusion',)), 'it reads UV1')
        front = recipe(uv_nodes() + [node('input', name='SV_IsFrontFace', kind='b'), node('select', 2, 0, 1)], {'gloss': 3})
        self.assertIsNone(bake.blocker(front, ('gloss',)))   # a single-sided material: front faces only
        self.assertEqual(bake.blocker(front, ('gloss',), two_sided=True),   # its back faces differ
                         'it reads the front-face flag of a double-sided material')
        self.assertEqual(bake.blocker(r, ('alpha',)), 'the program has no such term')
        self.assertEqual(bake.blocker(None, ('alpha',)), 'the program does not translate')

    def test_a_branch_weighted_by_zero_folds_out(self):
        # albedo = uv.x + uv1.x x 0 (a damage overlay at weight 0); gloss = rcp(uv1.x) x 0 (may be infinite: kept)
        nodes = uv_nodes() + [node('input', name='TEXCOORD1.x'), node('const', value=0.0), node('mul', 2, 3),
                              node('add', 0, 4), node('rcp', 2), node('mul', 6, 3), node('neg', 3), node('sat', 8)]
        r = recipe(nodes, {'albedo.r': 5, 'gloss': 7, 'occlusion': 9})
        self.assertEqual(bake.blocker(r, ('albedo.r',)), 'it reads UV1')
        folded = bake.fold(r)
        self.assertEqual(folded['terms']['albedo.r'], 0)   # x + 0 is x
        self.assertIsNone(bake.blocker(folded, ('albedo.r',)))
        self.assertEqual(bake.blocker(folded, ('gloss',)), 'it reads UV1')
        self.assertEqual(folded['nodes'][9], dict(op='const', type='f', value=0.0, args=[]))
        self.assertIsNone(bake.fold(None))
        self.assertEqual(r['nodes'][4]['op'], 'mul')   # the input recipe is not changed
        out = bake.bake(folded, ('albedo.r',), sampler({}), (2, 1))
        np.testing.assert_allclose(out['albedo.r'], [[0.25, 0.75]])

    def test_cube_maps_and_volumes_do_not_bake(self):
        r = tinted()
        r['nodes'][2]['dimension'] = 6
        self.assertIn('cube map', bake.blocker(r, ('albedo.r',)))
        r['nodes'][2]['dimension'] = 5
        self.assertIn('volume', bake.blocker(r, ('albedo.r',)))


class BakeTests(unittest.TestCase):
    def test_terms_at_the_texel_centres(self):
        texels = [[[10, 20, 30, 40], [50, 60, 70, 80]], [[90, 100, 110, 120], [130, 140, 150, 255]]]
        s = sampler({'c.texture': texture(texels)})
        out = bake.bake(tinted(), ('albedo.r', 'albedo.g', 'albedo.b', 'gloss', 'emission'), s,
                        bake.grid_size(s, ['textureUnit5']))
        c = np.asarray(texels, np.float32) / 255
        np.testing.assert_allclose(out['albedo.r'], c[..., 0] * 2, rtol=1e-6)
        np.testing.assert_allclose(out['albedo.b'], c[..., 2] * 0.5, rtol=1e-6)
        np.testing.assert_allclose(out['gloss'], c[..., 3], rtol=1e-6)
        np.testing.assert_allclose(out['emission'], c[..., 0] * 3, rtol=1e-6)
        self.assertEqual(s.read, {'c.texture'})

    def test_srgb_textures_decode_before_filtering(self):
        s = sampler({'c.texture': texture([[[188, 0, 0, 255]]], srgb=True)})
        out = bake.bake(tinted(), ('albedo.r',), s, (1, 1))
        self.assertAlmostEqual(float(out['albedo.r'][0, 0]), 2 * ((188 / 255 + 0.055) / 1.055) ** 2.4, places=5)

    def test_a_tiled_read_is_bilinear_with_wrap(self):
        """u x 1.5 on a 2-texel-wide texture (red 0 and 200) over a 4-texel grid: texel coordinates -0.125, 0.625, 1.375
        and 2.125, so 7/8 of texel 0 across the wrap, then 5/8, 3/8 and 1/8 of texel 1."""
        scale = [node('const', value=1.5), node('mul', 0, 2)]
        s = sampler({'c.texture': texture([[[0, 0, 0, 0], [200, 0, 0, 0]]])})
        out = bake.bake(tinted(u=3, extra=scale), ('albedo.r',), s, (4, 1))
        red = np.array([25, 125, 125, 25]) / 255
        np.testing.assert_allclose(out['albedo.r'][0], red * 2, rtol=1e-5)

    def test_lookup_nodes_are_tables(self):
        nodes = uv_nodes() + [node('const', value=4.0), node('mul', 0, 2), node('round_ni', 3), node('const', value=-1.0),
                              node('lookup', 4, 5, low=1, table=[10.0, 20.0, 0.0], present=[True, True, False])]
        out = bake.bake(recipe(nodes, {'gloss': 6}), ('gloss',), bake.Sampler(None, {}), (4, 1))
        self.assertEqual(out['gloss'][0].tolist(), [-1.0, 10.0, 20.0, -1.0])   # floor(4u) = 0, 1, 2, 3; 3 is not present

    def test_chunks_and_threads_give_the_same_values(self):
        rng = np.random.default_rng(3)
        s = sampler({'c.texture': texture(rng.integers(0, 256, (16, 16, 4)))})
        names = ('albedo.r', 'albedo.g', 'gloss')
        whole = bake.bake(tinted(), names, s, (16, 16))
        with patch.object(bake, 'CHUNK', 7), patch.object(bake, 'THREADS', 3):
            parts = bake.bake(tinted(), names, s, (16, 16))
        for name in names:
            np.testing.assert_array_equal(whole[name], parts[name])


@unittest.skipUnless(GAME, 'RCRA_GAME is not set')
class InstallTests(unittest.TestCase):
    def test_the_bake_equals_the_scalar_evaluator(self):
        import tempfile
        from rc2016.games import rcra_export
        from rc2016.games.rcra import surface as rsurface
        from rc2016.games.rcra.source import RcraSource
        with tempfile.TemporaryDirectory() as tmp:
            source = RcraSource(Path(GAME), Path(tmp))
            path = 'material/characters/hero/hero_ratchet_head/hero_ratchet_head.material'
            key = next(k for k in source.resolve(path) if source.entries[k]['type'] == 'material')
            d = rcra_export.decoded_maps(source, key)
            names = [n for g, why in d['bake'].items() if why is None and g != 'alpha' for n in rcra_export.BAKE_GROUPS[g]]
            self.assertIn('gloss', names)
            s = bake.Sampler(source, d['bindings'])
            size = (256, 256)
            values = bake.bake(d['surface'], names, s, size)
            rng = random.Random(1)
            for _ in range(8):
                x, y = rng.randrange(size[0]), rng.randrange(size[1])
                inputs = {'TEXCOORD.x': float(np.float32((x + 0.5) / size[0])),
                          'TEXCOORD.y': float(np.float32((y + 0.5) / size[1])), 'SV_IsFrontFace': True}
                ref = rsurface.evaluate_recipe(d['surface'], inputs, lambda n, a: [
                    float(c[0]) for c in s(n, [np.asarray([v]) for v in a])])
                for name in names:
                    self.assertAlmostEqual(float(values[name][y, x]), ref[name], places=5, msg=name)


if __name__ == '__main__':
    unittest.main()
