"""Rift Apart surface terms for Blender: unpacking checked against the writer's own G-buffer bits, the texture writer,
the export wiring, the blend and lit decode, forward recipes of blended materials, and (with Blender) the emitted node
trees baked against the recipes.

Real-install tests read RCRA_GAME; the Blender test also needs Blender (RC2016_BLENDER, else the default install path).
"""
import math
import os
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

from rc2016.games.rcra import blender_check, blender_export
from rc2016.games.rcra import surface as rsurface
from rc2016.shaders import sm5

GAME = os.environ.get('RCRA_GAME')
BLENDER = os.environ.get('RC2016_BLENDER') or r'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe'
GRAPHS = ('required/materials/basic_normal_gloss.materialgraph',
          'materialgraph/global/gbl_rgb_blend/gbl_rgb_blend.materialgraph',
          'materialgraph/characters/hero/hero_armor_palette/hero_armor_palette.materialgraph',
          'required/materials/basic_normal_gloss_emissive.materialgraph',
          'required/materials/snow_two_layer.materialgraph',   # a select between two packed words, material LOD records
          'materialgraph/global/gbl_rgb_blend/sav_rgb_blend_scale_sand_up_01.materialgraph')   # triplanar axis cases
CHECKED = 'material/environment/megalopolis/prop/meg_prop_plastic_trims/meg_prop_plastic_ship_002.material'
DOME = 'characters/enemy/enm_neo_nefarious_trooper/materials/enm_neo_nefarious_trooper_dome.material'
# (material, blend, lit) as the flags word decodes them.
BLENDS = ((DOME, 'blended', True), ('material/characters/npc/wine_glass/wine_glass.material', 'additive', True),
          ('material/visualeffect/global/gbl_glow_sphere_mask_01_w_depthsoftness/gbl_glow_sphere_mask_01_modulate.material',
           'modulate', False),
          ('material/visualeffect/global/exp_quad/exp_quad_thk_02_unlit.material', 'blended', False), (CHECKED, 'opaque', True))


class TexelTests(unittest.TestCase):
    def test_missing_channels_follow_d3d(self):
        two = dict(format=49, width=2, height=1, pixels=bytes([10, 20, 30, 40]))
        data = blender_export.texel_array(two)
        self.assertEqual(data.shape, (1, 2, 4))
        np.testing.assert_allclose(data[0, 0], [10 / 255, 20 / 255, 0, 1])
        floats = dict(format=16, width=1, height=1, pixels=np.array([0.25, -0.5], '<f4').tobytes())
        np.testing.assert_allclose(blender_export.texel_array(floats)[0, 0], [0.25, -0.5, 0, 1])

    def test_byte_atlas_is_the_float_atlas_rounded(self):
        """The viewer's 8-bit atlas, built in bytes, equals the float path it replaced (texel_array x 255, rounded)."""
        from rc2016.games.rcra import viewer_materials
        rng = np.random.default_rng(1)
        for fmt, size in ((49, 2), (28, 4)):   # R8G8 (no blue stored) and R8G8B8A8
            slices = [rng.integers(0, 256, 3 * 2 * size, dtype=np.uint8).tobytes() for _ in range(3)]
            texture = dict(format=fmt, width=3, height=2, pixels=slices[0], slices=slices)
            out, grid = viewer_materials._build_atlas(texture, 't/a')
            float_atlas = blender_export.volume_atlas([blender_export.texel_array(texture, k) for k in range(3)], grid['columns'])
            self.assertEqual(out['pixels'], np.round(np.clip(float_atlas, 0, 1) * 255).astype(np.uint8).tobytes())
            self.assertEqual((out['width'], out['height']), (float_atlas.shape[1], float_atlas.shape[0]))

    def test_cached_translation_keeps_infinite_and_nan_constants(self):
        from rc2016.games.rcra import viewer_materials
        value = dict(nodes=[dict(op='const', value=float('inf')), dict(op='const', value=float('-inf')),
                            dict(op='const', value=float('nan')), dict(op='const', value=0.5)], t=('a', 1.0))
        stored = viewer_materials._floats(value)
        import json
        back = viewer_materials._floats(json.loads(json.dumps(stored, allow_nan=False)), decode=True)
        self.assertEqual([n['value'] for n in back['nodes']][:2], [float('inf'), float('-inf')])
        self.assertTrue(math.isnan(back['nodes'][2]['value']))
        self.assertEqual(back['nodes'][3]['value'], 0.5)

    def test_slices(self):
        nodes = [dict(op='const', type='f', value=1.0), dict(op='input', type='f', name='TEXCOORD.x'),
                 dict(op='sample', binding='textureUnit5', args=[1, 1, 0]), dict(op='sample', binding='textureUnit6', args=[1, 1, 1])]
        self.assertEqual(blender_export._slices(dict(nodes=nodes)), {'textureUnit5': 1, 'textureUnit6': blender_export.ALL_SLICES})

    def test_bilinear_matches_texel_centres(self):
        image = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
        np.testing.assert_allclose(blender_check.bilinear(image, 0.25, 0.25), image[0, 0])
        np.testing.assert_allclose(blender_check.bilinear(image, 0.5, 0.25), (image[0, 0] + image[0, 1]) / 2)

    def test_blend_and_lit_bits(self):
        from unittest import mock
        from rc2016.games.rcra import material as rmat
        for flags, blend, lit in ((0x1810, 'opaque', 0x400), (0x1110, 'blended', 0), (0x887, 'additive', 0x400),
                                  (0x1211, 'modulate', 0), (0x918, 'opaque', 0x400), (None, 'opaque', 0)):
            with mock.patch.object(rmat, 'flags_word', return_value=flags):
                self.assertEqual(rmat.material_blend(b''), blend, hex(flags or 0))
                self.assertEqual(rmat.draw_flags(b''), lit, hex(flags or 0))
        with mock.patch.object(rmat, 'flags_word', return_value=0x180):
            with self.assertRaises(ValueError):
                rmat.material_blend(b'')

    def test_export_wiring(self):
        from rc2016.export.models import FORMATS
        from rc2016.games import rcra_game
        self.assertTrue(FORMATS['blend'][2])
        self.assertIn('blender_materials', rcra_game.GAME.hooks)


class CaseTests(unittest.TestCase):
    def test_integer_cases_become_selects_of_constants(self):
        """A triplanar axis index (integer selects and compares of two conditions) is replaced exactly."""
        g = sm5.Graph()
        x, y = g.input('TEXCOORD.x'), g.input('TEXCOORD.y')
        a, b = g.op('lt', x, y), g.op('lt', y, g.const('f', 0.25))
        axis = g.op('iadd', g.select(a, g.const('i', 2), g.const('i', 0)), g.select(b, g.const('i', 1), g.const('i', 0)))
        weight = g.op('itof', g.op('ineg', axis))
        colour = g.op('mul', g.select(g.op('ieq', axis, g.const('i', 3)), x, y), weight)
        terms = {'colour': colour}
        self.assertTrue({g.nodes[i]['op'] for i in g.reachable([colour])} - rsurface.PORTABLE_OPS)
        out = rsurface._enumerate_cases(g, terms)
        self.assertFalse({g.nodes[i]['op'] for i in g.reachable(list(out.values()))} - rsurface.PORTABLE_OPS)
        for u in (0.1, 0.2, 0.6):
            for v in (0.0, 0.3, 0.9):
                inputs = {'TEXCOORD.x': u, 'TEXCOORD.y': v}
                self.assertEqual(sm5.evaluate(g, [out['colour']], inputs)[out['colour']],
                                 sm5.evaluate(g, [colour], inputs)[colour], (u, v))

    def test_integer_expressions_of_inputs_stay(self):
        g = sm5.Graph()
        n = g.op('ftoi', g.input('TEXCOORD.x'))
        terms = {'colour': g.op('itof', g.op('iadd', n, g.const('i', 1)))}
        self.assertEqual(rsurface._enumerate_cases(g, terms), terms)

    def _in_float(self, g, terms):
        out = dict(terms)
        packed = False
        for _round in range(rsurface.INT_FLOAT_ROUNDS):
            out, changed, used = rsurface._integers_in_float(g, out)
            packed = packed or used
            if not changed:
                break
        self.assertFalse({g.nodes[i]['op'] for i in g.reachable(list(out.values()))} - rsurface.PORTABLE_OPS)
        return out, packed

    def test_remainder_of_constant_cases_folds(self):
        """urem by a non-power of two of a select tree of constants (a cell index tested against each case) folds exactly."""
        g = sm5.Graph()
        x = g.input('TEXCOORD.x')
        index = g.select(g.op('lt', x, g.const('f', 0.3)), g.const('i', 180),
                         g.select(g.op('lt', x, g.const('f', 0.6)), g.const('i', 156), g.const('i', 1001)))
        cell = g.op('urem', index, g.const('i', 12))
        colour = g.select(g.op('ieq', cell, g.const('i', 0)), g.const('f', 1.0),
                          g.op('itof', g.op('iadd', cell, g.const('i', 1))))
        out, packed = self._in_float(g, {'colour': colour})
        self.assertFalse(packed)
        for u in (0.1, 0.5, 0.9):
            self.assertEqual(sm5.evaluate(g, [out['colour']], {'TEXCOORD.x': u})[out['colour']],
                             sm5.evaluate(g, [colour], {'TEXCOORD.x': u})[colour], u)

    def test_select_chain_of_constants_becomes_a_lookup(self):
        """A table the compiler unrolled (select(eq(X, k), table[k], ...)) is one lookup node with the same values: the
        outer test of a repeated key wins, keys missing from the span fall through, and so do non-integer and outside X."""
        g = sm5.Graph()
        x = g.op('mul', g.input('TEXCOORD.x'), g.const('f', 1.0))
        rest = g.op('mul', x, g.const('f', 0.5))
        chain = rest
        keys = [k for k in range(3, 30) if k != 17] + [5]   # the inner 5 is shadowed by the outer one
        for k in reversed(keys):
            chain = g.select(g.op('eq', x, g.const('f', float(k))), g.const('f', float(k * k % 23)), chain)
        terms = {'colour': g.op('add', chain, g.input('TEXCOORD.y'))}
        recipe = rsurface.export_recipe(dict(graph=g, terms=terms, textures={}, inputs=[], limitations=[]))
        out = rsurface.lookup_tables(recipe)
        lookups = [n for n in out['nodes'] if n['op'] == 'lookup']
        self.assertEqual(len(lookups), 1)
        self.assertEqual((lookups[0]['low'], len(lookups[0]['table']), lookups[0]['present'][17 - 3]), (3, 27, False))
        self.assertLess(len(out['nodes']), 12)
        for u in [float(k) for k in range(-2, 34)] + [4.5, 17.25]:
            inputs = {'TEXCOORD.x': u, 'TEXCOORD.y': 0.25}
            self.assertEqual(rsurface.evaluate_recipe(out, inputs, None), rsurface.evaluate_recipe(recipe, inputs, None), u)

    def test_constant_nan_folds_by_d3d_rules(self):
        """A NaN profile constant (a normalised zero vector) times the normal, saturated, then a clip test: the folded
        recipe has no NaN left and evaluates the same."""
        g = sm5.Graph()
        nan = g.const('f', float('nan'))
        n = g.op('add', g.op('mul', g.input('TEXCOORD2.x'), nan), g.op('mul', g.input('TEXCOORD2.y'), nan))
        clip = g.op('lt', g.op('add', g.op('sat', n), g.const('f', -0.001)), g.const('f', 0.0))
        low = g.op('max', g.op('mul', nan, g.input('TEXCOORD.x')), g.input('TEXCOORD.y'))
        terms = {'clip': clip, 'low': low, 'never': g.op('eq', n, g.const('f', 0.0))}
        recipe = rsurface.export_recipe(dict(graph=g, terms=terms, textures={}, inputs=[], limitations=[]))
        out = rsurface.blender_recipe(recipe)
        reach = sm5.Graph()
        reach.nodes = out['nodes']
        live = reach.reachable(list(out['terms'].values()))
        self.assertFalse([i for i in live if out['nodes'][i]['op'] == 'const' and out['nodes'][i]['value'] != out['nodes'][i]['value']])
        inputs = {'TEXCOORD2.x': 0.0, 'TEXCOORD2.y': 1.0, 'TEXCOORD.x': 0.5, 'TEXCOORD.y': 0.25}
        self.assertEqual(rsurface.evaluate_recipe(out, inputs, None), rsurface.evaluate_recipe(recipe, inputs, None))
        self.assertEqual(rsurface.evaluate_recipe(out, inputs, None), {'clip': True, 'low': 0.25, 'never': False})

    def test_fields_packed_and_read_back_are_their_values(self):
        """A word packed from quantised floats (shifted fields, a bfi flag) and read back by and, ubfe and ushr."""
        g = sm5.Graph()
        x, y, z = g.input('TEXCOORD.x'), g.input('TEXCOORD.y'), g.input('TEXCOORD.z')
        q = lambda v, s, o: g.op('ftou', g.op('add', g.op('mul', v, g.const('f', s)), g.const('f', o)))  # noqa: E731
        word = g.op('or', g.op('ishl', q(g.op('sat', z), 127.0, 0.5), g.const('i', 25)),
                    g.op('or', g.op('bfi', g.const('i', 1), g.const('i', 18),
                                    g.op('b2i', g.op('lt', x, g.const('f', 0.0))), g.const('i', 0)),
                         g.op('or', g.op('ishl', q(y, 722.66, 256.0), g.const('i', 9)), q(x, 722.66, 256.0))))
        low = g.op('utof', g.op('and', word, g.const('i', 511)))
        mid = g.op('utof', g.op('ubfe', g.const('i', 9), g.const('i', 9), word))
        top = g.op('utof', g.op('ushr', word, g.const('i', 25)))
        flag = g.select(g.op('ine', g.op('ubfe', g.const('i', 1), g.const('i', 18), word), g.const('i', 0)),
                        g.const('f', 1.0), g.const('f', 0.0))
        terms = {'low': low, 'mid': mid, 'top': top, 'flag': flag}
        out, packed = self._in_float(g, terms)
        self.assertTrue(packed)
        for u, v, w in ((-0.3, 0.2, 0.5), (0.35, -0.35, 1.2), (0.0, 0.1, -0.4)):
            inputs = {'TEXCOORD.x': u, 'TEXCOORD.y': v, 'TEXCOORD.z': w}
            got = sm5.evaluate(g, list(out.values()), inputs)
            want = sm5.evaluate(g, list(terms.values()), inputs)
            self.assertEqual({k: got[out[k]] for k in out}, {k: want[terms[k]] for k in terms}, (u, v, w))


class VolumeTests(unittest.TestCase):
    def test_atlas_tiles_wrap(self):
        slices = [np.full((2, 3, 4), k, np.float32) + np.arange(3, dtype=np.float32)[None, :, None] for k in range(5)]
        atlas = blender_export.volume_atlas(slices, 3)
        self.assertEqual(atlas.shape, (2 * 4, 3 * 5, 4))
        tile = atlas[4:8, 5:10]                       # slice 4: row 1, column 1
        np.testing.assert_array_equal(tile[1:3, 1:4], slices[4])
        np.testing.assert_array_equal(tile[1:3, 0], slices[4][:, 2])   # left border is the right edge
        np.testing.assert_array_equal(tile[0, 1:4], slices[4][1])      # top border is the bottom edge

    @unittest.skipUnless(os.path.isfile(BLENDER), 'Blender is not installed')
    def test_blender_volume_lookup_is_trilinear(self):
        """A synthetic 3D texture, its depth coordinate running over several wraps: the bake matches trilinear wrap."""
        rng = np.random.default_rng(7)
        slices = [rng.random((6, 5, 4)).astype(np.float32) for _ in range(7)]
        texture = dict(format=28)
        nodes = [dict(op='input', type='f', name='TEXCOORD.x', args=[]), dict(op='input', type='f', name='TEXCOORD.y', args=[]),
                 dict(op='const', type='f', value=3.7, args=[]), None, None,
                 dict(op='const', type='f', value=0.0, args=[]), dict(op='const', type='f', value=1.0, args=[])]
        nodes[3] = dict(op='mul', type='f', args=[0, 2])
        nodes[4] = dict(op='sample', type='v', args=[0, 1, 3, 0], binding='textureUnit5', opcode='sample', dimension=5,
                        sampler=None, offsets=[], controls=None)
        nodes += [dict(op='channel', type='f', args=[4], index=c) for c in range(3)]   # 7, 8, 9
        terms = {'albedo.r': 7, 'albedo.g': 8, 'albedo.b': 9, 'f0.r': 5, 'f0.g': 5, 'f0.b': 5, 'gloss': 5, 'normal.x': 5,
                 'normal.y': 6, 'normal.z': 5, 'occlusion': 6, 'emission': 5, 'clip': 5}
        with tempfile.TemporaryDirectory() as tmp:
            columns = 3
            row = blender_export._save(blender_export.volume_atlas(slices, columns), texture, 'volume', Path(tmp) / 'units',
                                       'synthetic')
            row['volume'] = dict(depth=7, columns=columns, width=5, height=6)
            entry = dict(recipe=dict(version=1, nodes=nodes, terms=terms, textures={}, inputs=[], limitations=[]),
                         images={'textureUnit5': row})
            want = blender_check.expected(entry, 23)
            got = blender_check.bake(entry, BLENDER, 23, tmp)
        self.assertGreater(float(np.ptp(want['albedo'])), 0.3)
        self.assertLess(float(np.abs(got['albedo'][:, :3] - want['albedo']).max()), 3e-3)   # 8-bit atlas, float bake

    @unittest.skipUnless(os.path.isfile(BLENDER), 'Blender is not installed')
    def test_blender_division_by_a_constant_zero_is_infinite(self):
        """D3D x / 0 is sign(x) inf, so sat(x / 0) is a step; Blender's Divide alone would return 0."""
        nodes = [dict(op='input', type='f', name='TEXCOORD.x', args=[]), dict(op='const', type='f', value=-0.5, args=[]),
                 dict(op='const', type='f', value=0.0, args=[]), dict(op='const', type='f', value=1.0, args=[])]
        nodes.append(dict(op='add', type='f', args=[0, 1]))   # 4: u - 0.5
        nodes.append(dict(op='div', type='f', args=[4, 2]))   # 5: (u - 0.5) / 0
        nodes.append(dict(op='sat', type='f', args=[5]))      # 6
        terms = {'albedo.r': 6, 'albedo.g': 6, 'albedo.b': 6, 'f0.r': 2, 'f0.g': 2, 'f0.b': 2, 'gloss': 2, 'normal.x': 2,
                 'normal.y': 3, 'normal.z': 2, 'occlusion': 3, 'emission': 2, 'clip': 2}
        entry = dict(recipe=dict(version=1, nodes=nodes, terms=terms, textures={}, inputs=[], limitations=[]), images={})
        with tempfile.TemporaryDirectory() as tmp:
            want = blender_check.expected(entry, 9)
            got = blender_check.bake(entry, BLENDER, 9, tmp)
        self.assertEqual(sorted(set(want['albedo'][:, 0])), [0.0, 1.0])
        np.testing.assert_array_equal(got['albedo'][:, :3], want['albedo'])

    @unittest.skipUnless(os.path.isfile(BLENDER), 'Blender is not installed')
    def test_blender_srgb_units_filter_in_linear(self):
        """Between contrasting sRGB texels the GPU filters decoded values; the written unit makes Cycles do the same."""
        tex = np.array([[[0.1, 0.1, 0.1, 1], [0.9, 0.9, 0.9, 1]], [[0.9, 0.9, 0.9, 1], [0.1, 0.1, 0.1, 1]]], np.float32)
        nodes = [dict(op='const', type='f', value=0.0, args=[]), dict(op='const', type='f', value=1.0, args=[]),
                 dict(op='const', type='b', value=False, args=[]),
                 dict(op='sample', type='v', args=[0, 0, 0, 0], binding='textureUnit5', opcode='sample', dimension=3,
                      sampler=None, offsets=[], controls=None)]   # 3: the wrap corner, all four texels at 1/4
        nodes.append(dict(op='channel', type='f', args=[3], index=0))   # 4
        terms = {'albedo.r': 4, 'albedo.g': 4, 'albedo.b': 4, 'f0.r': 0, 'f0.g': 0, 'f0.b': 0, 'gloss': 0, 'normal.x': 0,
                 'normal.y': 1, 'normal.z': 0, 'occlusion': 1, 'emission': 0, 'clip': 2}
        with tempfile.TemporaryDirectory() as tmp:
            row = blender_export._save(tex, dict(format=29), 'flat', Path(tmp) / 'units', 'synthetic')
            entry = dict(recipe=dict(version=1, nodes=nodes, terms=terms, textures={}, inputs=[], limitations=[]),
                         images={'textureUnit5': row})
            want = blender_check.expected(entry, 5)
            got = blender_check.bake(entry, BLENDER, 5, tmp)
        decoded = blender_export._srgb_to_linear(np.round(np.array([0.1, 0.9]) * 255) / 255)
        self.assertAlmostEqual(float(want['albedo'][0, 0]), float(decoded.mean()), places=5)
        np.testing.assert_allclose(got['albedo'][:, :3], want['albedo'], atol=1e-6)

    def test_non_finite_sample_coordinates_are_undefined(self):
        """A coordinate divided by a neutral 0 (here u * inf) samples nowhere: the pixel is left out, not a crash."""
        nodes = [dict(op='input', type='f', name='TEXCOORD.x', args=[]), dict(op='input', type='f', name='TEXCOORD.y', args=[]),
                 dict(op='const', type='f', value=math.inf, args=[]), dict(op='const', type='f', value=0.0, args=[]),
                 dict(op='const', type='f', value=1.0, args=[])]
        nodes.append(dict(op='mul', type='f', args=[0, 2]))                                                # 5
        nodes.append(dict(op='sample', type='v', args=[5, 1, 3, 3], binding='textureUnit5', opcode='sample',
                          dimension=3, sampler=None, offsets=[], controls=None))                          # 6
        nodes += [dict(op='channel', type='f', args=[6], index=c) for c in range(3)]                      # 7, 8, 9
        terms = {'albedo.r': 7, 'albedo.g': 8, 'albedo.b': 9, 'f0.r': 3, 'f0.g': 3, 'f0.b': 3, 'gloss': 3, 'normal.x': 3,
                 'normal.y': 4, 'normal.z': 3, 'occlusion': 4, 'emission': 3, 'clip': 3}
        with tempfile.TemporaryDirectory() as tmp:
            row = blender_export._save(np.full((4, 4, 4), 0.5, np.float32), dict(format=28), 'flat', Path(tmp) / 'units',
                                       'synthetic')
            entry = dict(recipe=dict(version=1, nodes=nodes, terms=terms, textures={}, inputs=[], limitations=[]),
                         images={'textureUnit5': row})
            want = blender_check.expected(entry, 5)
        self.assertEqual(want['albedo'].shape, (25, 3))
        self.assertTrue(np.isnan(want['albedo']).all() and np.isnan(want['normal']).all())


def _q(x):
    return int(math.floor(min(max(x, 0.0), 1.0) * 255.0 + 0.5))


@unittest.skipUnless(GAME, 'RCRA_GAME is not set')
class RealSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rc2016.games.rcra.source import RcraSource
        cls.temp = tempfile.TemporaryDirectory()
        cls.source = RcraSource(GAME, Path(cls.temp.name) / 'out')

    @classmethod
    def tearDownClass(cls):
        cls.source.close()
        cls.temp.cleanup()

    def material_of(self, graph_path):
        for key in self.source.ids_of_type('material'):
            if any(r['path'] == graph_path for r in self.source.asset_references(key)):
                return key
        self.fail('no material uses ' + graph_path)

    def test_view_direction_writer_takes_the_forward_view(self):
        """The dimensional glass normalises the direction to the camera: under review values that is NaN, so it is translated
        with VIEW and CAMERA inputs, labelled, and no term depends on NaN."""
        graph_path = 'materialgraph/global/basic_glass_dimensional/basic_glass_dimensional.materialgraph'
        gkey = next(k for k in self.source.resolve(graph_path) if self.source.entries[k]['type'] == 'materialgraph')
        result = rsurface.surface(self.source.asset_data(gkey), self.source.asset_data(self.material_of(graph_path)))
        self.assertIn(rsurface.VIEW_NOTE, result['limitations'])
        self.assertTrue({'VIEW.x', 'VIEW.y'} <= set(result['inputs']))
        self.assertFalse(rsurface._reads_nan(result))

    def test_terms_repack_to_the_writers_bits(self):
        """Each term, packed again with the engine's formulas, gives the writer's own G-buffer words."""
        rng = np.random.default_rng(3)
        for graph_path in GRAPHS:
            gkey = next(k for k in self.source.resolve(graph_path) if self.source.entries[k]['type'] == 'materialgraph')
            result = rsurface.surface(self.source.asset_data(gkey), self.source.asset_data(self.material_of(graph_path)))
            g, terms, outputs = result['graph'], result['terms'], result['outputs']
            roots = list(terms.values()) + [outputs[('o', r, c)] for r in (0, 1) for c in range(4)]
            for _ in range(25):
                n = rng.normal(size=3)
                n /= np.linalg.norm(n)
                t = np.cross(n, rng.normal(size=3))
                t /= np.linalg.norm(t)
                inputs = defaultdict(float, {'TEXCOORD.x': rng.random(), 'TEXCOORD.y': rng.random(), 'SV_IsFrontFace': True,
                                             'TEXCOORD3.w': 1.0, **{'TEXCOORD2.' + a: n[i] for i, a in enumerate('xyz')},
                                             **{'TEXCOORD3.' + a: t[i] for i, a in enumerate('xyz')},
                                             **{'TEXCOORD4.' + a: 1.0 for a in 'xyzw'}})
                texels = defaultdict(lambda: [float(x) for x in rng.random(4)])
                values = sm5.evaluate(g, roots, inputs, lambda node, args: texels[node['binding']])
                v = {k: values[i] for k, i in terms.items()}
                word = {(r, c): values[outputs[('o', r, c)]] for r in (0, 1) for c in range(4)}
                self.assertEqual(word[(0, 2)], _q(math.sqrt(v['f0.r'])) << 8 | _q(math.sqrt(v['f0.g'])), graph_path)
                self.assertEqual(word[(0, 3)] & 0xFF, _q(v['gloss']), graph_path)
                self.assertEqual(word[(0, 3)] >> 8, _q(math.sqrt(v['f0.b'])), graph_path)
                weight = 1.0 / math.sqrt(8.0 * abs(v['normal.z']) + 8.0)
                for c, axis in ((0, 'normal.x'), (1, 'normal.y')):
                    packed = int(v[axis] * weight * rsurface.NORMAL_SCALE + 2048.0)
                    self.assertLessEqual(abs((word[(0, c)] & 0xFFF) - packed), 1, graph_path)
                self.assertEqual(bool(word[(0, 1)] & 0x8000), v['normal.z'] < 0, graph_path)
                for c, axis in enumerate(('albedo.r', 'albedo.g', 'albedo.b')):
                    self.assertAlmostEqual(min(max(word[(1, c)], 0.0), 1.0), v[axis], places=6)
                alpha = min(max(word[(1, 3)], 0.0), 1.0)
                if word[(0, 1)] & 0x2000:   # occlusion mode
                    self.assertAlmostEqual(alpha, min(max(v['occlusion'], 0.0), 1.0), places=6, msg=graph_path)
                    self.assertEqual(v['emission'], 0.0, graph_path)
                else:
                    self.assertEqual(v['occlusion'], 1.0, graph_path)
                    self.assertAlmostEqual(alpha, min(max(0.2 * math.log2(min(v['emission'], 31.0) + 1.0), 0.0), 1.0),
                                           places=5, msg=graph_path)

    def test_blend_modes_and_forward_recipes(self):
        from rc2016.games.rcra import material as rmat
        for path, blend, lit in BLENDS:
            data = self.source.asset_data(self.source.resolve(path)[0])
            self.assertEqual(rmat.material_blend(data), blend, path)
            self.assertEqual(bool(rmat.draw_flags(data)), lit, path)
        with tempfile.TemporaryDirectory() as tmp:
            entry = blender_export.material_entry(self.source, DOME, Path(tmp))
        self.assertEqual(entry['blend'], 'blended')
        self.assertIn('recipe', entry)
        self.assertTrue({'VIEW.x', 'VIEW.y', 'VIEW.z'} <= set(entry['forward']['inputs']))   # the rim reads the view

        def alpha(view):
            inputs = {'TEXCOORD.x': 0.3, 'TEXCOORD.y': 0.3, 'TEXCOORD2.x': 0.0, 'TEXCOORD2.y': 1.0, 'TEXCOORD2.z': 0.0,
                      'TEXCOORD3.x': 1.0, 'TEXCOORD3.y': 0.0, 'TEXCOORD3.z': 0.0, 'TEXCOORD3.w': 1.0, 'SV_IsFrontFace': True,
                      'VIEW.x': view[0], 'VIEW.y': view[1], 'VIEW.z': view[2], 'CAMERA.x': 0.0, 'CAMERA.y': 0.0, 'CAMERA.z': 0.0}
            terms = rsurface.evaluate_recipe(entry['forward'], inputs, lambda node, args: [0.5, 0.5, 0.5, 1.0])
            return terms['alpha']
        self.assertLess(alpha((0.0, -3.0, 0.0)), alpha((2.9, -0.3, 0.0)))   # glass is clearer face-on than at its rim

    def forward_of(self, path):
        key = self.source.resolve(path)[0]
        graph = next(r['path'] for r in self.source.asset_references(key) if r['path'].endswith('.materialgraph'))
        gkey = next(k for k in self.source.resolve(graph) if self.source.entries[k]['type'] == 'materialgraph')
        return rsurface.forward(self.source.asset_data(gkey), self.source.asset_data(key))

    def test_object_record_reads_the_neutral_object(self):
        result = self.forward_of('material/environment/clankpuzzles/cpz_plasma_ball.material')
        self.assertTrue(any('plain object' in line for line in result['limitations']))

    def test_scene_behind_splits_into_a_background_tint(self):
        result = self.forward_of('material/visualeffect/global/gbl_fire_heat_distortion_01/gbl_fire_heat_distortion_01.material')
        g, terms = result['graph'], result['terms']
        self.assertEqual([g.value(terms['background.' + c]) for c in 'rgb'], [1.0, 1.0, 1.0])   # passes the scene through
        self.assertEqual([g.value(terms['colour.' + c]) for c in 'rgb'], [0.0, 0.0, 0.0])
        self.assertTrue(any('distortion' in line for line in result['limitations']))

    def test_lit_material_passes_the_scene_behind_unlit(self):
        result = self.forward_of('material/visualeffect/global/gbl_shockwave_hemisphere_01/fx_ground_wave_01.material')
        g, terms = result['graph'], result['terms']
        self.assertTrue(any(rsurface.UNLIT_BACKGROUND_NOTE in line for line in result['limitations']))
        self.assertTrue(all(g.value(terms['background.' + c]) is None for c in 'rgb'))   # a tint expression, not 0
        self.assertNotIn('lit_background.r', terms)

    def test_lit_material_lights_the_scene_behind_as_albedo(self):
        result = self.forward_of('material/visualeffect/global/gbl_heat_distortion_01/gbl_heat_distortion_01_nodistancefade'
                                 '.material')
        g, terms = result['graph'], result['terms']
        self.assertTrue(any(rsurface.LIT_BACKGROUND_NOTE in line for line in result['limitations']))
        self.assertTrue(all(g.value(terms['lit_background.' + c]) is None for c in 'rgb'))
        recipe = rsurface.export_recipe(result)
        folded = rsurface.unit_irradiance(recipe)   # Blender: background + lit_background
        self.assertNotIn('lit_background.r', folded['terms'])
        inputs = defaultdict(float, {'TEXCOORD.x': 0.3, 'TEXCOORD.y': 0.6, 'SV_IsFrontFace': True, 'VIEW.y': -2.0})
        sample = lambda node, args: [0.5, 0.5, 0.5, 1.0]   # noqa: E731
        lit = rsurface.evaluate_recipe(recipe, inputs, sample)
        unit = rsurface.evaluate_recipe(folded, inputs, sample)
        for c in 'rgb':
            self.assertEqual(lit['background.' + c], 0.0)   # nothing passes unlit
            self.assertEqual(unit['background.' + c], lit['background.' + c] + lit['lit_background.' + c])

    def test_background_lit_by_specular_light_is_unsupported(self):
        with self.assertRaisesRegex(rsurface.Unsupported, 'more than one light factor'):
            self.forward_of('material/visualeffect/character/enemy/enm_fx_goo_decal_simple_alt/enm_cord_juice_decal'
                            '.material')

    @unittest.skipUnless(os.path.isfile(BLENDER), 'Blender is not installed')
    def test_blender_forward_nodes_bake_to_the_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = blender_check.check(self.source, DOME, BLENDER, tmp, size=13)
        self.assertLess(report['forward_colour']['max'], 1e-4)
        self.assertLess(report['forward_alpha']['max'], 1e-5)

    @unittest.skipUnless(os.path.isfile(BLENDER), 'Blender is not installed')
    def test_blender_nodes_bake_to_the_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = blender_check.check(self.source, CHECKED, BLENDER, tmp, size=19)
        for term in ('albedo', 'f0', 'normal'):
            self.assertLess(report[term]['max'], 1e-3, term)
        self.assertLessEqual(report['scalars']['max'], 2.5 / 255)   # 8-bit rounding ties
        self.assertEqual(report['alpha']['max'], 0.0)


if __name__ == '__main__':
    unittest.main()
