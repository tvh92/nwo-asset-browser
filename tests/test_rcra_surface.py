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

from nwo.games.rcra import blender_check, blender_export
from nwo.games.rcra import surface as rsurface
from nwo.games.rcra import texture as rtexture
from nwo.shaders import sm5

GAME = os.environ.get('RCRA_GAME')
FUR = 'material/characters/hero/hero_ratchet_head/hero_ratchet_head_fur.material'   # no graph: the engine fur writer
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
        from nwo.games.rcra import viewer_materials
        rng = np.random.default_rng(1)
        for fmt, size in ((49, 2), (28, 4)):   # R8G8 (no blue stored) and R8G8B8A8
            slices = [rng.integers(0, 256, 3 * 2 * size, dtype=np.uint8).tobytes() for _ in range(3)]
            texture = dict(format=fmt, width=3, height=2, pixels=slices[0], slices=slices)
            out, grid = viewer_materials._build_atlas(texture, 't/a')
            float_atlas = blender_export.volume_atlas([blender_export.texel_array(texture, k) for k in range(3)], grid['columns'])
            self.assertEqual(out['pixels'], np.round(np.clip(float_atlas, 0, 1) * 255).astype(np.uint8).tobytes())
            self.assertEqual((out['width'], out['height']), (float_atlas.shape[1], float_atlas.shape[0]))

    def test_cached_translation_keeps_infinite_and_nan_constants(self):
        from nwo.games.rcra import viewer_materials
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
        zero = [dict(op='const', type='f', value=0.0), dict(op='input', type='f', name='TEXCOORD.x'),
                dict(op='sample', binding='textureUnit7', args=[1, 1, 0]), dict(op='sample', binding='textureUnit7', args=[1, 1, 0]),
                dict(op='sample', binding='textureUnit8', args=[1, 1])]
        # slice 0, read once or twice, and a sample without a slice lane stay slice 0
        self.assertEqual(blender_export._slices(dict(nodes=zero)), {'textureUnit7': 0, 'textureUnit8': 0})

    def test_bilinear_matches_texel_centres(self):
        image = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
        np.testing.assert_allclose(blender_check.bilinear(image, 0.25, 0.25), image[0, 0])
        np.testing.assert_allclose(blender_check.bilinear(image, 0.5, 0.25), (image[0, 0] + image[0, 1]) / 2)

    def test_blend_and_lit_bits(self):
        from unittest import mock
        from nwo.games.rcra import material as rmat
        for flags, blend, lit in ((0x1810, 'opaque', 0x400), (0x1110, 'blended', 0), (0x887, 'additive', 0x400),
                                  (0x1211, 'modulate', 0), (0x918, 'opaque', 0x400), (None, 'opaque', 0)):
            with mock.patch.object(rmat, 'flags_word', return_value=flags):
                self.assertEqual(rmat.material_blend(b''), blend, hex(flags or 0))
                self.assertEqual(rmat.draw_flags(b''), lit, hex(flags or 0))
        with mock.patch.object(rmat, 'flags_word', return_value=0x180):
            with self.assertRaises(ValueError):
                rmat.material_blend(b'')

    def test_export_wiring(self):
        from nwo.export.models import FORMATS
        from nwo.games import rcra_game
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

    def test_loop_flag_in_a_float_register_is_pruned_where_unused(self):
        """An unrolled loop's exit flag kept in a float register (bits2f(b2i(counter >= 16)), NaN when set) and carried
        into a register the program reads only when the loop did not exit: the NaN is taken out, the values stay. A
        term that does read the flag keeps its NaN."""
        g = sm5.Graph()
        x = g.input('TEXCOORD.x')
        broke = g.op('lt', x, g.const('f', 0.5))
        counter = g.op('iadd', g.select(broke, g.const('i', 18), g.const('i', 0)), g.const('i', 1))
        flag = g.op('bits2f', g.op('b2i', g.op('ige', counter, g.const('i', 16))))
        reg = g.select(g.op('b_not', broke), g.op('mul', x, g.const('f', 2.0)), flag)
        dead = g.select(broke, x, g.op('add', reg, g.const('f', 1.0)))
        live = g.op('add', reg, g.const('f', 0.0))
        terms = {'dead': dead, 'live': live}
        out, _packed = self._in_float(g, terms)
        pruned = rsurface._prune_nan(g, out)
        nan = lambda i: rsurface._is_nan(g.nodes[i])  # noqa: E731
        self.assertTrue(any(nan(i) for i in g.reachable([out['dead']])))
        self.assertFalse(any(nan(i) for i in g.reachable([pruned['dead']])))
        self.assertTrue(any(nan(i) for i in g.reachable([pruned['live']])))
        for u in (0.1, 0.49, 0.5, 0.9):
            want = sm5.evaluate(g, list(terms.values()), {'TEXCOORD.x': u})
            got = sm5.evaluate(g, list(pruned.values()), {'TEXCOORD.x': u})
            for k in terms:
                a, b = got[pruned[k]], want[terms[k]]
                self.assertTrue(a == b or (a != a and b != b), (k, u, a, b))

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

    def test_summed_fields_high_masks_and_unsigned_tests(self):
        """A word summed from a shifted field and constant bits above 2^24 (the foliage writers' iadd(imul(q, 2^18), c)),
        read back by ubfe, a single high-bit mask, and an unsigned compare of a field."""
        g = sm5.Graph()
        y = g.input('TEXCOORD.y')
        field = g.op('ftou', g.op('add', g.op('mul', g.op('sat', y), g.const('f', 127.0)), g.const('f', 0.5)))
        word = g.op('iadd', g.op('imul', field, g.const('i', 1 << 18)), g.const('i', 0x580094C0))
        terms = {'low': g.op('utof', g.op('ubfe', g.const('i', 6), g.const('i', 6), word)),
                 'field': g.op('utof', g.op('ubfe', g.const('i', 7), g.const('i', 18), word)),
                 'bit': g.select(g.op('ine', g.op('and', word, g.const('i', 0x40000000)), g.const('i', 0)),
                                 g.const('f', 1.0), g.const('f', 0.0)),
                 'small': g.select(g.op('ult', g.op('ubfe', g.const('i', 7), g.const('i', 18), word), g.const('i', 64)),
                                   g.const('f', 1.0), g.const('f', 0.0))}
        out, packed = self._in_float(g, terms)
        self.assertTrue(packed)
        for v in (0.0, 0.3, 0.51, 1.0):
            got = sm5.evaluate(g, list(out.values()), {'TEXCOORD.y': v})
            want = sm5.evaluate(g, list(terms.values()), {'TEXCOORD.y': v})
            self.assertEqual({k: got[out[k]] for k in out}, {k: want[terms[k]] for k in terms}, v)


class FactorSplitTests(unittest.TestCase):
    """surface._factor_split: F0 + sum Y x L, exact, with L the maximal light subexpressions."""

    @staticmethod
    def _light(g):
        return rsurface._reads(g, lambda name: name.startswith(('ENGINE.', 'L.', 'LIGHT.')))

    @staticmethod
    def _total(g, split):
        f0, factors = split
        total = f0
        for light, y in factors.items():
            total = g.op('add', total, g.op('mul', y, light))
        return total

    def _same(self, g, a, b, names):
        for k, v in enumerate((0.0, 0.3, 0.7, 1.2)):
            inputs = {n: v + 0.11 * j for j, n in enumerate(names)}
            inputs['ENGINE.shadow'] = 1.0 if k % 2 else -1.0
            got, want = (sm5.evaluate(g, [x], inputs)[x] for x in (a, b))
            self.assertAlmostEqual(got, want, places=5)

    def test_select_on_a_light_condition_splits_per_coefficient(self):
        """select(shadowed, Y key + spec, spec): with expand the albedo Y stays a coefficient; without, the select is one
        factor."""
        g = sm5.Graph()
        albedo, key, spec = g.input('TEXCOORD.x'), g.input('LIGHT.r'), g.input('ENGINE.g_BrdfLookup.0')
        lit = g.op('lt', g.const('f', 0.0), g.input('ENGINE.shadow'))
        colour = g.select(lit, g.op('add', g.op('mul', albedo, key), spec), spec)
        self.assertEqual(list(rsurface._factor_split(g, colour, self._light(g))[1].values()), [g.const('f', 1.0)])
        split = rsurface._factor_split(g, colour, self._light(g), expand=True)
        self.assertIn(albedo, split[1].values())
        self._same(g, self._total(g, split), colour, ['TEXCOORD.x', 'LIGHT.r', 'ENGINE.g_BrdfLookup.0'])

    def test_products_of_light_terms_multiply_out(self):
        """(key x shadow) x (albedo x (1 - ao)): with expand the coefficient is the albedo; without, the product stays whole."""
        g = sm5.Graph()
        albedo, key, ao = g.input('TEXCOORD.x'), g.input('LIGHT.r'), g.input('ENGINE.ao')
        colour = g.op('mul', g.op('mul', key, g.input('ENGINE.shadow')),
                      g.op('mul', albedo, g.op('add', g.const('f', 1.0), g.op('neg', ao))))
        plain = rsurface._factor_split(g, colour, self._light(g))
        self.assertEqual(list(plain[1].values()), [g.const('f', 1.0)])
        expanded = rsurface._factor_split(g, colour, self._light(g), expand=True)
        self.assertTrue(all(g.value(y) is None for y in expanded[1].values()))   # every coefficient carries the albedo
        self._same(g, self._total(g, expanded), colour, ['TEXCOORD.x', 'LIGHT.r', 'ENGINE.ao'])

    def test_set_inputs_binds_the_review_direction(self):
        g = sm5.Graph()
        term = g.op('sat', g.op('add', g.input('L.y'), g.input('TEXCOORD.x')))
        bound = rsurface._review_direction(g, {'t': term})['t']
        names = {g.nodes[i]['name'] for i in g.reachable([bound]) if g.nodes[i]['op'] == 'input'}
        self.assertEqual(names, {'TEXCOORD.x'})
        self.assertAlmostEqual(sm5.evaluate(g, [bound], {'TEXCOORD.x': 0.0})[bound], rsurface.REVIEW_KEY_DIRECTION[1],
                               places=6)


class ConstantNormalTests(unittest.TestCase):
    def test_constant_words_invert_the_packing(self):
        for n in ((0.6, 0.0, 0.8), (-0.2, 0.7, -0.68556546), (0.70710677, 0.70710677, 0.0), (0.0, 0.0, 1.0)):
            weight = 1.0 / math.sqrt(8.0 * abs(n[2]) + 8.0)
            x, y = (int(c * weight * rsurface.NORMAL_SCALE + 2048.0) for c in n[:2])   # ftou truncates
            got = rsurface.constant_normal(x | 0x1000, y | (0x8000 if n[2] < 0 else 0) | 0x2000)   # flag bits ignored
            for a, b in zip(got, n):
                self.assertAlmostEqual(a, b, delta=2e-3)   # 12-bit quantisation


class FurLayerMapTests(unittest.TestCase):
    def test_layer_map_matches_the_start_up_routine(self):
        """fur.layer_map against the start-up routine's per-texel loop (x outer, y inner), for the first columns."""
        from nwo.games.rcra import fur
        f, size = np.float32, fur.LAYER_SIZE
        slices = [np.frombuffer(s, np.uint8).reshape(size, size) for s in fur.layer_map()]
        self.assertEqual((len(slices), slices[0].shape), (fur.LAYER_SLICES, (size, size)))
        state, lengths = fur.XORSHIFT_SEED, set()
        for x in range(3):
            for y in range(size):
                value, state = fur._xorshift128(state)
                r = f(value >> 8)
                n = min(int(f(f(r * r) * f(2.0 ** -43))) + 1, fur.LAYER_SLICES)
                lengths.add(n)
                for d in range(fur.LAYER_SLICES):
                    h = max(f(f(f(f(d + 1) + f(d + 1)) * (f(1.0) / f(n))) - f(1.0)), f(0.0))
                    level = int(f(min(max(f(1.0) - f(f(h * h) * f(0.8)), f(0.0)), f(1.0)) * f(255.0))) if d < n else 0
                    self.assertEqual(int(slices[d][y, x]), level, (x, y, d))
        self.assertGreater(len(lengths), 10)   # the columns cover many strand lengths


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

    @unittest.skipUnless(os.path.isfile(BLENDER), 'Blender is not installed')
    def test_blender_tangent_frame_on_front_and_back_faces(self):
        """The decoded tangent (glTF attribute _RCRA_TANGENT) rebuilt by the material from Blender's UV0 frame, baked in
        Cycles: a front-facing triangle, and one whose corner normals oppose its winding, which Cycles shades as a back
        face (as on the helper bot's folded triangles). The Normal Map node negates its output there."""
        import json
        import subprocess
        script = r'''
import json, sys
import bpy
import numpy as np
sys.path.insert(0, sys.argv[sys.argv.index('--') + 1])
from nwo.export.rcra_blender import TANGENT_ATTRIBUTE, _Emitter, store_tangent_frame
bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new('frame')
mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0), (3, 0, 0), (2, 1, 0)], [], [(0, 1, 2), (3, 4, 5)])
mesh.uv_layers.new(name='UVMap').data.foreach_set('uv', [0, 0, 1, 0, 0, 1] * 2)
mesh.normals_split_custom_set([(0, 0, 1)] * 3 + [(0, 0, -1)] * 3)
tangent = np.array([[0.8, 0.5, 0.2]] * 3 + [[0.6, -0.7, -0.3]] * 3)
tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
native = np.stack((tangent[:, 0], tangent[:, 2], -tangent[:, 1]), 1)   # Blender (x, y, z) = game (x, -z, y)
packed = np.column_stack((native, [2.0] * 3 + [-3.0] * 3)).astype(np.float32)
mesh.attributes.new(TANGENT_ATTRIBUTE, 'FLOAT_COLOR', 'POINT').data.foreach_set('color', packed.ravel())
assert store_tangent_frame(mesh)
obj = bpy.data.objects.new('frame', mesh)
bpy.context.scene.collection.objects.link(obj)
material = bpy.data.materials.new('probe')
material.use_nodes = True
tree = material.node_tree
tree.nodes.clear()
frame = _Emitter(tree, dict(nodes=[], terms={}), {}, ['UVMap'], None, tangent=True).frame()
out, emit = tree.nodes.new('ShaderNodeOutputMaterial'), tree.nodes.new('ShaderNodeEmission')
tree.links.new(emit.outputs[0], out.inputs['Surface'])
combine, remap = tree.nodes.new('ShaderNodeCombineXYZ'), tree.nodes.new('ShaderNodeVectorMath')
remap.operation = 'MULTIPLY_ADD'
remap.inputs[1].default_value = remap.inputs[2].default_value = (0.5, 0.5, 0.5)
tree.links.new(combine.outputs[0], remap.inputs[0])
tree.links.new(remap.outputs[0], emit.inputs['Color'])
mesh.materials.append(material)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 1
scene.cycles.device = 'CPU'
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
baked = {}
for name, sockets in (('tangent', frame['tangent']), ('w', [frame['sign']] * 3)):
    for k in range(3):
        for link in list(combine.inputs[k].links):
            tree.links.remove(link)
        tree.links.new(sockets[k], combine.inputs[k])
    attribute = mesh.color_attributes.new('bake_' + name, 'FLOAT_COLOR', 'CORNER')
    mesh.color_attributes.active_color = attribute
    bpy.ops.object.bake(type='EMIT', target='VERTEX_COLORS')
    values = np.empty(len(mesh.loops) * 4, np.float32)
    attribute.data.foreach_get('color', values)
    baked[name] = (values.reshape(-1, 4)[:, :3] * 2.0 - 1.0).tolist()
print('FRAME ' + json.dumps(dict(baked=baked, native=native.tolist())))
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'frame.py'
            path.write_text(script)
            run = subprocess.run([BLENDER, '--background', '--factory-startup', '--python-exit-code', '1', '--python',
                                  str(path), '--', str(Path(__file__).resolve().parents[1])],
                                 capture_output=True, text=True, timeout=600)
        line = next((x for x in run.stdout.splitlines() if x.startswith('FRAME ')), None)
        self.assertIsNotNone(line, run.stdout[-2000:] + run.stderr[-2000:])
        result = json.loads(line[6:])
        np.testing.assert_allclose(result['baked']['tangent'], result['native'], atol=0.01)
        np.testing.assert_allclose(np.asarray(result['baked']['w'])[:, 0], [2.0] * 3 + [-3.0] * 3, atol=0.01)

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
        from nwo.games.rcra.source import RcraSource
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

    def test_g_map_layout_follows_7107A2D8(self):
        """Below 0.5 (or in a graph without the key) the _g map holds gloss² in red and a grey F0 in green; at or above it F0
        in rgb and gloss in alpha."""
        from nwo.games.rcra import material as rmat, shaders as rshaders
        cases = {'material/environment/sargasso/prop/uwb_prop_tube_bubbles_01/uwb_prop_tube_bubbles_01_opaque.material': False,
                 'material/characters/hero/hero_rivet_weteye/hero_rivet_weteye.material': False,   # basic_normal_gloss, key 0
                 'material/environment/underwaterbase/gameplay/uwb_gp_fuel_crate_frost_01/uwb_gp_fuel_crate_frost_01.material': True}
        for path, rgb in cases.items():
            key = next(k for k in self.source.resolve(path) if self.source.entries[k]['type'] == 'material')
            graph_path = rmat._graphs(self.source, key)[0]
            graph = self.source.asset_data(next(k for k in self.source.resolve(graph_path)
                                                if self.source.entries[k]['type'] == 'materialgraph'))
            data = self.source.asset_data(key)
            rows, offsets, defaults = rshaders._layout(graph)
            default = defaults[offsets[0x7107A2D8] // 4] if 0x7107A2D8 in offsets else 0.0   # absent: the red and green layout
            mode = rmat.material_parameters(data).get(0x7107A2D8, (default,))[0]
            self.assertEqual(mode >= 0.5, rgb, path)
            unit = 'textureUnit%d' % (rshaders.FIRST_MATERIAL_UNIT + next(u for u, k, _ in rows if k == rmat.SLOT_G))
            result = rsurface.surface(graph, data, texture_sizes=lambda p: rtexture.texture_size(self.source, p))
            g, terms = result['graph'], result['terms']
            texel = (0.36, 0.25, 0.64, 0.81)
            values = sm5.evaluate(g, list(terms.values()), defaultdict(float, {'TEXCOORD2.z': 1.0, 'TEXCOORD3.x': 1.0,
                                                                               'TEXCOORD3.w': 1.0}),
                                  lambda node, args: list(texel) if node['binding'] == unit else [0.5, 0.5, 0.5, 1.0])
            v = {k: values[i] for k, i in terms.items()}
            gloss, f0 = (texel[3], texel[:3]) if rgb else (math.sqrt(texel[0]), (texel[1],) * 3)
            self.assertAlmostEqual(v['gloss'], gloss, delta=1 / 255, msg=path)
            for name, want in zip(('f0.r', 'f0.g', 'f0.b'), f0):
                self.assertAlmostEqual(v[name], want, delta=0.01, msg=path)

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
        from nwo.games.rcra import material as rmat
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
        from nwo.games.rcra.texture import texture_size
        return rsurface.forward(self.source.asset_data(gkey), self.source.asset_data(key),
                                texture_sizes=lambda texture: texture_size(self.source, texture))   # as the export does

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

    def surface_of(self, path):
        from nwo.games.rcra.texture import texture_size
        key = self.source.resolve(path)[0]
        graph = next(r['path'] for r in self.source.asset_references(key) if r['path'].endswith('.materialgraph'))
        gkey = next(k for k in self.source.resolve(graph) if self.source.entries[k]['type'] == 'materialgraph')
        return rsurface.surface(self.source.asset_data(gkey), self.source.asset_data(key),
                                texture_sizes=lambda texture: texture_size(self.source, texture))

    def test_fur_material_draws_the_engine_fur_writer(self):
        from nwo.games.rcra import fur, material
        from nwo.games.rcra.texture import texture_size
        key = self.source.resolve(FUR)[0]
        data = self.source.asset_data(key)
        self.assertEqual(material._graphs(self.source, key), [])
        record = fur.fur_record(data)
        self.assertEqual([p.rsplit('/', 1)[-1] for p in record[2]],
                         ['hero_ratchet_head_c.texture', 'hero_ratchet_head_n.texture', 'hero_ratchet_head_g.texture',
                          'hero_ratchet_head_fur_control.texture'])
        self.assertEqual(dict(material.material_slots(data))[material.SLOT_BASE_COLOUR], record[2][0])
        program = fur.fur_program(self.source, data)
        result = rsurface.surface(None, data, program=program,
                                  texture_sizes=lambda texture: texture_size(self.source, texture))
        g, terms = result['graph'], result['terms']
        self.assertEqual(set(result['textures']), {'g_BaseMap', 'g_FurGlossMap', 'g_FurControlMap'})
        self.assertIs(g.value(terms['clip']), False)   # the shell cut is not applied
        self.assertIn(fur.FUR_NOTE, result['limitations'])
        with tempfile.TemporaryDirectory() as tmp:
            entry = blender_export.material_entry(self.source, FUR, Path(tmp))
        self.assertIn('recipe', entry)

    def test_missing_textures_get_the_engine_default(self):
        from nwo.games.rcra.texture import UNBOUND, texture_header
        with tempfile.TemporaryDirectory() as tmp:
            # The glove's _c is not installed (its graph default is); the shadow's texture and its default are not.
            glove = blender_export.material_entry(
                self.source, 'material/character/hero/hero_ratchet_protosuit_glove/hero_ratchet_protosuit_glove.material', Path(tmp))
            shadow = blender_export.material_entry(
                self.source, 'material/characters/hero/hero_platform_shadow/hero_platform_shadow.material', Path(tmp))
        for entry in (glove, shadow):
            self.assertNotIn('error', entry)
            self.assertTrue(any('install does not hold' in line and 'Default 2D, 128/255' in line
                                for line in entry['limitations']))
        self.assertEqual(texture_header(self.source, UNBOUND)['format'], 2)

    def test_graph_named_by_file_name(self):
        from nwo.games.rcra import material
        key = self.source.resolve('material/environment/kalebo/ground/kal_gnd_floor_bare_01/'
                                  'meg_kalebo_float44_gnd_floor_bare_01.material')[0]
        self.assertEqual([p.rsplit('/', 1)[-1] for p in material._graphs(self.source, key)], ['basic_normal_gloss.materialgraph'])
        missing = self.source.resolve('material/environment/northern_forest/ground/nor_gnd_snow_01/nor_gnd_snow_01.material')[0]
        self.assertIn('not in the install', material.no_graph_reason(self.source.asset_data(missing)))

    def test_writer_with_a_fixed_normal(self):
        result = self.surface_of('material/environment/global/prop/gbl_prop_smash_button_glow_01/'
                                 'gbl_prop_smash_button_glow_02.material')
        g, terms = result['graph'], result['terms']
        normal = [g.value(terms['normal.' + c]) for c in 'xyz']
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in normal)), 1.0, places=3)
        self.assertEqual(g.value(terms['occlusion']), 1.0)   # y bit 13: occlusion mode, alpha 1

    def test_byte_scale_with_a_folded_factor(self):
        result = self.surface_of('source/material/environment/zurkon/prop/zur_gp_arena_energy_barrier_outer/'
                                 'zur_gp_arena_energy_barrier_outer.material')
        self.assertIn('gloss', result['terms'])

    def test_scene_read_other_than_as_a_tint_stays_an_input(self):
        result = self.forward_of('material/characters/npc/npc_zur_pub_hologram_01/npc_zur_pub_hologram_01.material')
        g, terms = result['graph'], result['terms']
        recipe = rsurface.export_recipe(result)
        self.assertTrue(rsurface.reads_scene(recipe))
        self.assertTrue(any('BACKGROUND.rgba' in line for line in result['limitations']))
        self.assertEqual([g.value(terms['background.' + c]) for c in 'rgb'], [0.0, 0.0, 0.0])
        with tempfile.TemporaryDirectory() as tmp:
            entry = blender_export.material_entry(
                self.source, 'material/characters/npc/npc_zur_pub_hologram_01/npc_zur_pub_hologram_01.material', Path(tmp))
        self.assertNotIn('forward', entry)   # Cycles cannot sample the scene behind
        self.assertTrue(any('Cycles has no node' in line for line in entry.get('limitations', [])) or 'error' in entry)

    def test_lit_particles_pass_the_scene_unlit_as_an_input(self):
        result = self.forward_of('material/visualeffect/environment/blizar_embers_atmo_01.material')
        self.assertTrue(rsurface.reads_scene(rsurface.export_recipe(result)))
        self.assertTrue(any(rsurface.UNLIT_BACKGROUND_NOTE in line for line in result['limitations']))

    def test_background_specular_factor_is_left_out(self):
        """The goo decal also passes the scene behind through a specular factor: the profile's zero specular light leaves
        it out, and the diffuse factor lights the scene behind as albedo."""
        result = self.forward_of('material/visualeffect/character/enemy/enm_fx_goo_decal_simple_alt/enm_cord_juice_decal'
                                 '.material')
        self.assertIn('lit_background.r', result['terms'])
        self.assertTrue(any(rsurface.LIT_BACKGROUND_SPECULAR_NOTE in line for line in result['limitations']))

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


class SharedForwardSplitTests(unittest.TestCase):
    def test_one_split_per_graph_and_material_inside_the_block(self):
        from unittest import mock
        outcome = ('g', 'tr', 'bindings', {}, [])
        with mock.patch.object(rsurface, '_split_forward', return_value=outcome) as split:
            rsurface._forward_split(b'graph', b'material', None)
            rsurface._forward_split(b'graph', b'material', None)
            self.assertEqual(split.call_count, 2)   # outside a block: no memo
            with rsurface.shared_forward_split():
                with rsurface.shared_forward_split():   # nested: the outer memo
                    self.assertIs(rsurface._forward_split(b'graph', b'material', None), outcome)
                self.assertIs(rsurface._forward_split(b'graph', b'material', None), outcome)
                rsurface._forward_split(b'graph', b'other', None)
            self.assertEqual(split.call_count, 4)
            rsurface._forward_split(b'graph', b'material', None)
            self.assertEqual(split.call_count, 5)   # the block ended: its memo is gone

    def test_a_failed_split_fails_again_without_translating(self):
        from unittest import mock
        with mock.patch.object(rsurface, '_split_forward', side_effect=rsurface.Unsupported('no')) as split:
            with rsurface.shared_forward_split():
                for _ in range(2):
                    with self.assertRaisesRegex(rsurface.Unsupported, 'no'):
                        rsurface._forward_split(b'g', b'm', None)
            self.assertEqual(split.call_count, 1)


if __name__ == '__main__':
    unittest.main()
