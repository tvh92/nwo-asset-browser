"""Rift Apart materials the plain model-instance draw never draws: a blended material whose alpha (flags block f32 5) is
not above 0 (rc2016.games.rcra.material.draw_skip_note), left out by the viewer and the model export.

Real-install tests read RCRA_GAME and skip without it.
"""
import os
import struct
import unittest
from pathlib import Path

from rc2016.games.rcra import material as rmat

GAME = os.environ.get('RCRA_GAME')
ROOT = Path(__file__).resolve().parents[1]
HEAD = 'characters/hero/hero_ratchet/bangle_set/bangle_head_default/hero_ratchet_bangle_head_default.model'
SHELL = 'material/characters/hero/hero_ratchet_compositeshell/hero_ratchet_compositeshell.material'


def material_bytes(flags, alpha):
    """A material asset holding only the 40-byte flags block (section 0xE1275683)."""
    words = [0, 0, flags, 0, 0x0EB42269, struct.unpack('<I', struct.pack('<f', alpha))[0], 0, 0, 0x80, 0]
    block = struct.pack('<10I', *words)
    offset = 16 + 12
    dat1 = b'1TAD' + struct.pack('<IIHH', rmat.MATERIAL_MAGIC, offset + len(block), 1, 0)
    dat1 += struct.pack('<III', rmat.FLAGS, offset, len(block)) + block
    return bytes(36) + dat1


class DrawSkipRuleTests(unittest.TestCase):
    def test_alpha_word(self):
        self.assertEqual(rmat.material_alpha(material_bytes(0x1914, 0.0)), 0.0)
        self.assertAlmostEqual(rmat.material_alpha(material_bytes(0x1110, 0.5)), 0.5)
        self.assertEqual(rmat.material_alpha(bytes(36) + b'1TAD' + struct.pack('<IIHH', rmat.MATERIAL_MAGIC, 16, 0, 0)), 1.0)

    def test_only_blended_paths_with_alpha_zero_are_skipped(self):
        cases = ((0x1914, 0.0, True),     # compositeshell: blended, alpha 0
                 (0x880, 0.0, True),      # additive, alpha 0 (the hidden rift shards)
                 (0x1110, float('nan'), True),
                 (0x1110, 0.05, False),   # a faint blended material is still drawn
                 (0x1810, 0.0, False),    # opaque: the draw at full instance alpha takes the opaque path
                 (0x918, 0.0, False),     # bit 0x8 keeps it opaque
                 (0x21110, 0.0, False),   # distance fade (0x20000) may lift it above 0: not decided here
                 (0x5110, 0.0, False))    # fur skips the test
        for flags, alpha, skipped in cases:
            note = rmat.draw_skip_note(material_bytes(flags, alpha))
            self.assertEqual(bool(note), skipped, (hex(flags), alpha))
            if note:
                self.assertIn('Not drawn', note)

    def test_draw_alpha_is_the_packed_material_alpha(self):
        # the draw packs trunc(alpha x 4096) into 16 bits; the vertex program reads it back as TEXCOORD.w
        for alpha, expected in ((1.0, 1.0), (0.5, 0.5), (0.3, 1228 / 4096), (0.99, 4055 / 4096), (0.005, 20 / 4096)):
            self.assertEqual(rmat.draw_alpha(material_bytes(0x1110, alpha)), expected, alpha)
            self.assertEqual(rmat.draw_interpolants(material_bytes(0x1110, alpha)), {'TEXCOORD.w': expected})
        for alpha in (0.0, 0.0001, float('nan')):   # a zero field: the vertex program reads the records (profile 1)
            self.assertIsNone(rmat.draw_alpha(material_bytes(0x1110, alpha)), alpha)
            self.assertEqual(rmat.draw_interpolants(material_bytes(0x1110, alpha)), {})

    def test_technique_flags(self):
        def graph(*words):
            records = b''.join(bytes(8 * rmat.TECHNIQUE_SLOTS) + struct.pack('<6I', 0, 5, w, 0, 0x40800000, 0) for w in words)
            body = b'1TAD' + struct.pack('<IIHH', 0x1592401A, 28 + len(records), 1, 0)
            body += struct.pack('<III', rmat.GRAPH_SWITCHES, 28, len(records)) + records
            return bytes(36) + body
        self.assertEqual(rmat.technique_flags(graph(0x11210)), [0x11210])
        self.assertEqual(rmat.technique_flags(graph(0x8011810, 0x11210)), [0x8011810, 0x11210])
        self.assertTrue(rmat.forward_technique(graph(0x11210)))           # basic_eye
        self.assertFalse(rmat.forward_technique(graph(0x8011810)))        # basic_normal_gloss
        self.assertEqual(rmat.graph_switches(graph(0x11210, 0x11210)), {})

    def test_front_ends_leave_them_out(self):
        viewport = (ROOT / 'rc2016/viewer/viewport.py').read_text(encoding='utf-8')
        self.assertIn('.get("not_drawn")]', viewport)
        export = (ROOT / 'rc2016/export/models.py').read_text(encoding='utf-8')
        self.assertIn("decoded.get('undrawn_materials')", export)


@unittest.skipUnless(GAME, 'RCRA_GAME is not set')
class InstallDrawSkipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        from rc2016.games.rcra.source import RcraSource
        cls.temp = tempfile.TemporaryDirectory()
        cls.source = RcraSource(GAME, os.path.join(cls.temp.name, 'out'))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_ratchet_head_shell_is_not_drawn(self):
        from rc2016.games.rcra.model import decode_for_viewer
        key = next(k for k in self.source.resolve(HEAD) if self.source.entries[k]['type'] == 'model')
        decoded, _data = decode_for_viewer(self.source, key, HEAD, True)
        undrawn = list(decoded['undrawn_materials'])   # the model's own spelling (hero_Ratchet_...)
        self.assertEqual([p.lower() for p in undrawn], [SHELL])
        self.assertIn(decoded['undrawn_materials'][undrawn[0]], decoded['material_notes'][undrawn[0]])
        cache = {}
        entries = rmat.material_textures(self.source, decoded['materials'], cache)
        hidden = {i for i, e in entries.items() if e.get('not_drawn')}
        self.assertEqual(hidden, {m['index'] for m in decoded['materials'] if m['path'].lower() == SHELL})

    def test_named_hidden_materials_are_skipped(self):
        for path in ('material/global/invisible_projectile_model/invisible_projectile_model.material',
                     'material/visualeffect/global/gbl_fx_spark_random_3d/null_model.material',
                     SHELL):
            data = self.source.asset_data(self.source.resolve(path)[0])
            self.assertTrue(rmat.draw_skip_note(data), path)
        eyeshadow = 'material/characters/hero/hero_rivet_eyeshadow/hero_rivet_eyeshadow.material'
        self.assertIsNone(rmat.draw_skip_note(self.source.asset_data(self.source.resolve(eyeshadow)[0])))

    def test_eye_is_a_forward_technique(self):
        from rc2016.games.rcra import surface as rsurface
        from rc2016.games.rcra.texture import texture_size

        def graph(path):
            return self.source.asset_data(next(k for k in self.source.resolve(path)
                                               if self.source.entries[k]['type'] == 'materialgraph'))
        eye_graph = graph('required/materials/basic_eye.materialgraph')
        self.assertTrue(rmat.forward_technique(eye_graph))
        self.assertFalse(rmat.forward_technique(graph('required/materials/basic_normal_gloss.materialgraph')))
        eye = self.source.asset_data(self.source.resolve('material/characters/hero/hero_ratchet_eyes/ratchet_eye.material')[0])
        result = rsurface.surface(eye_graph, eye, texture_sizes=lambda p: texture_size(self.source, p))
        # the albedo is the forward program's diffuse coefficient: the eye texture at the refracted coordinate
        self.assertIn(rsurface.FORWARD_ALBEDO_NOTE, result['limitations'])
        self.assertIn('characters/hero/hero_ratchet/textures/ratchet_eye_c.texture', result['textures'].values())
        recipe = rsurface.fold_nan(rsurface.export_recipe(result))
        values = {name: 0.3 for name in recipe['inputs']}
        values.update({'VIEW.z': 2.0, 'SV_IsFrontFace': True})
        out = rsurface.evaluate_recipe(recipe, values, lambda node, args: [0.6, 0.4, 0.2, 1.0])
        self.assertEqual([round(out['albedo.' + c], 6) for c in 'rgb'], [0.6, 0.4, 0.2])

        # its own specular term: of the key light only, 0 without it, a bright highlight with N = L = -V
        spec = rsurface.forward_specular(eye_graph, eye, texture_sizes=lambda p: texture_size(self.source, p))
        self.assertIn(rsurface.FORWARD_SPECULAR_NOTE, spec['limitations'])
        self.assertTrue({'L.x', 'LIGHT.r'} <= set(spec['inputs']))
        self.assertFalse([n for n in spec['inputs'] if n.startswith('ENGINE.')])
        recipe = rsurface.fold_nan(rsurface.export_recipe(spec))
        values = {name: 0.0 for name in recipe['inputs']}
        values.update({'TEXCOORD2.z': 1.0, 'TEXCOORD3.x': 1.0, 'TEXCOORD3.w': 1.0, 'VIEW.z': -2.0, 'L.z': 1.0,
                       'SV_IsFrontFace': True})
        for scale, check in ((1.0, lambda v: self.assertGreater(v, 1.0)), (0.0, lambda v: self.assertEqual(v, 0.0))):
            values.update({'LIGHT.' + c: scale for c in 'rgb'})
            out = rsurface.evaluate_recipe(recipe, values, lambda node, args: [0.5, 0.5, 0.5, 1.0])
            for c in 'rgb':
                check(out['colour.' + c])

    def test_emissive_forward_technique_draws_its_unlit_part(self):
        """A forward technique whose colour has an unlit part (an emissive eye) carries it as the terms unlit.r/g/b."""
        from rc2016.games.rcra import surface as rsurface
        from rc2016.games.rcra.texture import texture_size
        key = self.source.resolve('material/characters/npc/npc_dino_large_horns/npc_dino_large_eye.material')[0]
        self.assertTrue(rmat._graphs(self.source, key)[0].endswith('/eye_emissive.materialgraph'))
        graph = self.source.asset_data(next(k for k in self.source.resolve(rmat._graphs(self.source, key)[0])
                                            if self.source.entries[k]['type'] == 'materialgraph'))
        sizes = lambda p: texture_size(self.source, p)   # noqa: E731
        result = rsurface.surface(graph, self.source.asset_data(key), texture_sizes=sizes)
        self.assertIn(rsurface.FORWARD_UNLIT_NOTE, result['limitations'])
        recipe = rsurface.fold_nan(rsurface.export_recipe(result))
        self.assertTrue({'unlit.r', 'unlit.g', 'unlit.b'} <= set(recipe['terms']))
        self.assertFalse([n for n in recipe['inputs'] if n.startswith(('ENGINE.', 'L.', 'LIGHT.', 'BACKGROUND.'))])
        values = {name: 0.3 for name in recipe['inputs']}
        values.update({'TEXCOORD2.z': 1.0, 'TEXCOORD3.x': 1.0, 'TEXCOORD3.w': 1.0, 'VIEW.z': -2.0, 'SV_IsFrontFace': True})
        out = rsurface.evaluate_recipe(recipe, values, lambda node, args: [0.5, 0.5, 0.5, 1.0])
        self.assertGreater(max(out['unlit.' + c] for c in 'rgb'), 0.0)
        self.assertTrue(all(out['unlit.' + c] >= 0.0 for c in 'rgb'))

    def test_forward_alpha_carries_the_draw_alpha(self):
        from rc2016.games.rcra import surface as rsurface
        from rc2016.games.rcra.texture import texture_size
        path = 'material/environment/megalopolis/architecture/meg_arch_gc_float/meg_arch_gc_float_glass_01.material'
        key = self.source.resolve(path)[0]
        data = self.source.asset_data(key)
        graph_path = rmat._graphs(self.source, key)[0]
        graph = self.source.asset_data(next(k for k in self.source.resolve(graph_path)
                                            if self.source.entries[k]['type'] == 'materialgraph'))
        self.assertAlmostEqual(rmat.material_alpha(data), 0.3, places=6)
        result = rsurface.forward(graph, data, texture_sizes=lambda p: texture_size(self.source, p))
        g = result['graph']
        constants = {g.nodes[i].get('value') for i in g.reachable([result['terms']['alpha']]) if g.nodes[i]['op'] == 'const'}
        self.assertIn(1228 / 4096, constants)
        self.assertTrue(any('draw alpha 0.299805' in line for line in result['limitations']))


if __name__ == '__main__':
    unittest.main()
