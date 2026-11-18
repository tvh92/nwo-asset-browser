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


if __name__ == '__main__':
    unittest.main()
