"""Rift Apart textures and material slots: synthetic assets, plus real-install checks (RCRA_GAME)."""
import os
import struct
import tempfile
import unittest
from pathlib import Path

from nwo.app.browser import rule_text
from nwo.games.rcra import material as rmat
from nwo.games.rcra import texture as rt

GAME = os.environ.get('RCRA_GAME')


def dat1(kind, sections, tail=b''):
    header = 16 + 12 * len(sections)
    table, body = b'', b''
    for tag, data in sections.items():
        table += struct.pack('<III', tag, header + len(body), len(data))
        body += data
    return struct.pack('<4sIIHH', b'1TAD', kind, header + len(body), len(sections), 0) + table + body + tail


def texture_asset(fmt=71, sd=(8, 8), sd_mips=2, hd=(16, 16), hd_mips=1, array=1, faces=1):
    sd_len = sum(rt.chain_sizes(fmt, *sd, sd_mips)) * array * faces
    hd_len = sum(rt.chain_sizes(fmt, *hd, hd_mips)) * array * faces if hd_mips else 0
    header = bytearray(44)
    struct.pack_into('<IIHHHHHBBH', header, 0, sd_len, hd_len, *(hd if hd_mips else sd), *sd, array, 16, 0, fmt)
    header[30], header[32] = sd_mips, hd_mips
    payload = bytes(i % 251 for i in range(sd_len))
    body = dat1(rt.TEXTURE_MAGIC, {rt.HEADER_SECTION: bytes(header)})
    return struct.pack('<III', rt.TEXTURE_MAGIC, len(body), sd_len) + bytes(24) + body + payload, hd_len


def material_asset(slots):
    strings, records = b'', b''
    for key, path in slots:
        records += struct.pack('<II', len(strings), key)
        strings += path.encode() + b'\0'
    start = 40
    end = start + len(records)
    head = struct.pack('<10I', end + len(strings), 0, 0, 0, start, len(slots), start, end, 0, 0)
    body = dat1(rmat.MATERIAL_MAGIC, {rmat.FLAGS: struct.pack('<I', 5) + bytes(36), rmat.SERIALIZED: head + records + strings})
    return struct.pack('<III', rmat.MATERIAL_MAGIC, len(body), 0) + bytes(24) + body


class FakeSource:
    """The slice of RcraSource the texture/material functions use."""

    def __init__(self, assets, streams=None, refs=None):
        self.assets, self.streams, self.refs = assets, streams or {}, refs or {}
        self.entries = {p: dict(type=p.rsplit('.', 1)[-1]) for p in assets}

    def resolve(self, path):
        return [path] if path in self.assets else []

    def asset_data(self, key):
        return self.assets[key]

    def texture_stream(self, key):
        return self.streams.get(key)

    def asset_references(self, key):
        return [dict(path=p) for p in self.refs.get(key, ())]


class TextureTests(unittest.TestCase):
    def test_header_and_sd_top_mip(self):
        data, _hd = texture_asset(hd_mips=0)
        info = rt.parse_header(data)
        self.assertEqual((info['format'], info['sd_width'], info['sd_mips'], info['faces'], info['stream_size']), (71, 8, 2, 1, 0))
        source = FakeSource({'t.texture': data})
        texture = rt.read_top_mip(source, 't.texture', full=True)
        self.assertEqual((texture['width'], texture['height'], len(texture['pixels'])), (8, 8, 32))
        self.assertEqual(texture['pixels'], data[info['payload_offset']:info['payload_offset'] + 32])

    def test_hd_stream_is_used_for_full_and_small_sd(self):
        data, hd_len = texture_asset()
        stream = bytes(range(hd_len % 256)) * 0 + bytes((i * 3) % 256 for i in range(hd_len))
        source = FakeSource({'t.texture': data}, {'t.texture': stream})
        texture = rt.read_top_mip(source, 't.texture', full=False)    # SD 8 px < MIN_PREVIEW_SIZE: stream
        self.assertEqual((texture['width'], texture['pixels']), (16, stream[:128]))
        self.assertEqual(texture['full_size'], (16, 16))
        self.assertEqual(rt.texture_header(source, 't.texture'),
                         dict(format=71, hd_width=16, hd_height=16, sd_width=8, sd_height=8, array_size=1))

    def test_arrays_and_cube_faces_are_slices(self):
        data, _ = texture_asset(hd_mips=0, array=2)
        self.assertEqual(len(rt.read_top_mip(FakeSource({'a.texture': data}), 'a.texture', True)['slices']), 2)
        data, _ = texture_asset(hd_mips=0, faces=6)
        info = rt.parse_header(data)
        self.assertEqual(info['faces'], 6)
        self.assertEqual(len(rt.read_top_mip(FakeSource({'c.texture': data}), 'c.texture', True)['slices']), 6)

    def test_rejects_wrong_sizes(self):
        data, _ = texture_asset(hd_mips=0)
        with self.assertRaisesRegex(ValueError, 'bytes after its header'):
            rt.parse_header(data + b'x')
        with self.assertRaises(ValueError):
            rt.parse_header(b'\0' * 80)

    def test_data_formats_size_but_do_not_preview(self):
        data, _ = texture_asset(fmt=16, hd_mips=0)
        self.assertEqual(rt.parse_header(data)['sd_size'], (64 + 16) * 8)
        with self.assertRaisesRegex(ValueError, 'not previewable'):
            rt.load_texture(FakeSource({'v.texture': data}), 'v.texture')

    def test_unit_float_and_16_bit_textures_convert_to_rgba8(self):
        from nwo.games.rcra import unit_texture as ut
        rg32f = dict(format=16, width=2, height=1, pixels=struct.pack('<4f', 0.0, 0.5, 128 / 255, 1.0))
        out = ut.as_rgba8(rg32f)
        self.assertEqual((out['format'], out['stored_format'], out['srgb']), (28, 16, False))
        self.assertEqual(list(out['pixels']), [0, 128, 0, 255, 128, 255, 0, 255])   # D3D defaults: B 0, A 1
        r16 = dict(format=56, width=2, height=1, pixels=struct.pack('<2H', 65535, 32896), slices=[struct.pack('<2H', 65535, 32896)])
        self.assertEqual(list(ut.as_rgba8(r16)['pixels']), [255, 0, 0, 255, 128, 0, 0, 255])
        with self.assertRaisesRegex(ValueError, 'from -1 to 2'):
            ut.unit_rgba(dict(format=41, width=2, height=1, pixels=struct.pack('<2f', -1.0, 2.0)))
        with self.assertRaisesRegex(ValueError, '1 non-finite'):
            ut.unit_rgba(dict(format=41, width=2, height=1, pixels=struct.pack('<2f', float('nan'), 0.0)))
        self.assertFalse(ut.is_data(28))

    def test_unit_data_textures_preview_and_export_as_16_bit_png(self):
        from nwo.games import rcra_game
        data, _ = texture_asset(fmt=35, hd_mips=0)   # R16G16_UNORM, 8x8 with 2 mips: 320 bytes
        top = struct.pack('<128H', *[i * 512 for i in range(128)])
        data = data[:-320] + top + bytes(64)
        source = FakeSource({'v.texture': data})
        texture = rcra_game.load_texture(source, 'v.texture')
        self.assertEqual((texture['format'], texture['stored_format'], texture['width']), (28, 35, 8))
        self.assertEqual(texture['pixels'][:8], bytes([0, 2, 0, 255, 4, 6, 0, 255]))   # round(i * 512 / 65535 * 255)
        with tempfile.TemporaryDirectory() as folder:
            rows = rt.export_textures(source, ['v.texture'], folder, {'png', 'dds'})
            self.assertIsNone(rows[0]['error'])
            png = (Path(folder) / 'v.png').read_bytes()
            self.assertEqual(png[24:26], bytes([16, 6]))   # 16-bit RGBA
            import zlib
            idat = png.index(b'IDAT')
            rows16 = zlib.decompress(png[idat + 4:idat + 4 + struct.unpack('>I', png[idat - 4:idat])[0]])
            self.assertEqual(struct.unpack('>4H', rows16[1:9]), (0, 512, 0, 65535))   # UNORM16 kept exactly
        half, _ = texture_asset(fmt=54, hd_mips=0, sd_mips=1)   # R16F, every value 2.0
        bad = half[:-128] + struct.pack('<64e', *[2.0] * 64)
        with self.assertRaisesRegex(ValueError, 'outside the 0..1'):
            rcra_game.load_texture(FakeSource({'b.texture': bad}), 'b.texture')
        with tempfile.TemporaryDirectory() as folder:
            row = rt.export_textures(FakeSource({'b.texture': bad}), ['b.texture'], folder, {'dds', 'png'})[0]
            self.assertIn('outside the 0..1', row['error'])
            self.assertEqual([Path(f).name for f in row['files']], ['b.dds'])   # the DDS keeps the stored values

    def test_export_writes_asset_stream_and_dds(self):
        data, hd_len = texture_asset()
        stream = bytes(hd_len)
        source = FakeSource({'x/t_c.texture': data}, {'x/t_c.texture': stream})
        with tempfile.TemporaryDirectory() as folder:
            rows = rt.export_textures(source, ['x/t_c.texture'], folder, {'texture', 'dds'})
            self.assertIsNone(rows[0]['error'])
            names = sorted(Path(f).name for f in rows[0]['files'])
            self.assertEqual(names, ['t_c.dds', 't_c.texture', 't_c.texture.stream'])
            self.assertEqual((Path(folder) / 't_c.texture.stream').read_bytes(), stream)


class EngineDefaultTests(unittest.TestCase):
    def test_engine_defaults_are_constant_rgba8(self):
        from nwo.games.rcra import blender_export as bexp
        for path, value, faces in ((rt.ENGINE_DEFAULT_2D, 128, 1), (rt.ENGINE_DEFAULT_3D, 255, 1),
                                   (rt.ENGINE_DEFAULT_CUBE, 128, 6)):
            texture = rt.read_top_mip(None, path, full=True)
            self.assertEqual((texture['width'], texture['height'], texture['format'], texture['faces']), (4, 4, 28, faces))
            self.assertEqual(len(texture['slices']), faces)
            self.assertEqual(set(texture['pixels']), {value})
            self.assertEqual(rt.texture_size(None, path), (4, 4, 1, 1))
            self.assertEqual(rt.texture_header(None, path)['format'], 28)
            self.assertTrue((bexp.texel_array(texture) == value / 255.0).all())
        self.assertEqual(bexp.engine_default(rmat.SLOT_TYPE_1D), rt.ENGINE_DEFAULT_2D)
        self.assertEqual(bexp.engine_default(rmat.SLOT_TYPE_3D), rt.ENGINE_DEFAULT_3D)
        self.assertEqual(bexp.engine_default(rmat.SLOT_TYPES_CUBE[1]), rt.ENGINE_DEFAULT_CUBE)
        self.assertEqual(bexp.engine_default(0x49A7866B), rt.ENGINE_DEFAULT_2D)


class MaterialTests(unittest.TestCase):
    def setUp(self):
        self.tex, _ = texture_asset(hd_mips=0)
        self.material = material_asset([(rmat.SLOT_G, 'm/a_g.texture'), (rmat.SLOT_NORMAL, 'm/a_n.texture'),
                                        (rmat.SLOT_BASE_COLOUR, 'm/a_c.texture')])
        self.source = FakeSource({'m/a.material': self.material, 'm/a_c.texture': self.tex, 'm/a_n.texture': self.tex},
                                 refs={'m/a.material': ['g/basic.materialgraph', 'm/a_c.texture'],
                                       'g/basic.materialgraph': ['g/default_c.texture']})
        self.source.entries['g/basic.materialgraph'] = dict(type='materialgraph')
        self.source.assets['g/basic.materialgraph'] = b''

    def test_slots_and_flags(self):
        self.assertEqual(rmat.material_slots(self.material)[2], (rmat.SLOT_BASE_COLOUR, 'm/a_c.texture'))
        self.assertEqual(rmat.material_flags(self.material), 5)

    def test_preview_binds_only_colour_and_normal(self):
        cache = {}
        entries = rmat.material_textures(self.source, [dict(index=0, path='m/a.material', name='a')], cache)
        entry = entries[0]
        self.assertEqual(entry['_c']['path'], 'm/a_c.texture')
        self.assertEqual(entry['_n']['path'], 'm/a_n.texture')
        self.assertIsNone(entry['_g'])
        self.assertTrue(entry['rcra_preview'])
        self.assertIn('m/a_c.texture', cache)

    def test_texture_list_and_rules(self):
        rows = {r['path']: r for r in rmat.model_texture_list(self.source, [dict(index=0, path='m/a.material', name='a')])}
        self.assertEqual(set(rows), {'m/a_g.texture', 'm/a_n.texture', 'm/a_c.texture', 'g/default_c.texture'})
        self.assertTrue(rows['g/default_c.texture']['via_graph'])
        rules = rmat.material_rules(self.source, [dict(index=0, path='m/a.material')])
        self.assertEqual(rules['m/a.material']['rule_kind'], rmat.RULE_KIND)
        self.assertIn('Rift Apart preview', rule_text(rules['m/a.material']))
        self.assertIn('basic', rule_text(rules['m/a.material']))


@unittest.skipUnless(GAME, 'RCRA_GAME is not set')
class RealTextureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nwo.games.rcra.source import RcraSource
        cls.temp = tempfile.TemporaryDirectory()
        cls.source = RcraSource(GAME, Path(cls.temp.name) / 'out')

    @classmethod
    def tearDownClass(cls):
        cls.source.close()
        cls.temp.cleanup()

    def test_ratchet_boots(self):
        material = 'material/characters/hero/hero_ratchet_boots/hero_ratchet_boots.material'
        slots = dict(rmat.material_slots(self.source.asset_data(self.source.resolve(material)[0])))
        self.assertTrue(slots[rmat.SLOT_BASE_COLOUR].endswith('hero_ratchet_boots_c.texture'))
        texture = rt.load_texture(self.source, slots[rmat.SLOT_BASE_COLOUR], hd=True)
        self.assertEqual((texture['width'], texture['height'], texture['format'], texture['srgb']), (2048, 2048, 99, True))
        from nwo.viewer.viewport import ModelViewport
        levels = ModelViewport._mip_chain_levels(texture)   # the game's own chain, every level the size BC7 needs
        self.assertEqual(levels[0][:2], (1024, 1024))
        self.assertEqual([w for w, _h, _p in levels], [1024 >> i for i in range(8)])   # HD then SD mips, down to 8 x 8
        normal = rt.load_texture(self.source, slots[rmat.SLOT_NORMAL], hd=True)
        self.assertFalse(normal['srgb'])

    def test_cube_map(self):
        info = rt.parse_header(self.source.asset_data(self.source.resolve('required/atmospheres/default_day/default_day.texture')[0]))
        self.assertEqual(info['faces'], 6)


class PreviewMipTests(unittest.TestCase):
    def test_preview_mip_is_the_first_within_the_cap(self):
        self.assertEqual(rt.preview_mip(2048, 1024, 12, cap=0), 0)
        self.assertEqual(rt.preview_mip(2048, 1024, 12, cap=512), 2)
        self.assertEqual(rt.preview_mip(512, 512, 10, cap=512), 0)
        self.assertEqual(rt.preview_mip(2048, 2048, 2, cap=512), 1)   # the chain ends first: its last mip
        self.assertEqual(rt.preview_mip(2048, 2048, 1, cap=512), 0)

    def test_capped_read_takes_the_smaller_mip_of_each_slice(self):
        data, _ = texture_asset(sd=(16, 16), sd_mips=3, hd_mips=0, array=2)
        info = rt.parse_header(data)
        payload = data[info['payload_offset']:]
        sizes = rt.chain_sizes(71, 16, 16, 3)
        source = FakeSource({'t.texture': data})
        saved = rt.MAX_PREVIEW_SIZE
        rt.MAX_PREVIEW_SIZE = 4
        try:
            texture = rt.read_top_mip(source, 't.texture', full=False)
        finally:
            rt.MAX_PREVIEW_SIZE = saved
        stride = sum(sizes)
        self.assertEqual((texture['width'], texture['height'], texture['full_size']), (4, 4, (16, 16)))
        self.assertEqual(texture['slices'], [payload[i * stride + sizes[0] + sizes[1]:(i + 1) * stride] for i in range(2)])
        self.assertEqual(rt.read_top_mip(source, 't.texture', full=False)['width'], 16)   # uncapped: the top mip


class MipChainTests(unittest.TestCase):
    def test_hd_stream_then_the_sd_chain(self):
        data, hd_len = texture_asset(hd_mips=2, hd=(32, 32))   # HD 32 and 16, SD 8 and 4
        info = rt.parse_header(data)
        stream = bytes((i * 7) % 256 for i in range(hd_len))
        source = FakeSource({'t.texture': data}, {'t.texture': stream})
        texture = rt.load_texture(source, 't.texture', hd=True)
        sd = data[info['payload_offset']:]
        self.assertEqual(texture['width'], 32)
        self.assertEqual(texture['mip_chain'], [stream[512:640], sd[:32], sd[32:40]])
        self.assertNotIn('mip_chain', rt.read_top_mip(source, 't.texture', full=True))   # export reads stay top-mip only

    def test_sd_only_and_arrays(self):
        data, _ = texture_asset(sd=(16, 16), sd_mips=3, hd_mips=0)
        sd = data[rt.parse_header(data)['payload_offset']:]
        self.assertEqual(rt.load_texture(FakeSource({'t.texture': data}), 't.texture')['mip_chain'], [sd[128:160], sd[160:168]])
        data, _ = texture_asset(sd=(16, 16), sd_mips=3, hd_mips=0, array=2)
        self.assertNotIn('mip_chain', rt.load_texture(FakeSource({'a.texture': data}), 'a.texture'))

    def test_a_stream_the_sd_chain_does_not_continue_stops_at_its_own_mips(self):
        data, hd_len = texture_asset(hd_mips=1, hd=(32, 32))   # HD 32 only; SD starts at 8, not 16
        source = FakeSource({'t.texture': data}, {'t.texture': bytes(hd_len)})
        self.assertNotIn('mip_chain', rt.load_texture(source, 't.texture', hd=True))

    def test_viewport_accepts_only_matching_level_sizes(self):
        from nwo.viewer.viewport import ModelViewport
        data, hd_len = texture_asset(hd_mips=2, hd=(32, 32))
        texture = rt.load_texture(FakeSource({'t.texture': data}, {'t.texture': bytes(hd_len)}), 't.texture', hd=True)
        levels = ModelViewport._mip_chain_levels(texture)
        self.assertEqual([(w, h, len(p)) for w, h, p in levels], [(16, 16, 128), (8, 8, 32), (4, 4, 8)])
        self.assertIsNone(ModelViewport._mip_chain_levels(dict(texture, width=64, height=64)))


if __name__ == '__main__':
    unittest.main()
