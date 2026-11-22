"""Rift Apart visual effects: synthetic .visualeffect sections, curve evaluation, and real-install checks of the weapons
vendor's effects and conduit (RCRA_GAME, skipped without it)."""
import os
import struct
import tempfile
import unittest

import numpy as np

from nwo.games.rcra import vfx
from nwo.games.rcra.toc import path_crc64_many

GAME = os.environ.get('RCRA_GAME')
HEADER = bytes(36)


def dat1(section_map, strings=b''):
    """A DAT1 of the given {tag: bytes}; ``strings`` sit in the string pool (offsets from the DAT1 start)."""
    head = 16 + 12 * len(section_map)
    offset = head + len(strings)
    offset += -offset % 16
    table, body = b'', b''
    for tag, data in section_map.items():
        body += bytes(-len(body) % 16)
        table += struct.pack('<III', tag, offset + len(body), len(data))
        body += data
    pool = strings + bytes(offset - head - len(strings))
    return struct.pack('<4sIIHH', b'1TAD', 0xF05EF819, offset + len(body), len(section_map), 0) + table + pool + body


def half_key(base, time, rng, mode):
    """One curve key: base xyz and a u16 time, range xyz and a mode half."""
    halves = np.array(list(base) + [0.0] + list(rng) + [mode], np.float16).view(np.uint16)
    halves[3] = time
    return halves.tobytes()


def node(kind, asset_id=0, material_id=0, emitter_hash=0, position=(0, 0, 0), scale=(1, 1, 1), slots=None, order=0):
    record = bytearray(vfx.RECORD)
    struct.pack_into('<QQI', record, 0, asset_id, material_id, emitter_hash)
    struct.pack_into('<9f', record, 0x20, 1, 0, 0, 0, 1, 0, 0, 0, 1)
    struct.pack_into('<3f', record, 0x44, *position)
    struct.pack_into('<3f', record, 0xC4, *scale)
    record[0x17A] = kind
    record[0x17F] = order
    for i, word in enumerate(slots or [0] * 8):
        struct.pack_into('<I', record, 0x180 + 8 * i, word)
    return bytes(record)


def emitter(name_hash, slots):
    record = bytearray(vfx.RECORD)
    struct.pack_into('<9f', record, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
    struct.pack_into('<3f', record, 0x24, 0.0, 2.0, 0.0)
    struct.pack_into('<I', record, 0x100, name_hash)
    for i, word in enumerate(slots):
        struct.pack_into('<I', record, 0x120 + 8 * i, word)
    return bytes(record)


def effect():
    """Two materials (one inline), one model, a SingleModel node with curves and a MultiQuads node on an emitter."""
    strings = b'VisualEffect Built File\0'
    offsets = {}
    for name in ('material\\fx\\glow.material', 'materialgraph\\fx\\flash.materialgraph',
                 'materialnode: material_2, materialgraph\\fx\\flash.materialgraph', 'objects\\fx\\plane.model'):
        offsets[name] = 16 + 12 * 7 + len(strings)
        strings += name.encode() + b'\0'
    materials = struct.pack('<Ii', offsets['material\\fx\\glow.material'], -1)
    materials += struct.pack('<Ii', offsets['materialgraph\\fx\\flash.materialgraph'],
                             offsets['materialnode: material_2, materialgraph\\fx\\flash.materialgraph'])
    assets = struct.pack('<III', offsets['objects\\fx\\plane.model'], 0, 0xA4070B70)
    glow, inline, plane = (int(v) for v in path_crc64_many([
        'material/fx/glow.material', 'materialnode: material_2, materialgraph/fx/flash.materialgraph',
        'objects/fx/plane.model']))
    # curve 0: constant colour; curve 1: random scalar; curve 2: alpha ramp 0 -> 1 -> 0 at times 0, 0.5, 1.
    keys = half_key((0, 0.5, 1), 0, (0, 0, 0), 1) + half_key((1, 1, 1), 0, (-0.5, -0.5, -0.5), 2)
    keys += half_key((0, 0, 0), 0, (0, 0, 0), 1) + half_key((1, 0, 0), 0x8000, (0, 0, 0), 1)
    keys += half_key((0, 0, 0), 0xFFFF, (0, 0, 0), 1)
    curves = struct.pack('<IIIf', 0x11, 0, 1, 1.0) + struct.pack('<IIIf', 0x22, 1, 1, 1.0)
    curves += struct.pack('<IIIf', 0x33, 2, 3, 1.0)
    slots = [1, 1, 1, 0, 1, 1, 0, 2 | (6 << 16)]            # the alpha slot carries a flag in its high half
    nodes = node(11, plane, glow, position=(0, 0.725, 0), scale=(2, 2, 1), slots=slots, order=3)
    nodes += node(1, 0, inline, emitter_hash=0xABCD)
    emitters = emitter(0xABCD, [1] * 18 + [0, 2])
    settings = struct.pack('<3fI4I', 1, 1, 1, 1, 0, 0, 0, 0)
    sections = {vfx.MATERIALS: materials, vfx.ASSETS: assets, vfx.CURVES: curves, vfx.KEYS: keys,
                vfx.NODES: nodes, vfx.EMITTERS: emitters, vfx.SETTINGS: settings}
    assert len(sections) == 7
    return HEADER + dat1(sections, strings), dict(glow=glow, inline=inline, plane=plane)


class EffectFormatTest(unittest.TestCase):
    def test_nodes_resolve_assets_materials_and_emitters(self):
        data, ids = effect()
        decoded = vfx.decode_effect(data)
        self.assertEqual([m['path'] for m in decoded['materials']],
                         ['material/fx/glow.material', 'materialgraph/fx/flash.materialgraph'])
        self.assertEqual(decoded['assets'][0]['kind'], 'model')
        single, multi = decoded['nodes']
        self.assertEqual(single['render_type_name'], 'SingleModel')
        self.assertEqual(single['asset'], 'objects/fx/plane.model')
        self.assertEqual(single['material'], 'material/fx/glow.material')
        self.assertEqual(single['position'], (0.0, 0.7250000238418579, 0.0))
        self.assertEqual(single['scale'], (2.0, 2.0, 1.0))
        self.assertEqual(single['sort_order'], 3)
        self.assertIsNone(single['emitter'])
        self.assertEqual(single['curves']['alpha'], (2, 6))
        self.assertEqual(multi['render_type_name'], 'MultiQuads')
        self.assertEqual(multi['material'], 'materialnode: material_2, materialgraph/fx/flash.materialgraph')
        self.assertIsNone(multi['curves'])
        self.assertEqual(multi['emitter'], 0)
        self.assertEqual(decoded['emitters'][0]['position'], (0.0, 2.0, 0.0))
        self.assertEqual(decoded['emitters'][0]['curves']['color'], (0, 0))
        self.assertEqual(decoded['settings'], (1.0, 1.0, 1.0))

    def test_curve_keys_and_values(self):
        data, _ids = effect()
        curves = vfx.decode_effect(data)['curves']
        self.assertEqual(curves[0]['keys'][0]['base'], (0.0, 0.5, 1.0))
        self.assertEqual(curves[0]['keys'][0]['time'], 0.0)
        self.assertEqual([k['time'] for k in curves[2]['keys']], [0.0, 0x8000 / 65535, 1.0])
        self.assertEqual(vfx.curve_value(curves[1], 0.0, (0.5, 0.5, 0.5)), (0.75, 0.75, 0.75))
        self.assertAlmostEqual(vfx.curve_value(curves[2], 0.25)[0], 0.25 / (0x8000 / 65535), places=6)
        self.assertEqual(vfx.curve_value(curves[2], 1.0)[0], 0.0)
        self.assertEqual(vfx.curve_value(curves[2], -1.0)[0], 0.0)

    def test_broken_curve_chain_fails(self):
        data, _ids = effect()
        dat = bytearray(data)
        table = vfx.sections(bytes(dat), 36)
        offset, _size = table[vfx.CURVES]
        struct.pack_into('<I', dat, 36 + offset + 16 + 4, 2)          # curve 1 starts at key 2, skipping key 1
        with self.assertRaises(ValueError):
            vfx.decode_effect(bytes(dat))

    def test_slot_outside_the_curves_fails(self):
        data, _ids = effect()
        dat = bytearray(data)
        offset, _size = vfx.sections(bytes(dat), 36)[vfx.NODES]
        struct.pack_into('<I', dat, 36 + offset + 0x180, 9)
        with self.assertRaises(ValueError):
            vfx.decode_effect(bytes(dat))

    def test_not_dat1(self):
        with self.assertRaises(ValueError):
            vfx.decode_effect(bytes(64))


@unittest.skipUnless(GAME, 'RCRA_GAME is not set')
class VendorEffectsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nwo.games.rcra.source import RcraSource
        cls.temp = tempfile.TemporaryDirectory()
        cls.source = RcraSource(GAME, os.path.join(cls.temp.name, 'out'))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def data(self, path):
        return self.source.asset_data(self.source.resolve(path)[0])

    def test_holo_logo(self):
        decoded = vfx.decode_effect(self.data(
            'visualeffect/characters/npc/npc_weapon_vendor_holo_logo/npc_weapon_vendor_holo_logo.visualeffect'))
        light, plane = decoded['nodes']
        self.assertEqual(light['render_type_name'], 'SingleLight')
        self.assertTrue(light['material'].endswith('npc_weapon_vendor_holo_logo_gobo.material'))
        self.assertEqual(plane['render_type_name'], 'SingleModel')
        self.assertEqual(plane['asset'], 'objects/fx/fx_1x1_plane_rt.model')
        self.assertTrue(plane['material'].endswith('npc_weapon_vendor_holo_logo.material'))
        self.assertAlmostEqual(plane['position'][1], 0.725, places=6)
        self.assertEqual(plane['scale'], (2.0, 2.0, 1.0))
        color = vfx.curve_value(decoded['curves'][plane['curves']['color'][0]])
        self.assertEqual(color[0], 0.0)
        self.assertAlmostEqual(color[1], 0.652, places=3)
        self.assertEqual(vfx.curve_value(decoded['curves'][plane['curves']['luminance'][0]]), (10.0, 10.0, 10.0))

    def test_projector_links_emitters(self):
        decoded = vfx.decode_effect(self.data(
            'visualeffect/characters/npc/npc_weapon_vendor_holo_logo_projector/'
            'npc_weapon_vendor_holo_logo_projector.visualeffect'))
        kinds = [(n['render_type_name'], n['emitter']) for n in decoded['nodes']]
        self.assertEqual(kinds, [('SingleQuad', None), ('MultiModels', 0), ('MultiQuads', 1)])
        self.assertEqual(decoded['nodes'][1]['asset'], 'visualeffect/model/ammo_pip_hologram_source_rays.model')

    def test_vendor_conduit_attachments(self):
        found = vfx.conduit_effects(self.data('conduit/characters/npc/npc_civ_robot_04_vfx/weapons_vendor_vfx.conduit'))
        started = {(f['effect'].rsplit('/', 1)[-1], f['locator']) for f in found
                   if f['event'] == 'ActivatedEvent' and f['start']}
        self.assertEqual(started, {('weapons_vendor_hover_01.visualeffect', 'igLoc_hover'),
                                   ('npc_weapon_vendor_holo_logo.visualeffect', 'igLoc_projection_vfx'),
                                   ('npc_weapon_vendor_holo_logo_projector.visualeffect', 'igLoc_projection_vfx')})
        self.assertTrue(any(not f['start'] for f in found if f['event'] == 'DeactivatedEvent'))


if __name__ == '__main__':
    unittest.main()
