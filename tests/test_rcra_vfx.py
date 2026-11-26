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


def node(kind, asset_id=0, material_id=0, emitter_hash=0, position=(0, 0, 0), scale=(1, 1, 1), slots=None, order=0,
         rates=None):
    record = bytearray(vfx.RECORD)
    struct.pack_into('<QQI', record, 0, asset_id, material_id, emitter_hash)
    struct.pack_into('<9f', record, 0x20, 1, 0, 0, 0, 1, 0, 0, 0, 1)
    struct.pack_into('<3f', record, 0x44, *position)
    struct.pack_into('<3f', record, 0xC4, *scale)
    record[0x17A] = kind
    record[0x17F] = order
    for i, word in enumerate(slots or [0] * 8):
        struct.pack_into('<I', record, 0x180 + 8 * i, word)
    for i, rate in enumerate(rates or []):
        struct.pack_into('<e', record, 0x180 + 8 * i + 6, rate)
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
    nodes = node(11, plane, glow, position=(0, 0.725, 0), scale=(2, 2, 1), slots=slots, order=3,
                 rates=[0, 0, 0.5, 0, 0, 0, 0, -2])
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
        self.assertEqual((single['rates']['luminance'], single['rates']['alpha'], single['rates']['color']), (0.5, -2.0, 0.0))
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
        self.assertEqual(vfx.curve_value(curves[1], 0.0, 0.5), (0.75, 0.75, 0.75))   # mode 2 reads the instance value
        self.assertAlmostEqual(vfx.curve_value(curves[2], 0.25)[0], 0.25 / (0x8000 / 65535), places=6)
        self.assertEqual(vfx.curve_value(curves[2], 1.0)[0], 0.0)
        self.assertEqual(vfx.curve_value(curves[2], -1.0)[0], 0.0)

    def test_curve_scale_modes_and_time_rates(self):
        """0x14138D140 and 0x14138D0E0: |scale| multiplies, the mode picks the random value (1 the particle's, 2 the
        instance's, 3 a coin over 0.5), and a slot's rate loops, plays once or keeps the normalised age."""
        key = dict(base=(1.0, 2.0, 3.0), time=0.0, range=(2.0, 2.0, 2.0), mode=1.0)
        curve = dict(hash=0, scale=-10.0, keys=[key])
        self.assertEqual(vfx.curve_value(curve, 0.0, 0.25, 0.75), (15.0, 25.0, 35.0))
        self.assertEqual(vfx.curve_value(dict(curve, keys=[dict(key, mode=2.0)]), 0.0, 0.25, 0.75), (25.0, 35.0, 45.0))
        coin = dict(curve, keys=[dict(key, mode=3.0)])
        self.assertEqual(vfx.curve_value(coin, 0.0, 0.5, 0.9), (10.0, 20.0, 30.0))
        self.assertEqual(vfx.curve_value(coin, 0.0, 0.6, 0.0), (30.0, 40.0, 50.0))
        self.assertAlmostEqual(vfx.curve_time(0.4, 3.0, 0.9), 0.2)
        self.assertEqual(vfx.curve_time(-2.0, 0.25, 0.9), 0.5)
        self.assertEqual(vfx.curve_time(-2.0, 3.0, 0.9), 1.0)
        self.assertEqual(vfx.curve_time(0.0, 3.0, 0.9), 0.9)
        self.assertEqual(vfx.curve_time(0.0, 3.0, 1.5), 1.0)

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


class EffectMeshTest(unittest.TestCase):
    def test_object_override_reaches_object_words(self):
        from types import SimpleNamespace
        from nwo.games.rcra import shaders as rshaders
        members = [('m_Alpha', 0, None), ('m_ColorRGFp16', 4, None)]
        layout = SimpleNamespace(size=8, variables=[SimpleNamespace(members=members)])
        reflection = SimpleNamespace(constant_buffer=lambda name: layout)
        neutral = rshaders.object_words(reflection)
        self.assertEqual(neutral[4], rshaders.HALF_ONE | rshaders.HALF_ONE << 16)
        with rshaders.draw_override({'object': {'m_ColorRGFp16': (rshaders.half_pair(0.0, 6.5),)}}):
            words = rshaders.object_words(reflection)
        self.assertEqual(words[4], rshaders.half_pair(0.0, 6.5))
        self.assertEqual(words[0], neutral[0])
        self.assertEqual(rshaders.object_words(reflection), neutral)
        self.assertEqual(rshaders.object_text('m_ColorRGFp16', (rshaders.half_pair(0.0, 6.5),)), '(0, 6.5)')

    def test_emitter_fields_and_names(self):
        data, _ids = effect()
        decoded = vfx.decode_effect(data)
        emitter = decoded['emitters'][0]
        self.assertEqual(len(set(vfx.EMITTER_CURVES)), 20)
        self.assertEqual((emitter['curves']['speed'], emitter['curves']['velocity']), ((1, 0), (1, 0)))
        self.assertEqual((emitter['gravity'], emitter['lifetime'], emitter['shape'], emitter['space']), (0.0, 0.0, 0, 0))
        self.assertEqual(decoded['shapes'], [])

    def test_particle_replay(self):
        """Rate 20 per second, life 0.5 s, speed 2 along +Z: at age 0.5 about ten particles, each 2 x its age along the
        emitter axis; gravity pulls along world -Y."""
        from nwo.games.rcra import particles
        constant = [dict(hash=0, scale=1.0, keys=[dict(base=(v, v, v), time=0.0, range=(0, 0, 0), mode=1.0)])
                    for v in (0.0, 1.0, 20.0, 0.5, 2.0)]
        slots = dict.fromkeys(vfx.EMITTER_CURVES, 1)
        slots.update(rate=2, life=3, speed=4, burst=0, per_metre=0, gap=0, emitter_rotation=0, emitter_position=0,
                     follow=0, acceleration=0, turbulence=0, custom=0)
        emitter = dict(index=0, rotation=(1, 0, 0, 0, 1, 0, 0, 0, 1), position=(0.0, 0.0, 0.0),
                       curves={k: (v, 0) for k, v in slots.items()}, rates=dict.fromkeys(vfx.EMITTER_CURVES, 0.0),
                       accel_direction=(0.0, 0.0, 0.0), gravity=0.0, drag=0.0, lifetime=0.0, rotation_range=(0, 0, 0),
                       rotation_base=(0, 0, 0), shape=0, max_alive=8192)
        effect = dict(curves=constant, shapes=[])
        self.assertEqual(particles.still_age(effect, emitter), 0.5)
        live, notes = particles.simulate(effect, emitter, np.eye(3))
        self.assertEqual(notes, [])
        self.assertIn(len(live), (9, 10, 11))
        for p in live:
            self.assertAlmostEqual(p['position'][2], 2.0 * p['age'], delta=2 * particles.STEP * 2.0)
            self.assertAlmostEqual(p['t'], p['age'] / 0.5)
        falling, _ = particles.simulate(effect, dict(emitter, gravity=9.8), np.eye(3))
        oldest = max(falling, key=lambda p: p['age'])
        self.assertLess(oldest['position'][1], -0.5 * 9.8 * oldest['age'] ** 2 * 0.8)

    def test_quads_face_the_view_or_lie_flat(self):
        from nwo.games.rcra import effects
        quads = [dict(centre=(0.0, 1.0, 2.0), size=(2.0, 1.0), angle=0.0)]
        flat = effects.quad_mesh(quads, 'local')
        self.assertEqual((len(flat['positions']), len(flat['triangles'])), (4, 2))
        self.assertEqual(set(flat['positions'][:, 2].tolist()), {2.0})        # in the local XY plane
        self.assertNotIn('billboard', flat)
        facing = effects.quad_mesh(quads, 'camera')
        np.testing.assert_allclose(facing['positions'], [(0.0, 1.0, 2.0)] * 4)
        np.testing.assert_allclose(facing['billboard'], [(-1, -0.5, 0, 1), (1, -0.5, 0, 1), (1, 0.5, 0, 1), (-1, 0.5, 0, 1)])
        self.assertEqual(effects.quad_mesh(quads, 'billboard')['billboard'][0, 3], 2.0)
        values = dict(colour=np.array([0.0, 2.0, 4.0]), alpha=0.5, alpha_test=0.1, custom=np.array([1.0, 2.0, 3.0]), t=0.25)
        bound = effects.quad_interpolants(values)
        self.assertEqual((bound['TEXCOORD4.y'], bound['TEXCOORD4.w'], bound['TEXCOORD.w'], bound['TEXCOORD.z']),
                         (2.0, 1.0, 0.5, 0.1))
        self.assertEqual((bound['TEXCOORD6.y'], bound['TEXCOORD7.z']), (0.25, 3.0))

    def test_facing_model_keeps_its_offsets(self):
        from nwo.games.rcra import effects
        plane = dict(positions=np.array([[-0.5, -0.5, 0], [0.5, 0.5, 0]], float), normals=np.array([[0, 0, 1]] * 2, float),
                     tangents=np.array([[1, 0, 0]] * 2, float))
        mesh = effects.facing_mesh(plane, (0, 3, 0), np.diag((2.0, 2.0, 1.0)), 'billboard')
        np.testing.assert_allclose(mesh['positions'], [(0, 3, 0)] * 2)
        np.testing.assert_allclose(mesh['billboard'], [(-1, -1, 0, 2), (1, 1, 0, 2)])
        np.testing.assert_allclose(mesh['normals'], [(0, 0, 1)] * 2)

    def test_light_values(self):
        from nwo.games.rcra import effects
        light = effects.light_values(dict(colour=np.array([0.0, 0.5, 1.0]), size=np.array([2.0, 1.0, 1.0]), alpha=0.5), 1)
        y = 0.7152 * 0.5 + 0.0722
        np.testing.assert_allclose(light['colour'], (0.0, 0.5 / y, 1.0 / y))
        self.assertAlmostEqual(light['intensity'], 2.0 * np.sqrt(1.25) * 0.5)
        self.assertEqual(effects.light_values(dict(colour=np.ones(3), size=np.ones(3), alpha=1.0), 0)['intensity'], 0.0)

    def test_mesh_is_placed_and_weighted_to_the_joint(self):
        from nwo.games.rcra import effects
        node = dict(rotation=(1, 0, 0, 0, 1, 0, 0, 0, 1), position=(0, 0.725, 0), scale=(2, 2, 1))
        attach = np.eye(4)
        attach[:3, 3] = (0, 1.5, 0)
        placement = attach @ effects.node_matrix(node)
        plane = dict(index=0, positions=np.array([[0, 0, 0], [1, 1, 0]], float), normals=np.array([[0, 0, 1]] * 2, float),
                     tangents=np.array([[1, 0, 0]] * 2, float), material=None, weights=[None, None])
        material = dict(index=4, path='material/fx/logo.material')
        mesh = effects._mesh(plane, placement, 60, material, 13, True)
        np.testing.assert_allclose(mesh['positions'], [[0, 2.225, 0], [2, 4.225, 0]])
        np.testing.assert_allclose(mesh['normals'], [[0, 0, 1]] * 2)
        self.assertEqual((mesh['index'], mesh['material_index']), (60, 4))
        self.assertEqual(mesh['skin_joint_array'][:, 0].tolist(), [13, 13])
        self.assertEqual(mesh['skin_weight_array'][:, 0].tolist(), [1.0, 1.0])
        self.assertEqual(list(mesh['weights'][0]), [(13, 1.0)])
        static = effects._mesh(plane, placement, 61, material, None, True)
        self.assertNotIn('skin_joint_array', static)
        self.assertFalse(static['skinned'])


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

    def test_vendor_view_decode_carries_the_logo(self):
        from nwo.games import rcra_game
        from nwo.games.rcra import material as rmat
        path = 'characters/weapons_vendor/weapons_vendor.model'
        key = next(k for k in self.source.resolve(path) if self.source.entries[k]['type'] == 'model')
        decoded, _data = rcra_game.decode_model(self.source, key, path, True)
        extra = [m for m in decoded['meshes'] if m['index'] >= decoded['mesh_record_count']]
        by_name = {}
        for mesh in extra:
            by_name.setdefault(mesh['material']['path'].rsplit('/', 1)[-1], []).append(mesh)
        logo, = by_name['npc_weapon_vendor_holo_logo.material']
        rings, = by_name['enm_neo_nefarious_swarmer_thrust_ring.material']
        self.assertEqual(len(rings['positions']), 20)                         # 5 particle quads, kLocal: stored flat
        self.assertIsNone(rings.get('billboard'))
        self.assertEqual(set(rings['skin_joint_array'][:, 0].tolist()), {0})   # igLoc_hover's joint
        self.assertEqual(set(logo['skin_joint_array'][:, 0].tolist()), {13})   # igLoc_projection_vfx's joint
        self.assertGreater(np.asarray(logo['positions'])[:, 1].min(), 2.0)     # above the head
        self.assertEqual(set(np.asarray(logo['billboard'])[:, 3].tolist()), {2.0})   # kBillboard: faces the view
        smoke, = by_name['smk_quad_thn_01_lit.material']
        self.assertEqual(set(np.asarray(smoke['billboard'])[:, 3].tolist()), {1.0})  # kCamera particles
        self.assertEqual(len(by_name['gbl_hologram_source_rays_cone.material']), 2)   # MultiModels: one per particle
        for look in decoded['looks']:
            self.assertTrue(all(logo['index'] in level['meshes'] for level in look['lods'] if level['count']))
        rows = {r['effect'].rsplit('/', 1)[-1]: r for r in decoded['effects']}
        self.assertEqual(rows['npc_weapon_vendor_holo_logo.visualeffect']['drawn'], ['1 SingleModel'])
        self.assertEqual(rows['npc_weapon_vendor_holo_logo.visualeffect']['lights'][0]['type'], 'spot')
        self.assertEqual(rows['npc_weapon_vendor_holo_logo_projector.visualeffect']['drawn'],
                         ['0 SingleQuad', '1 MultiModels', '2 MultiQuads'])
        self.assertIn('0 MultiQuads', rows['weapons_vendor_hover_01.visualeffect']['drawn'])
        entry = rmat.material_textures(self.source, [logo['material']], {})[logo['material']['index']]
        self.assertEqual(entry['forward']['kind'], rmat.TRANSLATED_KIND)
        self.assertIn('m_ColorRGFp16 (0, 6.52)', entry['forward']['limitation'])
        export, _data = rcra_game.decode_model(self.source, key, path, False)
        self.assertNotIn('effects', export)                       # the export decode leaves effects out
        self.assertEqual(len(export['meshes']), len(decoded['meshes']) - len(extra))


if __name__ == '__main__':
    unittest.main()
