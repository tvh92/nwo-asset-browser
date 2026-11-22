"""Additive layers over a base clip: ``viewer.animation.layer_over``, the Rift Apart animation graph
(``rcra.animation.driver_graph``, ``clip_references``, ``RcraAnimationLibrary.layer_partners`` and ``layer_weight``)
and the real-install partners (RCRA_GAME, skipped without it)."""
import math
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nwo.games.rcra import animation as ra
from nwo.games.rcra.clip_library import RcraAnimationLibrary
from nwo.viewer.animation import AnimationLibrary, layer_over, on_rest

GAME = os.environ.get('RCRA_GAME')


def row(hash_, t=(0, 0, 0), q=(0, 0, 0, 1), s=(1, 1, 1)):
    return dict(hash=hash_, translation=list(t), rotation_xyzw=list(q) if q is not None else None, scale=list(s))


def zq(degrees):
    half = math.radians(degrees) / 2
    return (0.0, 0.0, math.sin(half), math.cos(half))


def dat1(section_map):
    head = 16 + 12 * len(section_map)
    offset = head + (-head % 16)
    table, body = b'', b''
    for tag, data in section_map.items():
        body += bytes(-len(body) % 16)
        table += struct.pack('<III', tag, offset + len(body), len(data))
        body += data
    return (struct.pack('<4sIIHH', b'1TAD', 0xD1AD9F7C, offset + len(body), len(section_map), 0) + table
            + bytes(offset - head) + body)


class LayerOverTests(unittest.TestCase):
    rows = [dict(hash='a', parent=-1, translation=[0, 1, 0], rotation_xyzw=[0, 0, 0, 1], scale=[1, 1, 1]),
            dict(hash='b', parent=0, translation=[0, 2, 0], rotation_xyzw=[0, 0, 0, 1], scale=[1, 1, 1])]

    def test_adds_on_the_base_and_loops_the_shorter(self):
        layer = [[row('b', (1, 0, 0), zq(90), (2, 2, 2))], [row('b', (2, 0, 0), None)]]
        base = [[row('a', (0, 5, 0)), row('b', (0, 3, 0), zq(30))] for _ in range(3)]
        out = layer_over(self.rows, layer, base)
        self.assertEqual(len(out), 3)
        first = {r['hash']: r for r in out[0]}
        self.assertEqual(first['a']['translation'], [0, 5, 0])                  # a joint the layer leaves as the base
        self.assertEqual(first['b']['translation'], [1, 3, 0])                  # base T + delta T
        self.assertEqual(first['b']['scale'], [2, 2, 2])
        self.assertAlmostEqual(first['b']['rotation_xyzw'][2], math.sin(math.radians(60)), places=6)   # 30 then 90
        second = {r['hash']: r for r in out[1]}
        self.assertEqual(second['b']['rotation_xyzw'], list(zq(30)))           # no rotation delta: the base's
        third = {r['hash']: r for r in out[2]}
        self.assertEqual(third['b']['translation'], [1, 3, 0])                  # the layer loops

    def test_rest_values_where_the_base_does_not_sample(self):
        layer = [[row('b', (1, 0, 0)), row('zzz', (9, 9, 9))]]
        out = layer_over(self.rows, layer, [[row('a')]])
        joints = {r['hash']: r for r in out[0]}
        self.assertEqual(joints['b']['translation'], [1, 2, 0])                 # rest T + delta T, as on_rest
        self.assertNotIn('zzz', joints)                                          # not a model joint
        self.assertEqual(joints['b']['translation'], on_rest(self.rows, layer)[0][0]['translation'])

    def test_weight_scales_the_delta(self):
        layer = [[row('b', (2, 0, 0), zq(90), (4, 4, 4))]]
        base = [[row('a'), row('b', (0, 3, 0), zq(30))]]
        b = {r['hash']: r for r in layer_over(self.rows, layer, base, 0.5)[0]}['b']
        self.assertEqual(b['translation'], [1, 3, 0])                          # half the delta T
        self.assertEqual(b['scale'], [2, 2, 2])                                # delta S to the power 0.5
        self.assertAlmostEqual(b['rotation_xyzw'][2], math.sin(math.radians(37.5)), places=6)   # 30 then 45
        full = {r['hash']: r for r in layer_over(self.rows, layer, base)[0]}['b']
        self.assertEqual(full['translation'], [2, 3, 0])

    def test_library_hook_defaults(self):
        self.assertEqual(AnimationLibrary.layer_partners(SimpleNamespace(), 0), [])
        self.assertEqual(AnimationLibrary.layer_weight(SimpleNamespace(), 0, 1), 1.0)


def node(name, kind_word, payload, ident=0, phase=0.0, rate=1.0):
    head = struct.pack('<IHHfIfIffII', name, kind_word, ident, phase, 0, rate, 0, -1.0, -1.0, 0, 0xFFFFFFFF)
    return head + payload


PATHS = ['set/base_a.animclip', 'set/layer_l.animclip', 'set/base_b.animclip']


def graph_animset():
    """One class: blend space @344 (root, bound to variable 0xABCD) over [pair @288, group clip @192]; the pair adds
    layer L (@96) over base A (@0) at weight 0.5; the group holds base B. A second, empty class."""
    from nwo.games.rcra.toc import path_crc64_many
    a, l_, b = (int(i) for i in path_crc64_many(PATHS))
    clip = lambda ref, flags=0: node(0x10, 1 | flags << 8, struct.pack('<II', 0, ref) + bytes(48))  # noqa: E731
    data = (clip(0x1111) + clip(0x2222, 1) + clip(0x3333)
            + node(0x20, 2, struct.pack('<HH4BIf', 0, 24, 2, 2, 2, 2, 2, 0.5))
            + node(0x30, 5, struct.pack('<HHHHII', 0, 2, 101, 102, 0, 0xFFFF) + bytes(4)))
    assert len(data) == 404
    data += struct.pack('<HH', 72, 48) + struct.pack('<4f', 0, 1, 0, 1)           # child list @404, spans @408
    data += struct.pack('<IHHI', 0xABCD, 9, 110, 0x101) + bytes(4)                # binding @424, its list @440
    data += struct.pack('<HH', 98, 0) + bytes(4)                                  # bound field: @344 + 48
    built = struct.pack('<14I4f2I', 0xC1A55, 9, 512, len(data), 344, 5, 1, 424, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    built += struct.pack('<14I4f2I', 0xE3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    lookup = struct.pack('<4I', 0, 0, 0, 0) + struct.pack('<4I', 0xFFFFFFFF, 0, 0, 0)
    group = dat1({ra.GROUP_ENTRIES: struct.pack('<8I', 1, 0, 0, 0, 0, 0x201, 1, 0),
                  ra.GROUP_LISTS: struct.pack('<4I', 0x77, 0xFFFFFFFF, 0xFFFFFFFF, 0x4444)})
    rows = b''.join(struct.pack('<IIiQI', h, k, off, asset, 0) for h, k, off, asset in
                    ((0x1111, 0, -1, a), (0x2222, 0, -1, l_), (0x4444, 0x8000, -1, b), (0x3333, 1, 0, 0)))
    return bytes(36) + dat1({ra.DRIVER_BUILT: built, ra.DRIVER_LOOKUP: lookup, ra.DRIVER_DATA: data,
                             ra.CLIP_LOOKUP: rows, ra.CLIP_GROUPS: group}), (a, l_, b)


class DriverGraphTests(unittest.TestCase):
    def test_nodes_links_and_bindings(self):
        asset, _ids = graph_animset()
        first, empty = ra.driver_graph(asset)
        self.assertEqual(empty, dict(name=0xE3, root=None, nodes={}, bindings=[]))
        self.assertEqual(first['root'], 344)
        nodes = first['nodes']
        self.assertEqual(sorted(nodes), [0, 96, 192, 288, 344])
        self.assertEqual((nodes[96]['kind'], nodes[96]['flags'], nodes[96]['ref']), (1, 1, 0x2222))
        pair = nodes[288]
        self.assertEqual((pair['kind'], pair['children'], pair['mode'], pair['weight']), (2, [0, 96], 2, 0.5))
        space = nodes[344]
        self.assertEqual((space['children'], space['spans']), ([288, 192], [(0.0, 1.0), (0.0, 1.0)]))
        self.assertEqual(space['bound'], {48: 0xABCD})
        self.assertEqual(first['bindings'], [dict(variable=0xABCD, slot=9, fields=[392])])
        self.assertEqual(ra.graph_reach(first), [344, 288, 0, 96, 192])
        self.assertEqual(ra.graph_reach(first, 288), [288, 0, 96])
        self.assertEqual(ra.driver_graph(bytes(36) + dat1({ra.DRIVER_BUILT: bytes(80)})), [])

    def test_clip_references_resolve_groups(self):
        asset, (a, l_, b) = graph_animset()
        self.assertEqual(ra.clip_references(asset), {0x1111: [a], 0x2222: [l_], 0x4444: [b], 0x3333: [b]})

    def test_partners_come_from_pair_nodes(self):
        asset, _ids = graph_animset()
        lib = RcraAnimationLibrary.__new__(RcraAnimationLibrary)
        lib.clips = [dict(name=Path(p).stem, path=p, additive='layer' in p, animset='set', source='animset')
                     for p in PATHS]
        lib._partners = None
        lib.source = SimpleNamespace(bytes=lambda key: (asset, None))
        import nwo.games.rcra.clip_library as cl
        saved = cl._key
        cl._key = lambda source, path: path
        try:
            self.assertEqual(lib.layer_partners(1), [0])        # B sits beside the pair, not under its child 0
            self.assertEqual(lib.layer_weight(1, 0), 0.5)
            self.assertEqual(lib.layer_weight(1, 2), 1.0)       # no pair holds them
            self.assertEqual(lib.layer_partners(0), [])         # not additive
        finally:
            cl._key = saved


@unittest.skipUnless(GAME, 'RCRA_GAME is not set')
class RealPartnerTests(unittest.TestCase):
    MODEL = 'characters/enemy/enm_thug_brawler/enm_thug_brawler.model'

    def test_thug_aim_layer_over_its_idle(self):
        import numpy as np

        from nwo.games.rcra.clip_library import animation_library
        from nwo.games.rcra.source import RcraSource
        with tempfile.TemporaryDirectory() as tmp:
            source = RcraSource(GAME, str(Path(tmp) / 'out'))
            try:
                lib = animation_library(source, source.bytes(source.resolve(self.MODEL)[0])[0], self.MODEL)
                names = [c['name'] for c in lib.clips]
                layer = names.index('Shooter_Aim_Idle_Fwd_Pitch_thug_brawler')
                partners = lib.layer_partners(layer)
                self.assertEqual([names[i] for i in partners], ['shooter_aim_idle_fwd_thug_brawler'])   # its pair
                left = lib.layer_partners(names.index('Shooter_Aim_Idle_L090_Pitch_thug_brawler'))
                self.assertEqual([names[i] for i in left], ['shooter_aim_idle_l090_thug_brawler'])
                over = lib.pose(layer, base=partners[0])
                base = lib.pose(partners[0])
                alone = lib.pose(layer)
                self.assertEqual(over['base'], 'shooter_aim_idle_fwd_thug_brawler')
                self.assertEqual(len(over['frames']), max(len(base['frames']), len(alone['frames'])))
                a, b = np.stack(over['frames']), np.stack(base['frames'])
                self.assertGreater(float(np.abs(a - b).max()), 0.1)   # the layer moves the base
                adds = [i for i, c in enumerate(lib.clips) if c.get('additive')]
                self.assertGreater(sum(1 for i in adds if lib.layer_partners(i)), 60)   # measured 72 of 160
                model = 'characters/enemy/enm_grunthor_rcf/enm_grunthor_rcf.model'
                grunthor = animation_library(source, source.bytes(source.resolve(model)[0])[0], model)
                eyes = [c['name'] for c in grunthor.clips].index('eyelook_horizontal_enm_grunthor_rcf')
                under = grunthor.layer_partners(eyes)
                self.assertEqual(grunthor.layer_weight(eyes, under[0]), 0.5)   # the pair's stored weight
                self.assertEqual(grunthor.pose(eyes, base=under[0])['weight'], 0.5)
            finally:
                source.close()


if __name__ == '__main__':
    unittest.main()
