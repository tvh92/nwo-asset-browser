"""Additive layers over a base clip: ``viewer.animation.layer_over``, Rift Apart driver classes
(``rcra.animation.driver_classes``, ``RcraAnimationLibrary.layer_partners``) and the real-install partners
(RCRA_GAME, skipped without it)."""
import math
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rc2016.games.rcra import animation as ra
from rc2016.games.rcra.clip_library import RcraAnimationLibrary
from rc2016.viewer.animation import AnimationLibrary, layer_over, on_rest

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

    def test_library_hook_defaults(self):
        self.assertEqual(AnimationLibrary.layer_partners(SimpleNamespace(), 0), [])


class DriverClassTests(unittest.TestCase):
    def test_classes_split_at_lookup_offsets(self):
        data_words = [11, 22, 33, 44, 55]
        built = bytes(80 * 3)
        lookup = struct.pack('<4I', 0, 0, 0, 0) + struct.pack('<4I', 12, 0, 0, 0) + struct.pack('<4I', 20, 0, 0, 0)
        asset = bytes(36) + dat1({ra.DRIVER_BUILT: built, ra.DRIVER_LOOKUP: lookup,
                                  ra.DRIVER_DATA: struct.pack('<5I', *data_words)})
        self.assertEqual(ra.driver_classes(asset), [(11, 22, 33), (44, 55), ()])
        self.assertEqual(ra.driver_classes(bytes(36) + dat1({ra.DRIVER_BUILT: built})), [])

    def test_partners_are_class_mates_ranked_by_name(self):
        lib = RcraAnimationLibrary.__new__(RcraAnimationLibrary)
        names = ['Aim_Idle_Fwd_Pitch', 'aim_idle_l090', 'aim_idle_fwd', 'run', 'Twitch']
        lib.clips = [dict(name=n, hash=100 + i, additive=n[0].isupper(), animset='set', source='animset')
                     for i, n in enumerate(names)]
        lib._partners = None
        lib.source = None
        classes = [(7, 100, 9, 101, 102), (104, 103)]
        import rc2016.games.rcra.clip_library as cl
        saved = cl._key, cl.driver_classes
        cl._key, cl.driver_classes = (lambda source, path: path), (lambda data: classes)
        lib.source = SimpleNamespace(bytes=lambda key: (b'', None))
        try:
            self.assertEqual(lib.layer_partners(0), [2, 1])   # fwd shares more words than l090
            self.assertEqual(lib.layer_partners(4), [3])
            self.assertEqual(lib.layer_partners(2), [])       # not additive
        finally:
            cl._key, cl.driver_classes = saved


@unittest.skipUnless(GAME, 'RCRA_GAME is not set')
class RealPartnerTests(unittest.TestCase):
    MODEL = 'characters/enemy/enm_thug_brawler/enm_thug_brawler.model'

    def test_thug_aim_layer_over_its_idle(self):
        import numpy as np

        from rc2016.games.rcra.clip_library import animation_library
        from rc2016.games.rcra.source import RcraSource
        with tempfile.TemporaryDirectory() as tmp:
            source = RcraSource(GAME, str(Path(tmp) / 'out'))
            try:
                lib = animation_library(source, source.bytes(source.resolve(self.MODEL)[0])[0], self.MODEL)
                names = [c['name'] for c in lib.clips]
                layer = names.index('Shooter_Aim_Idle_Fwd_Pitch_thug_brawler')
                partners = lib.layer_partners(layer)
                self.assertEqual(names[partners[0]], 'shooter_aim_idle_fwd_thug_brawler')
                self.assertEqual({names[i] for i in partners[:3]},
                                 {f'shooter_aim_idle_{d}_thug_brawler' for d in ('fwd', 'l090', 'r090')})
                over = lib.pose(layer, base=partners[0])
                base = lib.pose(partners[0])
                alone = lib.pose(layer)
                self.assertEqual(over['base'], 'shooter_aim_idle_fwd_thug_brawler')
                self.assertEqual(len(over['frames']), max(len(base['frames']), len(alone['frames'])))
                a, b = np.stack(over['frames']), np.stack(base['frames'])
                self.assertGreater(float(np.abs(a - b).max()), 0.1)   # the layer moves the base
                adds = [i for i, c in enumerate(lib.clips) if c.get('additive')]
                self.assertGreater(sum(1 for i in adds if lib.layer_partners(i)), 40)   # measured 55 of 160
            finally:
                source.close()


if __name__ == '__main__':
    unittest.main()
