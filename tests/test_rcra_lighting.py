"""Rift Apart lighting classes (rc2016.games.rcra.lighting), the forward programs' own key light
(rc2016.games.rcra.surface.forward_lighting), their viewer emission (rc2016.viewer.rcra_glsl) and the translator's
tessellation, groupshared and atomic support (rc2016.shaders.sm5).

Real-install tests read RCRA_GAME (the Rift Apart install folder) and skip without it.
"""
import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

from rc2016.games.rcra import lighting
from rc2016.viewer import rcra_glsl

GAME = os.environ.get('RCRA_GAME')
CLOTH = 'required/materials/basic_normal_gloss_cloth.materialgraph'
FOLIAGE = 'required/materials/basic_normal_gloss_foliage.materialgraph'
FUR = 'material/characters/hero/hero_ratchet_head/hero_ratchet_head_fur.material'
DOME = 'characters/enemy/enm_neo_nefarious_trooper/materials/enm_neo_nefarious_trooper_dome.material'
JELLYFISH = 'materialgraph/characters/ambient/amb_uwb_jellyfish/amb_uwb_jellyfish.materialgraph'


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _inputs(rng, cls):
    n = _unit([rng.uniform(-1, 1) for _ in range(3)])
    light = _unit([a + 1.5 * b for a, b in zip(_unit([rng.uniform(-1, 1) for _ in range(3)]), n)])
    values = {'normal.' + c: v for c, v in zip('xyz', n)}
    values.update({'L.' + c: v for c, v in zip('xyz', light)})
    values.update({'VIEW.' + c: rng.uniform(-3, 3) for c in 'xyz'}, DEPTH=rng.uniform(0.5, 8.0))
    values.update({'LIGHT.r': 1.0, 'LIGHT.g': 0.8, 'LIGHT.b': 0.6}, gloss=rng.randrange(256) / 255.0,
                  occlusion=rng.uniform(0.2, 1.0))
    values.update({'albedo.' + c: rng.uniform(0.05, 0.95) for c in 'rgb'})
    values.update({'f0.' + c: (rng.randrange(256) / 255.0) ** 2 for c in 'rgb'})
    values.update({'extra.%d' % k: float(rng.randrange(1 << w)) for k, (_o, w) in enumerate(lighting.EXTRA_FIELDS.get(cls, ()))})
    return values


def _game_brdf(v):
    """The class-0 key light as read from the apply program: GGX alpha = r^4, Smith joint approximation with a = r^4,
    Schlick, Lambert without 1/pi."""
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))   # noqa: E731
    n, light = [v['normal.' + c] for c in 'xyz'], [v['L.' + c] for c in 'xyz']
    view = _unit([-v['VIEW.' + c] for c in 'xyz'])
    h = _unit([a + b for a, b in zip(view, light)])
    nl, nv = max(min(dot(n, light), 1), 0), abs(dot(n, view))
    nh, vh = max(min(dot(n, h), 1), 0), max(min(dot(view, h), 1), 0)
    a = (0.9725 - 0.7514 * v['gloss']) ** 4
    d = nh * nh * (a * a - 1) + 1
    vis = 0.5 / (nl * (nv * (1 - a) + a) + nv * (nl * (1 - a) + a))
    return [v['LIGHT.' + c] * (v['albedo.' + c] * nl + a * a / (d * d) * vis * nl *
                               (v['f0.' + c] + (1 - v['f0.' + c]) * (1 - vh) ** 5)) for c in 'rgb']


class ClassTests(unittest.TestCase):
    def test_class_bits(self):
        self.assertEqual(lighting.class_of(0x9000), 0)    # bit 12 set: class 0 whatever bits 13 to 15 hold
        self.assertEqual(lighting.class_of(0x2000), 1)
        self.assertEqual(lighting.class_of(0xA000), 5)
        self.assertEqual(lighting.class_of(0x6000), 3)

    def test_extra_fields_tile_the_word(self):
        for cls, fields in lighting.EXTRA_FIELDS.items():
            bits = 0
            for offset, width in fields:
                mask = ((1 << width) - 1) << offset
                self.assertFalse(bits & mask, cls)
                bits |= mask
            self.assertLessEqual(bits, 0xFFFFFFFF)
        word = lighting.pack_extra(2, (255, 3, 31, 0, 7))
        self.assertEqual((word & 0xFF, word >> 8 & 0xFF, word >> 16 & 31, word >> 27), (255, 3, 31, 7))


def _light_recipe():
    """A lighting recipe: diffuse = N.L x LIGHT, specular = extra.0 x LIGHT."""
    names = ['normal.x', 'normal.y', 'normal.z', 'L.x', 'L.y', 'L.z', 'LIGHT.r', 'LIGHT.g', 'LIGHT.b', 'extra.0']
    nodes = [dict(op='input', type='f', name=n, args=[]) for n in names]
    nodes.append(dict(op='mul', type='f', args=[0, 3]))
    nodes.append(dict(op='mul', type='f', args=[1, 4]))
    nodes.append(dict(op='add', type='f', args=[10, 11]))
    nodes.append(dict(op='mul', type='f', args=[2, 5]))
    nodes.append(dict(op='add', type='f', args=[12, 13]))   # 14: N.L
    terms = {}
    for k, c in enumerate('rgb'):
        nodes.append(dict(op='mul', type='f', args=[14, 6 + k]))
        terms['diffuse.' + c] = len(nodes) - 1
        nodes.append(dict(op='mul', type='f', args=[9, 6 + k]))
        terms['specular.' + c] = len(nodes) - 1
    return dict(version=1, nodes=nodes, terms=terms, textures={}, inputs=names, limitations=['class note'], offset=0x10,
                lighting_class=2)


class EmitterTests(unittest.TestCase):
    def test_light_function_and_class_lighting(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_rcra_viewer import _receipt
        receipt = _receipt('opaque', forward=False)
        receipt['lighting'] = _light_recipe()
        source = rcra_glsl.fragment_source(receipt)['source']
        self.assertIn(rcra_glsl.LIGHT_SIGNATURE + ' {', source)
        self.assertIn('return rA * vec3(', source)
        self.assertIn('rcraClassLit(rcraNormal, V, rcraAlbedo, rcraF0, rcraGloss, rcraOcclusion, rcraWorldN', source)
        self.assertEqual(source.count('rcraLight(n, view, depth'), 3)   # key, fill, preview
        plain = rcra_glsl.fragment_source(_receipt('opaque', forward=False))['source']
        self.assertNotIn('rcraClassLit(rcraNormal', plain)
        self.assertIn('float rcraAlpha(float gloss) { float r = 0.9725 - 0.7514 * clamp(gloss, 0.0, 1.0); r *= r; return r * r; }',
                      plain)

    def test_forward_light_is_emitted_per_light_sharing_independent_statements(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_rcra_viewer import _recipe, _receipt
        receipt = _receipt('blended')
        # colour = sample + L.x x LIGHT.r (the sample reads no light: emitted once)
        lit = _recipe({}, extra=[dict(op='input', type='f', name='L.x', args=[]),
                                 dict(op='input', type='f', name='LIGHT.r', args=[]),
                                 dict(op='mul', type='f', args=[7, 8]), dict(op='add', type='f', args=[3, 9])])
        lit['terms'] = {'colour.r': 10, 'colour.g': 3, 'colour.b': 3}
        receipt['forward']['lit'] = lit
        source = rcra_glsl.fragment_source(receipt)['source']
        self.assertEqual(source.count('rcraOwn += '), 3)
        self.assertEqual(source.count('texture2D(uNativeTextextureUnit5'), 3)   # surface, forward, and once for the lights
        self.assertIn('rcraOwn + (uLightingEnabled ? rcraAmbient(', source)


@unittest.skipUnless(GAME, 'RCRA_GAME is not set')
class RealLightingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))
        from rc2016.games.rcra.source import RcraSource
        cls.temp = tempfile.TemporaryDirectory()
        cls.source = RcraSource(GAME, Path(cls.temp.name) / 'out')

    @classmethod
    def tearDownClass(cls):
        cls.source.close()
        cls.temp.cleanup()

    def key(self, path, kind):
        return next(k for k in self.source.resolve(path) if self.source.entries[k]['type'] == kind)

    def test_every_class_has_an_apply_program(self):
        programs = lighting.programs(GAME)
        self.assertEqual(sorted(programs), list(lighting.CLASSES))
        self.assertEqual(len({offset for offset, _blob in programs.values()}), 6)

    def test_class_0_is_ggx_with_alpha_r4(self):
        recipe = lighting.direct(GAME, 0)
        rng = random.Random(2)
        for _ in range(40):
            v = _inputs(rng, 0)
            for got, want in zip(lighting.evaluate(recipe, v), _game_brdf(v)):
                self.assertAlmostEqual(got, want, delta=0.01 * max(abs(want), 1e-3))

    def test_class_recipes_are_portable_and_read_their_fields(self):
        for cls in lighting.CLASSES:
            recipe = lighting.direct(GAME, cls)
            names = {n['name'] for n in recipe['nodes'] if n['op'] == 'input'}
            self.assertTrue({'L.x', 'LIGHT.r', 'normal.x', 'gloss'} <= names, cls)
            fields = {'extra.%d' % k for k in range(len(lighting.EXTRA_FIELDS.get(cls, ())))}
            self.assertTrue(fields <= names | {'extra.%d' % k for k in range(5)}, cls)
            v = _inputs(random.Random(cls), cls)
            self.assertTrue(all(math.isfinite(x) for x in lighting.evaluate(recipe, v)), cls)
            dark = dict(v, **{'LIGHT.r': 0.0, 'LIGHT.g': 0.0, 'LIGHT.b': 0.0})
            self.assertEqual(lighting.evaluate(recipe, dark), [0.0, 0.0, 0.0], cls)

    def test_writers_store_their_class_and_extra_fields(self):
        from rc2016.games.rcra import material as rmat, surface
        from rc2016.games.rcra.texture import texture_size
        for graph_path, cls in ((CLOTH, 2), (FOLIAGE, 5)):
            graph = self.source.asset_data(self.key(graph_path, 'materialgraph'))
            material = next(k for k in self.source.ids_of_type('material')
                            if graph_path in rmat._graphs(self.source, k))
            r = surface.surface(graph, self.source.asset_data(material),
                                texture_sizes=lambda p: texture_size(self.source, p))
            g = r['graph']
            self.assertEqual(g.value(r['terms']['class']), float(cls))
            word = r['outputs'][('o', 3, 0)]
            texels = {}
            rng = random.Random(4)

            def sample(node, _args):
                return texels.setdefault(node['binding'], [rng.random() for _ in range(4)])
            from rc2016.shaders import sm5
            names = ['extra.%d' % k for k in range(len(lighting.EXTRA_FIELDS[cls]))]
            for _ in range(10):
                texels.clear()
                values = {n: rng.uniform(-1, 1) for n in surface.GEOMETRY_INPUTS}
                values.update({'SV_IsFrontFace': True, 'VIEW.x': 0.3, 'VIEW.y': -1.0, 'VIEW.z': 2.0,
                               'CAMERA.x': 0.0, 'CAMERA.y': 0.0, 'CAMERA.z': 0.0})
                out = sm5.evaluate(g, [word] + [r['terms'][n] for n in names], values, sample)
                for n, (offset, width) in zip(names, lighting.EXTRA_FIELDS[cls]):
                    self.assertEqual(out[r['terms'][n]], (out[word] & 0xFFFFFFFF) >> offset & ((1 << width) - 1))

    def test_fur_writer_is_hair(self):
        from rc2016.games.rcra import fur, surface
        r = surface.surface(None, self.source.asset_data(self.key(FUR, 'material')),
                            program=fur.fur_program(self.source, self.source.asset_data(self.key(FUR, 'material'))))
        self.assertEqual(r['graph'].value(r['terms']['class']), 3.0)

    def test_forward_program_lights_itself_linearly(self):
        from rc2016.games.rcra import material as rmat, surface
        from rc2016.games.rcra.texture import texture_size
        key = self.key(DOME, 'material')
        data = self.source.asset_data(key)
        graph = self.source.asset_data(self.key(rmat._graphs(self.source, key)[0], 'materialgraph'))
        lit = surface.forward_lighting(graph, data, texture_sizes=lambda p: texture_size(self.source, p))
        names = {lit['graph'].nodes[i]['name'] for i in lit['graph'].reachable(list(lit['terms'].values()))
                 if lit['graph'].nodes[i]['op'] == 'input'}
        self.assertTrue({'L.x', 'LIGHT.r'} <= names)

    def test_domain_programs_translate(self):
        from rc2016.games.rcra import shaders as rshaders
        data = self.source.asset_data(self.key(JELLYFISH, 'materialgraph'))
        shader = rshaders.pass_shader(data, 2)
        self.assertEqual(shader.stage, 'ds')
        tr = rshaders.translate(shader)
        self.assertEqual(tr.status(), 'ready')
        names = {tr.graph.nodes[i]['name'] for i in tr.graph.reachable(list(tr.outputs.values()))
                 if tr.graph.nodes[i]['op'] == 'input'}
        self.assertTrue(any(n.startswith('CP0.') for n in names) and any(n.startswith('PATCH.') for n in names))
        self.assertIn('vDomain.x', names)

    def test_engine_compute_programs_with_groupshared_memory_translate(self):
        """Every engine compute program that declares groupshared memory runs through the translator (stores and atomics
        recorded, a value another thread wrote unknown) without a translator error."""
        import struct
        from rc2016.shaders import dxbc, sm5
        data = (Path(GAME) / lighting.EXECUTABLE).read_bytes()
        count = 0
        for at, size in dxbc.find_containers(data):
            try:
                shader = dxbc.shader(data[at:at + size])
            except dxbc.DxbcError:
                continue
            if shader.stage != 'cs' or not any(d.opcode.startswith('dcl_tgsm') for d in shader.program.declarations):
                continue
            if len(shader.program.instructions) > 400:
                continue
            tr = sm5.Translator(shader).run()
            self.assertFalse([u for u in tr.unsupported if 'groupshared' in u or 'atomic' in u], at)
            count += 1
            if count >= 12:
                break
        self.assertGreater(count, 0)
        del struct


if __name__ == '__main__':
    unittest.main()
