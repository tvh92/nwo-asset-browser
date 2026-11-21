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

import numpy as np

from rc2016.games.rcra import lighting
from rc2016.viewer import rcra_glsl

GAME = os.environ.get('RCRA_GAME')
BLENDER = os.environ.get('RC2016_BLENDER') or r'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe'
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

    def test_cycles_anisotropic_keeps_the_class_4_alphas(self):
        """Cycles' alpha_x = R^2 / sqrt(1 - 0.9 A) on its tangent, alpha_y = R^2 sqrt(1 - 0.9 A): the class-4 lobe's a_t
        along the tangent and a_b across it, the tangent turned a quarter when a_b is the larger, up to the cap of 1."""
        rng = random.Random(4)
        capped = 0
        for _ in range(500):
            gloss, second, turn = rng.randrange(256) / 255.0, rng.randrange(128), rng.randrange(64)
            a_t, a_b, angle = lighting.anisotropic_lobes(gloss, second, turn)
            rough, aniso, rot1, rot2 = lighting.cycles_anisotropic(gloss, second, turn)
            aspect = math.sqrt(1.0 - 0.9 * aniso)
            ax, ay = rough * rough / aspect, rough * rough * aspect
            self.assertAlmostEqual(ax * ay, a_t * a_b, delta=1e-9)
            along, across = (a_b, a_t) if rot1 else (a_t, a_b)
            self.assertGreaterEqual(along, across)
            if aniso < lighting.CYCLES_MAX_ANISOTROPY:
                self.assertAlmostEqual(ax / along, 1.0, delta=1e-9)
                self.assertAlmostEqual(ay / across, 1.0, delta=1e-9)
            else:
                capped += 1
                self.assertLess(ax / ay, along / across)
            self.assertAlmostEqual((rot2 - rot1) * 2 * math.pi, angle, delta=1e-12)
        self.assertLess(capped, 500)

    def test_blender_constants_follow_the_reader(self):
        text = (Path(__file__).resolve().parents[1] / 'rc2016' / 'export' / 'rcra_blender.py').read_text()
        line = next(x for x in text.splitlines() if x.startswith('TANGENT_STEP, TANGENT_BIAS, ROTATION_STEP'))
        values = [eval(v) for v in line.split('=', 1)[1].split(',')]   # noqa: S307 (the module's own literal line)
        self.assertEqual(values, [lighting.TANGENT_STEP, lighting.TANGENT_BIAS, lighting.ROTATION_STEP,
                                  lighting.SECOND_GLOSS_STEP])
        self.assertIn('ANISOTROPIC_CLASS = %d' % lighting.ANISOTROPIC, text)

    @unittest.skipUnless(os.path.isfile(BLENDER), 'Blender is not installed')
    def test_blender_anisotropic_group_renders_the_class_4_lobes(self):
        """The Blender group, rendered in Cycles on a plane seen along its normal under one sun, against the analytic
        Cycles lobes of lighting.cycles_anisotropic on the decoded tangent: the group feeds Cycles what the mapping says."""
        import json
        import subprocess
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / 'aniso.py'
            script.write_text(ANISOTROPIC_SCRIPT)
            run = subprocess.run([BLENDER, '--background', '--factory-startup', '--python-exit-code', '1', '--python',
                                  str(script), '--', str(Path(__file__).resolve().parents[1]), folder,
                                  json.dumps(ANISOTROPIC_CASES)], capture_output=True, text=True, timeout=600)
        line = next((x for x in run.stdout.splitlines() if x.startswith('ANISO ')), None)
        self.assertIsNotNone(line, run.stdout[-2000:] + run.stderr[-2000:])
        for case, got in zip(ANISOTROPIC_CASES, json.loads(line[6:])):
            gloss, second, turn, fx, fy, sign, theta, phi = case
            g = lighting.anisotropic_tangent(fx, fy, sign)
            tangent = (g[0], -g[2], g[1])   # game (x, y, z) -> Blender (x, -z, y)
            t, p = math.radians(theta), math.radians(phi)
            light = (math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t))
            rough, aniso, rot1, rot2 = lighting.cycles_anisotropic(gloss, second, turn)
            want = 0.5 * (_cycles_lobe(rough, aniso, rot1, tangent, light) + _cycles_lobe(rough, aniso, rot2, tangent, light))
            self.assertAlmostEqual(got, want, delta=1e-4 * max(want, 1.0), msg=str(case))

    def test_probe_records_follow_the_tables(self):
        from rc2016.games.rcra import probe
        r = probe.record(2, 3)
        self.assertEqual((r[0], r[4], r[8], r[12], r[16], r[48], r[52]), (0.0, 3.0, probe.RECORD_TABLES['a'][2], 0.08, 1 / 32,
                                                                         128.0, 7.0))
        faces = probe.face_directions(4)
        self.assertTrue(np.allclose(np.linalg.norm(faces, axis=-1), 1.0))
        from rc2016.games.rcra.blender_export import cube_lookup
        marked = np.zeros((6, 4, 4, 3))
        marked[1, 2, 3] = 1.0
        self.assertAlmostEqual(float(cube_lookup(marked, faces[1, 2, 3])[0]), 1.0)   # a texel's direction finds that texel

    def test_ambient_emitter_reads_the_probe_at_its_lod(self):
        nodes = [dict(op='input', type='f', name=n, args=[]) for n in ('normal.x', 'normal.y', 'normal.z', 'gloss')]
        nodes.append(dict(op='const', type='f', value=0.0, args=[]))
        nodes.append(dict(op='sample', type='f', binding='g_EnvProbeDefault', opcode='sample_l', args=[0, 1, 2, 0, 3, 3, 3, 3]))
        nodes.append(dict(op='sample', type='f', binding='g_BrdfLookup', opcode='sample_l', args=[3, 3, 4, 3, 4, 4, 4, 4]))
        nodes.append(dict(op='channel', type='f', index=0, args=[5]))
        nodes.append(dict(op='channel', type='f', index=1, args=[6]))
        nodes.append(dict(op='mul', type='f', args=[7, 8]))
        recipe = dict(nodes=nodes, terms={'ambient.r': 9, 'ambient.g': 9, 'ambient.b': 9}, lighting_class=0, offset=0)
        text = rcra_glsl.ambient_source(recipe)
        self.assertIn('rcraProbe(vec3(rN.x, rN.y, rN.z), rG)', text)
        self.assertIn('rcraBrdf(rG, rG)', text)
        nodes[6]['args'][2] = 0   # a slice other than the constant 0
        with self.assertRaises(ValueError):
            rcra_glsl.ambient_source(recipe)

    def test_profile_texel_is_a_d3d_load(self):
        texels = [(k / 255.0, 0.0, 0.0, 1.0) for k in range(128)]
        self.assertEqual(lighting.profile_texel(texels, 3), (67 / 255.0, 0.0, 0.0, 1.0))
        self.assertEqual(lighting.profile_texel(texels[:70], 9), (0.0, 0.0, 0.0, 0.0))   # past the end reads 0

    def test_anisotropic_tangent_is_unit(self):
        for fx, fy in ((0, 0), (255, 255), (511, 0), (300, 211)):
            for sign in (0, 1):
                t = lighting.anisotropic_tangent(fx, fy, sign)
                q = (fx * lighting.TANGENT_STEP + lighting.TANGENT_BIAS) ** 2 + (fy * lighting.TANGENT_STEP + lighting.TANGENT_BIAS) ** 2
                if q < 2.0:   # inside the octahedral disc the writers fill
                    self.assertAlmostEqual(sum(x * x for x in t), 1.0, delta=1e-9)
                    self.assertEqual(t[2] < 0, bool(sign))


# Cases for the Blender group render: (gloss, second gloss field, rotation field, tangent fields x, y, z sign, L theta, phi).
ANISOTROPIC_CASES = ((0.3, 70, 0, 400, 255, 0, 25.0, 0.0), (0.3, 70, 10, 400, 255, 1, 35.0, 60.0),
                     (0.6, 30, 40, 255, 255, 0, 30.0, 120.0), (0.6, 30, 0, 120, 255, 1, 20.0, -40.0),
                     (0.9, 0, 25, 330, 255, 0, 30.0, 10.0))
ANISOTROPIC_SCRIPT = r'''
import json, math, os, sys
import bpy
import numpy as np
root, out_dir, cases = sys.argv[sys.argv.index('--') + 1:]
sys.path.insert(0, root)
from rc2016.export.rcra_blender import build_anisotropic_group
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device, scene.cycles.samples, scene.cycles.use_denoising, scene.cycles.max_bounces = 'CPU', 1, False, 0
scene.render.resolution_x = scene.render.resolution_y = 4
scene.render.image_settings.file_format, scene.render.image_settings.color_depth = 'OPEN_EXR', '32'
scene.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('w')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.0
scene.world = world
bpy.ops.mesh.primitive_plane_add(size=100)
material = bpy.data.materials.new('aniso')
material.use_nodes = True
tree = material.node_tree
tree.nodes.clear()
out = tree.nodes.new('ShaderNodeOutputMaterial')
group = tree.nodes.new('ShaderNodeGroup')
group.node_tree = build_anisotropic_group()
tree.links.new(group.outputs['Shader'], out.inputs['Surface'])
group.inputs['Albedo'].default_value = (0, 0, 0, 1)
group.inputs['F0'].default_value = (1, 1, 1, 1)
normal = tree.nodes.new('ShaderNodeCombineXYZ')
normal.inputs[2].default_value = 1.0
tree.links.new(normal.outputs[0], group.inputs['Normal'])
bpy.context.object.data.materials.append(material)
camera = bpy.data.objects.new('c', bpy.data.cameras.new('c'))
camera.data.type, camera.data.ortho_scale, camera.location = 'ORTHO', 0.01, (0, 0, 5)
scene.collection.objects.link(camera)
scene.camera = camera
sun = bpy.data.objects.new('s', bpy.data.lights.new('s', 'SUN'))
sun.data.energy, sun.data.angle = 1.0, 0.0
scene.collection.objects.link(sun)
from mathutils import Vector
values = []
for gloss, second, turn, fx, fy, sign, theta, phi in json.loads(cases):
    for name, value in (('Gloss', gloss), ('Second gloss field', second), ('Rotation field', turn), ('Tangent field x', fx),
                        ('Tangent field y', fy), ('Tangent z sign', sign)):
        group.inputs[name].default_value = value
    t, p = math.radians(theta), math.radians(phi)
    sun.rotation_euler = Vector((math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t))).to_track_quat('Z', 'Y').to_euler()
    scene.render.filepath = os.path.join(out_dir, 'aniso.exr')
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath)
    values.append(float(np.array(image.pixels[:]).reshape(-1, 4)[:, 0].mean()))
    bpy.data.images.remove(image)
print('ANISO ' + json.dumps(values))
'''


def _cycles_lobe(rough, aniso, rotation, tangent, light):
    """Cycles' anisotropic GGX (exact height-correlated Smith, F = 1) x N.L for V = N = +z, a unit sun of direction light."""
    cs, sn = math.cos(2 * math.pi * rotation), math.sin(2 * math.pi * rotation)
    t = (tangent[0] * cs - tangent[1] * sn, tangent[0] * sn + tangent[1] * cs)   # turned about +z, made orthogonal to it
    x = _unit([t[0], t[1], 0.0])
    y = (-x[1], x[0], 0.0)
    aspect = math.sqrt(1.0 - 0.9 * min(max(aniso, 0.0), 1.0))
    ax, ay = rough * rough / aspect, rough * rough * aspect
    h = _unit([light[0], light[1], light[2] + 1.0])
    hx, hy = sum(a * b for a, b in zip(h, x)), sum(a * b for a, b in zip(h, y))
    d = (hx / ax) ** 2 + (hy / ay) ** 2 + h[2] ** 2
    lx, ly = sum(a * b for a, b in zip(light, x)), sum(a * b for a, b in zip(light, y))
    lam = (-1.0 + math.sqrt(1.0 + ((ax * lx) ** 2 + (ay * ly) ** 2) / light[2] ** 2)) / 2.0
    return 1.0 / (math.pi * ax * ay * d * d) / (1.0 + lam) / 4.0


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

    def test_class_4_is_two_anisotropic_ggx_lobes(self):
        """The class-4 recipe equals the reading lighting.py gives (and the Blender export maps onto Cycles): Lambert, plus
        the average of two anisotropic GGX lobes on the tangent and on it turned by the rotation field."""
        recipe = lighting.direct(GAME, 4)
        rng = random.Random(9)
        dot = lambda a, b: sum(x * y for x, y in zip(a, b))   # noqa: E731
        cross = lambda a, b: (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])   # noqa: E731
        checked = 0
        while checked < 60:
            v = _inputs(rng, 4)
            n, light = [v['normal.' + c] for c in 'xyz'], [v['L.' + c] for c in 'xyz']
            t = lighting.anisotropic_tangent(v['extra.0'], v['extra.1'], v['extra.2'])
            if not all(math.isfinite(x) for x in t) or abs(dot(t, n)) > 0.99:
                continue
            view = _unit([-v['VIEW.' + c] for c in 'xyz'])
            h = _unit([a + b for a, b in zip(view, light)])
            nl, vh = max(min(dot(n, light), 1), 0), dot(view, h)
            a_t, a_b, angle = lighting.anisotropic_lobes(v['gloss'], v['extra.4'], v['extra.3'])
            nt, cs, sn = dot(n, t), math.cos(angle), math.sin(angle)
            t2 = [ni * nt * (1 - cs) + ti * cs + ci * sn for ni, ti, ci in zip(n, t, cross(n, t))]
            lobes = 0.0
            for tk in (t, t2):
                bk = _unit(cross(n, tk))
                s = dot(h, n) ** 2 + dot(h, tk) ** 2 / (a_t * a_t) + dot(h, bk) ** 2 / (a_b * a_b)
                lobes += 0.5 / (a_t * a_b * s * s)
            k = (0.75 * min(a_t, a_b) + 0.25 * max(a_t, a_b)) * 0.353553295 + 0.353553295
            vis = 0.25 / ((vh * (1 - k * k) + k * k) * (nl * (1 - k * k) + k * k))
            want = [v['LIGHT.' + c] * (v['albedo.' + c] * nl + nl * lobes * vis *
                                       (v['f0.' + c] + (1 - v['f0.' + c]) * (1 - vh) ** 5)) for c in 'rgb']
            for got, w in zip(lighting.evaluate(recipe, v), want):
                self.assertAlmostEqual(got, w, delta=0.01 * max(abs(w), 1e-3))
            checked += 1

    def test_class_1_reads_the_skin_profile_only_through_shadow_transmission(self):
        """The class-1 program loads texel 64 + the profile field (checked inside lighting.translate); with the review
        lights unshadowed the profile does not change the light, so the recipe folds it to 0 and says why."""
        recipe = lighting.direct(GAME, 1)
        names = {n['name'] for n in recipe['nodes'] if n['op'] == 'input'}
        self.assertFalse(names & set(lighting.PROFILE_INPUTS))
        self.assertIn('char-skin-profiles-srgb_1d.texture', ' '.join(recipe['limitations']))
        from rc2016.games.rcra.texture import texture_header
        header = texture_header(self.source, lighting.PROFILE_TEXTURE)
        self.assertEqual((header['format'], header['sd_width'], header['sd_height']), (29, 128, 1))   # RGBA8 sRGB, 1D

    def _furnace(self, cls, albedo, f0, gloss, extra=None, probe=(0.5, 0.7, 1.0)):
        """The class's ambient over a constant probe, with the engine's BRDF lookup (bilinear, clamped), over the probe."""
        from rc2016.games.rcra import engine_tables, surface
        lut = engine_tables.brdf_lookup(GAME)[0]

        def bilinear(u, v):
            x, y = min(max(u * 64 - 0.5, 0.0), 63.0), min(max(v * 64 - 0.5, 0.0), 63.0)
            x0, y0 = int(x), int(y)
            x1, y1 = min(x0 + 1, 63), min(y0 + 1, 63)
            fx, fy = x - x0, y - y0
            return list((lut[y0, x0] * (1 - fx) + lut[y0, x1] * fx) * (1 - fy) +
                        (lut[y1, x0] * (1 - fx) + lut[y1, x1] * fx) * fy)
        values = {'normal.x': 0.0, 'normal.y': 1.0, 'normal.z': 0.0, 'VIEW.x': 0.6, 'VIEW.y': -1.8, 'VIEW.z': 0.3, 'DEPTH': 2.0,
                  'gloss': gloss, 'occlusion': 1.0}
        values.update({'albedo.' + c: albedo for c in 'rgb'}, **{'f0.' + c: f0 for c in 'rgb'})
        values.update(extra or {'extra.%d' % k: 0.0 for k in range(5)})
        out = surface.evaluate_recipe(lighting.ambient(GAME, cls), values,
                                      lambda n, a: bilinear(a[0], a[1]) if n['binding'] == 'g_BrdfLookup' else list(probe) + [1.0])
        return [out['ambient.' + c] / p for c, p in zip('rgb', probe)]

    def test_ambient_reads_only_the_probe_and_brdf_lookup(self):
        for cls in lighting.CLASSES:
            recipe = lighting.ambient(GAME, cls)
            bindings = {n['binding'] for n in recipe['nodes'] if n['op'] == 'sample'}
            self.assertTrue(bindings <= set(lighting.AMBIENT_SAMPLED), cls)
            self.assertIn('g_EnvProbeDefault', bindings, cls)
            names = {n['name'] for n in recipe['nodes'] if n['op'] == 'input'}
            self.assertFalse(names & {'L.x', 'L.y', 'L.z', 'LIGHT.r'}, cls)

    def test_ambient_furnace(self):
        """A white diffuse surface under a constant probe reflects about the probe (the diffuse takes 1 - the lookup's
        specular part); a white metal (F0 1) about the probe at high gloss. Hair has its own transmission model."""
        for cls in (0, 1, 2, 4, 5):
            for gloss in (0.1, 0.5, 0.9):
                extra = {'extra.%d' % k: 0.0 for k in range(5)}
                if cls == 4:   # an isotropic lobe: second gloss = gloss
                    extra.update({'extra.1': 255.0, 'extra.4': float(round(gloss * 127))})
                for got in self._furnace(cls, 1.0, 0.04, gloss, extra):
                    self.assertAlmostEqual(got, 1.0, delta=0.08, msg=(cls, gloss))
        for got in self._furnace(0, 0.0, 1.0, 0.9):
            self.assertAlmostEqual(got, 1.0, delta=0.02)

    def test_brdf_lookup_is_read_through_its_loader(self):
        from rc2016.games.rcra import engine_tables
        lut = engine_tables.brdf_lookup(GAME)
        self.assertEqual(lut.shape, (4, 64, 64, 4))
        self.assertFalse(lut[3].any())
        self.assertFalse(lut[0, :, :, 2:].any())
        self.assertTrue((lut[0, :, :, :2] >= 0).all() and (lut[0, :, :, :2] <= 1).all())
        with open(os.path.join(GAME, engine_tables.EXECUTABLE), 'rb') as f:
            data = bytearray(f.read())
        image = engine_tables._Image(bytes(data))
        at = engine_tables.BRDF_LEAS[2][0] - image.base
        offset = next(raw + at - start for start, _v, raw, rsize in image.sections if start <= at < start + rsize)
        data[offset + 3] ^= 0x40   # move one loader displacement: a different build
        with self.assertRaises(ValueError):
            engine_tables.brdf_tables(bytes(data))

    def test_probe_convolution_keeps_a_constant_environment(self):
        """The engine's probe program over a constant source returns it unchanged at every output mip (energy is kept)."""
        import numpy as np
        from rc2016.games.rcra import probe
        _offset, blob = probe.program(GAME)
        chain = probe.box_chain(np.ones((6, probe.SIZE, probe.SIZE, 3), np.float32) * np.array([0.5, 1.0, 2.0], np.float32))
        for mip in (0, 4):
            out = probe.convolve(blob, chain, mip, faces=[2])[2]
            np.testing.assert_allclose(out.reshape(-1, 3), np.broadcast_to([0.5, 1.0, 2.0], (out.shape[0] * out.shape[1], 3)),
                                       rtol=1e-4)

    def test_forward_ambient_reads_one_review_probe(self):
        """The dome's forward ambient term: the probe lists translate with one review probe (the list loop's final
        iteration), its reads are the review probe, the probe array at cube 0 and the BRDF lookup, and the probe array
        looks along the same direction as the default probe's reflection (no parallax correction)."""
        from collections import defaultdict
        from rc2016.games.rcra import surface
        from rc2016.games.rcra.material import _graphs, _material_key
        from rc2016.games.rcra.texture import texture_size
        from rc2016.shaders import sm5
        key = _material_key(self.source, DOME)
        graph = self.source.asset_data(self.key(list(_graphs(self.source, key))[0], 'materialgraph'))
        result = surface.forward_ambient(graph, self.source.asset_data(key), lambda p: texture_size(self.source, p))
        recipe = surface.export_recipe(result)
        nodes = recipe['nodes']
        samples = [n for n in nodes if n['op'] == 'sample' and n['binding'] in surface.AMBIENT_ENGINE]
        self.assertEqual({n['binding'] for n in samples}, set(surface.AMBIENT_ENGINE))
        for n in samples:
            if n['binding'] == 'g_EnvProbeArray':
                self.assertEqual(nodes[n['args'][3]].get('value'), 0.0)
        g = sm5.Graph()
        g.nodes = [dict(n) for n in nodes]
        sharp = [n for n in samples if n['binding'] != 'g_BrdfLookup' and nodes[n['args'][4]].get('value') != 5.0]
        rng = random.Random(4)
        for _trial in range(4):
            values = defaultdict(float, {name: rng.uniform(0.1, 0.9) for name in recipe['inputs']})
            values.update({'VIEW.x': rng.uniform(-1, 1), 'VIEW.y': rng.uniform(-1, 1), 'VIEW.z': rng.uniform(1, 3)})
            roots = [a for n in sharp for a in n['args'][:3]]
            out = sm5.evaluate(g, roots, values, lambda n, a: [0.5, 0.5, 0.5, 1.0])
            directions = [_unit([out[a] for a in n['args'][:3]]) for n in sharp]
            for d in directions[1:]:
                for x, y in zip(d, directions[0]):
                    self.assertAlmostEqual(x, y, delta=1e-3)

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


class ProgramPlaceCacheTests(unittest.TestCase):
    class Disk:
        def __init__(self):
            self.store = {}

        def key(self, data, graphs, variant='', own_code=False):
            return repr((data, graphs, variant, own_code))

        def has(self, key):
            return key in self.store

        def get(self, key):
            return self.store.get(key)

        def put(self, key, value):
            self.store[key] = value

    def test_places_are_kept_and_read_back(self):
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as folder:
            exe = Path(folder) / lighting.EXECUTABLE
            blobs = {c: b'DXBC' + bytes([c]) * (8 + c) for c in lighting.CLASSES}
            body, found = bytearray(b'MZ' + bytes(30)), {}
            for c, blob in blobs.items():
                found[c] = (len(body), blob)
                body += blob + bytes(5)
            exe.write_bytes(bytes(body))
            disk = self.Disk()
            lighting._PROGRAMS.clear()
            with mock.patch.object(lighting, 'find_programs', return_value=found) as scan:
                self.assertEqual(lighting.programs(folder, disk), found)
                lighting._PROGRAMS.clear()
                self.assertEqual(lighting.programs(folder, disk), found)   # a new session: from the stored places
                self.assertEqual(scan.call_count, 1)
                stored = next(iter(disk.store.values()))
                self.assertEqual(stored, {str(c): [o, len(b)] for c, (o, b) in found.items()})
                disk.store[next(iter(disk.store))] = dict(stored, **{'0': [3, 8]})   # not a DXBC container: scan again
                lighting._PROGRAMS.clear()
                self.assertEqual(lighting.programs(folder, disk), found)
                self.assertEqual(scan.call_count, 2)
            lighting._PROGRAMS.clear()

    def test_read_places_rejects_incomplete_records(self):
        self.assertIsNone(lighting._read_places('missing', None))
        self.assertIsNone(lighting._read_places('missing', {'0': [0, 4]}))


if __name__ == '__main__':
    unittest.main()
