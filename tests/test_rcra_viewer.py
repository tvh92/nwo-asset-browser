"""Rift Apart translated materials in the viewer: the GLSL emitter (rc2016.viewer.rcra_glsl), the receipts
(rc2016.games.rcra.viewer_materials), the forward rule toggle, and (Windows, real install) a draw through the model viewport.

Real-install tests read RCRA_GAME and skip without it; the viewport test also needs Windows OpenGL and uses a temporary APPDATA.
"""
import ctypes
import os
import tempfile
import unittest
from unittest import mock

from rc2016.viewer import rcra_glsl
from rc2016.viewer.viewport import forward_rule, split_forward, TRANSLATED_FORWARD

GAME = os.environ.get('RCRA_GAME')
DOME = 'characters/enemy/enm_neo_nefarious_trooper/materials/enm_neo_nefarious_trooper_dome.material'
EYE = 'characters/hero/hero_clank/materials/hero_clank_eyeinner.material'
VOLUME = 'material/characters/hero/hero_mj_invisible_transition_01/dimension_swap_proto_frame_grab_01.material'
PLACEHOLDER = 'characters/hero/hero_clank/materials/hero_clank_antennaglass.material'   # binds the float default_pink_shader_n
SWARMER = 'characters/enemy/enm_neo_nefarious_swarmer/enm_neo_nefarious_swarmer.model'
BUTTON = 'environment/clankpuzzles/cpz_gp_pressure_button_01.model'


def _recipe(terms, extra=()):
    """A recipe whose terms are constants or one 2D sample of textureUnit5 at UV0."""
    nodes = [dict(op='input', type='f', name='TEXCOORD.x', args=[]), dict(op='input', type='f', name='TEXCOORD.y', args=[]),
             dict(op='sample', type='v', args=[0, 1], binding='textureUnit5', dimension=8),
             dict(op='channel', type='f', args=[2], index=0), dict(op='const', type='f', value=0.0, args=[]),
             dict(op='const', type='f', value=0.5, args=[]), dict(op='const', type='b', value=False, args=[])] + list(extra)
    names = {'sample': 3, 'zero': 4, 'half': 5, 'false': 6}
    return dict(version=1, nodes=nodes, terms={k: names[v] if isinstance(v, str) else v for k, v in terms.items()},
                textures={'textureUnit5': 'a.texture'}, inputs=[], limitations=[])


SURFACE = dict(zip(('albedo.r', 'albedo.g', 'albedo.b', 'f0.r', 'f0.g', 'f0.b', 'gloss', 'normal.x', 'normal.y', 'normal.z',
                    'occlusion', 'emission', 'clip'),
                   ('sample', 'sample', 'sample', 'zero', 'zero', 'zero', 'half', 'zero', 'half', 'zero', 'half', 'zero',
                    'false')))
FORWARD = {'colour.r': 'sample', 'colour.g': 'sample', 'colour.b': 'sample', 'alpha': 'half', 'clip': 'false',
           'background.r': 'zero', 'background.g': 'zero', 'background.b': 'half'}


def _receipt(blend, surface=True, forward=True):
    layouts = {'textureUnit5': dict(kind='slice')}
    s = _recipe(SURFACE) if surface else None
    f = _recipe(FORWARD) if forward else None
    return dict(kind='rcra', blend=blend, surface=s, forward=f, layouts=layouts,
                translation_sha256=rcra_glsl.receipt_sha256(s, f, blend, layouts))


class EmitterTests(unittest.TestCase):
    def test_opaque_program_lights_the_surface_and_discards(self):
        source = rcra_glsl.fragment_source(_receipt('opaque', forward=False))['source']
        self.assertIn('uniform sampler2D uNativeTextextureUnit5;', source)
        self.assertEqual(source.count('texture2D(uNativeTextextureUnit5'), 1)   # one sample, read by three terms
        self.assertIn('rcraDiffuse(rcraNormal, rcraAlbedo, rcraOcclusion)', source)
        self.assertIn('discard;', source)
        self.assertIn('vNativeToView * rcraWorldN', source)

    def test_blend_modes_choose_their_passes(self):
        blended = rcra_glsl.fragment_source(_receipt('blended'))['source']
        self.assertIn('mix(vec3(1.0), rcraBackground, rcraAlpha)', blended)
        self.assertIn('(rcraDiffuse(rcraNormal, rcraAlbedo, rcraOcclusion) + rcraSpecular', blended)
        additive = rcra_glsl.fragment_source(_receipt('additive', surface=False))['source']
        self.assertIn('vec3(1.0) + rcraAlpha * rcraBackground', additive)
        self.assertIn('rcraColour * rcraAlpha', additive)
        self.assertNotIn('rcraDiffuse(rcraNormal', additive)
        modulate = rcra_glsl.fragment_source(_receipt('modulate', surface=False))['source']
        self.assertIn('mix(vec3(1.0), rcraColour, rcraAlpha)', modulate)
        for source in (blended, additive):
            self.assertIn('if (%s == 0)' % rcra_glsl.PASS_UNIFORM, source)

    def test_forward_receipt_keeps_the_forward_discard_only(self):
        source = rcra_glsl.fragment_source(_receipt('blended'))['source']
        self.assertEqual(source.count('discard;'), 1)

    def test_atlas_layouts_and_unknown_inputs(self):
        receipt = _receipt('opaque', forward=False)
        receipt['surface']['nodes'][2]['args'] = [0, 1, 5]
        receipt['layouts'] = {'textureUnit5': dict(kind='volume', depth=7, columns=3, width=5, height=6)}
        self.assertIn('rcraVolume(uNativeTextextureUnit5', rcra_glsl.fragment_source(receipt)['source'])
        receipt['layouts']['textureUnit5']['kind'] = 'array'
        self.assertIn('rcraArray(uNativeTextextureUnit5', rcra_glsl.fragment_source(receipt)['source'])
        receipt['surface']['nodes'][0]['name'] = 'TEXCOORD7.x'
        with self.assertRaises(ValueError):
            rcra_glsl.fragment_source(receipt)

    def test_vertex_colour_reaches_the_program_squared(self):
        colour = dict(op='input', type='f', name='TEXCOORD4.y', args=[])
        recipe = _recipe(dict(SURFACE, **{'albedo.g': 7}), extra=[colour])
        layouts = {'textureUnit5': dict(kind='slice')}
        receipt = dict(kind='rcra', blend='opaque', surface=recipe, forward=None, layouts=layouts,
                       translation_sha256=rcra_glsl.receipt_sha256(recipe, None, 'opaque', layouts))
        source = rcra_glsl.fragment_source(receipt)['source']
        self.assertIn('varying vec4 vRcraColour;', source)
        self.assertIn('vRcraColour.y', source)
        self.assertEqual({rcra_glsl.GEOMETRY['TEXCOORD4.' + c] for c in 'xyzw'}, {'vRcraColour.' + c for c in 'xyzw'})
        vertex = rcra_glsl.vertex_source()
        self.assertIn('attribute vec4 aRcraColour;', vertex)
        self.assertIn('vRcraColour = aRcraColour * aRcraColour;', vertex)   # the engine squares before interpolation


class VertexColourPayloadTests(unittest.TestCase):
    """The loader passes a Rift Apart model's colour bytes on (loader._vertex_colours); R&C 2016 payloads do not change."""

    def test_colour_bytes_follow_the_drawn_meshes(self):
        from rc2016.games.rcra import model as rm
        from rc2016.viewer.loader import mesh_payload
        from tests.test_rcra_model import build_model
        for arrays in (True, False):
            decoded = rm.decode_model('t', data=build_model(), as_arrays=arrays)
            colours = mesh_payload(decoded, 'Highest', 0)['vertex_colours']
            self.assertEqual(colours.dtype.name, 'uint8')
            self.assertEqual(colours.tolist(), [[255, 0, 0, 128]] * 4)
        decoded = rm.decode_model('t', data=build_model(), as_arrays=True)
        self.assertNotIn('vertex_colours', mesh_payload(dict(decoded, game='rc2016'), 'Highest', 0))
        for mesh in decoded['meshes']:
            mesh['colors'] = None
        self.assertNotIn('vertex_colours', mesh_payload(decoded, 'Highest', 0))   # the viewer reads 1


class ForwardRuleTests(unittest.TestCase):
    def test_translated_rule_stands_only_with_native_translation(self):
        glass = dict(kind='rcra_arch_glass', blend='alpha_over')
        textures = {0: dict(forward=dict(kind=TRANSLATED_FORWARD, blend='blended', fallback=glass)),
                    1: dict(forward=dict(kind=TRANSLATED_FORWARD, blend='additive', fallback=None)), 2: {}}
        batches = [(0, 0, 3), (1, 3, 3), (2, 6, 3)]
        self.assertEqual(split_forward(batches, textures), ([(2, 6, 3)], [(0, 0, 3), (1, 3, 3)]))
        self.assertEqual(split_forward(batches, textures, native=False), ([(1, 3, 3), (2, 6, 3)], [(0, 0, 3)]))
        self.assertIs(forward_rule(textures[0], native=False), glass)


@unittest.skipUnless(GAME, 'RCRA_GAME is not set')
class RealReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rc2016.games.rcra.source import RcraSource
        cls.temp = tempfile.TemporaryDirectory()
        cls.source = RcraSource(GAME, os.path.join(cls.temp.name, 'out'))

    @classmethod
    def tearDownClass(cls):
        cls.source.close()
        cls.temp.cleanup()

    def receipt(self, path):
        from rc2016.games.rcra.material import _material_key
        from rc2016.games.rcra.viewer_materials import material_receipt
        return material_receipt(self.source, _material_key(self.source, path))

    def test_dome_is_blended_with_its_lit_surface(self):
        receipt, textures = self.receipt(DOME)
        self.assertEqual(receipt['blend'], 'blended')
        self.assertTrue(receipt['surface'] and receipt['forward'])
        self.assertEqual(set(textures), set(receipt['layouts']))
        rcra_glsl.fragment_source(receipt)

    def test_modulate_eye_has_no_surface(self):
        receipt, _textures = self.receipt(EYE)
        self.assertEqual(receipt['blend'], 'modulate')
        self.assertIsNone(receipt['surface'])

    def test_volume_texture_uploads_as_an_unmipmapped_atlas(self):
        receipt, textures = self.receipt(VOLUME)
        volumes = [b for b, layout in receipt['layouts'].items() if layout['kind'] == 'volume']
        self.assertTrue(volumes)
        for binding in volumes:
            layout, texture = receipt['layouts'][binding], textures[binding]
            self.assertTrue(texture['no_mipmaps'])
            rows = -(-layout['depth'] // layout['columns'])
            self.assertEqual((texture['width'], texture['height']),
                             (layout['columns'] * (layout['width'] + 2), rows * (layout['height'] + 2)))

    def test_float_placeholder_uploads_as_float(self):
        _receipt_, textures = self.receipt(PLACEHOLDER)
        floats = [t for t in textures.values() if t.get('gl_float')]
        self.assertTrue(floats)
        self.assertEqual(len(floats[0]['pixels']), 16 * floats[0]['width'] * floats[0]['height'])


@unittest.skipUnless(GAME and os.name == 'nt', 'RCRA_GAME is not set or not Windows')
class ViewportDrawTests(unittest.TestCase):
    def test_swarmer_draws_translated_materials(self):
        with tempfile.TemporaryDirectory() as appdata, mock.patch.dict(os.environ, {'APPDATA': appdata}):
            import tkinter as tk
            from rc2016.viewer.loader import SourcePreviewLoader
            from rc2016.viewer.viewport import ModelViewport
            loader = SourcePreviewLoader()
            archive = loader._ensure_archive(GAME)
            key = next(k for k in archive.resolve(SWARMER) if archive.entries[k]['type'] == 'model')
            payload = loader.load(GAME, key, SWARMER, output_root=os.path.join(appdata, 'out'))
            blends = sorted(e['native_translation']['blend'] for e in payload['textures'].values()
                            if e.get('native_translation'))
            self.assertIn('blended', blends)
            root = tk.Tk()
            try:
                root.geometry('320x320')
                view = ModelViewport(root)
                view.pack(fill='both', expand=True)
                root.update()
                if view._context_error or view._shading is None:
                    self.skipTest(view._context_error or 'no shading context')
                programs = []
                real = view._shading.native_program
                view._shading.native_program = lambda t, *a, **k: programs.append(t.get('kind')) or real(t, *a, **k)
                view.set_mode('Shaded')
                view.set_mesh(payload)
                view._draw()
                self.assertIsNone(view.shading_error)
                self.assertIn('rcra', programs)
            finally:
                root.destroy()

    def test_button_reads_its_vertex_colour(self):
        """The pressure button's base material (sav_gp_templegate_start_base_button) takes albedo and emission from
        TEXCOORD4.x; its model carries a colour section. The frame changes when the colour is replaced by the constant 1."""
        from rc2016.viewer.shading import RCRA_COLOUR_ATTRIBUTE
        with tempfile.TemporaryDirectory() as appdata, mock.patch.dict(os.environ, {'APPDATA': appdata}):
            import tkinter as tk
            from rc2016.viewer.loader import SourcePreviewLoader
            from rc2016.viewer.viewport import ModelViewport
            loader = SourcePreviewLoader()
            archive = loader._ensure_archive(GAME)
            key = next(k for k in archive.resolve(BUTTON) if archive.entries[k]['type'] == 'model')
            payload = loader.load(GAME, key, BUTTON, output_root=os.path.join(appdata, 'out'))
            self.assertEqual(len(payload['vertex_colours']), len(payload['vertices']))
            reads = [e for e in payload['textures'].values() if e.get('native_translation')
                     and any(n['op'] == 'input' and n['name'] == 'TEXCOORD4.x'
                             for n in (e['native_translation']['surface'] or {}).get('nodes', []))]
            self.assertTrue(reads)
            root = tk.Tk()
            try:
                root.geometry('240x240')
                view = ModelViewport(root)
                view.pack(fill='both', expand=True)
                root.update()
                if view._context_error or view._shading is None:
                    self.skipTest(view._context_error or 'no shading context')
                pointers = []
                real_pointer = view._shading.glVertexAttribPointer
                view._shading.glVertexAttribPointer = lambda *a: pointers.append(a[:5]) or real_pointer(*a)
                view.set_mode('Shaded')
                view.set_mesh(payload)
                self.assertEqual(bytes(view._rcra_colour), payload['vertex_colours'].tobytes())
                coloured = _frame(view)
                self.assertIsNone(view.shading_error)
                self.assertIn((RCRA_COLOUR_ATTRIBUTE, 4, view.GL_UNSIGNED_BYTE, 1, 0), pointers)   # normalized bytes
                view._rcra_colour = None
                self.assertNotEqual(coloured, _frame(view))
            finally:
                root.destroy()


def _frame(view):
    """The viewport's back buffer (RGB bytes) read just before the swap of one draw."""
    real, grabbed = view._gdi32, {}

    class Grab:
        def __getattr__(self, name):
            return getattr(real, name)

        def SwapBuffers(self, hdc):
            gl = view._gl
            w, h = max(1, view.winfo_width()), max(1, view.winfo_height())
            pixels = (ctypes.c_ubyte * (w * h * 3))()
            gl.glPixelStorei(0x0D05, 1)   # GL_PACK_ALIGNMENT
            gl.glReadBuffer(0x0405)       # GL_BACK
            gl.glReadPixels.argtypes = [ctypes.c_int] * 4 + [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
            gl.glReadPixels(0, 0, w, h, 0x1907, 0x1401, pixels)   # GL_RGB, GL_UNSIGNED_BYTE
            grabbed['pixels'] = bytes(pixels)
            return real.SwapBuffers(hdc)
    view._gdi32 = Grab()
    try:
        view._draw()
    finally:
        view._gdi32 = real
    return grabbed['pixels']


if __name__ == '__main__':
    unittest.main()
