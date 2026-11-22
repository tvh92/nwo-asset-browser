"""The game seam (nwo.games): folder detection, the 2016 adapter, capability gating in the window."""
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nwo import games
from nwo.games import RC2016, RCRA
from nwo.games.base import CACHE_ROOT, Game, GameUnavailable


def _rc2016_folder(root: Path) -> Path:
    """A minimal extracted R&C 2016 game folder: one archive root two levels down."""
    archive = root / 'CUSA01073' / 'Image0' / 'asset_archive'
    archive.mkdir(parents=True)
    for name in ('layout.csv', 'toc', 'dag', 'chunkmap.txt', 'a00s000'):
        (archive / name).write_bytes(b'\0' * 8)
    return root / 'CUSA01073'


def _rcra_folder(root: Path, magic=0x34E89035, dag=True, data_folder=True) -> Path:
    folder = root / 'Ratchet & Clank - Rift Apart'
    folder.mkdir(parents=True)
    (folder / 'toc').write_bytes(struct.pack('<II', magic, 0) + b'1TAD')
    if dag:
        (folder / 'dag').write_bytes(struct.pack('<I', 0xB8EF3955))
    if data_folder:
        (folder / 'd').mkdir()
    return folder


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # locate() remembers found roots in a sidecar; keep the test's out of the shared cache.
        sidecar = patch('nwo.formats.archive.LOCATE_SIDECAR', self.root / 'located_roots.json')
        sidecar.start()
        self.addCleanup(sidecar.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def test_extracted_ps4_archive_is_rc2016(self):
        self.assertIs(games.detect(_rc2016_folder(self.root)), RC2016)

    def test_install_with_toc_magic_dag_and_d_is_rift_apart(self):
        self.assertIs(games.detect(_rcra_folder(self.root)), RCRA)

    def test_wrong_toc_magic_or_missing_parts_are_not_rift_apart(self):
        for number, kwargs in enumerate((dict(magic=0x77AF12AF), dict(dag=False), dict(data_folder=False))):
            folder = _rcra_folder(self.root / str(number), **kwargs)
            self.assertFalse(RCRA.detect(folder), kwargs)
            with self.assertRaises(ValueError):
                games.detect(folder)

    def test_unknown_folder_names_the_supported_games(self):
        (self.root / 'empty').mkdir()
        with self.assertRaises(ValueError) as caught:
            games.detect(self.root / 'empty')
        self.assertIn('R&C 2016', str(caught.exception))
        self.assertIn('Rift Apart', str(caught.exception))
        with self.assertRaises(ValueError):
            games.detect(self.root / 'missing')

    def test_rift_apart_without_its_reader_says_so(self):
        folder = _rcra_folder(self.root)
        with patch.dict(sys.modules, {'nwo.games.rcra.source': None}):
            with self.assertRaisesRegex(GameUnavailable, 'Rift Apart reader not installed'):
                games.open_source(folder, 'source_preview', lambda *_a, **_k: None)

    def test_relative_output_is_under_the_game_cache(self):
        opened = []
        game = Game(id='fake', title='Fake', detect=lambda folder: True,
                    open_source=lambda folder, output, log: opened.append(output) or 'source')
        with patch('nwo.games._DETECT_ORDER', (game,)):
            self.assertEqual(games.open_source(self.root, 'source_preview', None), (game, 'source'))
            games.open_source(self.root, self.root / 'absolute', None)
        self.assertEqual(opened, [CACHE_ROOT / 'fake' / 'source_preview', self.root / 'absolute'])


class GameRecordTests(unittest.TestCase):
    def test_rc2016_declares_everything_but_root_motion_and_faces_and_keeps_its_caches_in_place(self):
        self.assertEqual(RC2016.capabilities, games.CAPABILITIES - {'root_motion', 'face_performances', 'layer_partners',
                                                                       'vertex_animation'})
        # The exact folder the loader and level viewer used before the seam, so nothing is rescanned.
        self.assertEqual(RC2016.cache_dir(), Path(__file__).resolve().parents[1] / '.cache')
        self.assertEqual(RCRA.cache_dir(), CACHE_ROOT / 'rcra')

    def test_rift_apart_capabilities(self):
        self.assertEqual(RCRA.capabilities, {'models', 'textures', 'export', 'animation', 'root_motion', 'face_performances',
                                             'layer_partners', 'vertex_animation'})
        self.assertTrue(RCRA.has_animation_library())
        self.assertFalse(RC2016.has_animation_library())   # 2016 finds its animsets through the scan's index
        for capability in ('levels', 'flash', 'pipeline', 'decoded_shading'):
            self.assertFalse(RCRA.has(capability), capability)
        self.assertEqual(RCRA.unavailable_tip, 'Not yet available for Rift Apart')
        self.assertEqual(RCRA.window_title, 'NWO Asset Browser — Rift Apart')

    def test_unknown_capability_or_hook_is_rejected(self):
        with self.assertRaises(ValueError):
            Game(id='x', title='X', detect=bool, open_source=print, capabilities=frozenset({'sound'}))
        with self.assertRaises(ValueError):
            Game(id='x', title='X', detect=bool, open_source=print, hooks={'play': print})
        with self.assertRaises(ValueError):
            RC2016.has('sound')

    def test_missing_hooks_raise_not_implemented(self):
        bare = Game(id='bare', title='Bare', detect=bool, open_source=print)
        with self.assertRaisesRegex(NotImplementedError, 'Bare'):
            bare.texture_header(None, 'a.texture')
        with self.assertRaisesRegex(NotImplementedError, 'Bare'):
            bare.list_models('folder', 'output')
        self.assertFalse(bare.shadow_only('a.material'))   # no hook: the game cannot tell, so nothing is left out
        with self.assertRaises(NotImplementedError):
            RC2016.list_models('folder', 'output')   # 2016 lists its models through the pipeline scan

    def test_default_output_is_beside_a_source_and_under_documents_for_an_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            programs, documents = Path(tmp) / 'Programs', Path(tmp) / 'Docs'
            source = Path(tmp) / 'Downloads' / 'CUSA01073'
            with patch.dict(os.environ, {'ProgramFiles': str(programs)}), \
                    patch('nwo.games.base.documents_folder', return_value=documents):
                self.assertEqual(RC2016.default_output(source), source.parent / 'CUSA01073 Animation Output')
                install = programs / 'Steam' / 'steamapps' / 'common' / 'Ratchet & Clank - Rift Apart'
                self.assertEqual(RCRA.default_output(install), documents / 'NWO Asset Browser' / 'Rift Apart Output')
                library = Path(tmp) / 'SteamLibrary' / 'steamapps' / 'common' / 'Ratchet & Clank - Rift Apart'
                self.assertEqual(RCRA.default_output(library), documents / 'NWO Asset Browser' / 'Rift Apart Output')


class LoaderDispatchTests(unittest.TestCase):
    def test_decode_dispatches_through_the_game(self):
        from nwo.viewer.loader import SourcePreviewLoader
        calls = []

        def decode(source, model_id, source_path, as_arrays):
            calls.append((source, model_id, source_path, as_arrays))
            return dict(meshes=[], joint_count=0), b'data'
        game = Game(id='fake', title='Fake', detect=bool, open_source=print, hooks={'decode_model': decode})
        loader = SourcePreviewLoader()
        self.assertIs(loader.game, RC2016)   # the default before any archive opens
        entry = loader._decode('archive', 'id', 'a.model', 'export', {}, game)
        self.assertEqual(calls, [('archive', 'id', 'a.model', False)])
        self.assertEqual(entry['data'], b'data')
        self.assertEqual(entry['decoded']['_archive_path'], 'a.model')

    def test_rift_apart_decode_goes_to_its_model_decoder(self):
        with patch('nwo.games.rcra.model.decode_for_viewer', return_value=('decoded', b'data')) as decode:
            self.assertEqual(RCRA.decode_model('source', 'id', 'a.model', True), ('decoded', b'data'))
        decode.assert_called_once_with('source', 'id', 'a.model', True)

    def test_rift_apart_animation_library_hook_reaches_its_clip_library(self):
        with patch('nwo.games.rcra.clip_library.animation_library', return_value='library') as build:
            self.assertEqual(RCRA.animation_library('s', b'model', 'a.model', clips=False), 'library')
        build.assert_called_once_with('s', b'model', 'a.model', False)

    def test_loader_builds_a_rift_apart_library_through_the_game(self):
        from nwo.viewer.loader import SourcePreviewLoader
        library = SimpleNamespace(clips=['clip'], animset_data={'a.animset': None})
        library.with_cinematic = lambda archive, path: library
        library.with_performances = lambda: library   # the view also lists face performances
        loader = SourcePreviewLoader()
        loader.game, loader.archive = RCRA, 'archive'
        cached = dict(decoded=dict(joint_count=3, _archive_path='a.model'), data=b'model', purpose='view',
                      animation=None, animation_built=None)
        with patch('nwo.games.rcra.clip_library.animation_library', return_value=library) as build:
            loader._build_animation_locked(cached, None)
        build.assert_called_once_with('archive', b'model', 'a.model', True)
        self.assertIs(cached['animation'], library)
        self.assertEqual((cached['decoded']['animsets'], cached['decoded']['animation_note']), (['a.animset'], ''))

    def test_rift_apart_texture_and_material_hooks_reach_its_readers(self):
        for hook, module, name, args in (
                ('material_textures', 'material', 'material_textures', ('s', [], {}, None)),
                ('model_texture_list', 'material', 'model_texture_list', ('s', [])),
                ('material_rules', 'material', 'material_rules', ('s', [], None)),
                ('effect_preview', 'material', 'effect_preview', ('s', b'', 'a.model')),
                ('texture_header', 'texture', 'texture_header', ('s', 'a.texture')),
                ('load_texture', 'texture', 'load_texture', ('s', 'a.texture')),
                ('export_textures', 'texture', 'export_textures', ('s', ['a.texture'], 'out', {'dds'}))):
            with patch(f'nwo.games.rcra.{module}.{name}', return_value='result') as reader:
                self.assertEqual(getattr(RCRA, hook)(*args), 'result', hook)
            reader.assert_called_once()
        rows = [dict(path='b/y.texture', folder='b'), dict(path='a/x.texture', folder='a')]
        with patch('nwo.games.rcra.texture.archive_textures', return_value=rows):
            self.assertEqual(RCRA.texture_paths('s'), ['a/x.texture', 'b/y.texture'])

    def test_textures_tab_dispatches_through_the_game(self):
        from nwo.viewer.loader import SourcePreviewLoader
        calls = []
        game = Game(id='fake', title='Fake', detect=bool, open_source=print, hooks=dict(
            texture_paths=lambda source: ['a/x_c.texture'],
            load_texture=lambda source, path, hd=False: calls.append(('load', path, hd)) or dict(
                path=path, width=1, height=1, format=28, pixels=b'\x10\x20\x30\x40', full_size=(1, 1), srgb=True),
            export_textures=lambda *args: calls.append(('export',) + args[1:]) or []))
        loader = SourcePreviewLoader()
        loader.archive, loader.game, loader.source_root = 'source', game, Path.cwd().resolve()
        self.assertEqual(loader.archive_textures(Path.cwd()),
                         [dict(path='a/x_c.texture', suffix='_c', materials=[], via_graph=False)])
        loader.export_textures(['a/x_c.texture'], 'out', {'dds'})
        self.assertEqual(calls, [('export', ['a/x_c.texture'], 'out', {'dds'}, False, None)])

    def rift_apart_maps(self, decided):
        from nwo.export.models import _MapCache
        from nwo.games.rcra.material import SLOT_BASE_COLOUR, SLOT_G, SLOT_NORMAL
        textures = {
            'c.texture': dict(format=28, width=2, height=2, pixels=bytes([200, 100, 50, 255] * 4), srgb=False),
            'n.texture': dict(format=49, width=2, height=2, pixels=bytes([128, 128] * 4), srgb=False),
            'g.texture': dict(format=28, width=2, height=2, pixels=bytes([64, 10, 0, 255] * 4), srgb=False),
            'g_rgb.texture': dict(format=28, width=2, height=2, pixels=bytes([10, 10, 10, 128] * 4), srgb=False),
            'm.texture': dict(format=28, width=2, height=2, pixels=bytes([255, 0, 77, 255] * 4), srgb=False),
            'float_n.texture': dict(format=16, width=2, height=2, pixels=struct.pack('<8f', *[2.0] * 8), srgb=False),
            'pink_n.texture': dict(format=16, width=2, height=2, pixels=struct.pack('<8f', *[128 / 255] * 8), srgb=False)}
        slots = {'a.material': [(SLOT_BASE_COLOUR, 'c.texture'), (SLOT_NORMAL, 'n.texture'), (SLOT_G, 'g.texture')],
                 'b.material': [(SLOT_NORMAL, 'float_n.texture')], 'c.material': [(SLOT_NORMAL, 'pink_n.texture')]}
        source = SimpleNamespace(asset_data=lambda key: key)
        with patch('nwo.games.rcra.material._material_key', side_effect=lambda s, path: path), \
                patch('nwo.games.rcra.material.material_slots', side_effect=lambda data: slots[data]), \
                patch('nwo.games.rcra_export.decoded_maps', side_effect=lambda s, key: dict(decided[key])), \
                patch('nwo.games.rcra.texture.load_texture_full', side_effect=lambda s, path: dict(textures[path])):
            return {key: RCRA.material_maps(source, dict(index=i, path=key), {}, _MapCache()) for i, key in enumerate(decided)}

    def test_rift_apart_export_maps_without_a_plain_writer_are_the_stand_in(self):
        stand_in = dict(layout=None, occlusion=False, g=None, m=None, reason='the writer does not translate: x')
        maps = self.rift_apart_maps({'a.material': stand_in, 'b.material': stand_in, 'c.material': stand_in})
        a, b, c = maps['a.material'], maps['b.material'], maps['c.material']
        self.assertEqual(a['sources'], {'_c': 'c.texture', '_n': 'n.texture'})   # _g is not read without a plain writer
        self.assertEqual(a['base'].shape, (2, 2, 3))
        self.assertEqual((a['alpha_mode'], a['double_sided'], a['emissive'], a['occlusion']), ('OPAQUE', False, None, False))
        self.assertTrue((a['mr'][..., 2] == 0).all() and (a['mr'][..., 1] == 255).all())   # non-metallic, fully rough
        self.assertTrue((a['mr'][..., 0] == 255).all())
        self.assertIsNotNone(a['normal'])
        self.assertIn('fully rough (the writer does not translate: x)', ' '.join(a['notes']))
        self.assertIn('Rift Apart portable material', a['notes'][-1])
        self.assertEqual((b['sources'], b['normal'], b['base_factor']), ({}, None, [0.8, 0.8, 0.8]))
        self.assertTrue(any('float_n.texture left out' in note and 'from 2 to 2' in note for note in b['notes']))
        # An R32G32_FLOAT normal in 0..1 is its values rounded to 8 bits, like the R8G8 one (z rebuilt).
        self.assertEqual(c['sources'], {'_n': 'pink_n.texture'})
        self.assertEqual(c['normal'].tolist(), a['normal'].tolist())
        self.assertTrue(any('pink_n.texture: stored as DXGI 16' in note for note in c['notes']))

    def test_rift_apart_export_maps_take_the_decoded_g_and_m_channels(self):
        maps = self.rift_apart_maps({'a.material': dict(layout='grey', occlusion=True, g='g.texture', m='m.texture', reason=None),
                                     'b.material': dict(layout='colored', occlusion=False, g='g_rgb.texture', m=None, reason=None)})
        grey, colored = maps['a.material'], maps['b.material']
        self.assertEqual(grey['sources'], {'_c': 'c.texture', '_n': 'n.texture', '_g': 'g.texture', '_m': 'm.texture'})
        gloss = (64 / 255) ** 0.5   # gloss² in red
        self.assertEqual(int(grey['mr'][0, 0, 1]), round((0.9725 - 0.7514 * gloss) ** 2 * 255))
        self.assertEqual(int(grey['mr'][0, 0, 0]), 77)   # occlusion: _m blue in the glTF occlusion channel
        self.assertEqual(int(grey['mr'][0, 0, 2]), 0)    # a grey F0 of 10/255 is a dielectric
        self.assertTrue(grey['occlusion'])
        self.assertIn('gloss² in red, grey F0 in green', ' '.join(grey['notes']))
        self.assertEqual(int(colored['mr'][0, 0, 1]), round((0.9725 - 0.7514 * 128 / 255) ** 2 * 255))   # gloss in alpha
        self.assertEqual((int(colored['mr'][0, 0, 0]), colored['occlusion']), (255, False))

    def test_rift_apart_plain_writer_check(self):
        """A writer whose gloss is sqrt(_g.r), F0 _g.g and occlusion _m.b at UV0 is plain; scaled or moved UVs are not."""
        from nwo.games.rcra_export import _plain

        def recipe(gloss_op='sqrt', u_scale=None):
            nodes = [dict(op='input', name='TEXCOORD.x', type='f', args=[]), dict(op='input', name='TEXCOORD.y', type='f', args=[])]
            u = 0
            if u_scale:
                nodes += [dict(op='const', type='f', value=u_scale, args=[]), dict(op='mul', type='f', args=[0, 2])]
                u = 3
            base = len(nodes)
            nodes += [dict(op='sample', binding='textureUnit7', args=[u, 1]), dict(op='sample', binding='textureUnit8', args=[0, 1])]
            nodes += [dict(op='channel', index=i, type='f', args=[base]) for i in range(4)]       # _g r g b a
            nodes += [dict(op='channel', index=2, type='f', args=[base + 1])]                   # _m b
            nodes += [dict(op=gloss_op, type='f', args=[base + 2])]
            terms = {'gloss': len(nodes) - 1, 'f0.r': base + 3, 'f0.g': base + 3, 'f0.b': base + 3, 'occlusion': base + 6}
            return dict(nodes=nodes, terms=terms, inputs=['TEXCOORD.x', 'TEXCOORD.y'], textures={})
        self.assertEqual(_plain(recipe(), 'textureUnit7', 'textureUnit8')[:2], ('grey', True))
        self.assertEqual(_plain(recipe('sat'), 'textureUnit7', 'textureUnit8')[:2], (None, True))   # gloss is r, not sqrt(r)
        self.assertEqual(_plain(recipe(u_scale=2.0), 'textureUnit7', 'textureUnit8')[:2], (None, True))   # tiled _g
        self.assertEqual(_plain(recipe(), None, 'textureUnit8')[0], None)

    def test_rift_apart_emission_and_alpha_forms(self):
        """Emission K x _m.r with a plain _c albedo, a constant emission, a forward alpha K x _c.a; a mixed form is refused."""
        from nwo.games.rcra_export import decoded_alpha, decoded_emission

        def recipe(emission=None, alpha=None, albedo_scale=None):
            nodes = [dict(op='input', name='TEXCOORD.x', type='f', args=[]), dict(op='input', name='TEXCOORD.y', type='f', args=[]),
                     dict(op='sample', binding='textureUnit5', args=[0, 1]), dict(op='sample', binding='textureUnit8', args=[0, 1])]
            nodes += [dict(op='channel', index=i, type='f', args=[2]) for i in range(4)]      # _c r g b a: 4..7
            nodes += [dict(op='channel', index=0, type='f', args=[3])]                        # _m r: 8
            nodes += [dict(op='const', type='f', value=2.5, args=[])]                         # 9
            nodes += [dict(op='mul', type='f', args=[8, 9]), dict(op='mul', type='f', args=[7, 9])]   # 10: 2.5 _m.r, 11: 2.5 _c.a
            nodes += [dict(op='mul', type='f', args=[8, 7])]                                  # 12: _m.r _c.a
            albedo = [4, 5, 6]
            if albedo_scale:
                nodes += [dict(op='mul', type='f', args=[4, 9])]
                albedo = [len(nodes) - 1, 5, 6]
            terms = {'albedo.r': albedo[0], 'albedo.g': albedo[1], 'albedo.b': albedo[2]}
            if emission is not None:
                terms['emission'] = emission
            if alpha is not None:
                terms['alpha'] = alpha
            return dict(nodes=nodes, terms=terms, inputs=['TEXCOORD.x', 'TEXCOORD.y'], textures={})
        form, reason = decoded_emission(recipe(emission=10), 'textureUnit5', None, 'textureUnit8')
        self.assertEqual((form['binding'], form['channel'], reason), ('textureUnit8', 0, None))
        self.assertAlmostEqual(form['k'], 2.5, places=5)
        form, _ = decoded_emission(recipe(emission=9), 'textureUnit5', None, 'textureUnit8')
        self.assertEqual((form['binding'], form['k']), (None, 2.5))   # a constant emission
        self.assertEqual(decoded_emission(recipe(emission=12), 'textureUnit5', None, 'textureUnit8')[0], None)   # mixed
        self.assertIn('albedo', decoded_emission(recipe(emission=10, albedo_scale=True), 'textureUnit5', None,
                                                 'textureUnit8')[1])
        form, _ = decoded_alpha(recipe(alpha=11), 'textureUnit5')
        self.assertEqual(form['binding'], 'textureUnit5')
        self.assertEqual(form['k'], 1.0)   # 2.5 _c.a: the glTF factor is capped at 1
        self.assertIsNone(decoded_alpha(recipe(alpha=10), 'textureUnit5')[0])   # _m, not _c alpha

    def test_rift_apart_export_maps_carry_emission_and_alpha(self):
        maps = self.rift_apart_maps({
            'a.material': dict(layout=None, occlusion=False, g=None, m=None, reason='x', blend='blended',
                               emission=dict(binding='textureUnit8', channel=0, k=4.0, path='m.texture', label='_m.r'),
                               alpha=dict(binding='textureUnit5', k=0.5, label='0.5 x _c.a')),
            'b.material': dict(layout=None, occlusion=False, g=None, m=None, reason='x', blend='additive',
                               alpha_reason='glTF has no additive blend')})
        a, b = maps['a.material'], maps['b.material']
        self.assertEqual(a['base'].shape, (2, 2, 4))   # _c alpha kept for BLEND
        self.assertEqual((a['alpha_mode'], a['alpha'], a['emissive_strength']), ('BLEND', 0.5, 4.0))
        self.assertEqual(a['sources']['_e'], 'm.texture')
        self.assertEqual(tuple(int(x) for x in a['emissive'][0, 0]), tuple(int(x) for x in a['base'][0, 0, :3]))   # _m.r = 1
        self.assertIn('emission from the writer: _c x _m.r x 4', ' '.join(a['notes']))
        self.assertEqual((b['alpha_mode'], b['emissive']), ('OPAQUE', None))
        self.assertIn('drawn additive in the game; exported opaque: glTF has no additive blend', ' '.join(b['notes']))

    def test_rift_apart_export_maps_bake_the_programs_over_uv0(self):
        """albedo = _c.rgb x (2, 1, 0.5), F0 0.02, gloss _c.a, emission 3 _c.r in the writer; forward alpha 0.5 _c.g."""
        def n(op, *args, **extra):
            return dict(op=op, type='f', args=list(args), **extra)
        uv = [n('input', name='TEXCOORD.x'), n('input', name='TEXCOORD.y'),
              n('sample', 0, 1, opcode='sample', binding='textureUnit5', dimension=3, offsets=[0, 0, 0])]
        c = [n('channel', 2, index=i) for i in range(4)]                                   # 3..6
        k = [n('const', value=v) for v in (2.0, 1.0, 0.5, 3.0, 0.02)]                       # 7..11
        surface = dict(nodes=uv + c + k + [n('mul', 3 + i, 7 + i) for i in range(3)] + [n('mul', 3, 10)],
                       terms={'albedo.r': 12, 'albedo.g': 13, 'albedo.b': 14, 'f0.r': 11, 'f0.g': 11, 'f0.b': 11,
                              'gloss': 6, 'occlusion': 8, 'emission': 15})
        forward = dict(nodes=uv + c + k + [n('mul', 4, 9)], terms={'alpha': 12})
        plan = dict(albedo=None, f0gloss=None, occlusion=None, emission=None, alpha=None)
        maps = self.rift_apart_maps({'a.material': dict(
            layout=None, occlusion=False, g=None, m=None, reason='x', blend='blended', emission=None, alpha=None, bake=plan,
            bindings={'textureUnit5': 'c.texture'}, surface=surface, forward=forward)})
        a = maps['a.material']
        self.assertEqual(a['base'].shape, (2, 2, 4))
        self.assertEqual(int(a['base'][0, 0, 3]), round(0.5 * 100))   # 0.5 x 100/255 in the base map's alpha
        self.assertEqual((a['alpha_mode'], a['alpha'], a['occlusion']), ('BLEND', 1.0, False))   # occlusion 1: none
        self.assertEqual(int(a['mr'][0, 0, 1]), round((0.9725 - 0.7514) ** 2 * 255))            # gloss 1
        self.assertEqual(int(a['mr'][0, 0, 2]), 0)                                               # F0 0.02: dielectric
        self.assertAlmostEqual(a['emissive_strength'], 3 * 200 / 255, places=5)
        self.assertEqual(a['emissive'][0, 0].tolist(), [255, round(255 * 1.055 * (100 / 255) ** (1 / 2.4) - 14.025),
                                                        round(255 * 1.055 * (25 / 255) ** (1 / 2.4) - 14.025)])
        self.assertEqual(a['sources'], {'_n': 'n.texture', 'textureUnit5': 'c.texture'})   # _c and _g are not read
        notes = ' '.join(a['notes'])
        for text in ('albedo baked from the G-buffer writer over UV0 (2x2)', 'F0 and gloss baked',
                     'emission baked from the G-buffer writer: albedo x E, strength 2.35294',
                     "alpha-over (glTF BLEND): the forward program's alpha baked over UV0 (2x2)"):
            self.assertIn(text, notes)

    def test_gltf_occlusion_shares_the_metal_rough_image(self):
        import numpy as np
        from nwo.export.models import ModelExporter
        m = dict(obj_name='m', alpha=1.0, base=None, normal=None, emissive=None, emissive_strength=0.0, alpha_mode='OPAQUE',
                 double_sided=False, png={}, mr=np.full((1, 1, 3), 128, np.uint8), occlusion=True)
        doc = ModelExporter._build_gltf(None, 'x', [], {0: m}, None, []).doc
        material = doc['materials'][0]
        self.assertEqual(material['occlusionTexture'], material['pbrMetallicRoughness']['metallicRoughnessTexture'])

    def test_export_reads_materials_through_the_loaders_game(self):
        import inspect
        from nwo.export import models
        source = inspect.getsource(models.ModelExporter.export)
        for call in ('game.material_maps(', 'game.export_textures(', 'game.shadow_only('):
            self.assertIn(call, source)
        with patch('nwo.export.models.material_maps', return_value='maps') as maps:
            self.assertEqual(RC2016.material_maps('archive', {}, {}, None), 'maps')   # 2016 keeps its own function
        maps.assert_called_once_with('archive', {}, {}, None)

    def test_rift_apart_lists_its_models_in_process(self):
        closed = []
        # k2's toc header names no streamed sections (streamed size 0): a model without geometry.
        headers = {'k1': struct.pack('<III', 0x9D2C0FA9, 400, 1200) + bytes(24), 'k2': struct.pack('<III', 0x9D2C0FA9, 400, 0) + bytes(24)}
        source = SimpleNamespace(ids_of_type=lambda kind: {'model': ['k2', 'k1']}[kind],
                                 entries={'k1': dict(path='a/one.model'), 'k2': dict(path='b/two.model')},
                                 asset_header=headers.get, close=lambda: closed.append(True))
        with patch('nwo.games.rcra_game.open_source', return_value=source):
            rows = RCRA.list_models('install', 'output')
        self.assertEqual([(r.model_id, r.archive_path, r.no_geometry) for r in rows],
                         [('k1', 'a/one.model', False), ('k2', 'b/two.model', True)])
        self.assertEqual(closed, [True])

    def test_levels_need_the_levels_capability(self):
        from nwo.levels import scene
        with patch('nwo.levels.scene.detect_game', return_value=RCRA):
            with self.assertRaisesRegex(ValueError, 'Levels are not yet available for Rift Apart'):
                scene.open_archive(tempfile.gettempdir())


class CapabilityGatingTests(unittest.TestCase):
    """AssetBrowserApp._gate/_apply_game on real Tk widgets (the full window needs OpenGL)."""

    def setUp(self):
        try:
            import tkinter as tk
            from tkinter import ttk
            self.root = tk.Tk()
            self.root.withdraw()
        except Exception as error:  # no Tk runtime or no display
            self.skipTest(f'Tk unavailable: {error}')
        from nwo.app.window import ASSET_TYPE_CAPABILITY, HELP, AssetBrowserApp
        self.HELP = HELP
        shown = []
        app = SimpleNamespace(tk=tk, ttk=ttk, root=self.root, game=None, game_gates=[], options_changed=0,
                              asset_type_var=tk.StringVar(master=self.root, value='Level'),
                              shading_var=tk.StringVar(master=self.root, value='Generic reading'),
                              backend_var=tk.StringVar(master=self.root, value='Transform only'),
                              browser=SimpleNamespace(show_asset_type=shown.append))
        for name in ('_tip', '_gate', '_game_has', '_pipeline_backend', '_apply_game'):
            setattr(app, name, getattr(AssetBrowserApp, name).__get__(app))
        app._options_changed = lambda: setattr(app, 'options_changed', app.options_changed + 1)
        self.buttons = {kind: ttk.Radiobutton(self.root, text=kind) for kind in ASSET_TYPE_CAPABILITY}
        for kind, button in self.buttons.items():
            app._gate(ASSET_TYPE_CAPABILITY[kind], HELP['asset_type'], button)
        self.backend = ttk.Combobox(self.root, state='readonly')
        app._gate('pipeline', HELP['backend'], self.backend, 'readonly')
        self.animation = ttk.Checkbutton(self.root)
        app._gate('animation', HELP['animations'], self.animation, None)
        self.app, self.shown = app, shown

    def tearDown(self):
        self.root.destroy()

    def tips(self):
        return {id(tip.widget): tip.text for _cap, _widget, _state, tip, _text in self.app.game_gates}

    def test_everything_is_offered_before_a_game_is_recognised(self):
        self.assertTrue(self.app._game_has('levels'))
        self.assertFalse(self.app._pipeline_backend())   # Transform only, as chosen

    def test_rift_apart_disables_what_it_lacks_and_rc2016_restores_it(self):
        app = self.app
        app._apply_game(RCRA)
        self.assertEqual(self.root.title(), 'NWO Asset Browser — Rift Apart')
        tips = self.tips()
        for kind in ('Level', 'Flash UI'):
            self.assertTrue(self.buttons[kind].instate(['disabled']), kind)
            self.assertEqual(tips[id(self.buttons[kind])], 'Not yet available for Rift Apart')
        for kind in ('Model', 'Texture'):
            self.assertFalse(self.buttons[kind].instate(['disabled']), kind)
            self.assertEqual(tips[id(self.buttons[kind])], self.HELP['asset_type'])
        self.assertTrue(self.backend.instate(['disabled']))
        self.assertEqual(tips[id(self.animation)], self.HELP['animations'])   # Rift Apart animates
        self.assertEqual((app.asset_type_var.get(), self.shown), ('Model', ['Model']))   # Level was showing
        self.assertEqual(app.shading_var.get(), 'All')
        self.assertTrue(app._pipeline_backend())   # its models come from its own index, not Transform only
        self.assertEqual(app.options_changed, 1)

        app._apply_game(RC2016)
        self.assertEqual(self.root.title(), 'NWO Asset Browser — R&C 2016')
        tips = self.tips()
        for button in self.buttons.values():
            self.assertFalse(button.instate(['disabled']))
            self.assertEqual(tips[id(button)], self.HELP['asset_type'])
        self.assertTrue(self.backend.instate(['readonly']) and not self.backend.instate(['disabled']))
        self.assertEqual(tips[id(self.animation)], self.HELP['animations'])
        self.assertFalse(app._pipeline_backend())


if __name__ == '__main__':
    unittest.main()
