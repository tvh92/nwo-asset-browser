"""Native translations persist on disk across sessions and load back exactly."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nwo.viewer.native_glsl import TranslationDiskCache, prepare_material_translation


def translation():
    return dict(status='ready', nodes=[dict(op='const', args=[], value=1.0), dict(op='input', args=[], channel=0)],
                outputs=dict(albedo=[0, 0, 0], gloss=1), normal=None, cb2={5: 0.5, 7: -0.0},
                sample=(1, 2), nested=[dict(span=(0, 4))], tag={'__tuple__': 'literal'},
                translation_sha256='x', game_equivalence=False)


class Archive:
    """Two .material/.materialgraph entries; a fresh instance has an empty in-memory cache."""
    entries = {1: dict(type='material'), 2: dict(type='materialgraph')}

    def resolve(self, path):
        return [1] if path.endswith('.material') else [2]


class TranslationDiskCacheTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.disk = TranslationDiskCache(Path(self.folder.name))

    def test_round_trip_keeps_tuples_and_non_string_keys(self):
        key = self.disk.key(b'material', [b'graph'])
        self.disk.put(key, translation())
        loaded = self.disk.get(key)
        self.assertEqual(loaded, translation())
        self.assertIsInstance(loaded['sample'], tuple)
        self.assertIsInstance(loaded['nested'][0]['span'], tuple)
        self.assertEqual(set(loaded['cb2']), {5, 7})
        self.assertEqual(loaded['tag'], {'__tuple__': 'literal'})

    def test_key_follows_material_and_graph_bytes(self):
        key = self.disk.key(b'material', [b'graph'])
        self.assertEqual(key, self.disk.key(b'material', [b'graph']))
        self.assertNotEqual(key, self.disk.key(b'material2', [b'graph']))
        self.assertNotEqual(key, self.disk.key(b'material', [b'graph2']))
        self.assertNotEqual(key, self.disk.key(b'material', [b'graph', b'graph']))

    def test_variant_key_is_separate_and_default_is_unchanged(self):
        import hashlib
        from nwo.viewer import native_glsl
        key = self.disk.key(b'material', [b'graph'])
        self.assertEqual(key, self.disk.key(b'material', [b'graph'], variant=''))
        legacy = hashlib.sha256(repr((native_glsl.TRANSLATION_DISK_CACHE_VERSION, native_glsl._TRANSLATION_CODE_SHA256,
                                      hashlib.sha256(b'material').hexdigest(),
                                      [hashlib.sha256(b'graph').hexdigest()])).encode()).hexdigest()
        self.assertEqual(key, legacy)
        self.assertNotEqual(key, self.disk.key(b'material', [b'graph'], variant='world_position'))

    def test_unstorable_and_damaged_entries_are_ignored(self):
        key = self.disk.key(b'material', [b'graph'])
        self.disk.put(key, dict(status='ready', raw=b'bytes'))
        self.assertIsNone(self.disk.get(key))
        self.disk.put(key, dict(status='ready', value=float('nan')))
        self.assertIsNone(self.disk.get(key))
        self.disk.put(key, translation())
        self.disk._file(key).write_bytes(b'{not json')
        self.assertIsNone(self.disk.get(key))

    def test_stats_count_hits_and_misses(self):
        key = self.disk.key(b'material', [b'graph'])
        self.assertIsNone(self.disk.get(key))
        self.disk.put(key, translation())
        self.disk.get(key); self.disk.get(key)
        self.assertEqual(self.disk.stats(), dict(hits=2, misses=1))

    def test_a_new_session_reuses_the_stored_translation(self):
        calls = []

        def prepare(graphs, material, instance, **context):
            calls.append((graphs, material, context))
            return translation()
        with patch('nwo.formats.archive.asset_data', side_effect=lambda archive, i: b'material' if i == 1 else b'graph'), \
             patch('nwo.formats.archive.asset_references', return_value=[dict(path='g.materialgraph')]), \
             patch('nwo.formats.materials.extract_material_parameters', return_value={}), \
             patch('nwo.shaders.bindings.prepare_translation', side_effect=prepare):
            for _session in range(2):
                archive = Archive()
                archive.translation_disk_cache = self.disk
                self.assertEqual(prepare_material_translation(archive, 'm.material'), translation())
        self.assertEqual(calls, [([b'graph'], b'material',dict(
            source_graph_paths=['g.materialgraph'],source_material_path='m.material'))])

    def test_world_position_translation_is_reused_across_sessions(self):
        from nwo.viewer.experimental_materials import world_position_translation
        calls = []

        def prepare(graphs, material, instance, world_position=False):
            calls.append((graphs, material, world_position))
            return translation()
        with patch('nwo.formats.archive.asset_data', side_effect=lambda archive, i: b'material' if i == 1 else b'graph'), \
             patch('nwo.formats.archive.asset_references', return_value=[dict(path='g.materialgraph')]), \
             patch('nwo.formats.materials.extract_material_parameters', return_value={}), \
             patch('nwo.shaders.bindings.prepare_translation', side_effect=prepare):
            for _session in range(2):
                archive = Archive()
                archive.translation_disk_cache = self.disk
                self.assertEqual(world_position_translation(archive, 'm.material'), translation())
        self.assertEqual(calls, [([b'graph'], b'material', True)])


class TranslationCodeKeyTests(unittest.TestCase):
    def test_excluded_modules_are_not_imported_by_the_translator(self):
        """A module outside the code hash must not be reachable from prepare_translation."""
        import ast
        from nwo.viewer.native_glsl import TRANSLATION_CODE_EXCLUDED
        root = Path(__file__).resolve().parents[1]
        shaders = root / 'nwo' / 'shaders'
        seen, stack = set(), ['bindings']
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            for node in ast.walk(ast.parse((shaders / f'{name}.py').read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom) and node.module == 'nwo.shaders':
                    stack.extend(a.name for a in node.names if (shaders / f'{a.name}.py').exists())
                elif isinstance(node, ast.ImportFrom) and (node.module or '').startswith('nwo.shaders.'):
                    stack.append(node.module.split('.')[2])
                elif isinstance(node, ast.Import):
                    stack.extend(a.name.split('.')[2] for a in node.names if a.name.startswith('nwo.shaders.'))
        self.assertIn('translator', seen)
        self.assertEqual({f'{n}.py' for n in seen} & TRANSLATION_CODE_EXCLUDED, set())
        for name in TRANSLATION_CODE_EXCLUDED:
            self.assertTrue((shaders / name).exists(), name)

    def test_hash_ignores_the_excluded_modules_and_native_glsl(self):
        from nwo.viewer import native_glsl
        self.assertIsNotNone(native_glsl._TRANSLATION_CODE_SHA256)
        sources = Path(native_glsl.__file__).resolve().parents[1]
        with patch.object(Path, 'read_bytes', autospec=True,
                          side_effect=lambda self: (_ for _ in ()).throw(AssertionError(self.name))
                          if self.name in native_glsl.TRANSLATION_CODE_EXCLUDED | {'native_glsl.py'} else b'x'):
            self.assertEqual(len(native_glsl._translation_code_sha256()), 64)
        self.assertTrue((sources / 'shaders' / 'translator.py').exists())


def _rc2016_imports(path, package):
    """Package-relative files of the nwo modules ``path`` imports anywhere (module level or in functions)."""
    import ast
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        names = []
        if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('nwo'):
            names = [node.module] + [node.module + '.' + a.name for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names if a.name.startswith('nwo')]
        for name in names:
            relative = Path(*name.split('.')[1:])
            if (package / relative).with_suffix('.py').exists():
                found.add(relative.with_suffix('.py').as_posix())
    return found


class RcraReceiptKeyTests(unittest.TestCase):
    """The Rift Apart receipt cache key (viewer_materials.CODE_FILES) covers the code a translation runs and reads."""

    def test_key_files_import_only_keyed_readers_or_listed_modules(self):
        from nwo.games.rcra import viewer_materials as vm
        package = Path(vm.__file__).resolve().parents[2]
        known = set(vm.CODE_FILES) | set(vm.READERS) | set(vm.NOT_TRANSLATION)
        for name in vm.CODE_FILES + vm.READERS + vm.NOT_TRANSLATION:
            self.assertTrue((package / name).exists(), name)
        for name in vm.CODE_FILES:
            self.assertEqual(_rc2016_imports(package / name, package) - known, set(), name)
        for name in ('games/rcra/model.py', 'games/rcra/morph.py', 'games/rcra/vertex.py', 'viewer/loader.py'):
            self.assertNotIn(name, vm.CODE_FILES)   # editing these keeps the cache

    @unittest.skipUnless(os.environ.get('RCRA_GAME'), 'RCRA_GAME is not set')
    def test_translation_runs_only_keyed_code(self):
        """Every function that runs while materials translate (cache off) is in CODE_FILES or READERS."""
        import sys
        from nwo.games.rcra import viewer_materials as vm
        from nwo.games.rcra.source import RcraSource
        package = Path(vm.__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            source = RcraSource(os.environ['RCRA_GAME'], Path(folder) / 'out')
            try:
                ran, monitor = set(), sys.monitoring
                tool = monitor.PROFILER_ID
                monitor.use_tool_id(tool, 'receipt-key-test')

                def started(code, _offset):
                    path = Path(code.co_filename)
                    if path.is_relative_to(package):
                        ran.add(path.relative_to(package).as_posix())
                    return monitor.DISABLE
                monitor.register_callback(tool, monitor.events.PY_START, started)
                monitor.set_events(tool, monitor.events.PY_START)
                try:
                    for path in ('material/characters/hero/hero_ratchet_head/hero_ratchet_head_fur.material',
                                 'material/visualeffect/global/gbl_camera_pane/gbl_camera_pane_hologram.material',
                                 'material/character/enemy/enm_molonoth_robot/enm_dr_nefarious_laser_tube.material'):
                        key = next(k for k in source.resolve(path) if source.entries[k]['type'] == 'material')
                        for translate in (vm.translate, vm.translate_shell):
                            try:
                                translate(source, key)
                            except vm.NotTranslated:
                                pass
                finally:
                    monitor.set_events(tool, 0)
                    monitor.free_tool_id(tool)
            finally:
                source.close()
        self.assertIn('games/rcra/surface.py', ran)
        self.assertEqual(ran - set(vm.CODE_FILES) - set(vm.READERS), set())

    @unittest.skipUnless(os.environ.get('RCRA_GAME'), 'RCRA_GAME is not set')
    def test_blender_entries_run_only_keyed_code(self):
        """Every function that runs while Blender entries are made (cache off) is in the entry cache's key."""
        import sys
        from nwo.games.rcra import blender_export as bexp, viewer_materials as vm
        from nwo.games.rcra.source import RcraSource
        package = Path(vm.__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            source = RcraSource(os.environ['RCRA_GAME'], Path(folder) / 'out')
            try:
                ran, monitor = set(), sys.monitoring
                tool = monitor.PROFILER_ID
                monitor.use_tool_id(tool, 'entry-key-test')

                def started(code, _offset):
                    path = Path(code.co_filename)
                    if path.is_relative_to(package):
                        ran.add(path.relative_to(package).as_posix())
                    return monitor.DISABLE
                monitor.register_callback(tool, monitor.events.PY_START, started)
                monitor.set_events(tool, monitor.events.PY_START)
                try:
                    for path in ('material/characters/hero/hero_ratchet_head/hero_ratchet_head_fur.material',
                                 'material/visualeffect/global/gbl_camera_pane/gbl_camera_pane_hologram.material',
                                 'material/character/enemy/enm_molonoth_robot/enm_dr_nefarious_laser_tube.material'):
                        bexp.material_entry(source, path, Path(folder) / 'units')
                finally:
                    monitor.set_events(tool, 0)
                    monitor.free_tool_id(tool)
            finally:
                source.close()
        self.assertIn('games/rcra/surface.py', ran)
        self.assertEqual(ran - set(vm.CODE_FILES) - set(vm.READERS) - set(bexp.IMAGE_CODE_FILES), set())


class BlenderEntryCacheTests(unittest.TestCase):
    """blender_export.cached_material_entry: an entry and its images come back from disk into another folder."""

    def test_entry_and_images_are_reused_in_a_new_folder(self):
        from nwo.games.rcra import blender_export as bexp
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = type('Source', (), {})()
            source.translation_disk_cache = TranslationDiskCache(tmp / 'cache' / 'native_translation')
            calls = []

            def entry(_source, path, folder, *_args):
                calls.append(path)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'a_c.png').write_bytes(b'png bytes')
                return dict(images={'t0': dict(path=str(folder / 'a_c.png'), source='texture/a_c.texture')},
                            recipe=dict(nodes=[1, 2]), blend='opaque')

            with patch.object(bexp, 'entry_cache_key', return_value='k' * 64), patch.object(bexp, 'material_entry', entry):
                first = bexp.cached_material_entry(source, 'a.material', tmp / 'one' / 'units')
                second = bexp.cached_material_entry(source, 'a.material', tmp / 'two' / 'units')
                self.assertEqual(calls, ['a.material'])
                self.assertEqual(second['images']['t0']['path'], str(tmp / 'two' / 'units' / 'a_c.png'))
                self.assertEqual((tmp / 'two' / 'units' / 'a_c.png').read_bytes(), b'png bytes')
                self.assertEqual(second['recipe'], first['recipe'])
                # A stored image that is gone means translating again.
                for blob in (tmp / 'cache' / 'blender_images').rglob('*.blob'):
                    blob.unlink()
                bexp.cached_material_entry(source, 'a.material', tmp / 'three' / 'units')
                self.assertEqual(calls, ['a.material', 'a.material'])


if __name__ == '__main__':
    unittest.main()
