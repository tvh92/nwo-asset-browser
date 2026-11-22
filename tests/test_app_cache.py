"""The app's disk cache: size and Clear cache (nwo.app.cache)."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nwo.app import cache


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name) / '.cache'
        (self.root / 'rcra' / 'thumbnails').mkdir(parents=True)
        (self.root / 'rcra' / 'thumbnails' / 'a.png').write_bytes(b'x' * 100)
        (self.root / 'rcra' / 'index.pickle').write_bytes(b'y' * 50)
        (self.root / 'texture_preview').mkdir()
        (self.root / 'texture_preview' / 'b.rctx').write_bytes(b'z' * 25)
        (self.root / 'loose.json').write_bytes(b'{}')

    def test_usage_counts_every_file(self):
        self.assertEqual(cache.cache_usage(self.root), (177, 4))
        self.assertEqual(cache.cache_usage(self.root / 'missing'), (0, 0))

    def test_clear_removes_everything_and_keeps_the_folder(self):
        self.assertEqual(cache.clear_cache(self.root), dict(bytes=177, files=4, left=[]))
        self.assertTrue(self.root.is_dir())
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(cache.clear_cache(self.root / 'missing'), dict(bytes=0, files=0, left=[]))

    def test_an_entry_in_use_is_reported_and_the_rest_cleared(self):
        real = cache.shutil.rmtree

        def rmtree(path, *a, **k):
            if any(p.name == 'texture_preview' for p in [Path(path)]) or (Path(path) / 'b.rctx').exists():
                raise PermissionError('in use')
            return real(path, *a, **k)
        with mock.patch.object(cache.shutil, 'rmtree', side_effect=rmtree):
            result = cache.clear_cache(self.root)
        self.assertEqual(result['left'], ['texture_preview'])
        self.assertEqual(result['bytes'], 152)
        leftover = [p.name for p in self.root.iterdir()]
        self.assertEqual(len(leftover), 1)
        self.assertTrue(leftover[0].startswith(cache.TRASH_PREFIX))   # set aside: a reader never sees it half deleted
        self.assertEqual(cache.clear_cache(self.root)['left'], [])      # and removed on the next clear

    def test_size_text(self):
        self.assertEqual([cache.size_text(n) for n in (0, 812 * 1024, int(3.4 * 1024 ** 3))], ['0 B', '812 KB', '3.4 GB'])


if __name__ == '__main__':
    unittest.main()
