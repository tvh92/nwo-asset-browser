"""The build fingerprint follows every nwo module a pipeline build imports, not only formats/shaders/pipeline."""
from pathlib import Path
import tempfile
import unittest

from nwo.pipeline.run import ROOT, imported_sources


class ImportedSourcesTests(unittest.TestCase):
    def test_pipeline_build_reaches_the_viewer_effect_decoders(self):
        found = {p.relative_to(ROOT).as_posix() for p in imported_sources(ROOT, ['nwo/pipeline/holocon_effect.py'])}
        # ``from nwo.viewer import holocon_effect as fx`` inside a function, and its own imports in turn
        self.assertIn('nwo/viewer/holocon_effect.py', found)
        self.assertIn('nwo/viewer/idle_effect.py', found)

    def test_follows_imports_transitively_and_ignores_other_packages(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, text in {'nwo/__init__.py': '', 'nwo/a/__init__.py': '', 'nwo/b/__init__.py': '',
                               'nwo/a/start.py': 'import json\ndef f():\n    from nwo.b import middle\n',
                               'nwo/b/middle.py': 'from nwo.b.leaf import X\nimport numpy\n',
                               'nwo/b/leaf.py': 'X = 1\n', 'nwo/b/unused.py': 'Y = 2\n'}.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text(text)
            found = {p.relative_to(root.resolve()).as_posix() for p in imported_sources(root, ['nwo/a/start.py'])}
        self.assertEqual(found, {'nwo/b/__init__.py', 'nwo/b/middle.py', 'nwo/b/leaf.py'})


if __name__ == '__main__':
    unittest.main()
