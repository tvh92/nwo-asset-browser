"""Program links that do not wait: Shading's pending links and the model viewer's stand-in until they finish."""
import unittest

from nwo.viewer.shading import Shading, ShaderError
from nwo.viewer.viewport import ModelViewport


class FakeGLSL:
    """The Shading entry points a link uses; ``done`` and ``ok`` say what the driver reports for each program."""

    def __init__(self, shading, link_ok=True, compile_ok=True):
        self.next, self.done, self.calls = 1, set(), []
        self.link_ok, self.compile_ok = link_ok, compile_ok
        for name in ('glShaderSource', 'glCompileShader', 'glAttachShader', 'glBindAttribLocation', 'glLinkProgram',
                     'glDeleteShader', 'glDeleteProgram'):
            setattr(shading, name, self._recorder(name))
        shading.glCreateShader = shading.glCreateProgram = self._create
        shading.glGetProgramiv = self._program_iv
        shading.glGetShaderiv = lambda shader, pname, out: setattr(out._obj, 'value', int(self.compile_ok))
        shading.glGetShaderInfoLog = lambda shader, size, length, log: log.__setitem__(slice(0, 9), b'bad shade')
        shading.glGetProgramInfoLog = lambda program, size, length, log: log.__setitem__(slice(0, 8), b'bad link')
        shading.glGetUniformLocation = lambda program, name: len(name)

    def _recorder(self, name):
        return lambda *args: self.calls.append((name, args))

    def _create(self, *args):
        self.next += 1
        self.calls.append(('create', self.next))
        return self.next

    def _program_iv(self, program, pname, out):
        out._obj.value = int(program in self.done) if pname == 0x91B1 else int(self.link_ok)

    def count(self, name):
        return sum(1 for call in self.calls if call[0] == name)


def shading(parallel=True, **driver):
    sh = object.__new__(Shading)
    sh.programs, sh.pending, sh.failed, sh.binary, sh.parallel = {}, {}, {}, None, parallel
    return sh, FakeGLSL(sh, **driver)


class PendingLinkTests(unittest.TestCase):
    def test_parallel_link_is_issued_by_a_poll_and_finished_when_the_driver_is_done(self):
        sh, gl = shading()
        self.assertIsNone(sh._link('vertex', 'fragment', {'aPos': 0}, ('uA', 'uBB'), key='k', wait=False))
        self.assertEqual(gl.count('glLinkProgram'), 0)   # the caller pays nothing
        self.assertEqual(sh.poll_links(), 0)
        self.assertEqual(gl.count('glLinkProgram'), 1)
        program = sh.pending['k']['program']
        self.assertEqual(sh.poll_links(), 0)             # the driver is still working
        gl.done.add(program)
        self.assertEqual(sh.poll_links(), 1)
        self.assertEqual(sh.programs['k'], (program, {'uA': 2, 'uBB': 3}))
        self.assertEqual((sh.pending, gl.count('glDeleteShader')), ({}, 2))

    def test_a_poll_issues_at_least_one_and_more_within_its_budget(self):
        sh, gl = shading()
        for key in 'abc':
            sh._link('v', 'f' + key, {}, (), key=key, wait=False)
        sh.poll_links(budget=0.0)
        self.assertEqual(gl.count('glLinkProgram'), 1)
        sh.poll_links(budget=10.0)
        self.assertEqual(gl.count('glLinkProgram'), 3)

    def test_without_parallel_compiling_a_poll_links_one(self):
        sh, gl = shading(parallel=False)
        for key in 'ab':
            sh._link('v', 'f' + key, {}, (), key=key, wait=False)
        self.assertEqual(sh.poll_links(budget=0.0), 1)
        self.assertEqual((sorted(sh.programs), sorted(sh.pending)), (['a'], ['b']))

    def test_a_pending_program_stays_pending_for_the_frame_until_a_poll(self):
        sh, gl = shading()
        sh._link('v', 'f', {}, (), key='k', wait=False)
        sh.poll_links()
        gl.done.add(sh.pending['k']['program'])
        self.assertTrue(sh._waiting('k', wait=False))     # done, but only poll_links finishes it
        self.assertFalse(sh._waiting('k', wait=True))     # a caller that waits gets it now
        self.assertIn('k', sh.programs)

    def test_a_failed_link_keeps_its_compile_log_and_raises_when_asked_again(self):
        sh, gl = shading(link_ok=False, compile_ok=False)
        sh._link('v', 'f', {}, (), key='k', wait=False)
        sh.poll_links()
        gl.done.add(sh.pending['k']['program'])
        self.assertEqual(sh.poll_links(), 1)
        self.assertEqual(str(sh.failed['k']), 'bad shade')
        self.assertEqual(gl.count('glDeleteProgram'), 1)
        with self.assertRaisesRegex(ShaderError, 'bad shade'):
            sh._waiting('k', wait=False)


class FakeShading:
    def __init__(self):
        self.ready, self.pending, self.asked = False, {}, []

    def native_program(self, translation, srgb, instanced=False, wait=True, **options):
        self.asked.append((translation['translation_sha256'], wait, options))
        return ('program', {}) if self.ready or wait else None


def viewer():
    view = object.__new__(ModelViewport)
    view._shading, view.native_translation, view.experimental_materials = FakeShading(), True, False
    view.polls = []
    view.after = lambda ms, function: view.polls.append(function) or 'after#1'
    return view


class ModelViewerStandInTests(unittest.TestCase):
    entry = dict(native_translation=dict(translation_sha256='t'), native_textures={'g_Base': (5, True)},
                 alpha_pane=dict(alpha_api='g_Base', discard=0.5))

    def test_an_unlinked_surface_draws_as_untranslated_until_its_link_is_done(self):
        view = viewer()
        self.assertEqual(view._selected_surface(self.entry, 2), (None, {}, False))
        self.assertEqual(view.polls, [view._pump_links])
        self.assertEqual(view._shading.asked, [('t', False, dict(scene=False, alpha_api='g_Base', alpha_discard=0.5,
                                                                   alpha_opacity=1.0))])   # the draw's own variant
        view._shading.ready = True
        surface = view._selected_surface(self.entry, 2)
        self.assertIs(surface[0], self.entry['native_translation'])
        view._shading.ready = False   # remembered as linked until the memo is dropped
        self.assertIs(view._selected_surface(self.entry, 2)[0], self.entry['native_translation'])

    def test_captures_and_the_level_viewer_link_at_once(self):
        from nwo.app.thumbnails import thumbnail_viewport_class
        from nwo.levels.viewport import LevelViewport
        self.assertFalse(LevelViewport.ASYNC_LINKS)
        self.assertFalse(thumbnail_viewport_class().ASYNC_LINKS)
        view = viewer()
        view.ASYNC_LINKS = False
        self.assertIs(view._selected_surface(self.entry, 2)[0], self.entry['native_translation'])
        self.assertEqual(view._shading.asked, [])


if __name__ == '__main__':
    unittest.main()
