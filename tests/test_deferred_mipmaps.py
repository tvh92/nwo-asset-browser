"""Level loads queue mip generation and build it after the loader is done."""
import unittest
from collections import deque

from nwo.levels.viewport import LevelViewport
from nwo.viewer.viewport import ModelViewport


class FakeGL:
    def __init__(self):
        self.calls = []

    def glGetError(self):
        return 0

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))
        return call


def level_view():
    view = object.__new__(LevelViewport)
    view._gl = FakeGL()
    view._make_current = lambda: True
    view.built, view.scheduled, view.draws = [], [], 0
    view._build_mipmaps = lambda compressed: view.built.append((view._gl.calls[-1][1][1], compressed))
    view.after = lambda ms, function: view.scheduled.append(function) or f'after#{len(view.scheduled)}'
    view.after_cancel = lambda ident: view.scheduled.append(('cancel', ident))

    def draw():
        view.draws += 1
    view._schedule_draw = draw
    view._mipmap_queue, view._loading_done = deque([(5, True), (6, False), (7, True)]), False
    return view


class DeferredMipmapTests(unittest.TestCase):
    def test_upload_queues_instead_of_generating_when_deferred(self):
        self.assertFalse(ModelViewport.DEFER_MIPMAPS)
        self.assertTrue(LevelViewport.DEFER_MIPMAPS)

    def test_built_after_the_load_with_each_texture_bound(self):
        view = level_view()
        view._material_queue, view._compile_queue = {}, []
        view.finish_load()
        self.assertEqual(view.built, [(5, True), (6, False), (7, True)])
        self.assertEqual(view._gl.calls[-1], ('glBindTexture', (view.GL_TEXTURE_2D, 0)))
        self.assertEqual(view.draws, 1)

    def test_waits_for_queued_materials_then_the_pump_builds(self):
        view = level_view()
        view._material_queue, view._compile_queue = {'m': {}}, []
        view.finish_load()
        self.assertEqual(view.built, [])
        view._materials = {'m': {}}   # the pump drains its last queued material
        view._pump_materials()        # and, with the load done, starts the mip build
        self.assertEqual(view._material_queue, {})
        self.assertEqual(len(view.built), 3)

    def test_budget_continues_on_a_later_tick(self):
        view = level_view()
        view.build_deferred_mipmaps(budget=0.0)   # at least one per tick
        self.assertEqual(view.built, [(5, True)])
        self.assertEqual(view.scheduled, [view.build_deferred_mipmaps])
        view.build_deferred_mipmaps(budget=float('inf'))
        self.assertEqual(len(view.built), 3)

    def test_flush_for_captures_builds_everything(self):
        view = level_view()
        view._material_queue, view._compile_queue = {}, []
        view._mipmap_pump = 'after#9'
        view.flush_materials()
        self.assertIn(('cancel', 'after#9'), view.scheduled)
        self.assertEqual(len(view.built), 3)


    def test_waits_while_the_camera_moves(self):
        import time
        view = level_view()
        view._camera_changed_at = time.perf_counter()
        view.build_deferred_mipmaps()
        self.assertEqual(view.built, [])
        self.assertEqual(view.scheduled, [view.build_deferred_mipmaps])
        view._camera_changed_at = time.perf_counter() - 10
        view.build_deferred_mipmaps(budget=float('inf'))
        self.assertEqual(len(view.built), 3)


class GameMipChainTests(unittest.TestCase):
    """A texture with its own mip_chain uploads it instead of asking the driver to build one."""

    def view(self, errors=()):
        view = object.__new__(ModelViewport)
        view._gl = FakeGL()
        pending = list(errors)
        view._gl.glGetError = lambda: pending.pop(0) if pending else 0
        view.uploads, view.built = [], []
        view._compressed_upload = lambda *args: view.uploads.append(args[1:5])
        view._compressed_uploader = lambda: view._compressed_upload
        view._mipmap_support = dict(generate=object(), anisotropy=8.0, compressed=True)
        view._build_mipmaps = lambda compressed: view.built.append(compressed)
        return view

    @staticmethod
    def texture(chain=True):
        texture = dict(path='t', width=16, height=8, format=71, gl_format=0x83F1, pixels=bytes(32))   # BC1: 8 bytes a block
        if chain:
            texture['mip_chain'] = [bytes(16), bytes(8), bytes(8), bytes(8)]   # 8x4, 4x2, 2x1, 1x1
        return texture

    def test_chain_is_uploaded_level_by_level(self):
        view = self.view()
        view._upload_texture(self.texture())
        self.assertEqual([u[0] for u in view.uploads], [0, 1, 2, 3, 4])
        self.assertEqual(view.uploads[1][2:], (8, 4))
        self.assertEqual(view.built, [])
        params = [c[1] for c in view._gl.calls if c[0] == 'glTexParameteri']
        self.assertIn((view.GL_TEXTURE_2D, view.GL_TEXTURE_MAX_LEVEL, 4), params)
        self.assertIn((view.GL_TEXTURE_2D, view.GL_TEXTURE_MIN_FILTER, view.GL_LINEAR_MIPMAP_LINEAR), params)

    def test_no_chain_or_a_bad_one_generates(self):
        view = self.view()
        view._upload_texture(self.texture(chain=False))
        self.assertEqual(view.built, [True])
        bad = self.texture()
        bad['mip_chain'][1] = bytes(4)
        view._upload_texture(bad)
        self.assertEqual(view.built, [True, True])

    def test_a_refused_level_falls_back_to_generation(self):
        view = self.view(errors=(0, 0, 0x501))   # level 0 and its check pass; the chain's check reports the refusal
        view._upload_texture(self.texture())
        self.assertEqual(view.built, [True])
        params = [c[1] for c in view._gl.calls if c[0] == 'glTexParameteri']
        self.assertIn((view.GL_TEXTURE_2D, view.GL_TEXTURE_MAX_LEVEL, 1000), params)


if __name__ == '__main__':
    unittest.main()
