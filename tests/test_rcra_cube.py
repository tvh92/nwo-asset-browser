"""Rift Apart cube maps: the D3D lookup, the equirectangular image Blender reads, and the viewer's samplerCube (no game files)."""
import ctypes
import math
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from nwo.games.rcra import blender_export as bexp
from nwo.viewer import rcra_glsl


def face_direction(face, s, t):
    """The direction of texture coordinate (s, t) on a face, from the D3D cube table written out per face (t = 0 the top row)."""
    sc, tc = 2.0 * s - 1.0, 2.0 * t - 1.0
    return [(1.0, -tc, -sc), (-1.0, -tc, sc), (sc, 1.0, tc), (sc, -1.0, -tc), (sc, -tc, 1.0), (-sc, -tc, -1.0)][face]


def smooth_faces(n):
    """A cube whose texels hold 0.5 + 0.4 x the unit direction of their centre (alpha 1)."""
    faces = np.ones((6, n, n, 4), np.float32)
    for face in range(6):
        for row in range(n):
            for col in range(n):
                d = np.array(face_direction(face, (col + 0.5) / n, (row + 0.5) / n))
                faces[face, row, col, :3] = 0.5 + 0.4 * d / np.linalg.norm(d)
    return faces


def blender_uv(d):
    """Blender's equirectangular mapping of a direction (u from the left, v from the bottom)."""
    return -math.atan2(d[1], d[0]) / (2 * math.pi) + 0.5, math.atan2(d[2], math.hypot(d[0], d[1])) / math.pi + 0.5


def bilinear_wrap(image, u, v):
    """Blender's Linear lookup: u wraps, v is clamped, row 0 is the top (v = 1)."""
    h, w = image.shape[:2]
    x, y = u * w - 0.5, (1.0 - v) * h - 0.5
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    rows = [min(max(y0, 0), h - 1), min(max(y0 + 1, 0), h - 1)]
    cols = [x0 % w, (x0 + 1) % w]
    top = image[rows[0], cols[0]] * (1 - fx) + image[rows[0], cols[1]] * fx
    bottom = image[rows[1], cols[0]] * (1 - fx) + image[rows[1], cols[1]] * fx
    return top * (1 - fy) + bottom * fy


class CubeLookupTests(unittest.TestCase):
    def test_texel_centres_return_their_texel(self):
        rng = np.random.default_rng(1)
        faces = rng.random((6, 8, 8, 4)).astype(np.float32)
        for face in range(6):
            for row in range(8):
                for col in range(8):
                    d = face_direction(face, (col + 0.5) / 8, (row + 0.5) / 8)
                    got = bexp.cube_lookup(faces, np.array(d) * 3.7)
                    np.testing.assert_allclose(got, faces[face, row, col], atol=1e-6, err_msg=str((face, row, col)))

    def test_lookup_ignores_the_length_of_the_direction(self):
        faces = np.random.default_rng(2).random((6, 4, 4, 4))
        d = np.random.default_rng(3).normal(size=(50, 3))
        np.testing.assert_allclose(bexp.cube_lookup(faces, d), bexp.cube_lookup(faces, d * 0.01), atol=1e-9)


class EquirectTests(unittest.TestCase):
    def test_pixel_directions_invert_blenders_mapping(self):
        d = bexp.equirect_directions(32, 16)
        self.assertEqual(d.shape, (16, 32, 3))
        for row, col in ((0, 0), (3, 7), (8, 20), (15, 31), (5, 16)):
            u, v = blender_uv(d[row, col])
            self.assertAlmostEqual(u, (col + 0.5) / 32, places=9)
            self.assertAlmostEqual(v, 1.0 - (row + 0.5) / 16, places=9)

    def test_image_returns_the_cube_value_at_random_directions(self):
        n = 64
        image = bexp.cube_equirect(smooth_faces(n))
        self.assertEqual(image.shape, (2 * n, 4 * n, 4))
        rng = np.random.default_rng(4)
        errors = []
        for d in rng.normal(size=(400, 3)):
            u, v = blender_uv(d)
            want = 0.5 + 0.4 * d / np.linalg.norm(d)
            errors.append(np.abs(bilinear_wrap(image, u, v)[:3] - want).max())
        self.assertLess(float(np.median(errors)), 2e-3)
        self.assertLess(max(errors), 2e-2)

    def test_pixels_are_the_cube_lookup_of_their_direction(self):
        faces = np.random.default_rng(5).random((6, 8, 8, 4)).astype(np.float32)
        image = bexp.cube_equirect(faces)
        d = bexp.equirect_directions(32, 16)
        np.testing.assert_allclose(image, bexp.cube_lookup(faces, d), atol=1e-6)

    def test_srgb_is_filtered_in_linear_light(self):
        def decode(x):
            return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

        def encode(x):
            return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
        faces = np.random.default_rng(6).random((6, 8, 8, 4)).astype(np.float32)
        d = bexp.equirect_directions(32, 16)
        linear = faces.astype(np.float64)
        linear[..., :3] = decode(linear[..., :3])
        want = bexp.cube_lookup(linear, d)
        want[..., :3] = encode(want[..., :3])   # the alpha channel is linear data
        np.testing.assert_allclose(bexp.cube_equirect(faces, srgb=True), want, atol=1e-5)

    def test_layouts(self):
        cube = (bexp.CUBE, 0)
        self.assertEqual(bexp.merge_layouts(cube, cube), cube)
        self.assertIsNone(bexp.merge_layouts(cube, (bexp.CUBE, 1)))
        self.assertIsNone(bexp.merge_layouts(cube, 0))
        self.assertIsNone(bexp.merge_layouts(bexp.VOLUME, 0))
        self.assertEqual(bexp.merge_layouts(0, 1), bexp.ALL_SLICES)

    def test_cube_index_reads_the_constant_slice(self):
        nodes = [dict(op='const', value=1.0), dict(op='const', value=2.4), dict(op='input', name='x')]
        sample = dict(op='sample', dimension=10, binding='t', args=[0, 0, 0, 1])
        self.assertEqual(bexp.cube_index(dict(nodes=nodes), sample), 2)
        self.assertEqual(bexp.cube_index(dict(nodes=nodes), dict(sample, dimension=6)), 0)
        with self.assertRaises(ValueError):
            bexp.cube_index(dict(nodes=nodes), dict(sample, args=[0, 0, 0, 2]))


class CubeGlslTests(unittest.TestCase):
    def receipt(self, kind):
        nodes = [dict(op='const', type='f', value=v, args=[]) for v in (0.5, 0.25, 0.75, 0.0)] + [
                 dict(op='sample', args=[0, 1, 2, 3], binding='textureUnit12', dimension=10, opcode='sample'),
                 dict(op='channel', args=[4], index=0, type='f')]
        recipe = dict(nodes=nodes, terms={'albedo.r': 5})
        layouts = {'textureUnit12': dict(kind=kind, slice=0) if kind == 'cube' else dict(kind=kind)}
        return dict(surface=recipe, forward=None, blend='opaque', layouts=layouts, translation_sha256='test')

    def test_cube_layout_declares_a_samplercube_and_looks_up_the_game_direction(self):
        program = rcra_glsl.fragment_source(self.receipt('cube'))
        name = rcra_glsl.uniform('textureUnit12')
        self.assertIn('uniform samplerCube %s;' % name, program['source'])
        self.assertIn('textureCube(%s, vec3(0.5, 0.25, 0.75))' % name, program['source'])
        self.assertNotIn('sampler2D ' + name, program['source'])
        self.assertEqual(program['samplers'], [name])

    def test_a_slice_layout_stays_a_sampler2d(self):
        source = rcra_glsl.fragment_source(self.receipt('slice'))['source']
        self.assertIn('uniform sampler2D %s;' % rcra_glsl.uniform('textureUnit12'), source)
        self.assertNotIn('samplerCube', source)


@unittest.skipUnless(os.name == 'nt', 'needs Windows OpenGL')
class ViewportCubeTests(unittest.TestCase):
    """The viewport's GL cube map returns the D3D texel for the game direction (no game files)."""

    def test_gl_lookup_matches_the_d3d_table_at_texel_centres(self):
        n = 8
        faces = np.random.default_rng(7).integers(0, 256, (6, n, n, 4), dtype=np.uint8)
        textures = [dict(path='f%d' % k, width=n, height=n, format=28, gl_format=None, gl_pixel_format=0x1908, srgb=False,
                         pixels=faces[k].tobytes()) for k in range(6)]
        with tempfile.TemporaryDirectory() as appdata, mock.patch.dict(os.environ, {'APPDATA': appdata}):
            import tkinter as tk
            from nwo.viewer.viewport import ModelViewport
            root = tk.Tk()
            try:
                root.geometry('64x64')
                view = ModelViewport(root)
                view.pack(fill='both', expand=True)
                root.update()
                if view._context_error or view._shading is None or not view._make_current():
                    self.skipTest(view._context_error or 'no shading context')
                gl, sh = view._gl, view._shading
                name = view._upload_cube(dict(path='cube', width=n, height=n, format=28, srgb=False, cube=textures))
                self.assertTrue(name)
                program = sh._link_source(
                    'void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); gl_PointSize = 1.0; }',
                    'uniform samplerCube uCube; uniform vec3 uDir; '
                    'void main() { gl_FragColor = textureCube(uCube, uDir); }', {})
                sh.glUseProgram(program)
                sh.glActiveTexture(view.GL_TEXTURE0)
                gl.glBindTexture(view.GL_TEXTURE_CUBE_MAP, name)
                sh.glUniform1i(sh.glGetUniformLocation(program, b'uCube'), 0)
                direction = sh.glGetUniformLocation(program, b'uDir')
                gl.glDisable(0x0B71)   # GL_DEPTH_TEST
                gl.glDisable(0x0BE2)   # GL_BLEND
                gl.glViewport(0, 0, 63, 63)   # odd, so the centre vertex lands in the middle of pixel (31, 31)
                gl.glReadBuffer(0x0405)
                gl.glReadPixels.argtypes = [ctypes.c_int] * 4 + [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
                gl.glBegin.argtypes = [ctypes.c_uint]
                for face, row, col in ((0, 0, 0), (0, 7, 3), (1, 2, 6), (2, 5, 1), (3, 0, 7), (4, 3, 3), (5, 6, 2), (5, 0, 0),
                                       (2, 0, 7), (3, 7, 7), (4, 1, 5), (1, 6, 0)):
                    d = face_direction(face, (col + 0.5) / n, (row + 0.5) / n)
                    sh.glUniform3f(direction, *(3.0 * x for x in d))
                    gl.glClearColor.argtypes = [ctypes.c_float] * 4
                    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
                    gl.glClear(0x4000)
                    gl.glBegin(0x0000)   # GL_POINTS
                    gl.glVertex2f.argtypes = [ctypes.c_float, ctypes.c_float]
                    gl.glVertex2f(0.0, 0.0)
                    gl.glEnd()
                    pixel = (ctypes.c_ubyte * 4)()
                    gl.glReadPixels(31, 31, 1, 1, 0x1908, 0x1401, pixel)
                    self.assertEqual(list(pixel)[:3], faces[face, row, col, :3].tolist(), (face, row, col))   # the window has no alpha
                sh.glUseProgram(0)
            finally:
                root.destroy()


if __name__ == '__main__':
    unittest.main()
