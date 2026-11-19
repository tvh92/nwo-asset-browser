"""Check the viewer's translated Rift Apart programs against their recipes, texel by texel (the viewer's counterpart of
rc2016.games.rcra.blender_check).

    python -m rc2016.viewer.rcra_check --game <install> --output-root <folder> [--size 37] <material path> ...

For each material the viewer receipt (rc2016.games.rcra.viewer_materials.material_receipt) is emitted by rcra_glsl's own term
emitter into a check program that draws a UV window on a float target, four terms per pass. The inputs are those of
blender_check.expected: game UVs over the window, the plane normal game +y, tangent +x with TEXCOORD3.w 1, vertex colour 1,
front face, blender_check.FORWARD_VIEW, and 0 for everything else (UV1, the scene behind). The textures are the viewer's
uploads (the driver's decoded top mip, read back with glGetTexImage; sRGB ones decoded to linear), and the reference
evaluates the recipe with rc2016.shaders.sm5 on those texels: bilinear repeat for a slice, the unpadded tiles of an array or
volume atlas, the D3D cube lookup for a cube map. The window is small enough that GL magnifies, so it reads the top mip.

Prints the worst absolute difference per term group, its 99th percentile and the pixels over OVER. A pixel whose recipe samples at a non-finite coordinate is left out
and counted, as in blender_check. Screen derivatives (ddx, ddy) are the quad's in GL and 0 in the reference.
"""
import argparse
import ctypes
import json
import math
import os
import sys
import tempfile
from collections import defaultdict

import numpy as np

GROUPS = {'albedo': ('albedo.r', 'albedo.g', 'albedo.b'), 'f0': ('f0.r', 'f0.g', 'f0.b'),
          'scalars': ('gloss', 'occlusion', 'emission'), 'normal': ('normal.x', 'normal.y', 'normal.z'),
          'unlit': ('unlit.r', 'unlit.g', 'unlit.b'),
          'clip': ('clip',)}
FORWARD_GROUPS = {'forward_colour': ('colour.r', 'colour.g', 'colour.b'), 'forward_alpha': ('alpha', 'clip'),
                  'forward_background': ('background.r', 'background.g', 'background.b'),
                  'forward_lit_background': ('lit_background.r', 'lit_background.g', 'lit_background.b')}
GL_TEXTURE_2D, GL_RGBA, GL_FLOAT = 0x0DE1, 0x1908, 0x1406
GL_TEXTURE_CUBE_MAP_POSITIVE_X = 0x8515
OVER = 0.01   # 'over': pixels differing by more, relative above 1 (a threshold the GPU's rounding crosses, or a bug)

VERTEX = """varying vec2 q;
void main() { q = gl_Vertex.xy; gl_Position = vec4(gl_Vertex.xy * 2.0 - 1.0, 0.0, 1.0); }
"""


class UndefinedSample(ArithmeticError):
    pass


def _srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def window(widths, size):
    """(origin, span) of the UV window, as blender_check.window: each pixel covers at most a 32nd of a texel."""
    return 0.3, min(1.0, size / (32.0 * max(list(widths) + [1])))


def fragment_source(receipt, view_inputs):
    """(source, [(group, [term names])]): the check program of a receipt."""
    from rc2016.viewer import rcra_glsl
    layouts = receipt['layouts']
    lines = ['    vec2 b = uOrigin + uSpan * q;',
             '    vec2 vUV = vec2(b.x, 1.0 - b.y); vec2 vUV1 = vec2(0.0);',
             '    vec3 vTintWorldNrm = vec3(0.0, 1.0, 0.0); vec3 rcraTangent = vec3(1.0, 0.0, 0.0);',
             '    float rcraHandedness = 1.0; vec4 %s = vec4(1.0); vec4 rcraScene = vec4(0.0);' % rcra_glsl.COLOUR_VARYING,
             '    vec3 vTintViewOffset = vec3(%s, %s, %s);' % tuple(rcra_glsl._literal(view_inputs['VIEW.' + c]) for c in 'xyz'),
             '    vec3 rcraCamera = vec3(%s, %s, %s);' % tuple(rcra_glsl._literal(view_inputs['CAMERA.' + c]) for c in 'xyz')]
    outputs = []
    for key, groups, prefix in (('surface', GROUPS, 'g'), ('forward', FORWARD_GROUPS, 'f')):
        recipe = receipt.get(key)
        if not recipe:
            continue
        emitter = rcra_glsl._Emitter(recipe, layouts, prefix)
        for group, names in groups.items():
            names = [n for n in names if n in recipe['terms']]
            if names:
                outputs.append((group, names, [emitter.term(n) for n in names]))
        lines += emitter.lines
    passes = []
    for k, (group, names, values) in enumerate(outputs):
        values = values + ['0.0'] * (4 - len(values))
        lines.append('    if (uGroup == %d) { gl_FragColor = vec4(%s); return; }' % (k, ', '.join(values)))
        passes.append((group, names))
    helpers = rcra_glsl.HELPERS.split('// Rift Apart deferred BRDF')[0]
    declarations = ''.join('uniform %s %s;\n' % ('samplerCube' if layouts[b]['kind'] == 'cube' else 'sampler2D',
                                                  rcra_glsl.uniform(b)) for b in sorted(layouts))
    declarations += 'uniform vec2 uOrigin; uniform float uSpan; uniform int uGroup;\nvarying vec2 q;\n'
    if any('rcraInf' in line or 'rcraNan' in line for line in lines):
        declarations += rcra_glsl.NON_FINITE_DECLARATION
        lines.insert(0, rcra_glsl.NON_FINITE_LINE)
    source = declarations + helpers + '\nvoid main() {\n' + '\n'.join(lines) + '\n    gl_FragColor = vec4(0.0);\n}\n'
    return source, passes


class Checker:
    """One hidden viewport's GL context, reused for every material."""

    def __init__(self, size):
        import tkinter as tk
        from rc2016.viewer.viewport import ModelViewport
        self.size = size
        self.root = tk.Tk()
        self.root.geometry('64x64')
        self.view = ModelViewport(self.root)
        self.view.pack(fill='both', expand=True)
        self.root.update()
        if self.view._context_error or self.view._shading is None or not self.view._make_current():
            raise RuntimeError(self.view._context_error or 'no OpenGL shading context')
        self.functions = self.view._hdr_functions()
        if not self.functions:
            raise RuntimeError('no framebuffer support')
        gl = self.view._gl
        gl.glGetTexImage.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
        gl.glReadPixels.argtypes = [ctypes.c_int] * 4 + [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
        self.target = self._target()

    def close(self):
        self.root.destroy()

    def _target(self):
        gl, f, size = self.view._gl, self.functions, self.size
        colour = ctypes.c_uint(0)
        gl.glGenTextures(1, ctypes.byref(colour))
        gl.glBindTexture(GL_TEXTURE_2D, colour.value)
        for parameter in (0x2801, 0x2800):   # MIN, MAG filter: NEAREST
            gl.glTexParameteri(GL_TEXTURE_2D, parameter, 0x2600)
        gl.glTexImage2D(GL_TEXTURE_2D, 0, 0x8814, size, size, 0, GL_RGBA, GL_FLOAT, None)   # RGBA32F
        frame = ctypes.c_uint(0)
        f['glGenFramebuffers'](1, ctypes.byref(frame))
        f['glBindFramebuffer'](0x8D40, frame.value)
        f['glFramebufferTexture2D'](0x8D40, 0x8CE0, GL_TEXTURE_2D, colour.value, 0)
        if f['glCheckFramebufferStatus'](0x8D40) != 0x8CD5:
            raise RuntimeError('float framebuffer incomplete')
        return frame.value

    def _read_texture(self, target, width, height):
        pixels = np.zeros((height, width, 4), np.float32)
        self.view._gl.glGetTexImage(target, 0, GL_RGBA, GL_FLOAT, pixels.ctypes.data)
        return pixels

    def upload(self, receipt, textures):
        """({binding: GL name}, {binding: reference unit}) for the receipt's textures, as the viewport uploads them."""
        from rc2016.games.rcra import blender_check
        view, gl = self.view, self.view._gl
        names, units = {}, {}
        for binding, texture in textures.items():
            layout = receipt['layouts'][binding]
            decode = bool(texture.get('srgb'))
            if texture.get('cube'):
                name = view._upload_cube(texture, decode_srgb=decode)
                gl.glBindTexture(0x8513, name)
                faces = np.stack([self._read_texture(GL_TEXTURE_CUBE_MAP_POSITIVE_X + k, face['width'], face['height'])
                                  for k, face in enumerate(texture['cube'])])
                if decode:
                    faces[..., :3] = _srgb_to_linear(faces[..., :3])
                units[binding] = blender_check.CubeFaces(faces)
            else:
                name = view._upload_texture(texture, decode_srgb=decode, mipmaps=False)
                gl.glBindTexture(GL_TEXTURE_2D, name)
                image = self._read_texture(GL_TEXTURE_2D, texture['width'], texture['height'])
                if decode:
                    image[..., :3] = _srgb_to_linear(image[..., :3])
                if layout['kind'] in ('array', 'volume'):
                    w, h, columns = layout['width'], layout['height'], layout['columns']
                    tiles = [image[r * (h + 2) + 1:r * (h + 2) + 1 + h, c * (w + 2) + 1:c * (w + 2) + 1 + w]
                             for r, c in (divmod(k, columns) for k in range(layout['depth']))]
                    units[binding] = blender_check.Volume(tiles) if layout['kind'] == 'volume' else tiles
                else:
                    units[binding] = image
            if not name:
                raise RuntimeError('texture upload failed: ' + binding)
            names[binding] = name
        return names, units

    def draw(self, receipt, textures):
        """{group: (size*size, k) GL values}, the passes, the units and the UV window."""
        from rc2016.games.rcra.blender_check import FORWARD_VIEW
        from rc2016.viewer import rcra_glsl
        view, gl, shading, size = self.view, self.view._gl, self.view._shading, self.size
        names, units = self.upload(receipt, textures)
        widths = [u.faces.shape[2] if hasattr(u, 'faces') else (u[0] if isinstance(u, list) else u).shape[1]
                  for u in units.values()]
        origin, span = window(widths, size)
        source, passes = fragment_source(receipt, FORWARD_VIEW)
        program = shading._link_source(VERTEX, source, {})
        shading.glUseProgram(program)
        location = lambda n: shading.glGetUniformLocation(program, n.encode())  # noqa: E731
        for unit, binding in enumerate(sorted(receipt['layouts'])):
            shading.glActiveTexture(0x84C0 + unit)
            gl.glBindTexture(0x8513 if receipt['layouts'][binding]['kind'] == 'cube' else GL_TEXTURE_2D, names[binding])
            shading.glUniform1i(location(rcra_glsl.uniform(binding)), unit)
        shading.glActiveTexture(0x84C0)
        shading.glUniform2f(location('uOrigin'), origin, origin)
        shading.glUniform1f(location('uSpan'), span)
        self.functions['glBindFramebuffer'](0x8D40, self.target)
        gl.glViewport(0, 0, size, size)
        for cap in (0x0B71, 0x0BE2, 0x0B44):   # depth test, blend, cull face
            gl.glDisable(cap)
        out = {}
        for k, (group, terms) in enumerate(passes):
            shading.glUniform1i(location('uGroup'), k)
            gl.glClearColor(ctypes.c_float(-7.0), ctypes.c_float(-7.0), ctypes.c_float(-7.0), ctypes.c_float(-7.0))
            gl.glClear(0x4000)
            gl.glBegin(0x0007)   # GL_QUADS
            for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
                gl.glVertex2f(ctypes.c_float(x), ctypes.c_float(y))
            gl.glEnd()
            pixels = np.zeros((size, size, 4), np.float32)
            gl.glReadPixels(0, 0, size, size, GL_RGBA, GL_FLOAT, pixels.ctypes.data)
            out[group] = pixels.reshape(-1, 4)[:, :len(terms)]
        self.functions['glBindFramebuffer'](0x8D40, 0)
        shading.glUseProgram(0)
        shading.glDeleteProgram(program)
        gl.glDeleteTextures(len(names), (ctypes.c_uint * len(names))(*names.values()))
        if gl.glGetError():
            raise RuntimeError('OpenGL error in the check draw')
        return out, passes, units, (origin, span)


def expected(receipt, passes, units, window_, size):
    """{group: (size*size, k)} from the recipes evaluated on the uploaded texels."""
    from rc2016.games.rcra import blender_check, blender_export, surface as rsurface

    def sample(node, args):
        if not all(math.isfinite(a) for a in args[:3]):
            raise UndefinedSample(node['binding'])
        image = units[node['binding']]
        if isinstance(image, blender_check.CubeFaces):
            return [float(c) for c in blender_export.cube_lookup(image.faces, args[:3])]
        if isinstance(image, blender_check.Volume):
            s = args[2] * len(image) - 0.5
            k = math.floor(s)
            a = blender_check.bilinear(image[k % len(image)], args[0], args[1])
            b = blender_check.bilinear(image[(k + 1) % len(image)], args[0], args[1])
            return [x + (s - k) * (y - x) for x, y in zip(a, b)]
        if isinstance(image, list):
            image = image[int(min(max(math.floor(args[2] + 0.5), 0), len(image) - 1))]
        return blender_check.bilinear(image, args[0], args[1])
    origin, span = window_
    groups = {g: t for g, t in passes}
    out = defaultdict(list)
    for j in range(size):
        for i in range(size):
            ub, vb = origin + span * (i + 0.5) / size, origin + span * (j + 0.5) / size
            inputs = dict(blender_check.FORWARD_VIEW, **{
                'TEXCOORD.x': ub, 'TEXCOORD.y': 1.0 - vb, 'TEXCOORD2.x': 0.0, 'TEXCOORD2.y': 1.0, 'TEXCOORD2.z': 0.0,
                'TEXCOORD3.x': 1.0, 'TEXCOORD3.y': 0.0, 'TEXCOORD3.z': 0.0, 'TEXCOORD3.w': 1.0, 'TEXCOORD4.x': 1.0,
                'TEXCOORD4.y': 1.0, 'TEXCOORD4.z': 1.0, 'TEXCOORD4.w': 1.0, 'SV_IsFrontFace': True})
            for key, names in (('surface', GROUPS), ('forward', FORWARD_GROUPS)):
                recipe = receipt.get(key)
                mine = [g for g in names if g in groups]
                if not recipe or not mine:
                    continue
                try:
                    terms = rsurface.evaluate_recipe(recipe, inputs, sample)
                except UndefinedSample:
                    for g in mine:
                        out[g].append([math.nan] * len(groups[g]))
                    continue
                for g in mine:
                    out[g].append([float(terms[t]) for t in groups[g]])
    return {g: np.array(v, np.float64) for g, v in out.items()}


def check(checker, source, material_path):
    from rc2016.games.rcra.material import _material_key
    from rc2016.games.rcra.viewer_materials import material_receipt
    receipt, textures = material_receipt(source, _material_key(source, material_path))
    got, passes, units, window_ = checker.draw(receipt, textures)
    want = expected(receipt, passes, units, window_, checker.size)
    report = {}
    for group, values in want.items():
        defined = ~np.isnan(values).any(axis=1)
        if not defined.any():
            report[group] = dict(undefined=int(len(values)))
            continue
        a, b = got[group][defined].astype(np.float64), values[defined]
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        diff = np.where(same, 0.0, np.abs(a - b))
        diff = np.where(np.isnan(diff), np.inf, diff)
        relative = (diff / np.maximum(1.0, np.abs(b))).max(axis=1)
        diff = diff.max(axis=1)
        report[group] = dict(max=float(diff.max()), p99=float(np.percentile(diff, 99)), over=int((relative > OVER).sum()))
        if not defined.all():
            report[group]['undefined'] = int((~defined).sum())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--size', type=int, default=37)
    parser.add_argument('materials', nargs='+')
    args = parser.parse_args(argv)
    os.makedirs(args.output_root, exist_ok=True)
    from rc2016.games.rcra.source import RcraSource
    from rc2016.games.rcra.viewer_materials import NotTranslated
    source = RcraSource(args.game, args.output_root)
    results = {}
    with tempfile.TemporaryDirectory(dir=args.output_root) as appdata:
        os.environ['APPDATA'] = appdata   # the viewport's settings and program cache stay in the output root
        checker = Checker(args.size)
        try:
            for path in args.materials:
                try:
                    results[path] = check(checker, source, path)
                except (NotTranslated, RuntimeError, ValueError, KeyError) as error:
                    results[path] = dict(error='%s: %s' % (type(error).__name__, error))
                print(path, json.dumps(results[path]), flush=True)
        finally:
            checker.close()
    return results


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
