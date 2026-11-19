"""Rift Apart material terms baked over UV0, for the portable (glTF/OBJ) export.

A term of a translated program (a ``surface.surface`` or ``surface.forward`` recipe) whose operations reach no input but
UV0 (``TEXCOORD.xy``) and the front-face flag (front: the export is single-sided), and sample only 1D or 2D textures or
2D-array slices, takes the same value at every point of a mesh with the same UV0. Evaluating it at the texel centres of a
grid in UV0 space gives a texture the glTF material can carry: the game's own albedo, F0, gloss, occlusion, emission or
forward alpha, with every tint, mask, palette and blend the program applies there.

- The test is structural (``blocker``): a term counts only when no path through its operations reaches another input.
  Terms that read UV1, the normal, the tangent, the vertex colour or the view stay out, with the reason.
- The grid is the size of the largest texture the baked terms sample (each side at most ``MAX_SIZE``); a texture
  sampled at a scaled UV (a tiled detail map) is resampled at that grid, so detail finer than a grid texel is lost.
- Sampling is bilinear with wrap at the top mip, as the Blender export samples: the game's sampler states are not in the
  program, and a sample at an explicit mip level or with a bias or gradients reads the top mip (``notes`` say so).
  sRGB textures are decoded before filtering, as the GPU does. A load (``ld``) reads its texel, 0 outside the texture.
- NaN and infinity are what the program computes; ``bake`` returns them, and callers clamp as the G-buffer target does.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

UV0 = ('TEXCOORD.x', 'TEXCOORD.y')
FRONT_FACE = 'SV_IsFrontFace'
MAX_SIZE = 4096
CHUNK = 1 << 19            # texels evaluated at once
THREADS = max(1, min(4, (os.cpu_count() or 1) // 2))   # chunks evaluated in parallel (numpy releases the GIL)
NEAREST = 1e-4             # a read this close to a texel centre takes that texel
MAX_SLICES = 64            # distinct array slices one sample may read per chunk
SAMPLE_OPCODES = ('sample', 'sample_l', 'sample_b', 'sample_d')
DIMENSION_1D, DIMENSION_2D, DIMENSION_2D_ARRAY = 2, 3, 8   # a 1D texture is one texel high
INPUT_NAMES = {'TEXCOORD': 'the draw alpha', 'TEXCOORD1': 'UV1', 'TEXCOORD2': 'the world normal', 'TEXCOORD3': 'the tangent',
               'TEXCOORD4': 'the vertex colour', 'TEXCOORD5': 'the previous position', 'TEXCOORD6': 'the particle values',
               'TEXCOORD7': 'the particle values', 'VIEW': 'the view direction', 'CAMERA': 'the camera position',
               'SV_Position': 'the screen position', 'SV_POSITION': 'the screen position', 'BACKGROUND': 'the scene behind'}
MIP_NOTE = 'a sample at an explicit mip level, bias or gradient reads the top mip'


def _srgb_decode(x):
    """sRGB to linear, as the GPU decodes an sRGB texture before filtering (float32)."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


UNIT_TABLE = np.arange(256, dtype=np.float32) / 255.0
SRGB_TABLE = _srgb_decode(UNIT_TABLE)


def _reach(recipe, names):
    """(input names, sample/load nodes, other leaf reasons) the terms ``names`` reach."""
    nodes = recipe['nodes']
    stack = [recipe['terms'][n] for n in names if n in recipe['terms']]
    seen, inputs, samples, unknown = set(), set(), [], []
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        n = nodes[i]
        if n['op'] == 'input':
            inputs.add(n['name'])
        elif n['op'] in ('sample', 'load'):
            samples.append(n)
        elif n['op'] in ('unknown', 'cbload'):
            unknown.append(n.get('reason') or n['op'])
        stack.extend(n['args'])
    return inputs, samples, unknown


def describe(name):
    return INPUT_NAMES.get(name.split('.')[0], name)


def blocker(recipe, names):
    """None when the terms ``names`` of ``recipe`` bake over UV0; else why not."""
    from rc2016.games.rcra import blender_export as bexp
    if recipe is None:
        return 'the program does not translate'
    if not any(n in recipe['terms'] for n in names):
        return 'the program has no such term'
    inputs, samples, unknown = _reach(recipe, names)
    if unknown:
        return 'it reads a value the translation does not know (%s)' % unknown[0]
    other = sorted({describe(i) for i in inputs if i not in UV0 and i != FRONT_FACE})
    if other:
        return 'it reads %s' % ', '.join(other)
    try:
        layouts = bexp._slices(recipe)
    except ValueError as error:   # surface.Unsupported
        return str(error)
    for n in samples:
        if n['op'] == 'sample' and n.get('opcode') not in SAMPLE_OPCODES:
            return 'it samples with %s' % n.get('opcode')
        if n['op'] == 'load' and n.get('opcode') != 'ld':
            return 'it loads with %s' % n.get('opcode')
        if n.get('dimension') not in (DIMENSION_1D, DIMENSION_2D, DIMENSION_2D_ARRAY):
            return 'it samples a texture of dimension %s (a 1D array, volume or cube map)' % n.get('dimension')
        if n['op'] == 'sample' and isinstance(layouts.get(n['binding']), tuple):
            return 'it samples a cube map'
    return None


def sampled_bindings(recipe, names):
    return sorted({n['binding'] for n in _reach(recipe, names)[1]})


# ------------------------------------------------------------------------------------------------------- evaluation


def _typed(value, kind):
    """``rcra.vertex._typed`` without a copy when the value already has its type."""
    from rc2016.games.rcra import vertex as rvertex
    value = np.asarray(value)
    if value.dtype == rvertex.DTYPES[kind]:
        return value
    return rvertex._typed(value, kind)


def evaluate(nodes, roots, inputs, resource):
    """``rcra.vertex.evaluate`` over arrays, freeing each value after its last use, with ``lookup`` nodes as tables."""
    from rc2016.games.rcra import vertex as rvertex
    from rc2016.games.rcra.surface import Unsupported
    from rc2016.shaders import sm5
    needed, stack = set(), list(roots)
    while stack:
        i = stack.pop()
        if i not in needed:
            needed.add(i)
            stack.extend(nodes[i]['args'])
    order = sorted(needed)
    last = {}
    for i in order:
        for a in nodes[i]['args']:
            last[a] = i
    keep = set(roots)
    values = {}
    with np.errstate(all='ignore'):
        for i in order:
            n = nodes[i]
            op, kind = n['op'], n['type']
            args = [values[a] for a in n['args']]
            if op == 'const':
                v = n['value']
            elif op == 'input':
                if n['name'] not in inputs:
                    raise Unsupported('no value for ' + n['name'])
                v = inputs[n['name']]
            elif op == 'select':
                v = np.where(args[0], args[1], args[2])
            elif op == 'channel':
                v = args[0][n['index']]
            elif op == 'lookup':
                x = np.asarray(args[0], np.float64)
                table = np.asarray(n['table'] + [0.0], np.float32)
                present = np.asarray(n['present'] + [False])
                k = x - n['low']
                ok = np.isfinite(k) & (k == np.floor(k)) & (k >= 0) & (k < len(n['table']))
                index = np.where(ok, np.nan_to_num(k), len(n['table'])).astype(np.int64)
                v = np.where(ok & present[index], table[index], args[1])
            elif op in ('sample', 'load'):
                values[i] = resource(n, args)
                v = None
            elif op in rvertex.VECTOR_OPS:
                v = rvertex.VECTOR_OPS[op](*args)
            elif op in sm5.OPS and sm5.OPS[op][2] is not None:
                fn = sm5.OPS[op][2]
                v = np.frompyfunc(lambda *a, fn=fn: fn(list(a)), len(args), 1)(*[np.asarray(a).astype(object) for a in args])
            else:
                raise Unsupported('%s %s' % (op, n.get('reason', '')))
            if v is not None:
                values[i] = _typed(v, kind)
            for a in set(n['args']):
                if last.get(a) == i and a not in keep:
                    del values[a]
    return {r: values[r] for r in roots}


class Sampler:
    """The textures of one material, read as the bake samples them (module docstring). ``textures``: {binding: path}."""

    def __init__(self, source, textures, cache=None):
        self.source, self.textures = source, textures
        self.cache = cache if cache is not None else {}
        self.notes, self.read = set(), set()
        self.lock = threading.Lock()   # chunks run on several threads; textures decode once

    def texture(self, binding):
        from rc2016.games.rcra.texture import load_texture_full
        path = self.textures.get(binding)
        if path is None:
            raise ValueError('the program samples %s, which the material does not bind' % binding)
        with self.lock:
            if path not in self.cache:
                self.cache[path] = load_texture_full(self.source, path)
            self.read.add(path)
            return path, self.cache[path]

    def size(self, binding):
        _path, texture = self.texture(binding)
        return int(texture['width']), int(texture['height'])

    def texels(self, binding, index):
        """Float32 (4, h * w) channel planes of one slice as the sampler returns it, and (h, w): sRGB colour decoded,
        missing channels 0, alpha 1."""
        from rc2016.games.rcra import blender_export as bexp
        path, texture = self.texture(binding)
        index = bexp.slice_index(texture, index)
        key = ('texels', path, index)
        with self.lock:
            return self._texels(key, texture, index)

    def _texels(self, key, texture, index):
        from rc2016.games.rcra import blender_export as bexp
        from rc2016.viewer.textures import SRGB_FORMATS
        if key not in self.cache:
            srgb = bool(texture.get('srgb')) or texture['format'] in SRGB_FORMATS
            if texture['format'] in bexp.FLOAT_FORMATS:
                data = bexp.texel_array(texture, index)
                h, w = data.shape[:2]
                planes = np.ascontiguousarray(data.reshape(h * w, 4).T)
                if srgb:
                    planes[:3] = _srgb_decode(planes[:3])
            else:   # 8-bit: bexp.texel_array's values through a 256-entry table
                from rc2016.games.rcra.viewer_materials import _texel_bytes
                data = _texel_bytes(texture, index)
                h, w = data.shape[:2]
                flat = data.reshape(h * w, 4)
                planes = np.empty((4, h * w), np.float32)
                for c in range(4):
                    planes[c] = (SRGB_TABLE if srgb and c < 3 else UNIT_TABLE)[flat[:, c]]
            self.cache[key] = planes, (h, w)
        return self.cache[key]

    def _slices(self, node, args, count):
        """(per-texel slice indices or None, the distinct indices)."""
        if node.get('dimension') != DIMENSION_2D_ARRAY or len(args) < 3:
            return None, [0]
        z = np.asarray(args[2])
        if node['op'] == 'load':
            z = z.astype(np.uint32).view(np.int32).astype(np.int64)
        else:
            z = np.rint(np.nan_to_num(z.astype(np.float64), nan=0.0, posinf=1e9, neginf=-1e9)).astype(np.int64)
        if z.ndim == 0:   # a constant slice
            return None, [int(z)]
        z = np.broadcast_to(z, (count,))
        return z, np.unique(z)

    def __call__(self, node, args, chunk=None):
        """The four channels a sample or load returns; ``chunk`` (from ``bake``): the grid size, the texel range and the
        UV0 arrays being evaluated, for reads of a grid-sized texture at UV0."""
        count = max(np.size(a) for a in args)
        slices, unique = self._slices(node, args, count)
        if len(unique) > MAX_SLICES:
            raise ValueError('a sample reads %d array slices' % len(unique))
        out = np.zeros((4, count), np.float32)
        if node['op'] == 'load':
            x, y = (np.broadcast_to(np.asarray(a), (count,)).astype(np.uint32).view(np.int32).astype(np.int64)
                    for a in args[:2])
            mip = args[3] if len(args) > 3 else 0
            if np.any(np.asarray(mip) != 0):
                self.notes.add(MIP_NOTE)
            for index in unique:
                planes, (h, w) = self.texels(node['binding'], int(index))
                pick = (x >= 0) & (y >= 0) & (x < w) & (y < h)
                if slices is not None:
                    pick &= slices == index
                out[:, pick] = planes[:, y[pick] * w + x[pick]]
            return list(out)
        if node.get('opcode') != 'sample':
            self.notes.add(MIP_NOTE)
        offsets = node.get('offsets') or [0, 0, 0]
        for index in unique:
            planes, (h, w) = self.texels(node['binding'], int(index))
            if (chunk is not None and len(unique) == 1 and args[0] is chunk['u'] and args[1] is chunk['v']
                    and (w, h) == chunk['size'] and not any(offsets[:2])):
                return list(planes[:, chunk['start']:chunk['stop']])   # read at UV0 at the grid's own size: these texels
            pick = None if slices is None else slices == index
            u, v = (np.broadcast_to(np.asarray(a, np.float32), (count,)) for a in args[:2])
            if pick is not None:
                u, v = u[pick], v[pick]
            x0, tx = _texel_coordinates(u, w, offsets[0])
            y0, ty = _texel_coordinates(v, h, offsets[1])
            if np.all(np.minimum(tx, 1 - tx) < NEAREST) and np.all(np.minimum(ty, 1 - ty) < NEAREST):
                # Every read lands on a texel centre (a texture of the grid's size at UV0): that texel.
                x0[tx >= 0.5] += 1
                y0[ty >= 0.5] += 1
                value = planes[:, (y0 % h) * w + x0 % w]
            else:
                x1, y1 = (x0 + 1) % w, (y0 + 1) % h
                y0 *= w
                y1 *= w
                top = planes[:, y0 + x0]
                top += (planes[:, y0 + x1] - top) * tx
                bottom = planes[:, y1 + x0]
                bottom += (planes[:, y1 + x1] - bottom) * tx
                top += (bottom - top) * ty
                value = top
            if pick is None:
                out = value
            else:
                out[:, pick] = value
        return list(out)


def _texel_coordinates(u, size, offset):
    """(first texel index wrapped to the texture, int64; float32 weight of the next) of a bilinear read along one axis.
    A NaN or infinite address reads texel 0."""
    f = u * np.float32(size) + np.float32(offset - 0.5)
    np.nan_to_num(f, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(f, -2.0 ** 30, 2.0 ** 30, out=f)
    first = np.floor(f)
    weight = f - first
    return first.astype(np.int64) % size, weight


def grid_size(sampler, bindings):
    """(width, height) of the bake grid: the largest texture of ``bindings`` per side, each at most MAX_SIZE; (1, 1)
    without textures."""
    width = height = 1
    for binding in bindings:
        w, h = sampler.size(binding)
        width, height = max(width, w), max(height, h)
    return min(width, MAX_SIZE), min(height, MAX_SIZE)


def bake(recipe, names, sampler, size):
    """{term: float32 (height, width)} of the terms ``names`` at the texel centres of a ``size`` grid in UV0 space. Call
    only when ``blocker`` returns None."""
    width, height = size
    roots = {name: recipe['terms'][name] for name in names if name in recipe['terms']}
    out = {name: np.empty(width * height, np.float32) for name in roots}
    total = width * height
    xs = (np.arange(width, dtype=np.float64) + 0.5) / width
    ys = (np.arange(height, dtype=np.float64) + 0.5) / height
    def run(start):
        k = np.arange(start, min(total, start + CHUNK))
        inputs = {UV0[0]: xs[k % width].astype(np.float32), UV0[1]: ys[k // width].astype(np.float32),
                  FRONT_FACE: True}
        chunk = dict(size=(width, height), start=int(k[0]), stop=int(k[-1]) + 1, u=inputs[UV0[0]], v=inputs[UV0[1]])
        values = evaluate(recipe['nodes'], sorted(set(roots.values())), inputs,
                          lambda node, args: sampler(node, args, chunk))
        for name, i in roots.items():
            out[name][k] = np.broadcast_to(np.asarray(values[i], np.float32), k.shape)
    starts = range(0, total, CHUNK)
    if len(starts) == 1 or THREADS == 1:
        for start in starts:
            run(start)
    else:
        with ThreadPoolExecutor(THREADS) as pool:
            list(pool.map(run, starts))
    return {name: array.reshape(height, width) for name, array in out.items()}
