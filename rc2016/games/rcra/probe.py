"""Rift Apart environment probes: a review probe prefiltered by the engine's own probe convolution program.

The game lights its ambient and reflections from environment probes. Each probe is a 256 x 256 cube rendered from the
scene, whose mips 1 to 5 (128 down to 8 texels) RiftApart.exe prefilters on the GPU (render task "EnvProbeMipGen",
0x14108d140): per probe and face it dispatches one compute program (``find_program``: the one that writes
g_EnvProbeConvolveDst_0 to _4) with one 64-byte g_MiscArray record per output mip, which the CPU fills
(0x14108cd60) from four tables in the executable (RECORD_TABLES). The program samples the source cube with GGX importance
sampling, 2^n samples per texel, reading the source's mips by the sample's solid angle.

The viewer has no scene to render a probe from, so the review environment stands in for it, as the review lights stand in
for the game's lights: ``probe_levels`` turns the review HDRI (its residual after the sun is split out,
rc2016.viewer.environment) into the source cube, and runs the translated program over it for mips 1 to 5. The source's own
mip chain (the convolution's input lods) is a 2 x 2 box chain; how the game builds it is not traced (labelled,
SOURCE_MIPS_NOTE).

Cube faces are in D3D (and GL) order +X, -X, +Y, -Y, +Z, -Z, rows top first (blender_export.CUBE_FACE_AXES), in whatever
frame the directions are given; the convolution is the same in every frame.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
from pathlib import Path

import numpy as np

from rc2016.shaders import dxbc, sm5

EXECUTABLE = 'RiftApart.exe'
SIZE = 256        # the probe cube's top mip (the program's lod formula: 6 x 256^2 / 4 pi = 31291.13)
LEVELS = 6        # lod 0 the source, lods 1 to 5 the convolved mips
DESTINATIONS = tuple('g_EnvProbeConvolveDst_%d' % k for k in range(5))
SOURCE = 'g_EnvProbeArray'
RECORD_BUFFER = 'MiscCBufferArray'
# The per-mip tables the CPU copies into the records (RiftApart.exe, 0x14108cd60), by output mip 0 to 4 (probe mips 1 to 5).
# A at 0x1432d1f50, B at 0x1432d1f68 (f32), C at 0x1452f0b60, n at 0x1432d1ff0.
RECORD_TABLES = dict(a=(0.00036187001, 0.0054840399, 0.04155698, 0.20888358, 0.80000001),
                     b=(0.12, 0.12, 0.08, 0.04, 0.0),
                     c=(0.98217189, 0.91398168, 0.48696753, 0.0031622776, 0.0),
                     n=(6, 6, 7, 8, 8))
# Thread layout of one dispatch (the program's first instructions): output mip k covers threads x and y in
# [BASE[k], BASE[k] + size), size = 128 >> k.
BASE = (0, 128, 192, 224, 240)
SOURCE_MIPS_NOTE = ('The probe\'s source mips, which the convolution samples by solid angle, are a 2 x 2 box chain of the '
                    'source face; how the game builds them is not traced.')
PROBE_NOTE = ('The review environment (the HDRI without its sun) stands in for the game\'s environment probe: a 256 x 256 '
              'cube, its mips 1 to 5 prefiltered by the engine\'s own probe convolution program with its per-mip records.')
_LOCK = threading.Lock()
_UNROLL_LOCK = threading.Lock()
_PROGRAMS = {}


def record(mip, face, probe=0):
    """{byte offset: float} of the g_MiscArray record the CPU writes for output ``mip`` of ``face`` (0x14108cd60)."""
    t = RECORD_TABLES
    size = 256 >> (mip + 1)
    n = t['n'][mip]
    return {0: float(probe), 4: float(face), 8: t['a'][mip], 12: t['b'][mip], 16: 1.0 / size, 20: 0.5 / size,
            24: t['c'][mip], 28: 4.0 / (1.0 - t['c'][mip]), 32: 1.5, 36: 0.0, 40: 0.0, 44: 0.0, 48: float(1 << n),
            52: float(n), 56: 0.0, 60: 0.0}


def find_program(data):
    """(offset, DXBC bytes) of the probe convolution program in the executable bytes."""
    for at, size in dxbc.find_containers(data):
        blob = bytes(data[at:at + size])
        try:
            shader = dxbc.shader(blob)
        except dxbc.DxbcError:
            continue
        r = shader.reflection
        if shader.stage == 'cs' and r is not None and {b.name for b in r.bindings} >= set(DESTINATIONS) | {SOURCE}:
            return at, blob
    raise ValueError('Probe convolution program not found in the executable')


def program(root):
    path = os.path.join(str(root), EXECUTABLE)
    stat = os.stat(path)
    key = (path, stat.st_size, stat.st_mtime_ns)
    with _LOCK:
        if key not in _PROGRAMS:
            with open(path, 'rb') as f:
                _PROGRAMS[key] = find_program(f.read())
        return _PROGRAMS[key]


class _RecordBindings(sm5.Bindings):
    """One output mip's record as the dynamic g_MiscArray reads (so the sample loop has a constant count); the source cube
    is sampled (sample nodes), thread ids stay inputs."""

    def __init__(self, mip):
        self.mip = mip
        self.graph = None

    def constant(self, name):
        return None

    def dynamic_constant(self, buffer, offset, index):
        if buffer != RECORD_BUFFER:
            raise ValueError('Unexpected dynamic constant buffer ' + buffer)
        # The face is a run-time input (one translation serves the six faces); the rest is the mip's record.
        if offset == 4:
            return self.graph.input('FACE', 'f')
        return self.graph.const('f', record(self.mip, 0)[offset])

    def input(self, name):
        return None

    def resource(self, graph, op, binding, args):
        return None


def translate(blob, mip):
    """(graph, [r, g, b] value nodes) of output ``mip``'s store, for every face (input FACE) and thread (vThreadID.x/y)."""
    from rc2016.games.rcra import surface as rsurface
    bindings = _RecordBindings(mip)
    translator = sm5.Translator(dxbc.shader(blob), bindings)
    bindings.graph = translator.g
    # The sample loop runs 2^n times (a constant here); the translator's unroll cap is for unknown loops, so it is raised
    # to the count for this translation only.
    with _UNROLL_LOCK:
        saved = sm5.MAX_LOOP_ITERATIONS
        sm5.MAX_LOOP_ITERATIONS = max(saved, (1 << RECORD_TABLES['n'][mip]) + 1)
        try:
            tr = rsurface._run(translator)
        finally:
            sm5.MAX_LOOP_ITERATIONS = saved
    store = next(s for s in tr.stores if s['binding'] == DESTINATIONS[mip])
    values = [store['values'][c] for c in range(3)]
    unknown = tr.graph.unknowns(values)
    if unknown:
        raise rsurface.Unsupported('Probe convolution depends on unknown values: ' + ', '.join(unknown[:3]))
    return tr.graph, values


# ------------------------------------------------------------------------------------------------ cubes


def face_directions(n):
    """(6, n, n, 3) unit directions of the texel centres of an n x n cube, D3D face table, rows top first."""
    from rc2016.games.rcra.blender_export import CUBE_FACE_AXES
    s = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    out = np.zeros((6, n, n, 3))
    for f, ((ma, ma_sign), (sc, sc_sign), (tc, tc_sign)) in enumerate(CUBE_FACE_AXES):
        out[f, :, :, ma] = ma_sign
        out[f, :, :, sc] += sc_sign * s[None, :]
        out[f, :, :, tc] += tc_sign * s[:, None]
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def box_chain(faces):
    """[faces, faces / 2, ... 1 x 1] of (6, n, n, c) faces, each level the 2 x 2 mean of the one above."""
    chain = [np.asarray(faces, np.float32)]
    while chain[-1].shape[1] > 1:
        f = chain[-1]
        chain.append(0.25 * (f[:, 0::2, 0::2] + f[:, 1::2, 0::2] + f[:, 0::2, 1::2] + f[:, 1::2, 1::2]))
    return chain


def sample_chain(chain, direction, lod):
    """Trilinear lookup of a cube mip chain along ``direction`` (..., 3) at ``lod`` (...), as a D3D sample_l: the lod clamped
    to the chain, bilinear within a face (clamped at its edges), linear between the two mips."""
    from rc2016.games.rcra.blender_export import cube_lookup
    lod = np.clip(np.nan_to_num(np.asarray(lod, np.float64), nan=0.0), 0.0, len(chain) - 1.0)
    lo = np.floor(lod).astype(int)
    hi = np.minimum(lo + 1, len(chain) - 1)
    frac = (lod - lo)[..., None]
    out = np.zeros(np.shape(lod) + (chain[0].shape[-1],))
    for level in np.unique(np.concatenate([lo.ravel(), hi.ravel()])):
        texel = None
        for which, weight in ((lo, 1.0 - frac), (hi, frac)):
            mask = which == level
            if mask.any():
                if texel is None:
                    texel = cube_lookup(chain[level], direction)
                out += np.where(mask[..., None], weight * texel, 0.0)
    return out


def convolve(blob, chain, mip, faces=range(6)):
    """(6, size, size, 3) output ``mip`` (probe mip ``mip`` + 1) of the program over the source ``chain``."""
    from rc2016.games.rcra import bake
    graph, values = translate(blob, mip)
    size = 128 >> mip
    y, x = np.mgrid[0:size, 0:size]
    out = np.zeros((6, size, size, 3), np.float32)

    def resource(node, args):
        if node.get('binding') != SOURCE:
            raise ValueError('Unexpected texture read ' + str(node.get('binding')))
        d = np.stack([np.asarray(args[c], np.float64) for c in range(3)], -1)
        lod = np.asarray(args[4], np.float64) if len(args) > 4 else np.zeros(d.shape[:-1])
        texel = sample_chain(chain, d, lod)
        return [texel[..., c] if c < texel.shape[-1] else np.ones(texel.shape[:-1]) for c in range(4)]
    for face in faces:
        inputs = {'vThreadID.x': (x + BASE[mip]).astype(np.uint32).ravel(), 'vThreadID.y': (y + BASE[mip]).astype(np.uint32).ravel(),
                  'vThreadID.z': np.zeros(size * size, np.uint32), 'FACE': np.full(size * size, float(face), np.float32)}
        result = bake.evaluate(graph.nodes, values, inputs, resource)
        out[face] = np.stack([np.broadcast_to(np.asarray(result[v], np.float32), (size * size,)) for v in values],
                             -1).reshape(size, size, 3)
    return out


def source_faces(image, size=SIZE, frame=None):
    """(6, size, size, 3) source cube of an equirectangular image (rows top first, Blender's mapping as
    rc2016.viewer.environment): face texel direction d, in the cube's frame, looks up the image at frame @ d."""
    h, w, _c = image.shape
    d = face_directions(size)
    if frame is not None:
        d = d @ np.asarray(frame, np.float64).T
    u = 0.5 - np.arctan2(d[..., 1], d[..., 0]) / (2.0 * math.pi)
    v = np.arctan2(d[..., 2], np.hypot(d[..., 0], d[..., 1])) / math.pi + 0.5
    x = (u * w - 0.5) % w
    yy = np.clip((1.0 - v) * h - 0.5, 0.0, h - 1.0)
    x0, y0 = np.floor(x).astype(int), np.floor(yy).astype(int)
    x1, y1 = (x0 + 1) % w, np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[..., None], (yy - y0)[..., None]
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return (top * (1 - fy) + bottom * fy).astype(np.float32)


def probe_levels(root, image, frame=None):
    """[lod 0 .. lod 5] of the review probe, each (6, n, n, 3) float32: lod 0 the source cube of ``image``, lods 1 to 5 the
    engine's convolution of it."""
    _offset, blob = program(root)
    source = source_faces(image, SIZE, frame)
    chain = box_chain(source)
    return [source] + [convolve(blob, chain, mip) for mip in range(5)]


def _code_key():
    here = Path(__file__).resolve()
    return hashlib.sha256(b''.join(p.read_bytes() for p in (here, here.parents[2] / 'shaders' / 'sm5.py'))).hexdigest()


def cached_probe_levels(root, image, cache_dir, frame=None):
    """probe_levels kept in ``cache_dir`` by the image bytes, the frame, the program and this code."""
    _offset, blob = program(root)
    digest = hashlib.sha256(np.ascontiguousarray(image, np.float32).tobytes() + blob + _code_key().encode() +
                            np.asarray(frame if frame is not None else np.eye(3), np.float64).tobytes()).hexdigest()
    path = Path(cache_dir) / ('rcra_probe_%s.npz' % digest[:24])
    if path.is_file():
        try:
            with np.load(path) as stored:
                return [stored['lod%d' % k] for k in range(LEVELS)]
        except (OSError, ValueError, KeyError):
            pass
    levels = probe_levels(root, image, frame)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp.npz')
        np.savez(temporary, **{'lod%d' % k: level for k, level in enumerate(levels)})
        temporary.replace(path)
    except OSError:
        pass
    return levels
