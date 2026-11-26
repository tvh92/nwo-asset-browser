"""Rift Apart light grids: a level's baked bounce light (``.zonelightbin``), as lighting for the model viewer and the
Blender Scene export of an atmosphere.

**Asset.** A ``<level>/<name>_light_grid_<time of day>.zonelightbin`` has two toc rows. The headered one is a DAT1 (type
0xFA8D90B3) whose section 0x27204B67 lists brick centres (f32 x 3, game space) and 0xC72A514C one u32 offset per brick
plus the end, into the headerless row (``RcraSource.texture_stream``). A brick is 16 x 16 x 16 cells of one unit; cell
x + 16 y + 256 z covers [corner + (x, y, z), + 1), and the lighting programs read cell floor(p - 0.5).

**Brick** (``unpack_brick``, the CPU unpack at RiftApart.exe 0x1410a06a0): 1024 bytes of 2-bit codes (cell i in u64 i // 32,
high bits first); N = the cells with code 2 or 3; six u16 planes of N; N bytes; three bytes per code-3 cell. Code 0 is a
blocked cell (inside geometry), 1 a cell the GPU fills from its neighbours, 2 and 3 stored cells. The result is the
programs' ``g_LightGridData`` record: m_Radiance, three words of two 16-bit directions each (low half +axis, high half
-axis), and m_OcclAndScale (bits 6..9 the probe weight / 15, bits 10..13 the exponent e of the scale 2^(e - 10), the rest
the bleed term).

**Lighting** (``irradiance``, the apply programs): each 16-bit direction is Y (6 bits), b (5) and c (5). The programs sum
the three fields of the directions the normal faces, weighted by n|n|, then decode u = (b / 992 - 1)|.|, v = (c / 31744 -
1)|.| (signed squares) to (Y (1 - u + v), Y (1 + u), Y (1 - u - v)) x scale / 63, the diffuse light with no 1 / pi. They
blend it with the default probe by the probe weight max(cell weight << 6, lookup word & 0x3FC) / 960 and multiply by
m_LightGridIntensity.

**The spot** (``street_cell``): a model has no place in the level, so the light comes from one stored cell that stands on a
surface (a blocked cell directly below it, in the same brick), across every grid of the probe's level folder and time of
day: the one nearest the median log luminance of all such cells in each of the six directions. That choice is a
presentation choice; the cell's values are the game's.

Not reproduced: the GPU fill passes (the chosen cell is a stored one, so they do not change it), the trilinear blend and
bleed term between cells, the lookup words' probe weight (taken as 0: not traced), LightGridIntensity (1: its place in the
built atmosphere block is not traced), and the local probes and screen-space reflections the game reflects at street level.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

POSITIONS = 0x27204B67
OFFSETS = 0xC72A514C
BRICK = 16
CELLS = BRICK ** 3
CODE_BYTES = CELLS // 4
BLOCKED, FILL, STORED = 0x83F80000, 0x8008003F, 0x83F8003F
BLEED_MASK = 0xFFFFC03F
INTENSITY = 1.0           # m_LightGridIntensity (not traced; the apply programs' review value)
DIRECTIONS = ('+X', '-X', '+Y', '-Y', '+Z', '-Z')
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
VERSION = 1               # bump when street_cell's choice changes; it names the cache files
NOTE = ('Light grid: the level\'s bounce light at one stored cell standing on a surface, the one nearest the median of '
        'all such cells (the model has no place in the level). Not reproduced: the trilinear blend and bleed term between '
        'cells, LightGridIntensity (1), the lookup\'s probe weight (0), and the local probes and screen-space reflections '
        'the game reflects at street level.')


def _sections(dat1):
    from nwo.formats.container import read_container
    return {int(s['label_hash'], 16): (s['offset'], s['size']) for s in read_container(dat1, 0, {})['sections']}


def read_index(asset):
    """(brick centres (n, 3) float32 in game space, offsets (n + 1,) uint32) of a .zonelightbin asset (toc header +
    DAT1)."""
    dat1 = asset[36:]
    if dat1[:4] != b'1TAD':
        raise ValueError('not a Rift Apart light grid (no DAT1 after the toc header)')
    sections = _sections(dat1)
    if POSITIONS not in sections or OFFSETS not in sections:
        raise ValueError('light grid without its brick positions or offsets')
    po, ps = sections[POSITIONS]
    oo, os_ = sections[OFFSETS]
    centres = np.frombuffer(dat1, '<f4', ps // 4, po).reshape(-1, 3)
    offsets = np.frombuffer(dat1, '<u4', os_ // 4, oo)
    if len(offsets) != len(centres) + 1:
        raise ValueError('light grid with %d offsets for %d bricks' % (len(offsets), len(centres)))
    return centres, offsets


def codes(blob):
    """(4096,) uint8 cell codes of a brick blob."""
    words = np.frombuffer(blob[:CODE_BYTES], '<u8')
    shifts = np.arange(62, -1, -2, dtype=np.uint64)
    return ((words[:, None] >> shifts[None, :]) & np.uint64(3)).astype(np.uint8).ravel()


def unpack_brick(blob):
    """(cells (4096, 4) uint32 g_LightGridData records, codes (4096,)) of one brick blob, as the CPU unpack writes them."""
    code = codes(blob)
    stored = np.nonzero(code >= 2)[0]
    n = len(stored)
    third = code[stored] == 3
    n3 = int(third.sum())
    if len(blob) != CODE_BYTES + 13 * n + 3 * n3:
        raise ValueError('light grid brick of %d bytes for %d stored cells (%d with a bleed term)' % (len(blob), n, n3))
    planes = np.frombuffer(blob, '<u2', 6 * n, CODE_BYTES).reshape(6, n).astype(np.uint32)
    scale = np.frombuffer(blob, np.uint8, n, CODE_BYTES + 12 * n).astype(np.uint32)
    bleed = np.frombuffer(blob, np.uint8, 3 * n3, CODE_BYTES + 13 * n).reshape(n3, 3).astype(np.uint32)
    cells = np.zeros((CELLS, 4), np.uint32)
    cells[code == 0, 3] = BLOCKED
    cells[code == 1, 3] = FILL
    for axis in range(3):
        cells[stored, axis] = planes[2 * axis] | planes[2 * axis + 1] << 16
    word = np.full(n, STORED, np.uint32)
    word[third] = ((bleed[:, 0] << 24) | (bleed[:, 1] << 16) | (bleed[:, 2] << 8) | bleed[:, 2]) & BLEED_MASK
    cells[stored, 3] = word | scale << 6
    return cells, code


def fields(cells):
    """(Y, b, c) of the six directions (DIRECTIONS order) as the programs mask them: (n, 6, 3) float64."""
    out = np.empty((len(cells), 6, 3))
    for axis in range(3):
        for half in (0, 1):
            v = (cells[:, axis] >> (16 * half)) & 0xFFFF
            out[:, 2 * axis + half] = np.stack([v & 0x3F, v & 0x7C0, v & 0xF800], -1)
    return out


def scale(cells):
    return 2.0 ** (((cells[:, 3] >> 10) & 15).astype(np.float64) - 10)


def probe_weight(cells):
    """The cells' own probe weight (bits 6..9 << 6, / 960); the lookup word's part is not traced."""
    return ((cells[:, 3] >> 6) & 15).astype(np.float64) / 15.0


def decode(y, b, c):
    """RGB of summed (Y, b, c) fields, before the scale and the 1 / 63."""
    u = b * (1 / 992.0) - 1
    v = c * (1 / 31744.0) - 1
    u, v = np.abs(u) * u, np.abs(v) * v
    return np.stack([y * (1 - u + v), y * (1 + u), y * (1 - u - v)], -1)


def irradiance(cell, normals):
    """The light grid's diffuse light (no 1 / pi) of one cell (a ``street_cell`` record) for game-space unit ``normals``
    (..., 3), as the apply programs decode it, x INTENSITY."""
    f = np.asarray(cell['fields'], np.float64)            # (6, 3)
    n = np.asarray(normals, np.float64)
    w = np.abs(n) * n
    total = np.zeros(n.shape[:-1] + (3,))
    for axis in range(3):
        pos = w[..., axis:axis + 1] > 0
        total += np.abs(w[..., axis:axis + 1]) * np.where(pos, f[2 * axis], f[2 * axis + 1])
    return decode(total[..., 0], total[..., 1], total[..., 2]) * (cell['scale'] / 63.0 * INTENSITY)


def grids_for_probe(source, probe_path):
    """The light grid assets of an env probe's level: ``<level>/*_light_grid_<tod>.zonelightbin`` beside the probe's
    ``envprobes_<tod>`` folder, sorted by path."""
    probe = PurePosixPath(probe_path)
    folder = probe.parent
    if not folder.name.startswith('envprobes_'):
        return []
    tod = folder.name[len('envprobes_'):]
    level = str(folder.parent).replace('\\', '/') + '/'
    suffix = f'_light_grid_{tod}.zonelightbin'
    found = sorted({source.entries[k]['path'] for k in source.ids_of_type('zonelightbin')
                    if source.entries[k]['path'].startswith(level) and source.entries[k]['path'].endswith(suffix)
                    and '/' not in source.entries[k]['path'][len(level):]})
    return found


def bricks(source, path):
    """(centres, [brick blobs]) of the light grid at ``path``."""
    keys = [k for k in source.resolve(path) if source.entries[k]['type'] == 'zonelightbin'] or list(source.resolve(path))
    if not keys:
        raise ValueError(f'No light grid {path}')
    centres, offsets = read_index(source.asset_data(keys[0]))
    stream = source.texture_stream(keys[0])
    if stream is None or len(stream) < int(offsets[-1]):
        raise ValueError(f'{path} has no cell stream')
    return centres, [stream[offsets[i]:offsets[i + 1]] for i in range(len(centres))]


def ground_cells(source, paths):
    """Every stored cell with a blocked cell directly below it in its brick, over the grids at ``paths``:
    dict(fields (n, 6, 3), scale (n,), weight (n,), position (n, 3) game-space cell centres, grid (n,) index into paths)."""
    out = dict(fields=[], scale=[], weight=[], position=[], grid=[])
    for g, path in enumerate(paths):
        centres, blobs = bricks(source, path)
        for centre, blob in zip(centres, blobs):
            if len(blob) <= CODE_BYTES:
                continue
            cells, code = unpack_brick(blob)
            grid = code.reshape(BRICK, BRICK, BRICK)            # [z, y, x]
            ground = np.zeros_like(grid, bool)
            ground[:, 1:, :] = (grid[:, 1:, :] >= 2) & (grid[:, :-1, :] == 0)
            index = np.nonzero(ground.ravel())[0]
            if not len(index):
                continue
            z, y, x = np.unravel_index(index, grid.shape)
            out['fields'].append(fields(cells[index]))
            out['scale'].append(scale(cells[index]))
            out['weight'].append(probe_weight(cells[index]))
            out['position'].append(centre - BRICK / 2 + np.c_[x, y, z] + 0.5)
            out['grid'].append(np.full(len(index), g))
    if not out['fields']:
        return None
    return {k: np.concatenate(v) for k, v in out.items()}


def axis_colours(found):
    """(n, 6, 3) RGB of each cell for the six axis normals."""
    f = found['fields']
    return decode(f[..., 0], f[..., 1], f[..., 2]) * (found['scale'][:, None, None] / 63.0 * INTENSITY)


def pick(found):
    """Index of the cell nearest the median log luminance in all six directions."""
    lum = np.log(np.maximum(axis_colours(found) @ LUMINANCE, 1e-4))
    return int(np.argmin(((lum - np.median(lum, 0)) ** 2).sum(1)))


def street_cell(source, probe_path, cache_dir=None):
    """The light-grid cell that lights a model for the atmosphere probe at ``probe_path`` (see the module), or None when
    its level has no light grid: dict(fields (6, 3) summed-field units, scale, weight, position (game), grid path,
    cells (how many it was chosen from), axis_rgb (6, 3), sha256 of the grids' bytes)."""
    paths = grids_for_probe(source, probe_path)
    if not paths:
        return None
    ident = hashlib.sha256(json.dumps([[p, sorted(source.resolve(p))] for p in paths]).encode()).hexdigest()[:24]
    cache = Path(cache_dir) / f'light_grid_street_v{VERSION}_{ident}.json' if cache_dir else None
    if cache is not None and cache.is_file():
        try:
            return json.loads(cache.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
    found = ground_cells(source, paths)
    if found is None:
        return None
    i = pick(found)
    record = dict(fields=found['fields'][i].tolist(), scale=float(found['scale'][i]), weight=float(found['weight'][i]),
                  position=found['position'][i].tolist(), grid=paths[int(found['grid'][i])], cells=len(found['scale']),
                  axis_rgb=axis_colours({k: v[i:i + 1] for k, v in found.items()})[0].tolist(), grids=paths)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(record), encoding='utf-8')
    return record


def _lobes(count=96):
    """Fibonacci-sphere directions for the smooth world lobes."""
    i = np.arange(count) + 0.5
    z = 1 - 2 * i / count
    r = np.sqrt(1 - z * z)
    phi = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)


LOBE_POWER = 1


def _lobe_field(directions, lobes):
    return np.maximum(directions @ lobes.T, 0.0) ** LOBE_POWER


def fit_world(target, probe, directions, solid_angle, lobes=None, tolerance=0.005):
    """A Blender world L = k probe + sum_j a_j max(0, d . b_j)^LOBE_POWER (a_j >= 0) whose diffuse light (cosine convolution / pi)
    matches ``target`` (m, 3) at the unit Blender ``directions`` (m, 3) used as normals, with ``probe`` (m, 3) the probe
    radiance at the same directions (texel solid angles ``solid_angle``). k is the largest probe share (from 0 to the
    target's luminance over the probe's diffuse light, the game's reflection scale) whose relative RMS error stays within
    ``tolerance`` of k = 0. Returns dict(k, weights (lobes, 3), lobes (lobes, 3), rms, max) of relative errors."""
    from scipy.optimize import nnls
    lobes = _lobes() if lobes is None else lobes
    conv = np.maximum(directions @ directions.T, 0.0) * solid_angle[None, :] / np.pi   # normals x world texels
    basis = conv @ _lobe_field(directions, lobes)                                        # normals x lobes
    probe_diffuse = conv @ probe

    def solve(k):
        weights = np.zeros((len(lobes), 3))
        for c in range(3):
            weights[:, c] = nnls(basis, target[:, c] - k * probe_diffuse[:, c], maxiter=2000)[0]
        fitted = basis @ weights + k * probe_diffuse
        rel = np.abs(fitted - target) / np.maximum(target, 1e-6)
        return weights, float(np.sqrt((rel ** 2).mean())), float(rel.max())

    w0, rms0, max0 = solve(0.0)
    best = (0.0, w0, rms0, max0)
    high = float(np.clip((target @ LUMINANCE).mean() / max((probe_diffuse @ LUMINANCE).mean(), 1e-9), 0.0, 1.0))
    low = 0.0
    for _ in range(14):
        k = (low + high) / 2
        weights, rms, mx = solve(k)
        if rms <= rms0 + tolerance:
            low, best = k, (k, weights, rms, mx)
        else:
            high = k
    k, weights, rms, mx = best
    return dict(k=k, weights=weights, lobes=lobes, rms=rms, max=mx, rms_without_probe=rms0)


def world(cell, probe_equirect, turn, coarse=(32, 16)):
    """The probe equirect (Blender directions, rows top first, already turned by ``turn`` radians about the vertical)
    relit by the light-grid ``cell``: (equirect, record). The diffuse target is (1 - weight) x the cell's irradiance +
    weight x the probe's diffuse light; ``fit_world`` gives the probe share k and the smooth lobes, evaluated at full
    resolution."""
    from nwo.games.rcra import atmosphere as A
    from nwo.viewer.environment import equirect_directions
    h, w, _ = probe_equirect.shape
    cw, ch = coarse
    probe = probe_equirect.astype(np.float64).reshape(ch, h // ch, cw, w // cw, 3).mean((1, 3)).reshape(-1, 3)
    d = equirect_directions(cw, ch).reshape(-1, 3).astype(np.float64)
    lat = (0.5 - (np.arange(ch) + 0.5) / ch) * np.pi
    omega = np.repeat(np.cos(lat) * (np.pi / ch) * (2 * np.pi / cw), cw)
    game = A.turn_about_up(A.blender_to_game(d), turn)
    conv = np.maximum(d @ d.T, 0.0) * omega[None, :] / np.pi
    weight = float(cell['weight'])
    target = (1 - weight) * irradiance(cell, game) + weight * (conv @ probe)
    fit = fit_world(target, probe, d, omega)
    full = equirect_directions(w, h).reshape(-1, 3).astype(np.float64)
    lobe_light = (_lobe_field(full, fit['lobes']) @ fit['weights']).reshape(h, w, 3)
    out = (fit['k'] * probe_equirect.astype(np.float64) + lobe_light).astype(np.float32)
    record = dict(grid=cell['grid'], position=cell['position'], probe_weight=weight, cells=cell['cells'],
                  axis_rgb=cell['axis_rgb'], reflection_scale=fit['k'], fit_rms=fit['rms'], fit_max=fit['max'],
                  intensity=INTENSITY)
    return out, record
