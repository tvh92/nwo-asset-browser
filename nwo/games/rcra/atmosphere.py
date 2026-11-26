"""Rift Apart atmospheres: a level's key light, default environment probe and tone-map adaptation, read from the game's
``.atmosphere`` assets, as lighting for the model viewer and the Blender Scene export.

An atmosphere asset is 36 bytes of toc header and a DAT1 whose section 0x02F06D4E is the built parameter block
(RiftApart.exe loads it at 0x140fae710, ``kBuiltAtmosphereBaseVersion`` 229). Its field order follows the exe's
reflected ``AtmosphereDef`` / ``EnvLightingDef`` / ``ToneMapSettingsDef`` type records; ``FIELDS`` lists the offsets this
module reads. References are 24 bytes: a u64 asset id, 12 zero bytes and the tag 0x95A3A227.

What the lighting is made of:

- **Key light:** KeyLightColor x KeyLightIntensity is the apply programs' ``LIGHT``: diffuse is albedo x LIGHT x N.L with
  no 1/pi, so a Blender sun (or the viewer's sun, which divides by pi) gets irradiance pi x LIGHT. Its direction is
  (cos el sin az, sin el, cos el cos az) in game space, azimuth from +Z towards +X. That convention is fitted from where
  the probes are brightest (Megalopolis day: key 220 degrees, probe 209); the code that converts the angles is not traced.
- **Environment:** the atmosphere's FillLightCubeMap, the level's default env probe: a BC6H cube (nwo.formats.bc6h)
  in scene radiance. The viewer convolves it with the engine's probe program as it does the review HDRI.
- **Exposure:** the game meters the frame (CS_AverageLumAccum: centre-weighted mean of log2 luminance) and soft-clamps
  the result to the atmosphere's AdaptationAnchor +- AdaptationRange (CS_AverageLumFinalize); the display exposes by
  0.18 / adapted luminance; scene EV = log2 + 13. ``meter`` and ``adapted_log2`` port those programs.

The model has no place in the level, so the environment is turned about the vertical until the key light comes from the
review lighting's sun azimuth (``environment``): the same side of the model is lit as under the review HDRI. That turn
is a presentation choice.

- **Colour correction:** the ColorCorrection SourcePath (block offset 768) is a 32^3 or 33^3 R8G8B8A8_UNORM LUT
  (``read_lut``). Every frame the game blends the active atmospheres (0x1411d3500); its texture-field callback
  (0x1411d62b0) adds each atmosphere's LUT to a list weighted by how far that atmosphere is blended in, the list is
  copied to the post-process state (0x1411d3b3a) and the tone map binds up to four of them with their weights as m_B
  (0x1411db740). One fully active atmosphere therefore applies its LUT at weight 1 (display.colour_corrected).
  ColorCorrectionSystem (0x1411cb1a0) would layer more LUTs from system_ColorCorrection.config, which the PC install
  does not ship, so it changes nothing; a debug toggle (EnableColorCorrection) can clear the list.

Not reproduced: shadows from the level, the light grid, local lights, fog, the sky model, bloom settings per
atmosphere, AdaptationBias, ExposureBias and LuminanceKeyValue (LIMITATIONS).
"""
from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path

import numpy as np

DEFAULT = 'atmosphere/atm_malinon/atm_malinon_day.atmosphere'   # Corson V, Megalopolis by day
BLOCK = 0x02F06D4E
REFERENCE_TAG = 0x95A3A227
# Built-block byte offsets (f32 unless named a reference).
FIELDS = dict(sky=64, probe=88, key_colour=112, key_intensity=124, key_azimuth=128, key_elevation=132,
              anchor=636, range=640, bias=644, adjust_time=648, colour_correction=768)
MIDDLE_GREY = 0.18      # ToneMapCBuffer.m_AcesMiddleGray (the basic tone-map program inlines 0.18)
EV_OFFSET = 13.0        # displayed EV = log2(luminance) + 13 (RiftApart.exe 0x1412391e8)
ADAPT_SLOPE = 3.749     # CS_AverageLumFinalize: slope 1 at the anchor
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
PROBE_LEVEL = 1         # decode the probe's mip 1 (512 x 512 per face for the 1024 probes)
VERSION = 1             # bump when the baked environment changes; it names the cache files
LIMITATIONS = ('Atmosphere lighting: the level\'s key light and default env probe, with the game\'s auto-exposure. '
               'The environment is turned so the key light comes from the review sun\'s side (the model has no place '
               'in the level). Its colour-correction LUT is applied at weight 1, as for one fully active atmosphere. '
               'Not reproduced: the level\'s shadows, light grid, local lights, fog, AdaptationBias and ExposureBias.')


def _container(data):
    from nwo.formats.container import read_container
    return read_container(data, 0, {})


def block(source, path):
    """The built parameter block of the atmosphere at ``path``."""
    keys = [k for k in source.resolve(path) if source.entries[k]['type'] == 'atmosphere'] or list(source.resolve(path))
    if not keys:
        raise ValueError(f'No atmosphere {path}')
    data = source.asset_data(keys[0])[36:]
    section = next((s for s in _container(data)['sections'] if int(s['label_hash'], 16) == BLOCK), None)
    if section is None:
        raise ValueError(f'{path} has no built parameter block')
    return data[section['offset']:section['offset'] + section['size']]


def _reference(source, data, offset):
    asset, = struct.unpack_from('<Q', data, offset)
    if not asset:
        return None
    entry = source.entries.get('%016x' % asset)
    return entry['path'] if entry else None


def parse(data, resolve=lambda offset: None):
    """The fields of a built block ``data``; ``resolve(offset)`` names a reference."""
    f = lambda offset: struct.unpack_from('<f', data, offset)[0]   # noqa: E731
    o = FIELDS
    return dict(key_colour=[f(o['key_colour']), f(o['key_colour'] + 4), f(o['key_colour'] + 8)],
                key_intensity=f(o['key_intensity']), key_azimuth=f(o['key_azimuth']),
                key_elevation=f(o['key_elevation']), anchor=f(o['anchor']), range=f(o['range']), bias=f(o['bias']),
                adjust_time=f(o['adjust_time']), sky=resolve(o['sky']), probe=resolve(o['probe']),
                colour_correction=resolve(o['colour_correction']))


def read(source, path):
    """An atmosphere's lighting fields (``parse``) plus its ``path`` and the block's SHA-256."""
    data = block(source, path)
    fields = parse(data, lambda offset: _reference(source, data, offset))
    return dict(fields, path=path, sha256=hashlib.sha256(data).hexdigest())


def list_paths(source):
    """Every atmosphere asset path, sorted."""
    return sorted(p for p in source.by_path if p.endswith('.atmosphere'))


def label(path):
    """A short name: 'atm_malinon_day' for atmosphere/atm_malinon/atm_malinon_day.atmosphere."""
    return Path(path).stem


def key_light(atmosphere):
    """LIGHT = KeyLightColor x KeyLightIntensity (the apply programs' key-light colour)."""
    return np.asarray(atmosphere['key_colour'], np.float64) * float(atmosphere['key_intensity'])


def key_direction(azimuth, elevation):
    """Game-space (Y up) unit vector towards the key light (azimuth from +Z towards +X; inferred, see the module)."""
    a, e = math.radians(azimuth), math.radians(elevation)
    return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])


def game_to_blender(d):
    """Native Y-up directions to Blender world (Z up), as the glTF import turns the model."""
    d = np.asarray(d, np.float64)
    return np.stack([d[..., 0], -d[..., 2], d[..., 1]], -1)


def blender_to_game(d):
    d = np.asarray(d, np.float64)
    return np.stack([d[..., 0], d[..., 2], -d[..., 1]], -1)


def turn_about_up(d, radians):
    """Turn game-space directions about +Y."""
    c, s = math.cos(radians), math.sin(radians)
    d = np.asarray(d, np.float64)
    return np.stack([d[..., 0] * c + d[..., 2] * s, d[..., 1], -d[..., 0] * s + d[..., 2] * c], -1)


def probe_blocks(source, path, level=PROBE_LEVEL):
    """[(BC6H block bytes, edge)] of the six faces (D3D order +X, -X, +Y, -Y, +Z, -Z) of cube texture ``path`` at mip
    ``level``."""
    from nwo.games.rcra import texture
    data = source.asset_data(texture._entry(source, path))
    info = texture.parse_header(data)
    if info['format'] != 95 or info['faces'] != 6:
        raise ValueError(f'{path} is not a BC6H cube (format {info["format"]}, {info["faces"]} faces)')
    level = min(level, info['sd_mips'] - 1)
    faces = texture._slices(data[info['payload_offset']:], 95, info['sd_width'], info['sd_height'], info['sd_mips'],
                            6, 'SD', level)
    return [(bytes(face), info['sd_width'] >> level) for face in faces]


def probe_faces(source, path, level=PROBE_LEVEL, cache_dir=None):
    """(6, n, n, 3) float32 scene radiance of the probe's faces at mip ``level``, rows top first; cached in ``cache_dir``
    by the face bytes."""
    from nwo.formats.bc6h import decode_bc6h
    blocks = probe_blocks(source, path, level)
    digest = hashlib.sha256(b''.join(b for b, _ in blocks)).hexdigest()[:24]
    cache = Path(cache_dir) / f'atmosphere_probe_{digest}.npy' if cache_dir else None
    if cache is not None and cache.is_file():
        try:
            return np.load(cache)
        except (OSError, ValueError):
            pass
    faces = np.stack([decode_bc6h(b, n, n)[0] for b, n in blocks]).astype(np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, faces)
    return faces


def read_lut(source, path):
    """A colour-correction LUT texture (R8G8B8A8_UNORM, n x n x n, 32 or 33 in the install) as float (n, n, n, 3)
    indexed [r, g, b] in [0, 1]: x is red, y green, the depth slices blue (how the CC programs sample it)."""
    from nwo.games.rcra import texture
    data = source.asset_data(texture._entry(source, path))
    info = texture.parse_header(data)
    n = info['sd_width']
    if info['format'] != 28 or info['sd_height'] != n or info['array_size'] != n:
        raise ValueError(f'{path} is not an n^3 R8G8B8A8_UNORM LUT ({info["format"]}, '
                         f'{info["sd_width"]} x {info["sd_height"]} x {info["array_size"]})')
    raw = np.frombuffer(data, np.uint8, n ** 3 * 4, info['payload_offset'])
    return raw.reshape(n, n, n, 4)[..., :3].transpose(2, 1, 0, 3).astype(np.float64) / 255.0   # [b, g, r] -> [r, g, b]


def cube_sample(faces, d):
    """Bilinear lookup (within a face, clamped at its edges) of game-space directions ``d`` (..., 3) in a D3D cube."""
    n = faces.shape[1]
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    out = np.zeros(d.shape[:-1] + (3,), np.float32)
    with np.errstate(all='ignore'):
        sel = [((ax >= ay) & (ax >= az) & (x > 0), 0, -z / ax, -y / ax), ((ax >= ay) & (ax >= az) & (x <= 0), 1, z / ax, -y / ax),
               ((ay > ax) & (ay >= az) & (y > 0), 2, x / ay, z / ay), ((ay > ax) & (ay >= az) & (y <= 0), 3, x / ay, -z / ay),
               ((az > ax) & (az > ay) & (z > 0), 4, x / az, -y / az), ((az > ax) & (az > ay) & (z <= 0), 5, -x / az, -y / az)]
        for mask, face, u, v in sel:
            if not mask.any():
                continue
            fu = np.clip((u[mask] + 1) / 2 * n - 0.5, 0, n - 1)
            fv = np.clip((v[mask] + 1) / 2 * n - 0.5, 0, n - 1)
            j0, i0 = np.minimum(fu.astype(int), n - 2), np.minimum(fv.astype(int), n - 2)
            tu, tv = (fu - j0)[:, None], (fv - i0)[:, None]
            img = faces[face]
            out[mask] = ((img[i0, j0] * (1 - tu) + img[i0, j0 + 1] * tu) * (1 - tv)
                         + (img[i0 + 1, j0] * (1 - tu) + img[i0 + 1, j0 + 1] * tu) * tv)
    return out


def environment(source, path, sun_direction, width=2048, cache_dir=None):
    """The atmosphere at ``path`` as lighting in Blender world space, turned so its key light comes from the azimuth of
    ``sun_direction`` (Blender, the review sun's):

    {'atmosphere': read(), 'equirect': (width / 2, width, 3) float32 probe radiance in Blender's equirectangular layout
    (nwo.viewer.environment.equirect_directions, rows top first), 'sun_direction': Blender unit vector towards the key,
    'sun_irradiance': pi x LIGHT (RGB), 'turn_degrees', 'sha256': of the equirect}."""
    from nwo.viewer.environment import equirect_directions
    atm = read(source, path)
    if not atm['probe']:
        raise ValueError(f'{path} names no environment probe')
    key_game = key_direction(atm['key_azimuth'], atm['key_elevation'])
    key_b = game_to_blender(key_game)
    target = np.asarray(sun_direction, np.float64)
    turn = math.atan2(key_b[1], key_b[0]) - math.atan2(target[1], target[0])
    # A Blender turn by -turn about Z is a game turn by -turn about Y (Blender z = game y; x, y = game x, -z).
    sun_b = game_to_blender(turn_about_up(key_game, -turn))
    faces = probe_faces(source, atm['probe'], cache_dir=cache_dir)
    dirs = equirect_directions(width, width // 2).astype(np.float64)
    equi = cube_sample(faces, turn_about_up(blender_to_game(dirs), turn)).astype(np.float32)
    return dict(atmosphere=atm, equirect=equi, sun_direction=(sun_b / np.linalg.norm(sun_b)).astype(np.float32),
                sun_irradiance=(math.pi * key_light(atm)).astype(np.float32), turn_degrees=math.degrees(turn),
                sha256=hashlib.sha256(equi.tobytes()).hexdigest())


def meter(scene):
    """log2 of the frame's centre-weighted luminance (CS_AverageLumAccum): per pixel log2 of Rec.709 luminance, weighted by
    (1 - |x|)(1 - |y|) over the frame. ``scene`` is (h, w, 3) scene-linear radiance, rows in either order."""
    scene = np.asarray(scene, np.float64)
    h, w = scene.shape[:2]
    lum = np.maximum(scene[..., :3] @ LUMINANCE, 6.10352e-05)
    y, x = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing='ij')
    weight = (1 - np.abs(x)) * (1 - np.abs(y))
    return float((np.log2(lum) * weight).sum() / weight.sum())


def adapted_log2(measured, anchor_ev, range_ev):
    """CS_AverageLumFinalize's soft clamp of the measured log2 luminance to (anchor - 13) +- range (EVs)."""
    a = anchor_ev - EV_OFFSET
    t = min(max((measured - a) / (range_ev * ADAPT_SLOPE + 1e-6) + 0.5, 0.0), 1.0)
    s = t ** 3 * (t * (6 * t - 15) + 10)
    return a + (2 * s - 1) * range_ev


def exposure(adapted):
    """The display exposure m_AcesMiddleGray / adapted luminance for an adapted log2 luminance."""
    return MIDDLE_GREY / 2.0 ** adapted


__all__ = ['DEFAULT', 'LIMITATIONS', 'adapted_log2', 'cube_sample', 'environment', 'exposure', 'key_direction',
           'key_light', 'label', 'list_paths', 'meter', 'probe_blocks', 'probe_faces', 'read', 'read_lut']
