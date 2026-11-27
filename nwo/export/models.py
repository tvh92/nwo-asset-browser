"""Export one source model to portable files: glTF/GLB and OBJ directly, FBX/USD through Blender.

No Blender is needed for GLB, glTF or OBJ. The geometry, skeleton and clips come from the same
in-memory decoders as the 3D view (``source_preview``, ``animation_preview``); nothing else is
written. One folder per model::

    <output>/<model>/<model>.glb | .gltf + .bin | .obj + .mtl | .fbx | .usdc
    <output>/<model>/textures/   material PNGs (and the original DDS textures when asked)
    <output>/<model>/export.json source identity, options, clips and limitations
    <output>/<model>.blend         the Blender scene, when asked: beside the folder, not in it. When it is the only
                                   format, the model folder is not kept (no export.json either). With
                                   ExportOptions.scene it is render-ready: review lighting, HDRI, camera, Render
                                   Region (the Blender Scene option of a game without the research pipeline), or a
                                   Rift Apart atmosphere's lighting and the game's display (ExportOptions.atmosphere)

Coordinates stay native (Y-up, right-handed, game units), which is glTF's own convention; the
glTF importers in Blender and other tools turn that into their up axis. UVs are the game's (top
row first), textures are never flipped.

Materials use the 3D view's per-graph reading of ``_g`` (``texture_preview.material_texture_paths``):
grey (R gloss, G specular), skin (grey, cap 0.5) or coloured (rgb F0, gloss in alpha), with the
material's specular/gloss scales. The game's specular/gloss BRDF is converted per pixel to
glTF metal/roughness (Khronos spec-gloss conversion), roughness = 2^(-2.75 gloss) as in the
viewer and the RC PS4 Deferred BRDF. Self-lit is ``_c.rgb x mask x K`` with the mask from
``_c.a`` or ``_g.B``. Graph-specific extras (tints, palettes, detail maps, rims, glass,
animated terms) are not baked: a portable approximation, not game equivalence.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import weakref

import numpy as np

from nwo.viewer.animation import NotPlayable

FORMATS = {   # key: (label, file suffix, needs Blender)
    'glb': ('glTF binary (.glb)', '.glb', False),
    'gltf': ('glTF + textures (.gltf)', '.gltf', False),
    'obj': ('Wavefront (.obj + .mtl)', '.obj', False),
    'fbx': ('Autodesk FBX (.fbx)', '.fbx', True),
    'usd': ('Universal Scene Description (.usdc)', '.usdc', True),
    'blend': ('Blender scene (.blend)', '.blend', True),
}
BLENDER_SCRIPT = Path(__file__).with_name('blender_convert.py')


@dataclass
class ExportOptions:
    formats: tuple = ('glb',)
    textures: bool = True            # material textures (embedded in .glb, PNG files otherwise)
    source_textures: bool = False    # also every referenced game texture as DDS, unchanged
    animations: str = 'all'          # 'all', 'none' or 'clips'
    clips: tuple = ()                # clip labels (AnimationLibrary.labels()) for 'clips'
    lod: str = 'Highest'
    look: int = 0
    blender: str = ''
    skip_existing: bool = False      # keep a folder whose export.json matches source and options
    cinematic: bool = False          # also offer the model's real-time cutscene shots
    cinematic_only: bool = False     # only those shots (no animset clips, no reference index needed)
    borrowed: bool = False           # also offer the animsets the game pairs with another model of the same rig
    file_suffix: str = ''            # appended to the model stem in file names, e.g. '_cutscenes'
    skeleton: bool = True            # False: static meshes, no joints or skin weights, so no animations either
    attachments: bool = True         # also the model's bangle_set/*_attach parts, skinned to its rig (one armature)
    root_motion: bool = False        # move the skeleton by each clip's root motion (games that decode it: Rift Apart)
    performances: bool = False       # also the face performances (voice-line lip sync; Rift Apart)
    layer_bases: bool = False        # additive layers over their first partner clip (layer_partners; Rift Apart)
    vertex_animation: bool = False   # bake vertex-program motion over shader time as morph-target frames (Rift Apart)
    scene: bool = False              # the .blend as a render-ready scene: review lighting, HDRI, camera, Render Region
    atmosphere: str = ''             # with scene: a Rift Apart atmosphere asset lights it instead (games.rcra.atmosphere)

    def key(self):
        d = asdict(self)
        d.pop('blender'); d.pop('skip_existing')
        d['formats'] = sorted(d['formats']); d['clips'] = list(d['clips'])
        for name in ('cinematic', 'cinematic_only', 'borrowed', 'file_suffix', 'root_motion', 'performances',
                     'layer_bases', 'vertex_animation', 'scene', 'atmosphere'):   # absent from keys of exports that predate them
            if not d[name]:
                d.pop(name)
        if d['skeleton']:   # the default is left out, so keys of exports that predate the option still match
            d.pop('skeleton')
        return d   # 'attachments' stays in the key: exports that predate it left the parts as separate models


class Cancelled(Exception):
    pass


def safe_name(value, maximum=70):
    value = re.sub(r'[^A-Za-z0-9._-]+', '_', value).strip('._')
    return (value or 'model')[:maximum]


def material_name(material, slot):
    """A material slot's name, as the games' ``material_maps`` give it."""
    return material.get('name') or (Path(material['path']).stem if material.get('path') else f'material_{slot}')


def plain_maps(material, slot, note):
    """``material_maps``'s dict without textures: a white opaque material with ``note``."""
    return dict(name=material_name(material, slot), base=None, mr=None, normal=None, emissive=None, emissive_strength=0.0,
                alpha_mode='OPAQUE', alpha=1.0, double_sided=False, notes=[note], sources={}, png={})


# ---------------------------------------------------------------- textures / materials
def _srgb_to_linear(x):
    x = np.asarray(x)
    low = x <= 0.04045
    out = x / 12.92
    out[~low] = ((x[~low] + 0.055) / 1.055) ** 2.4   # the power only on the curved branch
    return out


def _linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


_SRGB_LUT = _srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0)   # float32 sRGB -> linear for 8-bit input


def _unit(u8):
    """uint8 -> float32 0..1 (exactly ``float32(v) / 255``)."""
    return u8.astype(np.float32) / 255.0


def _rgba_u8(texture, size=None):
    """uint8 RGBA array (h, w, 4), values as stored; optionally resized."""
    from PIL import Image
    from nwo.viewer.textures import to_image
    image = to_image(texture).convert('RGBA')
    if size and image.size != size:
        image = image.resize(size, Image.BILINEAR)
    return np.asarray(image)


def _rgba(texture, size=None):
    """Float RGBA array (h, w, 4) in 0..1, values as stored; optionally resized."""
    return _unit(_rgba_u8(texture, size))


def _u8(array):
    """0..1 float -> 8-bit, the quantisation every exported PNG uses."""
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def _png(array, mode='RGB'):
    from PIL import Image
    import io
    data = np.ascontiguousarray(array if array.dtype == np.uint8 else _u8(array))
    if mode == 'RGB' and data.ndim == 3 and data.shape[2] == 4:   # a base map with alpha (Rift Apart glTF BLEND)
        mode = 'RGBA'
    buffer = io.BytesIO()
    Image.fromarray(data, mode).save(buffer, 'PNG', optimize=False, compress_level=1)
    return buffer.getvalue()


class _MapCache:
    """Converted material maps (8-bit arrays plus their PNG bytes) keyed by source textures and mode, LRU by size."""

    def __init__(self, limit_bytes=768 << 20):
        from collections import OrderedDict
        self.items, self.limit = OrderedDict(), limit_bytes

    @staticmethod
    def size(entry):
        return sum(entry[k].nbytes for k in ('base', 'mr', 'normal', 'emissive') if entry[k] is not None)

    def get(self, key):
        entry = self.items.get(key)
        if entry is not None:
            self.items.move_to_end(key)
        return entry

    def put(self, key, entry):
        self.items[key] = entry
        while len(self.items) > 1 and sum(self.size(e) for e in self.items.values()) > self.limit:
            self.items.popitem(last=False)


def cached_png(material, key, array, mode='RGB'):
    """PNG bytes of one map, encoded once per converted-map entry (shared across formats and exports)."""
    store = material.get('png')
    png = store.get((key, mode)) if store is not None else None
    if png is None:
        png = _png(array, mode)
        if store is not None:
            store[(key, mode)] = png
    return png


def _perceived(c):
    return np.sqrt(0.299 * c[..., 0] ** 2 + 0.587 * c[..., 1] ** 2 + 0.114 * c[..., 2] ** 2)


def spec_gloss_to_metal_rough(diffuse, f0):
    """Khronos specular/glossiness -> metallic/roughness conversion, per pixel (linear inputs)."""
    dielectric = 0.04
    eps = 1e-6
    one_minus = 1.0 - f0.max(axis=-1)
    d, s = _perceived(diffuse), _perceived(f0)
    a = dielectric
    b = d * one_minus / (1.0 - dielectric) + s - 2.0 * dielectric
    c = dielectric - s
    disc = np.maximum(b * b - 4.0 * a * c, 0.0)
    metallic = np.clip((-b + np.sqrt(disc)) / (2.0 * a), 0.0, 1.0)
    metallic = np.where(s < dielectric, 0.0, metallic)
    m = metallic[..., None]
    from_diffuse = diffuse * (one_minus[..., None] / (1.0 - dielectric) / np.maximum(1.0 - m, eps))
    from_specular = (f0 - dielectric * (1.0 - m)) / np.maximum(m, eps)
    base = np.clip(from_diffuse + (from_specular - from_diffuse) * m * m, 0.0, 1.0)
    return base, metallic


def facing_glow_rgb(term, n_dot_v):
    """Linear RGB the facing-glow term adds (texture_preview 'facing_glow', gl_shading.FRAGMENT)."""
    f = max(min(max(n_dot_v, 0.0), 1.0) - term['bias'], 0.0) ** term['exponent']
    a = term['a_scale'] * f + term['a_offset']
    return [term['amplitude'] * 2.0 * a * (1.0 + a), term['amplitude'] * term['g_scale'] * f * (1.0 + a), 0.0]


def material_maps(archive, material, cache, map_cache=None):
    """PBR maps for one material slot: dict(name, base, mr, normal, emissive, factors, notes)."""
    from nwo.formats.archive import decompress
    from nwo.viewer.textures import (blend_state, load_texture_full, material_flags,
                                  material_texture_paths, SRGB_FORMATS)
    name = material.get('name') or (Path(material['path']).stem if material.get('path') else f"material_{material.get('index', 0)}")
    out = dict(name=name, base=None, mr=None, normal=None, emissive=None, emissive_strength=0.0,
               alpha_mode='OPAQUE', alpha=1.0, double_sided=False, notes=[], sources={})
    if not material.get('path'):
        out['notes'].append('no material file; neutral grey')
        out['base_factor'] = [0.5, 0.5, 0.5]
        return out
    try:
        paths = material_texture_paths(archive, material['path'])
    except (ValueError, KeyError, OSError) as error:
        paths = {}
        out['notes'].append(f'material unreadable: {error}')
    try:
        ids = [k for k in archive.resolve(material['path']) if archive.entries[k]['type'] == 'material']
        state = blend_state(material_flags(decompress(archive.bytes(ids[0])[0])[0])) if len(ids) == 1 else blend_state(None)
    except (ValueError, KeyError, OSError, IndexError):
        state = blend_state(None)
    out['double_sided'] = state['two_sided']
    if state['blend'] in ('alpha_over', 'additive'):
        out['alpha_mode'] = 'BLEND'; out['alpha'] = 0.4
        out['notes'].append(f"forward {state['blend']} material: fixed 40% opacity, not the game's glass/VFX shader")

    def texture(role):
        path = paths.get(role)
        if not path:
            return None
        if path not in cache:
            try:
                cache[path] = load_texture_full(archive, path)
            except (ValueError, KeyError, OSError) as error:
                cache[path] = None
                out['notes'].append(f'{role} texture unreadable: {error}')
        if cache[path] is not None:
            out['sources'][role] = path
        return cache[path]

    c_tex, g_tex, n_tex = texture('_c'), texture('_g'), texture('_n')
    glow = paths.get('facing_glow')
    if glow:
        out['base_factor'] = list(glow['albedo'])
        out['emissive_factor'] = facing_glow_rgb(glow, 1.0)
        out['notes'].append(f"white base (no albedo input) + facing glow of writer {glow['writer']}; glTF has no view-dependent "
                            f"emission, so the constant is the head-on value at the presentation amplitude {glow['amplitude']:g} "
                            f"({glow['amplitude_input']}, captured {glow['captured_amplitude']:g}); it falls to "
                            f"{facing_glow_rgb(glow, 0.0)[0]:.3f} red at grazing angles; blink timing not decoded")
    elif paths.get('flat_surface'):
        flat = paths['flat_surface']
        if flat.get('model') == 'basic_flat_color':
            out['base_factor'] = [0.0, 0.0, 0.0]
            emission = [float(x) * float(flat['emission_scale']) for x in flat['colour']]
            if all(x >= 0.0 for x in emission):
                strength = max(emission, default=0.0)
                out['emissive_factor'] = [x / strength for x in emission] if strength > 1.0 else emission
                out['emissive_strength'] = strength if strength > 1.0 else 1.0
                out['flat_source_emission_strength'] = strength
            else:
                out['notes'].append('basic_flat_color D169A814 has signed emission; glTF nonnegative emissiveFactor cannot represent '
                                    'this source term, so it is omitted from this export')
            out['notes'].append('basic_flat_color D169A814 bounded writer slice: zero base albedo and independent CB2 colour*K emission; '
                                'runtime light/probe/fog/debug branches and the grey F0/gloss BRDF are not reproduced')
        else:
            out['base_factor'] = [float(x) for x in flat['colour']]
        k = float(paths.get('k', 0.0) or 0.0)
        if k > 0.0 and flat.get('model') != 'basic_flat_color':
            out['emissive_factor'] = [min(1.0, float(x) * k) for x in flat['colour']]
        if flat.get('model') != 'basic_flat_color':
            out['notes'].append(f"constant colour of writer {flat['writer']} (no base colour texture)"
                                + ('; preview floor of a time- and noise-driven shader' if flat.get('floor') else ''))
    elif c_tex is None and g_tex is None:
        out['base_factor'] = [0.8, 0.8, 0.8]
        out['notes'].append('no base colour texture: the colour comes from a compiled shader this exporter does not run; white-grey placeholder')
    textured = c_tex is not None or g_tex is not None or n_tex is not None
    key = (paths.get('_c'), paths.get('_g'), paths.get('_n'), paths.get('g_mode', 'colored'), paths.get('spec', 1.0),
           paths.get('gloss', 1.0), float(paths.get('k', 0.0) or 0.0), paths.get('self_lit'))
    entry = map_cache.get(key) if map_cache is not None and textured else None
    if entry is None:
        entry = _convert_maps(c_tex, g_tex, n_tex, paths)
        entry['png'] = {}
        if map_cache is not None and textured:
            map_cache.put(key, entry)
    for name in ('base', 'mr', 'normal', 'emissive', 'emissive_strength', 'png'):
        out[name] = entry[name]
    if paths.get('flat_surface', {}).get('model') == 'basic_flat_color':
        out['emissive_strength'] = out.get('flat_source_emission_strength', 1.0)
        out.pop('flat_source_emission_strength', None)
    out['notes'] += entry['notes']
    return out


def rcra_roughness(gloss):
    """Rift Apart: glTF roughness whose GGX alpha (roughness²) is the game's r⁴, r = 0.9725 - 0.7514 gloss: roughness r²
    (docs/rift-apart.md)."""
    return np.clip(np.square(0.9725 - 0.7514 * gloss), 0.0, 1.0)


def _convert_maps(c_tex, g_tex, n_tex, paths, o_tex=None, e_tex=None):
    """The pixel work of ``material_maps``: 8-bit base / metal-rough / normal / emissive maps for one texture set.

    ``paths['roughness'] == 'rcra'`` takes Rift Apart's gloss curve (``rcra_roughness``) instead of R&C 2016's. ``o_tex``
    with ``paths['occlusion'] == 'blue'`` puts its blue channel (the sampled value) in the metal-rough map's red channel, the
    glTF occlusion channel, and sets ``occlusion``. The self-lit mask (``paths['self_lit']``) is _c alpha by default,
    'g_blue', 'one' (no mask), or 'texture': channel ``paths['self_lit_channel']`` of ``e_tex``. ``paths['base_alpha']``
    keeps _c alpha as the base map's alpha (glTF BLEND)."""
    from nwo.viewer.textures import SRGB_FORMATS
    out = dict(base=None, mr=None, normal=None, emissive=None, emissive_strength=0.0, occlusion=False, notes=[])
    size = None
    for tex in (c_tex, g_tex, o_tex):
        if tex is not None:
            size = max(size or (0, 0), (tex['width'], tex['height']))
    h_w = size
    if c_tex is not None:
        c = _rgba_u8(c_tex, h_w)
        albedo = _SRGB_LUT[c[..., :3]] if c_tex['format'] in SRGB_FORMATS else _unit(c[..., :3])
        c_alpha = _unit(c[..., 3])
    else:
        albedo = np.full((h_w[1], h_w[0], 3) if h_w else (1, 1, 3), 0.6, np.float32)
        c_alpha = np.zeros(albedo.shape[:2], np.float32)
    mode, spec, gloss_scale = paths.get('g_mode', 'colored'), paths.get('spec', 1.0), paths.get('gloss', 1.0)
    g_blue = None
    if g_tex is not None:
        g = _rgba_u8(g_tex, h_w)
        gl = _SRGB_LUT[g[..., :3]] if g_tex['format'] in SRGB_FORMATS else _unit(g[..., :3])
        g_blue = gl[..., 2]
        if mode == 'grey_skin':
            f0 = np.repeat(np.minimum(0.5, gl[..., 1:2]), 3, axis=-1)
            gloss = np.sqrt(np.clip(gl[..., 0], 0, 1))
        elif mode == 'grey':
            f0 = np.repeat(spec * gl[..., 1:2], 3, axis=-1)
            gloss = np.sqrt(np.clip(gloss_scale * gl[..., 0], 0, 1))
        elif mode == 'vehicle':
            f0 = np.broadcast_to(np.array([0.0, 0.0999756, 0.0999756], np.float32), gl.shape).copy()
            gloss = np.full(gl.shape[:2], 0.948681, np.float32)
        else:
            f0 = spec * gl
            gloss = np.sqrt(np.clip(gloss_scale * _unit(g[..., 3]) ** 2, 0, 1))
        f0 = np.clip(f0, 0.0, 1.0)
        base, metallic = spec_gloss_to_metal_rough(albedo, f0)
        rough = rcra_roughness(gloss) if paths.get('roughness') == 'rcra' else np.power(2.0, -2.75 * gloss)
        out['notes'].append(f'_g read as {mode} (specular x{spec:g}, gloss x{gloss_scale:g})')
    else:
        base, metallic, rough = albedo, np.zeros(albedo.shape[:2], np.float32), np.ones(albedo.shape[:2], np.float32)
        out['notes'].append('no _g texture: no game specular; exported fully rough')
    occlusion = np.ones_like(rough)
    if o_tex is not None and paths.get('occlusion') == 'blue':
        o = _rgba_u8(o_tex, (rough.shape[1], rough.shape[0]))
        occlusion = _SRGB_LUT[o[..., 2]] if o_tex['format'] in SRGB_FORMATS else _unit(o[..., 2])
        out['occlusion'] = True
    if c_tex is not None or g_tex is not None or out['occlusion']:
        out['base'] = _u8(_linear_to_srgb(base))
        out['mr'] = _u8(np.stack([occlusion, rough, metallic], axis=-1))
    if paths.get('base_alpha') and out['base'] is not None:
        out['base'] = np.concatenate([out['base'], _u8(c_alpha)[..., None]], axis=-1)
    k = float(paths.get('k', 0.0) or 0.0)
    self_lit = paths.get('self_lit')
    if self_lit == 'g_blue':
        mask = g_blue
    elif self_lit == 'one':
        mask = np.ones(albedo.shape[:2], np.float32)
    elif self_lit == 'texture':
        e = _rgba_u8(e_tex, (albedo.shape[1], albedo.shape[0])) if e_tex is not None else None
        channel = int(paths.get('self_lit_channel', 0))
        mask = None if e is None else (_SRGB_LUT[e[..., channel]] if e_tex['format'] in SRGB_FORMATS and channel < 3
                                       else _unit(e[..., channel]))
    else:
        mask = c_alpha if c_tex is not None else None
    if k > 0 and mask is not None and float(mask.max()) > 0.0:
        out['emissive'] = _u8(_linear_to_srgb(albedo * mask[..., None]))
        out['emissive_strength'] = k
        label = {'g_blue': '_g.B', 'one': '1', 'texture': paths.get('self_lit_label', 'mask')}.get(self_lit, '_c.a')
        out['notes'].append(f"self-lit = _c.rgb x {label} x K {k:g}")
    if n_tex is not None:
        n = _rgba(n_tex)
        x, y = n[..., 0] * 2 - 1, n[..., 1] * 2 - 1
        z = np.sqrt(np.clip(1 - x * x - y * y, 0, 1))
        out['normal'] = _u8(np.stack([n[..., 0], n[..., 1], z * 0.5 + 0.5], axis=-1))
    return out


# ---------------------------------------------------------------- geometry / skeleton / clips
def bind_worlds(rows):
    """World matrix of every joint in a model's bind pose: the local scale, rotation and translation rows chained down
    the parents (column vectors)."""
    worlds = []
    for row in rows:
        x, y, z, w = row['rotation_xyzw']
        rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        local = np.eye(4)
        local[:3, :3] = rotation * np.asarray(row['scale'])
        local[:3, 3] = row['translation']
        worlds.append(worlds[row['parent']] @ local if row['parent'] >= 0 else local)
    return worlds


def root_shift(part_rows, body_indices, body_worlds, tolerance=1e-3):
    """The translation that takes an attach part's bind pose onto the body's, or None when the joints disagree.

    A part's skeleton is the body's joints with the root moved to the origin: every joint has the body joint's
    rotation and scale and the same offset from it, and the part's vertices are relative to that moved root."""
    shifts = []
    for world, index in zip(bind_worlds(part_rows), body_indices):
        if np.abs(world[:3, :3] - body_worlds[index][:3, :3]).max() > tolerance:
            return None
        shifts.append(body_worlds[index][:3, 3] - world[:3, 3])
    if not shifts or any(np.abs(shift - shifts[0]).max() > tolerance for shift in shifts):
        return None
    return shifts[0]


def selected_meshes(decoded, lod, look):
    from nwo.formats.native_model import _selected_meshes
    chosen = set(_selected_meshes(decoded['looks'], decoded['mesh_record_count'], lod, look))
    return [m for m in decoded['meshes'] if m['index'] in chosen]


# The shape key that keeps a mesh's stored vertices when a decoder moved them (Rift Apart vertex programs at rest,
# nwo.games.rcra.vertex.STORED_SHAPE): a glTF morph target at weight 0, a shape key in Blender.
STORED_SHAPE = 'stored vertices'
VERTEX_SECONDS, VERTEX_FPS = 4.0, 15   # the shader time an export bakes (ExportOptions.vertex_animation)
VERTEX_ANIMATION = 'Vertex animation'   # its glTF animation


def vertex_frames(motion, parts, seconds=VERTEX_SECONDS, fps=VERTEX_FPS):
    """Bake ``motion`` (``rcra.vertex.VertexMotion``) at shader times 0, 1/fps, ... ``seconds``: each part whose mesh
    moves gets ``arrays['frames']`` [(position deltas, normal deltas or None)] from its exported vertices. Returns the
    times baked ([] when no part moves)."""
    times = np.arange(int(round(seconds * fps)) + 1, dtype=np.float64) / fps
    moving = {p.get('mesh_index'): p for p in parts if p.get('mesh_index') in set(motion.meshes)}
    if not moving:
        return []
    for part in moving.values():
        part['arrays']['frames'] = []
    for t in times:
        found = motion.at(float(t))
        for index, part in moving.items():
            a = part['arrays']
            positions, normals = found.get(index, (None, None))
            if positions is None or len(positions) != len(a['positions']):   # a time that does not evaluate: rest
                a['frames'].append((np.zeros_like(a['positions']), None))
                continue
            delta_n = None if a['normals'] is None else (_unit_normals(np.asarray(normals, np.float32)) - a['normals'])
            a['frames'].append((np.asarray(positions, np.float32) - a['positions'], delta_n))
    return [float(t) for t in times]
# Rift Apart's decoded vertex tangent (native axes) and its TEXCOORD3.w as one glTF custom attribute (xyz, w): Blender's
# importer keeps it as a point attribute of this name (nwo.export.rcra_blender.TANGENT_ATTRIBUTE). One attribute, since
# the importer pairs two custom attributes of a primitive with each other's data when its set order differs.
TANGENT_ATTRIBUTE = '_RCRA_TANGENT'


def _unit_normals(normals):
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.where(length > 1e-8, normals / np.maximum(length, 1e-8), np.array([0, 1, 0], np.float32))


def mesh_arrays(mesh, joint_count):
    """Native positions/normals, game UVs (top row first), triangles, skin sets (or None), the stored vertices and
    normals of a mesh the decoder moved (``stored``, or None), and a Rift Apart mesh's tangent and TEXCOORD3.w
    (``tangents``, ``tangent_w``, or None)."""
    positions = np.asarray(mesh['positions'], np.float32).reshape(-1, 3)
    normals = np.asarray(mesh['normals'], np.float32).reshape(-1, 3) if len(mesh.get('normals') if mesh.get('normals') is not None else ()) else None
    if normals is not None:
        normals = _unit_normals(normals)
    stored = None
    if mesh.get('stored_positions') is not None:
        stored = dict(positions=np.asarray(mesh['stored_positions'], np.float32).reshape(-1, 3),
                      normals=_unit_normals(np.asarray(mesh['stored_normals'], np.float32).reshape(-1, 3)))
    uvs = []
    for key in ('uv0', 'uv1'):
        uv = mesh.get(key)
        if uv and len(uv) == len(positions):
            uv = np.asarray(uv, np.float32).reshape(-1, 2)
            uvs.append(np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1))   # decoder flipped V for Blender
    triangles = np.asarray(mesh['triangles'], np.uint32).reshape(-1, 3)
    tangents = tangent_w = None
    if mesh.get('tangent_w') is not None and mesh.get('tangents') is not None and len(mesh['tangent_w']) == len(positions):
        tangents = np.asarray(mesh['tangents'], np.float32).reshape(-1, 3)
        tangent_w = np.asarray(mesh['tangent_w'], np.float32).reshape(-1)
    skin = None
    if joint_count and mesh.get('weights'):
        joints = np.zeros((len(positions), 8), np.uint16)
        weights = np.zeros((len(positions), 8), np.float32)
        for i, influences in enumerate(mesh['weights']):
            for slot, (joint, weight) in enumerate((influences or [])[:8]):
                if 0 <= joint < joint_count:
                    joints[i, slot], weights[i, slot] = joint, weight
        total = weights.sum(axis=1, keepdims=True)
        unweighted = int((total[:, 0] <= 0).sum())
        weights = np.where(total > 0, weights / np.maximum(total, 1e-12), 0.0)
        weights[total[:, 0] <= 0, 0] = 1.0   # unweighted vertices follow joint 0 (the root)
        sets = 2 if np.any(weights[:, 4:] > 0) else 1
        skin = dict(joints=joints[:, :4 * sets], weights=weights[:, :4 * sets].astype(np.float32), sets=sets, unweighted=unweighted)
    morphs = {number: (np.asarray(vertices, np.int64), np.asarray(deltas, np.float32), np.asarray(normal_deltas, np.float32))
              for number, vertices, deltas, normal_deltas in mesh.get('morphs') or ()}
    return dict(positions=positions, normals=normals, uvs=uvs, triangles=triangles, skin=skin, stored=stored,
                tangents=tangents, tangent_w=tangent_w, morphs=morphs)


def _quat_from_matrix(m):
    """Unit xyzw quaternions of rotation matrices (..., 3, 3); same branch rule as the classic trace method."""
    m = np.asarray(m, float)
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    tiny = 1e-300

    def root(v):
        return np.sqrt(np.maximum(v, tiny)) * 2
    t = m00 + m11 + m22
    s0 = root(t + 1.0)
    s1 = root(1.0 + m00 - m11 - m22)
    s2 = root(1.0 + m11 - m00 - m22)
    s3 = root(1.0 + m22 - m00 - m11)
    q0 = np.stack([(m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0, 0.25 * s0], axis=-1)
    q1 = np.stack([0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1, (m21 - m12) / s1], axis=-1)
    q2 = np.stack([(m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2, (m02 - m20) / s2], axis=-1)
    q3 = np.stack([(m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3, (m10 - m01) / s3], axis=-1)
    c0 = t > 0
    c1 = ~c0 & (m00 > m11) & (m00 > m22)
    c2 = ~c0 & ~c1 & (m11 > m22)
    q = np.where(c0[..., None], q0, np.where(c1[..., None], q1, np.where(c2[..., None], q2, q3)))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def decompose_batch(matrix):
    """(translation, rotation xyzw, scale) of affine 4x4 matrices (..., 4, 4); shear is dropped."""
    matrix = np.asarray(matrix, float)
    t = matrix[..., :3, 3].copy()
    basis = matrix[..., :3, :3]
    scale = np.linalg.norm(basis, axis=-2)
    scale[..., 0] = np.where(np.linalg.det(basis) < 0, -scale[..., 0], scale[..., 0])
    rotation = basis / np.where(np.abs(scale) > 1e-12, scale, 1.0)[..., None, :]
    u, _s, vt = np.linalg.svd(rotation)   # nearest pure rotation
    rotation = u @ vt
    flip = np.linalg.det(rotation) < 0
    if flip.any():
        u[..., :, -1] = np.where(flip[..., None], -u[..., :, -1], u[..., :, -1])
        rotation = u @ vt
    return t, _quat_from_matrix(rotation), scale


def decompose(matrix):
    """(translation, rotation xyzw, scale) of one affine 4x4; shear is dropped."""
    return decompose_batch(matrix)


def _hold_nearest(values, bad):
    """``values`` (F, J, k) with each ``bad`` (F, J) entry replaced by the same joint's previous good frame (the next
    one before the first good frame); a joint with no good frame keeps its values."""
    frames = np.arange(len(bad))[:, None]
    previous = np.maximum.accumulate(np.where(bad, -1, frames), axis=0)
    following = np.minimum.accumulate(np.where(bad, len(bad), frames)[::-1], axis=0)[::-1]
    source = np.where(previous >= 0, previous, np.where(following < len(bad), following, frames))
    return np.take_along_axis(values, source[..., None], axis=0)


def local_trs_batch(rows, worlds):
    """Local TRS arrays (T, R, S) of shape (F, J, 3/4/3) so the hierarchy reproduces ``worlds`` (F, J, 4, 4).

    A clip can scale a joint to zero to hide it (and everything under it) for some frames. The zero scale is kept,
    so the hierarchy still hides that subtree; the rotation of a zero-scale joint and the local TRS of a joint
    under a zero-scale parent are not defined by ``worlds``, and hold the joint's nearest defined frame instead."""
    worlds = np.asarray(worlds, float)
    parent = np.array([r['parent'] for r in rows], np.intp)
    local = worlds.copy()
    child = np.nonzero(parent >= 0)[0]
    orphan = np.zeros(worlds.shape[:2], bool)   # (F, J): the parent's world has no inverse
    if len(child):
        unique, inverse = np.unique(parent[child], return_inverse=True)
        parents = worlds[:, unique]
        singular = np.abs(np.linalg.det(parents[..., :3, :3])) < 1e-12
        if singular.any():
            parents = np.where(singular[..., None, None], np.eye(4), parents)
            orphan[:, child] = singular[:, inverse]
        inv = np.linalg.inv(parents)
        local[:, child] = inv[:, inverse] @ worlds[:, child]
    t, r, s = decompose_batch(local)
    unoriented = orphan | np.all(np.abs(s) <= 1e-12, axis=-1)
    if unoriented.any():
        r = _hold_nearest(r, unoriented)
        if orphan.any():
            t, s = _hold_nearest(t, orphan), _hold_nearest(s, orphan)
    return t, r, s


def local_trs(rows, world):
    """Per-joint local TRS so that the glTF hierarchy reproduces ``world`` (any inheritance rule)."""
    t, r, s = local_trs_batch(rows, np.asarray(world, float)[None])
    return [(t[0, j], r[0, j], s[0, j]) for j in range(len(rows))]


class Tracks:
    """Local TRS of every frame: translation (F, J, 3), rotation xyzw (F, J, 4), scale (F, J, 3); len = frames."""

    def __init__(self, translation, rotation, scale):
        self.translation, self.rotation, self.scale = translation, rotation, scale

    def __len__(self):
        return len(self.translation)


_TRACK_MEMO = weakref.WeakKeyDictionary()   # AnimationLibrary -> {(clip index, root motion): (kind, fps, Tracks)}


def clip_tracks(library, index, root_motion=False, base=None):
    """(kind, fps, Tracks) for one clip, or raises ValueError when unplayable. Memoised per library, clip,
    ``root_motion`` and ``base`` (the clip an additive layer plays over; ``AnimationLibrary.clip_frames``)."""
    memo = _TRACK_MEMO.setdefault(library, {})
    key = (index, bool(root_motion), base)
    if key not in memo:
        memo[key] = _clip_tracks(library, index, root_motion, base)
    return memo[key]


def _clip_tracks(library, index, root_motion=False, base=None):
    from nwo.formats.clips import worlds_batch
    if base is not None:
        clip, kind, frames = library.clip_frames(index, root_motion, base)
    else:
        clip, kind, frames = library.clip_frames(index, root_motion) if root_motion else library.clip_frames(index)
    fps = float(clip['metadata'].get('sample_rate_fps') or 30.0)
    tracks = Tracks(*local_trs_batch(library.rows, worlds_batch(library.rows, frames, library.inheritance)))
    return kind, fps, tracks


# ---------------------------------------------------------------- glTF writer
class GLTF:
    def __init__(self, generator):
        self.doc = dict(asset=dict(version='2.0', generator=generator), scene=0, scenes=[dict(nodes=[])],
                        nodes=[], meshes=[], materials=[], accessors=[], bufferViews=[], buffers=[])
        self.blob = bytearray()
        self.images = []   # (name, png bytes)
        self.locator_note = ''   # set by ModelExporter._build_gltf when it adds locator nodes

    def _view(self, data, target=None):
        while len(self.blob) % 4:
            self.blob.append(0)
        view = dict(buffer=0, byteOffset=len(self.blob), byteLength=len(data))
        if target:
            view['target'] = target
        self.blob += data
        self.doc['bufferViews'].append(view)
        return len(self.doc['bufferViews']) - 1

    def accessor(self, array, kind, target=None, minmax=False, normalized=False):
        array = np.ascontiguousarray(array)
        component = {np.dtype(np.float32): 5126, np.dtype(np.uint32): 5125, np.dtype(np.uint16): 5123,
                     np.dtype(np.uint8): 5121}[array.dtype]
        acc = dict(bufferView=self._view(array.tobytes(), target), componentType=component,
                   count=int(array.shape[0]), type=kind)
        if normalized:
            acc['normalized'] = True
        if minmax:
            flat = array.reshape(array.shape[0], -1)
            acc['min'] = [float(v) for v in flat.min(axis=0)]
            acc['max'] = [float(v) for v in flat.max(axis=0)]
        self.doc['accessors'].append(acc)
        return len(self.doc['accessors']) - 1

    def sparse(self, count, indices, values, kind, minmax=False):
        """Accessor of ``count`` elements that are zero except ``values`` at ``indices`` (glTF sparse storage); without
        indices, an all-zero accessor with no buffer view."""
        values = np.ascontiguousarray(values, np.float32)
        acc = dict(componentType=5126, count=int(count), type=kind)
        if len(indices):
            order = np.argsort(indices, kind='stable')
            indices, values = np.asarray(indices)[order], values[order]
            small = int(indices.max()) <= 65535
            acc['sparse'] = dict(count=int(len(indices)),
                                 indices=dict(bufferView=self._view(indices.astype(np.uint16 if small else np.uint32).tobytes()),
                                              componentType=5123 if small else 5125),
                                 values=dict(bufferView=self._view(values.tobytes())))
        if minmax:
            width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[kind]
            flat = values.reshape(len(values), width)
            if len(indices) < count:
                flat = np.vstack([flat, np.zeros((1, flat.shape[1]), np.float32)])
            acc['min'] = [float(v) for v in flat.min(axis=0)]
            acc['max'] = [float(v) for v in flat.max(axis=0)]
        self.doc['accessors'].append(acc)
        return len(self.doc['accessors']) - 1

    def image(self, name, png):
        self.images.append((name, png))
        doc = self.doc
        doc.setdefault('samplers', [dict(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)])
        doc.setdefault('images', []).append(dict(name=name))
        doc.setdefault('textures', []).append(dict(sampler=0, source=len(doc['images']) - 1))
        return len(doc['textures']) - 1

    def node(self, **values):
        self.doc['nodes'].append(values)
        return len(self.doc['nodes']) - 1

    def finish(self, embed, texture_dir=None):
        """Images go into the binary buffer (embed) or are referenced as files in ``texture_dir``."""
        doc = json.loads(json.dumps(self.doc))
        for number, (name, png) in enumerate(self.images):
            if embed:
                doc['images'][number] = dict(name=name, mimeType='image/png', bufferView=None)
        blob = bytearray(self.blob)
        if embed:
            for number, (name, png) in enumerate(self.images):
                while len(blob) % 4:
                    blob.append(0)
                doc['bufferViews'].append(dict(buffer=0, byteOffset=len(blob), byteLength=len(png)))
                blob += png
                doc['images'][number]['bufferView'] = len(doc['bufferViews']) - 1
        else:
            for number, (name, _png) in enumerate(self.images):
                doc['images'][number] = dict(name=name, uri=f'{texture_dir}/{name}.png')
        for key in ('materials', 'meshes', 'skins', 'animations'):
            if key in doc and not doc[key]:
                del doc[key]
        doc['buffers'] = [dict(byteLength=len(blob))]
        return doc, bytes(blob)


MODEL_TAG = 'rc2016_model'   # glTF node extras: the model a node belongs to, read back by the Blender import (model_collections)


def tag_model(document, model):
    """Tag every node that has no model yet, so an FBX or USD built from the file groups them per model."""
    for node in document.get('nodes', []):
        node.setdefault('extras', {}).setdefault(MODEL_TAG, model)


def write_glb(path, doc, blob):
    text = json.dumps(doc, separators=(',', ':')).encode('utf-8')
    text += b' ' * (-len(text) % 4)
    blob = blob + b'\0' * (-len(blob) % 4)
    with open(path, 'wb') as stream:
        stream.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(text) + 8 + len(blob)))
        stream.write(struct.pack('<II', len(text), 0x4E4F534A) + text)
        stream.write(struct.pack('<II', len(blob), 0x004E4942) + blob)


def write_gltf(path, doc, blob):
    path = Path(path)
    bin_name = path.with_suffix('.bin').name
    doc = dict(doc, buffers=[dict(byteLength=len(blob), uri=bin_name)])
    path.with_suffix('.bin').write_bytes(blob)
    path.write_text(json.dumps(doc, indent=1), encoding='utf-8')


# ---------------------------------------------------------------- OBJ writer
def write_obj(path, stem, parts, materials):
    lines = [f'# {stem}: native rest pose, Y-up, game units', f'mtllib {stem}.mtl']
    base = 1
    for part in parts:
        a = part['arrays']
        lines.append(f"o {part['name']}")
        lines += [f'v {x:.6f} {y:.6f} {z:.6f}' for x, y, z in a['positions']]
        has_uv = bool(a['uvs'])
        if has_uv:
            lines += [f'vt {u:.6f} {1.0 - v:.6f}' for u, v in a['uvs'][0]]   # OBJ: v up
        has_n = a['normals'] is not None
        if has_n:
            lines += [f'vn {x:.5f} {y:.5f} {z:.5f}' for x, y, z in a['normals']]
        lines.append(f"usemtl {materials[part['material']]['obj_name']}" if part['material'] in materials else 'usemtl default')
        for tri in a['triangles'] + base:
            if has_uv and has_n:
                lines.append('f ' + ' '.join(f'{i}/{i}/{i}' for i in tri))
            elif has_n:
                lines.append('f ' + ' '.join(f'{i}//{i}' for i in tri))
            elif has_uv:
                lines.append('f ' + ' '.join(f'{i}/{i}' for i in tri))
            else:
                lines.append('f ' + ' '.join(str(i) for i in tri))
        base += len(a['positions'])
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ---------------------------------------------------------------- driver
class ModelExporter:
    """Exports models through a private SourcePreviewLoader (separate from the 3D view's)."""

    def __init__(self, source_root, output_root, index_root=None, log=lambda message: None, cancelled=lambda: False):
        from nwo.viewer.loader import SourcePreviewLoader
        # Absolute: Blender runs in its own process and resolves relative texture and target paths elsewhere.
        self.source_root, self.output_root = Path(source_root), Path(output_root).resolve()
        # Animation sets are found through a scan's reference index (output_root/reports by default).
        self.index_root = Path(index_root) if index_root else self.output_root
        self.loader = SourcePreviewLoader()
        self.texture_cache = {}
        self.map_cache = _MapCache()   # converted material maps, shared by every export of this run
        self.log, self.cancelled = log, cancelled
        self.process = None
        self.pool = None   # viewer.loader.TranslationPool for cold Blender materials, started on first use

    def close(self):
        """Stop the translation workers (the exporter can still export; a later cold model starts them again)."""
        pool, self.pool = self.pool, None
        if pool is not None:
            pool.close()

    def _blender_materials(self, game, archive, materials, folder):
        """game.blender_materials, with the translations not cached yet made first in parallel worker processes
        (TranslationPool, game.blender_translation_units) that store them in the disk cache, so the game's hook reads
        them back; a single translation, a frozen build or a failed worker translates here."""
        self.log('  translating game materials for Blender')
        phases = game.blender_translation_units(archive, materials, folder)
        units = sum(len(phase) for phase in phases)
        if units > 1:
            if self.pool is None:
                from nwo.viewer.loader import TranslationPool, translation_workers
                workers = translation_workers()
                self.pool = TranslationPool(self.source_root, workers) if workers else None
            if self.pool is not None:
                self.log(f'  {units} translations on {min(units, self.pool.workers)} worker processes')
                for phase in phases:
                    self.pool.run(phase)
                    self._check()
        return game.blender_materials(archive, materials, folder)

    def _check(self):
        if self.cancelled():
            raise Cancelled()

    def model_folder(self, model_id, source_path):
        stem = safe_name(Path(source_path).stem)
        folder = self.output_root / stem
        info = folder / 'export.json'
        if info.is_file():
            try:
                if json.loads(info.read_text(encoding='utf-8')).get('model_id') != model_id:
                    folder = self.output_root / f'{stem}_{model_id[:8]}'
            except (OSError, ValueError):
                pass
        return stem, folder

    def animation_index(self):
        """The scan's reference index root that links models to animation sets, or None when it is missing or
        stale (its shared asset table gone or changed), in which case no animation set can be found."""
        if not (self.index_root / 'reports' / 'asset_references.sqlite').is_file():
            return None
        from nwo.formats.index import open_asset_index
        try:
            open_asset_index(self.index_root).close()
        except Exception:
            return None
        return self.index_root

    def clip_labels(self, model_id, source_path, options: ExportOptions, notes=False, seconds=False):
        """Animation clip labels that ``export`` offers for this model with these options (the same load, without
        writing anything). Empty when the export writes no animations. With ``notes``, (labels, clip notes):
        ``AnimationLibrary.clip_notes``, which marks copies, additive layers and clips that do not play. With
        ``seconds`` as well, (labels, notes, lengths): ``AnimationLibrary.clip_seconds``."""
        empty = ([], [], []) if notes and seconds else ([], []) if notes else []
        gltf_based = any(f in options.formats for f in ('glb', 'gltf', 'fbx', 'usd', 'blend'))   # Blender imports the GLB
        if options.animations == 'none' or not (options.skeleton and gltf_based):
            return empty
        index_root = self.animation_index()
        borrowed = options.borrowed and index_root is not None
        self.loader.load(self.source_root, model_id, source_path, options.lod, options.look,
                         index_root, False, purpose='export', borrowed=borrowed)
        library = self.loader.animation
        if library is not None and library.rows and options.cinematic:
            library = library.with_cinematic(self.loader.archive, self.loader.current['_archive_path'])   # as export() does
        if library is not None and library.rows and options.performances:
            library = library.with_performances()   # as export() does
        if library is None or not library.rows:
            return empty
        if notes and seconds:
            return library.labels(), library.clip_notes(), library.clip_seconds()
        return (library.labels(), library.clip_notes()) if notes else library.labels()

    def export(self, model_id, source_path, options: ExportOptions, cinematic_sources=None):
        """Export one model; returns a summary dict. Raises on failure, Cancelled on cancel.

        With ``options.cinematic`` or ``cinematic_only`` the model's real-time cutscene shots
        are an animation source (``cinematic_sources``: an ``animation_sources.cinematic_sources`` result the
        caller already has, otherwise scanned here). The summary lists every offered ``clips`` entry
        (name, kind, fps, frames, source, asset, offset) and ``skipped_clips`` with reasons.
        """
        stem, folder = self.model_folder(model_id, source_path)
        info_path = folder / 'export.json'
        if options.skip_existing and info_path.is_file():
            try:
                old = json.loads(info_path.read_text(encoding='utf-8'))
                if old.get('model_id') == model_id and old.get('options') == options.key() and \
                        all((folder / name).is_file() for name in old.get('files', [])):
                    return dict(status='skipped', folder=str(folder), message='already exported with these options')
            except (OSError, ValueError):
                pass
        needs_blender = [f for f in options.formats if FORMATS[f][2]]
        if needs_blender and not (options.blender and Path(options.blender).is_file()):
            raise ValueError(f"{', '.join(FORMATS[f][1] for f in needs_blender)} needs Blender: set its path in Options")
        self.log(f'Decoding {source_path}')
        index_root = self.animation_index()
        gltf_based = any(f in options.formats for f in ('glb', 'gltf', 'fbx', 'usd', 'blend'))   # Blender imports the GLB
        load_animation = options.animations != 'none' and options.skeleton and gltf_based   # OBJ is the static rest pose only
        cutscenes = (options.cinematic or options.cinematic_only) and load_animation
        animset_root = index_root if load_animation and not options.cinematic_only else None
        borrowed = options.borrowed and animset_root is not None
        self.loader.load(self.source_root, model_id, source_path, options.lod, options.look,
                         animset_root, False, purpose='export', borrowed=borrowed)
        decoded, library = self.loader.current, self.loader.animation if load_animation else None
        archive = self.loader.archive
        self._check()
        notes = []
        own_clips = self.loader.game.has_animation_library()   # Rift Apart finds its clips without a scan
        if load_animation and decoded['joint_count'] and animset_root is None and not options.cinematic_only and not own_clips:
            notes.append('Animations skipped: scan the archive into this output folder first (its reference index links models to animation sets).')
        joint_count = decoded['joint_count'] if library is not None and library.rows else 0
        if options.skeleton and decoded['joint_count'] and not joint_count:
            # Skeleton still exported for rigging even without a clip library.
            model_data = self.loader._recent[self.loader.identity]['data']
            if own_clips:
                library = self.loader.game.animation_library(archive, model_data, decoded['_archive_path'], clips=False)
            else:
                from nwo.viewer.animation import AnimationLibrary
                library = AnimationLibrary(archive, model_data, [])
            joint_count = len(library.rows) if library.rows else 0
        if cutscenes and library is not None and library.rows:
            self.log('  scanning real-time cutscenes')
            library = library.with_cinematic(archive, decoded['_archive_path'], replace=options.cinematic_only,
                                             sources=cinematic_sources)
            self._check()
        if options.performances and load_animation and library is not None and library.rows:
            library = library.with_performances()
        meshes = selected_meshes(decoded, options.lod, options.look)
        # Blend shapes (Rift Apart, rcra.model.attach_shapes): morph targets named in the model's shape order.
        shapes = [shape['name'] or f"shape_{shape['hash']:08X}" for shape in decoded.get('shapes') or ()]
        if shapes and any(mesh.get('morphs') for mesh in meshes):
            notes.append(f'{len(shapes)} blend shapes as morph targets (shape keys); clips with face motion animate their weights.')
        elif decoded.get('shape_note'):
            notes.append(decoded['shape_note'])
        game = self.loader.game   # the archive's game (nwo.games): its material reads and texture export
        material_paths = {m.get('index', i): m.get('path') for i, m in enumerate(decoded['materials'])}
        casters = [m for m in meshes if game.shadow_only(material_paths.get((m.get('material') or {}).get('index')))]
        if casters and len(casters) < len(meshes):
            meshes = [m for m in meshes if not any(m is c for c in casters)]
            notes.append(f'{len(casters)} shadow-cast-only mesh(es) left out: the game never draws them (material flags bit 0x80).')
        undrawn = decoded.get('undrawn_materials') or {}   # Rift Apart: no draw is issued (rcra.material.draw_skip_note)
        skipped = [m for m in meshes if material_paths.get((m.get('material') or {}).get('index')) in undrawn]
        if skipped and len(skipped) < len(meshes):
            meshes = [m for m in meshes if not any(m is s for s in skipped)]
            names = dict.fromkeys(Path(material_paths[m['material']['index']]).stem for m in skipped)
            notes.append(f"{len(skipped)} mesh(es) left out ({', '.join(names)}): "
                         + next(undrawn[material_paths[m['material']['index']]] for m in skipped))
        shown = {m['index'] for m in meshes}
        moved = {}
        for row in decoded.get('vertex_programs') or ():   # Rift Apart vertex programs (nwo.games.rcra.vertex)
            if row.get('note') and row['mesh'] in shown:
                moved.setdefault(row['note'], []).append(Path(row['material']).stem)
        notes += [f"{', '.join(dict.fromkeys(names))}: {note}" for note, names in moved.items()]
        if not meshes:
            raise ValueError('The selected look/LOD has no geometry')
        need_gltf = any(f in options.formats for f in ('glb', 'gltf', 'fbx', 'usd', 'blend'))   # Blender imports the GLB
        parts = []
        for mesh in meshes:
            material = (mesh.get('material') or {}).get('index')
            parts.append(dict(name=f"{stem}_mesh{mesh['index']:02d}", material=material, arrays=mesh_arrays(mesh, joint_count),
                              mesh_index=mesh['index']))
        baked = []
        if options.vertex_animation and decoded.get('vertex_motion') is not None:
            baked = vertex_frames(decoded['vertex_motion'], parts)
            if baked:
                notes.append(f'Vertex programs baked over {VERTEX_SECONDS:g} s of shader time at {VERTEX_FPS} fps: one morph '
                             f'target per frame ("time <seconds>") and the animation "{VERTEX_ANIMATION}" that steps '
                             'through them. Wind, force fields and other engine inputs stay at rest.')
        unweighted = sum((p['arrays']['skin'] or {}).get('unweighted', 0) for p in parts)
        if unweighted:
            notes.append(f'{unweighted} vertices without skin weights follow the root joint.')
        by_index = {m.get('index', i): m for i, m in enumerate(decoded['materials'])}
        attached = []
        # Attach parts are the R&C 2016 bangle_set/*_attach layout, read with its decoders; Rift Apart has none.
        if options.attachments and joint_count and need_gltf and not own_clips:
            attached, attachment_notes = self._attachment_parts(archive, source_path, library, parts, by_index, options)
            notes += attachment_notes
            parts += [part for entry in attached for part in entry['parts']]

        # Materials: only slots the selected meshes use.
        used = sorted({p['material'] for p in parts if p['material'] is not None})
        obj_names, names = {}, set()
        for slot in used:
            name = safe_name(material_name(by_index.get(slot, {}), slot), 50)
            while name in names:
                name += f'_{slot}'
            names.add(name); obj_names[slot] = name
        translated = None
        if needs_blender and options.textures and tuple(options.formats) == ('blend',):
            # The .blend alone: the game's translated material replaces the glTF one, so its glTF maps are only made for
            # the materials that do not translate.
            folder.mkdir(parents=True, exist_ok=True)
            translated = self._blender_materials(game, archive, {obj_names[slot]: by_index.get(slot, {}) for slot in used},
                                                 folder)
        maps = {}
        for slot in used:
            self._check()
            material = by_index.get(slot, {})
            if translated is not None and obj_names[slot] not in translated[1]:
                maps[slot] = plain_maps(material, slot, 'translated for Blender; no glTF maps')
            else:
                self.log(f"  material {slot}: {material.get('name') or material.get('path') or '(none)'}")
                maps[slot] = game.material_maps(archive, material, self.texture_cache, self.map_cache) if options.textures \
                    else plain_maps(material, slot, 'textures not exported')
            maps[slot]['obj_name'] = obj_names[slot]

        # Animations.
        clips_out, skipped = [], []
        if library is not None and library.rows and options.animations != 'none':
            labels = library.labels()
            wanted = range(len(labels)) if options.animations == 'all' else [i for i, l in enumerate(labels) if l in options.clips]
            if options.animations == 'clips':
                missing = [c for c in options.clips if c not in labels]
                if missing:
                    notes.append('Requested clips not found: ' + ', '.join(missing))
            layered = 0
            for i in wanted:
                self._check()
                try:
                    partners = library.layer_partners(i) if options.layer_bases and hasattr(library, 'layer_partners') else []
                    base = partners[0] if partners else None
                    kind, fps, tracks = clip_tracks(library, i, options.root_motion, base)
                    name = labels[i]
                    if base is not None and kind == 'relative/additive-style layer':
                        weight = library.layer_weight(i, base) if hasattr(library, 'layer_weight') else 1.0
                        at = '' if weight == 1.0 else f' at {weight:g}'
                        name, layered = f'{labels[i]} (over {library.clips[base]["name"]}{at})', layered + 1
                    morph_weights = library.morph_weights(i, len(tracks)) if shapes and hasattr(library, 'morph_weights') else None
                    clips_out.append(dict(name=name, kind=kind, fps=fps, tracks=tracks, morph_weights=morph_weights,
                                          source=library.clips[i].get('source', 'animset'),
                                          asset=library.clips[i]['animset'], offset=library.clips[i]['offset']))
                except Exception as error:   # one bad clip must not stop the export
                    skipped.append(dict(name=labels[i], reason=str(error), not_playable=isinstance(error, NotPlayable),
                                        source=library.clips[i].get('source', 'animset'),
                                        asset=library.clips[i]['animset'], offset=library.clips[i]['offset']))
            self.log(f'  {len(clips_out)} clip(s) decoded' + (f', {len(skipped)} skipped' if skipped else ''))
            if layered:
                notes.append(f'{layered} additive layer(s) play over the first clip an animation graph pair node adds '
                             'them to, at its weight (named "over ..."); the game picks among such clips at run time. The '
                             'others play on the rest pose.')
            if options.root_motion and clips_out:
                notes.append('Clips move by their root motion since their first frame, on the joints without a parent; '
                             'additive layers and clips without root motion play in place.')

        folder.mkdir(parents=True, exist_ok=True)
        written = []
        file_stem = stem + options.file_suffix
        doc = blob = None
        tex_dir = folder / 'textures'
        need_gltf = any(f in options.formats for f in ('glb', 'gltf', 'fbx', 'usd', 'blend'))   # Blender imports the GLB
        texture_files = {}
        if options.textures and any(f in options.formats for f in ('gltf', 'obj')):
            tex_dir.mkdir(exist_ok=True)
        if need_gltf:
            locators = decoded.get('locators') or []   # Rift Apart attach points (rcra.model._locators)
            gltf = self._build_gltf(stem, parts, maps, library if joint_count else None, clips_out, shapes, baked,
                                    locators=locators, joint_names=[j['name'] for j in decoded.get('joints') or []])
            if gltf.locator_note:
                notes.append(gltf.locator_note)
            doc, blob = gltf.finish(embed=True)
            if 'gltf' in options.formats:
                file_doc, file_blob = gltf.finish(embed=False, texture_dir='textures')
                for name, png in gltf.images:
                    (tex_dir / f'{name}.png').write_bytes(png); texture_files[name] = png
                write_gltf(folder / f'{file_stem}.gltf', file_doc, file_blob)
                written += [f'{file_stem}.gltf', f'{file_stem}.bin']
            if 'glb' in options.formats:
                write_glb(folder / f'{file_stem}.glb', doc, blob); written.append(f'{file_stem}.glb')
        if 'obj' in options.formats:
            self._write_obj_materials(folder, stem, maps, tex_dir if options.textures else None, texture_files)
            write_obj(folder / f'{stem}.obj', stem, parts, maps)
            written += [f'{stem}.obj', f'{stem}.mtl']
            if clips_out or joint_count:
                notes.append('OBJ holds the static rest pose only (no skeleton or animation).')
        if needs_blender:
            self._check()
            if translated is None and options.textures:
                translated = self._blender_materials(game, archive,
                                                     {m['obj_name']: by_index.get(slot, {}) for slot, m in maps.items()}, folder)
            if translated is not None:
                materials_json, untranslated = translated
                kept = ', '.join(f'{name} ({why})' for name, why in sorted(untranslated.items()))
                if untranslated:
                    notes.append(f'{len(untranslated)} material(s) keep the glTF material in Blender: {kept}')
                written.append(materials_json.name)
            atmosphere = None
            if options.scene and options.atmosphere and 'blend' in options.formats:
                atmosphere = self._atmosphere_bake(archive, options.atmosphere)
                notes.append(atmosphere['limitations'])
            written += self._blender_convert(folder, stem, doc, blob, needs_blender, options.blender,
                                             materials_json=translated[0] if translated else None,
                                             scene=options.scene, atmosphere=atmosphere,
                                             clips=[dict(name=c['name'], kind=c['kind'], fps=c['fps'], frames=len(c['tracks']))
                                                    for c in clips_out])
        if options.source_textures:
            paths = sorted({p for m in maps.values() for p in m['sources'].values()})
            rows = game.export_textures(archive, paths, tex_dir / 'game_dds', {'dds'})
            for row in rows:
                if row.get('error'):
                    notes.append(f"DDS export failed for {row['path']}: {row['error']}")
        for path in tex_dir.rglob('*') if tex_dir.is_dir() else ():
            if path.is_file():
                written.append(path.relative_to(folder).as_posix())
        info = dict(schema='rcps4-portable-export-v1', model_id=model_id, source_path=source_path,
                    source_packed_sha256=decoded['packed_sha256'], options=options.key(), files=sorted(set(written)),
                    look=options.look, lod=options.lod, meshes=[m['index'] for m in meshes],
                    joints=joint_count, attached_parts=[dict(path=e['path'], meshes=[m['index'] for m in e['meshes']]) for e in attached],
                    clips=[dict(name=c['name'], kind=c['kind'], fps=c['fps'], frames=len(c['tracks']),
                                                    **({} if c['source'] == 'animset' else dict(source=c['source']))) for c in clips_out],
                    skipped_clips=skipped,
                    materials={str(slot): dict(name=m['name'], textures=m['sources'], notes=m['notes']) for slot, m in maps.items()},
                    notes=notes, coordinates='native Y-up, right-handed, game units; glTF UVs top-left origin',
                    limitations='Portable approximation of the game material (metal/roughness conversion of the decoded specular/gloss); '
                                'terms the format cannot carry are left out, as each material\'s notes say. Not game-equivalent.')
        message = f"{len(info['files'])} files, {len(clips_out)} clip(s)" + (f", {len(skipped)} clip(s) skipped" if skipped else '')
        if tuple(options.formats) == ('blend',):
            # The .blend is the only deliverable and packs its own images, so the model folder (its export.json,
            # material list and any textures) goes; the .blend stays in the output root.
            shutil.rmtree(folder)
            folder = folder.parent / f'{folder.name}.blend'
        else:
            info_path.write_text(json.dumps(info, indent=1), encoding='utf-8')
        return dict(status='exported', folder=str(folder), message=message, notes=notes,
                    clips=[{k: v for k, v in c.items() if k != 'tracks'} | dict(frames=len(c['tracks'])) for c in clips_out],
                    skipped_clips=skipped)

    # -- attached parts
    ATTACH_FOLDER = 'bangle_set'   # <model folder>/bangle_set/<part>/<part>.model, named *_attach
    ATTACH_MARGIN = 0.05           # a part's bounds may exceed the body's by this share of the body's diagonal

    def _attachment_parts(self, archive, source_path, library, body_parts, body_materials, options):
        """The model's attach parts as extra skinned parts of its own rig, and notes for the parts left out.

        A part (``<model folder>/bangle_set/<name>_attach/<name>_attach.model``) carries a small skeleton of the
        body's own joints with the root moved to the origin, so on its own it cannot follow the body's pose, and
        its vertices are relative to that root. Adding the offset from the part's joints to the body's puts them in
        the body's bind space (checked below against the body's bounds), so skinning them to the body's joints with
        the body's inverse binds puts them where the part sits on the body and moves them with it.
        Returns ([{path, meshes, parts}], notes); every part that does not meet the checks is skipped with a note.
        """
        from nwo.formats.animation_transforms import model_skeleton
        folder = source_path.rsplit('/', 1)[0] + f'/{self.ATTACH_FOLDER}/'
        paths = sorted({e['path'] for e in archive.entries.values()
                        if e['type'] == 'model' and e['path'].startswith(folder) and 'attach' in Path(e['path']).stem.split('_')})
        if not paths:
            return [], []
        body_joint = {row['hash']: index for index, row in enumerate(library.rows)}
        body_worlds = bind_worlds(library.rows)
        lows = np.min([p['arrays']['positions'].min(axis=0) for p in body_parts], axis=0)
        highs = np.max([p['arrays']['positions'].max(axis=0) for p in body_parts], axis=0)
        margin = float(np.linalg.norm(highs - lows)) * self.ATTACH_MARGIN
        slot_of_material = {m.get('path'): slot for slot, m in body_materials.items() if m.get('path')}
        found, notes = [], []
        for number, path in enumerate(paths, start=1):
            self._check()
            name = safe_name(Path(path).stem)
            ids = archive.resolve(path)
            if len(ids) != 1:
                notes.append(f'Attach part {name} skipped: {len(ids)} archive entries for its path.')
                continue
            try:
                part_rows = model_skeleton(archive.decode(ids[0]))
                joints = [row['hash'] for row in part_rows]
                if not joints or any(h not in body_joint for h in joints):
                    notes.append(f'Attach part {name} skipped: its skeleton has joints that are not in the body rig.')
                    continue
                shift = root_shift(part_rows, [body_joint[h] for h in joints], body_worlds)
                if shift is None:
                    notes.append(f"Attach part {name} skipped: its skeleton is not the body's with a moved root.")
                    continue
                cached = self.loader._decode(archive, ids[0], path, 'export', self.texture_cache)['decoded']
                meshes = selected_meshes(cached, options.lod, options.look)
                remap = np.array([body_joint[h] for h in joints], np.uint16)
                part_materials = {m.get('index', i): m for i, m in enumerate(cached['materials'])}
                made = []
                for mesh in meshes:
                    arrays = mesh_arrays(mesh, len(joints))
                    if arrays['skin'] is None:
                        raise ValueError('it has no skin weights')
                    arrays['positions'] = (arrays['positions'] + shift).astype(np.float32)
                    if (arrays['positions'].min(axis=0) < lows - margin).any() or (arrays['positions'].max(axis=0) > highs + margin).any():
                        raise ValueError("its vertices lie outside the body's bind space")
                    arrays['skin']['joints'] = remap[arrays['skin']['joints']]
                    slot = (mesh.get('material') or {}).get('index')
                    material = part_materials.get(slot)
                    if material is not None:   # a material the body already uses keeps the body's slot, others get a slot of their own
                        slot = slot_of_material.get(material.get('path'), number * 1000 + slot)
                        body_materials.setdefault(slot, material)
                    made.append(dict(name=f"{name}_mesh{mesh['index']:02d}", material=slot, arrays=arrays, group=name))
            except Exception as error:   # one bad part must not stop the model's export
                notes.append(f'Attach part {name} skipped: {error}.')
                continue
            if made:
                found.append(dict(path=path, meshes=meshes, parts=made))
        if found:
            notes.append(f"{len(found)} attach part(s) skinned to the body rig: {', '.join(Path(e['path']).stem for e in found)}.")
        return found, notes

    # -- glTF assembly
    def _build_gltf(self, stem, parts, maps, library, clips, shapes=(), baked=(), locators=(), joint_names=()):
        """The glTF document. ``locators`` (Rift Apart): empty nodes named after each locator, children of their joint
        (matched by name in the skeleton) with the stored local transform, or of the model for one without a joint."""
        g = GLTF('R&C PS4 asset browser export')
        doc = g.doc
        material_ids = {}
        for slot, m in maps.items():
            pbr = dict(baseColorFactor=list(m.get('base_factor', [1.0, 1.0, 1.0])) + [m['alpha']],
                       metallicFactor=1.0 if m['mr'] is not None else 0.0, roughnessFactor=1.0)
            entry = dict(name=m['obj_name'], pbrMetallicRoughness=pbr)
            if m['base'] is not None:
                pbr['baseColorTexture'] = dict(index=g.image(f"{m['obj_name']}_basecolor", cached_png(m, 'base', m['base'])))
            if m['mr'] is not None:
                pbr['metallicRoughnessTexture'] = dict(index=g.image(f"{m['obj_name']}_metal_rough", cached_png(m, 'mr', m['mr'])))
                if m.get('occlusion'):   # occlusion in the red channel of the same image, as glTF reads it
                    entry['occlusionTexture'] = dict(pbr['metallicRoughnessTexture'])
            if m['normal'] is not None:
                entry['normalTexture'] = dict(index=g.image(f"{m['obj_name']}_normal", cached_png(m, 'normal', m['normal'])))
            if m['emissive'] is not None:
                entry['emissiveTexture'] = dict(index=g.image(f"{m['obj_name']}_emissive", cached_png(m, 'emissive', m['emissive'])))
                entry['emissiveFactor'] = [1.0, 1.0, 1.0]
                if m['emissive_strength'] != 1.0:
                    entry['extensions'] = dict(KHR_materials_emissive_strength=dict(emissiveStrength=m['emissive_strength']))
                    doc.setdefault('extensionsUsed', []).append('KHR_materials_emissive_strength')
            elif m.get('emissive_factor'):
                entry['emissiveFactor'] = [float(x) for x in m['emissive_factor']]
            if m['alpha_mode'] != 'OPAQUE':
                entry['alphaMode'] = m['alpha_mode']
            if m['double_sided']:
                entry['doubleSided'] = True
            doc['materials'].append(entry)
            material_ids[slot] = len(doc['materials']) - 1
        if 'extensionsUsed' in doc:
            doc['extensionsUsed'] = sorted(set(doc['extensionsUsed']))

        def primitive(part, skinned, targets=()):
            a = part['arrays']
            attributes = dict(POSITION=g.accessor(a['positions'], 'VEC3', 34962, minmax=True))
            if a['normals'] is not None:
                attributes['NORMAL'] = g.accessor(a['normals'], 'VEC3', 34962)
            for n, uv in enumerate(a['uvs']):
                attributes[f'TEXCOORD_{n}'] = g.accessor(uv.astype(np.float32), 'VEC2', 34962)
            if a.get('tangents') is not None:
                attributes[TANGENT_ATTRIBUTE] = g.accessor(np.column_stack((a['tangents'], a['tangent_w'])), 'VEC4', 34962)
            if skinned:
                skin = a['skin']
                for s in range(skin['sets']):
                    attributes[f'JOINTS_{s}'] = g.accessor(skin['joints'][:, 4 * s:4 * s + 4].astype(np.uint16), 'VEC4', 34962)
                    attributes[f'WEIGHTS_{s}'] = g.accessor(skin['weights'][:, 4 * s:4 * s + 4], 'VEC4', 34962)
            index = a['triangles'].reshape(-1)
            index = index.astype(np.uint16) if len(a['positions']) <= 65535 else index.astype(np.uint32)
            prim = dict(attributes=attributes, indices=g.accessor(index, 'SCALAR', 34963), mode=4)
            if part['material'] in material_ids:
                prim['material'] = material_ids[part['material']]
            if targets:   # every primitive of the mesh carries every target; unmoved ones a zero offset
                prim['targets'] = []
                for number in targets:
                    if number is None:   # the stored vertices of a mesh a vertex program moved
                        stored = a.get('stored')
                        target = dict(POSITION=g.accessor(stored['positions'] - a['positions'] if stored else np.zeros_like(a['positions']),
                                                          'VEC3', 34962, minmax=True))
                        if a['normals'] is not None:
                            target['NORMAL'] = g.accessor(stored['normals'] - a['normals'] if stored else np.zeros_like(a['normals']),
                                                          'VEC3', 34962)
                    elif isinstance(number, tuple):   # a baked vertex-program frame: its deltas, zero where still
                        frames, count = a.get('frames'), len(a['positions'])
                        if frames:
                            deltas, normal_deltas = frames[number[1]]
                            target = dict(POSITION=g.accessor(np.ascontiguousarray(deltas, np.float32), 'VEC3', 34962, minmax=True))
                            if a['normals'] is not None:
                                target['NORMAL'] = (g.accessor(np.ascontiguousarray(normal_deltas, np.float32), 'VEC3', 34962)
                                                    if normal_deltas is not None else g.sparse(count, (), np.zeros((0, 3)), 'VEC3'))
                        else:
                            target = dict(POSITION=g.sparse(count, (), np.zeros((0, 3)), 'VEC3', minmax=True))
                            if a['normals'] is not None:
                                target['NORMAL'] = g.sparse(count, (), np.zeros((0, 3)), 'VEC3')
                    else:                # a blend shape: sparse deltas at the vertices it moves
                        index, deltas, normal_deltas = a['morphs'].get(number, (np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 3))))
                        count = len(a['positions'])
                        target = dict(POSITION=g.sparse(count, index, deltas, 'VEC3', minmax=True))
                        if a['normals'] is not None:
                            target['NORMAL'] = g.sparse(count, index, normal_deltas, 'VEC3')
                    prim['targets'].append(target)
            return prim

        def mesh(name, members, skinned):
            """Append the glTF mesh of ``members``; returns its targets (None: stored vertices, n: blend shape n)."""
            targets = [None] if any(p['arrays'].get('stored') is not None for p in members) else []
            targets += sorted({n for p in members for n in p['arrays'].get('morphs') or {}})
            if any(p['arrays'].get('frames') for p in members):
                targets += [('frame', k) for k in range(len(baked))]
            entry = dict(name=name, primitives=[primitive(p, skinned, targets) for p in members])
            if targets:
                entry.update(weights=[0.0] * len(targets),
                             extras=dict(targetNames=[STORED_SHAPE if n is None else f'time {baked[n[1]]:.3f}'
                                                      if isinstance(n, tuple) else shapes[n] for n in targets]))
            doc['meshes'].append(entry)
            return targets

        root_children = []
        shaped = {}   # mesh node -> its morph targets (mesh())
        skinned_parts = [p for p in parts if p['arrays']['skin'] is not None] if library is not None else []
        rigid_parts = [p for p in parts if p not in skinned_parts]
        joint_nodes = []
        if library is not None:
            from nwo.formats.clips import worlds_batch
            rest = local_trs(library.rows, worlds_batch(library.rows, [()], library.rest_inheritance)[0])
            first = len(doc['nodes'])
            for row, (t, r, s) in zip(library.rows, rest):
                g.node(name=row['name'], translation=[float(v) for v in t], rotation=[float(v) for v in r],
                       scale=[float(v) for v in s])
            joint_nodes = list(range(first, first + len(library.rows)))
            for j, row in enumerate(library.rows):
                if row['parent'] >= 0:
                    doc['nodes'][first + row['parent']].setdefault('children', []).append(first + j)
            roots = [first + j for j, row in enumerate(library.rows) if row['parent'] < 0]
            armature = g.node(name=f'{stem}_skeleton', children=roots)
            root_children.append(armature)
            # glTF inverse binds are column-major.
            ibm = np.stack([np.asarray(m, np.float64).T for m in library.inverse]).astype(np.float32).reshape(-1, 16)
            doc['skins'] = [dict(name=f'{stem}_skin', joints=joint_nodes, skeleton=armature,
                                 inverseBindMatrices=g.accessor(ibm, 'MAT4'))]
        if skinned_parts:
            groups = {}
            for p in skinned_parts:   # the body first, then one node per attached part, so each can be hidden on its own
                groups.setdefault(p.get('group'), []).append(p)
            for group, members in groups.items():
                targets = mesh(f'{group}_skinned' if group else f'{stem}_skinned', members, True)
                root_children.append(g.node(name=group or stem, mesh=len(doc['meshes']) - 1, skin=0))
                shaped[root_children[-1]] = targets
        if rigid_parts:
            targets = mesh(f'{stem}_static' if skinned_parts else stem, rigid_parts, False)
            root_children.append(g.node(name=f'{stem}_static' if skinned_parts else stem, mesh=len(doc['meshes']) - 1))
            shaped[root_children[-1]] = targets
        if locators:
            by_name = {row['name']: joint_nodes[j] for j, row in enumerate(library.rows)} if library is not None else {}
            placed, unplaced = [], []
            for loc in locators:
                matrix = [float(v) for v in np.asarray(loc['matrix'], np.float64).T.reshape(-1)]   # glTF: column-major
                joint = loc.get('joint')
                if joint is None:
                    placed.append(g.node(name=loc['name'], matrix=matrix))
                elif joint < len(joint_names) and joint_names[joint] in by_name:
                    node = g.node(name=loc['name'], matrix=matrix)
                    doc['nodes'][by_name[joint_names[joint]]].setdefault('children', []).append(node)
                else:
                    unplaced.append(loc['name'])
            if placed:
                root_children.append(g.node(name=f'{stem}_locators', children=placed))
            g.locator_note = (f'{len(locators) - len(unplaced)} locator(s) as empty nodes, each a child of its joint with the '
                              'stored transform read column by column (the axis convention is not checked against a '
                              'runtime use).')
            if unplaced:
                g.locator_note += f' {len(unplaced)} left out: their joint is not in the exported skeleton.'
        top = g.node(name=stem, children=root_children)
        doc['scenes'][0]['nodes'] = [top]
        if clips:
            doc['animations'] = []
            for clip in clips:
                frames = len(clip['tracks'])
                times = g.accessor(np.arange(frames, dtype=np.float32) / np.float32(clip['fps']), 'SCALAR', minmax=True)
                single = g.accessor(np.zeros(1, np.float32), 'SCALAR', minmax=True)
                samplers, channels = [], []
                tr = clip['tracks']
                rotation = tr.rotation
                if frames > 1:   # keep neighbouring quaternions on one hemisphere
                    d = np.einsum('fjk,fjk->fj', rotation[1:], rotation[:-1])
                    sign = np.concatenate([np.ones((1, rotation.shape[1])), np.cumprod(np.where(d < 0, -1.0, 1.0), axis=0)])
                    rotation = rotation * sign[..., None]
                channels_by_path = (('translation', tr.translation, 'VEC3'), ('rotation', rotation, 'VEC4'), ('scale', tr.scale, 'VEC3'))
                constant = {path: np.all(np.abs(v - v[:1]) <= 1e-6 + 1e-5 * np.abs(v[:1]), axis=(0, 2)) for path, v, _k in channels_by_path}
                for j, node in enumerate(joint_nodes):
                    for path, v, kind in channels_by_path:
                        const = bool(constant[path][j])
                        values = v[:1, j] if const else v[:, j]
                        output = g.accessor(values.astype(np.float32), kind)
                        samplers.append(dict(input=single if const else times, output=output, interpolation='LINEAR'))
                        channels.append(dict(sampler=len(samplers) - 1, target=dict(node=node, path=path)))
                weights = clip.get('morph_weights')
                for node, targets in shaped.items():   # blend-shape weights; stored vertices and baked frames stay at 0
                    if weights is None or not any(isinstance(n, int) for n in targets):
                        continue
                    columns = np.stack([weights[:frames, n] if isinstance(n, int) else np.zeros(frames, np.float32)
                                        for n in targets], 1)
                    samplers.append(dict(input=times, output=g.accessor(columns.astype(np.float32).reshape(-1), 'SCALAR'),
                                         interpolation='LINEAR'))
                    channels.append(dict(sampler=len(samplers) - 1, target=dict(node=node, path='weights')))
                doc['animations'].append(dict(name=clip['name'], samplers=samplers, channels=channels))
        framed = {node: targets for node, targets in shaped.items() if any(isinstance(n, tuple) for n in targets)}
        if framed:   # the baked vertex-program frames: frame k at full weight at time k / fps, blended between frames
            times = g.accessor(np.asarray(baked, np.float32), 'SCALAR', minmax=True)
            samplers, channels = [], []
            for node, targets in framed.items():
                columns = np.zeros((len(baked), len(targets)), np.float32)
                for column, n in enumerate(targets):
                    if isinstance(n, tuple):
                        columns[n[1], column] = 1.0
                samplers.append(dict(input=times, output=g.accessor(columns.reshape(-1), 'SCALAR'), interpolation='LINEAR'))
                channels.append(dict(sampler=len(samplers) - 1, target=dict(node=node, path='weights')))
            doc.setdefault('animations', []).append(dict(name=VERTEX_ANIMATION, samplers=samplers, channels=channels))
        return g

    def _write_obj_materials(self, folder, stem, maps, tex_dir, texture_files):
        lines = [f'# {stem} materials (PBR maps; see export.json)']
        for slot, m in maps.items():
            name = m['obj_name']
            lines += ['', f'newmtl {name}']
            base = m.get('base_factor', [1.0, 1.0, 1.0])
            lines += [f'Kd {base[0]:.4f} {base[1]:.4f} {base[2]:.4f}', 'Ks 0 0 0', f"d {m['alpha']:.3f}", 'illum 2']
            if m.get('emissive_factor') and m['emissive'] is None:
                lines.append('Ke ' + ' '.join(f'{x:.4f}' for x in m['emissive_factor']))
            if tex_dir is None:
                continue

            def save(suffix, array, mode='RGB'):
                file = f'{name}_{suffix}'
                png = texture_files.get(file) or cached_png(m, suffix, array, mode)
                (tex_dir / f'{file}.png').write_bytes(png)
                texture_files[file] = png
                return f'textures/{file}.png'
            if m['base'] is not None:
                lines.append('map_Kd ' + save('basecolor', m['base']))
            if m['mr'] is not None:
                lines.append('map_Pr ' + save('roughness', m['mr'][..., 1], 'L'))
                lines.append('map_Pm ' + save('metallic', m['mr'][..., 2], 'L'))
            if m['normal'] is not None:
                lines.append('norm ' + save('normal', m['normal']))
                lines.append('map_Bump -bm 1 textures/' + f'{name}_normal.png')
            if m['emissive'] is not None:
                lines += ['Ke 1 1 1', 'map_Ke ' + save('emissive', m['emissive'])]
        (folder / f'{stem}.mtl').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def _atmosphere_bake(self, archive, path):
        """The lighting of Rift Apart atmosphere ``path`` for blender_convert.py --atmosphere: the probe turned so the key
        light comes from the review sun's azimuth, the key light and the tone-map adaptation (games.rcra.atmosphere)."""
        from nwo.games.rcra import atmosphere as A
        from nwo.viewer.environment import CACHE_DIR, environment_maps
        self.log(f'  lighting: atmosphere {path}')
        env = A.environment(archive, path, environment_maps()['sun_direction'], cache_dir=CACHE_DIR)
        atm = env['atmosphere']
        record = dict(path=path, block_sha256=atm['sha256'], probe=atm['probe'], anchor=atm['anchor'],
                      range=atm['range'], key_light=A.key_light(atm).tolist(), key_azimuth=atm['key_azimuth'],
                      key_elevation=atm['key_elevation'], turn_degrees=env['turn_degrees'], equirect_sha256=env['sha256'],
                      colour_correction=atm['colour_correction'], light_grid=env['light_grid'], metered=env['metered'])
        lut = A.read_lut(archive, atm['colour_correction']) if atm['colour_correction'] else None
        return dict(env, record=record, limitations=A.LIMITATIONS, colour_correction=lut)

    def _blender_convert(self, folder, stem, doc, blob, formats, blender, materials_json=None, scene=False,
                         atmosphere=None, clips=()):
        """GLB -> FBX/USD/.blend in a background Blender with factory settings and a private user folder; with
        ``materials_json`` the game's translated materials replace the imported ones; with ``scene`` the .blend gets
        the review lighting, camera and Render Region (blender_convert.py --scene), or with ``atmosphere`` (an
        _atmosphere_bake) a Rift Apart atmosphere's lighting and the game's display. ``clips`` (the GLB's animations in
        order: name, kind, fps, frames) are laid out in the .blend as the research pipeline does (blend_animation)."""
        with tempfile.TemporaryDirectory(prefix='rcps4-export-') as tmp:
            glb = Path(tmp) / f'{stem}.glb'
            tag_model(doc, stem)
            write_glb(glb, doc, blob)
            # The .blend sits in the output root beside the model folder, named after it (a folder with a
            # disambiguating model-ID suffix gives a distinct name, so two models with one stem do not collide).
            finals = {f: (folder.parent / f'{folder.name}.blend' if f == 'blend' else folder / f'{stem}{FORMATS[f][1]}')
                      for f in formats}
            # A .blend with translated materials is built as a candidate, reopened and validated in a second Blender,
            # then moved into place.
            candidate = materials_json is not None and 'blend' in finals
            targets = dict(finals, blend=Path(tmp) / f'{stem}.blend') if candidate else finals
            command = [str(blender), '--background', '--factory-startup', '--python-exit-code', '1', '--python', str(BLENDER_SCRIPT), '--',
                       '--glb', str(glb)] + [arg for f, path in targets.items() for arg in (f'--{f}', str(path))]
            if materials_json is not None:
                command += ['--materials', str(materials_json)]
            if scene and 'blend' in finals:
                command.append('--scene')
                if atmosphere is not None:
                    bake = Path(tmp) / 'atmosphere.npz'
                    np.savez(bake, equirect=atmosphere['equirect'], sun_direction=atmosphere['sun_direction'],
                             sun_irradiance=atmosphere['sun_irradiance'], record=json.dumps(atmosphere['record']),
                             colour_correction=np.zeros(0) if atmosphere['colour_correction'] is None
                             else atmosphere['colour_correction'].astype(np.float32))
                    command += ['--atmosphere', str(bake)]
            if clips and 'blend' in finals:
                clip_list = Path(tmp) / 'clips.json'
                clip_list.write_text(json.dumps(list(clips)), encoding='utf-8')
                command += ['--clips', str(clip_list)]
            env = os.environ.copy()
            env['BLENDER_USER_RESOURCES'] = str(Path(tmp) / 'user')
            env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2]) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
            self.log(f"  Blender: {', '.join(p.name for p in targets.values())}")
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                            encoding='utf-8', errors='replace', env=env,
                                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            output = []
            for line in self.process.stdout:
                output.append(line.rstrip())
                if line.startswith(('EXPORTED', 'TRANSLATED', 'SCENE', 'VALIDATED', 'Error', 'Traceback')):
                    self.log('  ' + line.rstrip())
            code = self.process.wait()
            self.process.stdout.close()
            self.process = None
            if self.cancelled():
                raise Cancelled()
            if code:
                raise RuntimeError('Blender conversion failed:\n' + '\n'.join(output[-15:]))
            missing = [p.name for p in targets.values() if not p.is_file()]
            if missing:
                raise RuntimeError('Blender did not write ' + ', '.join(missing))
            if candidate:
                check = subprocess.run([str(blender), '--background', '--factory-startup', '--python-exit-code', '1',
                                        '--python', str(BLENDER_SCRIPT), '--', '--validate', str(targets['blend'])],
                                       capture_output=True, text=True, encoding='utf-8', errors='replace', env=env,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                lines = (check.stdout + check.stderr).splitlines()
                for line in lines:
                    if line.startswith(('VALIDATED', 'PROBLEM')):
                        self.log('  ' + line)
                if check.returncode:
                    raise RuntimeError('The .blend failed its reopen check:\n' + '\n'.join(lines[-15:]))
                os.replace(targets['blend'], finals['blend'])
            # Names as export.json lists them, relative to the model folder (the .blend is '../<name>.blend').
            return [os.path.relpath(p, folder).replace(os.sep, '/') for p in finals.values()]

    def cancel(self):
        process = self.process
        if process and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
