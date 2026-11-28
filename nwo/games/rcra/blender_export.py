"""Rift Apart materials for Blender: each material's surface recipe and its unit textures, written beside an export.

``blender_materials`` is the ``Game.blender_materials`` hook. For every material it translates the graph's G-buffer
writer (nwo.games.rcra.surface), writes the textures the recipe samples and returns a JSON-ready entry that
nwo/export/rcra_blender.py turns into a node tree. A material whose writer is not supported gets an entry with its
reason and keeps the glTF importer's material.

A blended material (nwo.games.rcra.material.material_blend: additive, blended or modulate) also gets its forward program
translated (nwo.games.rcra.surface.forward) as ``forward``: the colour and alpha the game blends, with every light source
zero. Its G-buffer surface, when it translates, stays as ``recipe`` and is lit by the renderer; an additive material's
only when its forward program's key-light term is that surface under the class-0 apply program (``forward_lit``,
forward_light_matches_surface), as on the unicroid wasp's wings. A blended material whose
forward program does not translate keeps the opaque surface and says why.

Unit textures keep their stored values. 8-bit formats are written as RGBA PNG with the D3D defaults for channels a format
lacks (0 for green and blue, 1 for alpha); float formats are written as raw float32 RGBA (``.f32``, little endian, rows
top first) that Blender loads into a float image. Colourspace follows the DXGI format: sRGB formats are 'sRGB', everything
else 'Non-Color'. Texture arrays write the slice the recipe samples; an array sampled at a slice chosen at run time
writes every slice (up to MAX_SLICES), and the Blender material picks one by round(slice), clamped, as D3D does.
A 3D texture (flipbook volumes) is written as one atlas: its slices in a grid of ``columns``, each with a one-texel border
copied from the opposite edge, so a bilinear lookup inside a tile wraps as the texture does; the Blender material samples
the two slices around the depth coordinate and mixes them (trilinear, wrapping in depth).
A cube map (``cube_equirect``) is written as one equirectangular image of 4 x face size by 2 x face size, which Blender's
Environment Texture node reads with the game direction as its Vector: every pixel holds the D3D cube lookup of its direction
(a cube array's constant slice picks the cube). A cube whose alpha is not 1 gets a second image of its alpha.

A fur material (nwo.games.rcra.fur) also gets ``shells``: one entry per shell layer 1 .. Fur_LayerCount - 1 (layer 0
is the material itself), each with its own recipe (fur.shell_program: the writer at the layer's height, with a
``coverage`` term), its images, its height t and ``displace``, the metres the layer moves along the normal per unit of
``shell_height`` (the control map's blue channel, the shell vertex program's point-sampled offset scale). The Blender
side (nwo/export/rcra_blender.py build_fur_shells) draws each layer as a copy of the fur faces moved by that much.
"""
import json
import math
import os
import re
from pathlib import Path

import numpy as np

from nwo.formats.linear_png import linear_to_srgb as _linear_to_srgb, png16, srgb_to_linear as _srgb_to_linear
from nwo.games.rcra import lighting, surface as rsurface
from nwo.games.rcra.texture import texture_size

SRGB_FORMATS = frozenset({29, 72, 75, 78, 91, 93, 99})
# DXGI float/integer formats as (numpy dtype, channels).
FLOAT_FORMATS = {2: ('<f4', 4), 10: ('<f2', 4), 16: ('<f4', 2), 34: ('<f2', 2), 41: ('<f4', 1), 54: ('<f2', 1),
                 11: ('<u2', 4), 35: ('<u2', 2), 56: ('<u2', 1)}
UNORM16 = frozenset({11, 35, 56})
ENTRY_SCHEMA = 'rcra-blender-materials-v3'
# How each blend mode is composed over what is behind it. The engine names the modes ("Additive", "Lerp", "Modulate"
# blending into the alpha layer); the blend factors themselves are pipeline state outside the files, so these are the
# standard factors for those names, not decoded values.
BLEND_LIMITATIONS = {
    'blended': 'Blended (the engine\'s "Lerp" mode): colour x alpha over the background x (1 - alpha), the standard '
               'factors for that mode; the pipeline\'s blend state is not in the files.',
    'additive': 'Additive: colour x alpha added to the background, the standard factors for that mode; the pipeline\'s blend '
                'state is not in the files.',
    'modulate': 'Modulate: the background multiplied by the colour (lerped to white by alpha), the standard factors for that '
                'mode; the pipeline\'s blend state is not in the files.'}
MAX_SLICES = 32
# Recipes above this many operations (after lookup_tables) are not built: Blender takes minutes to build and compile them.
# Tables the compiler unrolled into select chains (512-entry noise permutations, about 21,000 operations each recipe) are
# one lookup each; the largest recipe is then 3,710 operations.
MAX_RECIPE_NODES = 10000
ALL_SLICES = 'all'
VOLUME = 'volume'
CUBE = 'cube'   # a layout is (CUBE, index of the cube in the array)
CLASS_LIMITATION = ('Blender lights this material with the class-0 BRDF: the game applies lighting class %d (%s) with its own '
                    'apply program, which Cycles has no closure for (the viewer draws it, nwo.games.rcra.lighting).')
# Classes 1, 2 and 5: their own Blender groups (nwo.export.rcra_closures, nwo.export.rcra_blender).
SUBSURFACE_LIMITATION = (
    'Lighting class 1 (subsurface): Blender mixes the class-1 program\'s two GGX lobes by the view depth (sat(0.15 - 0.03 '
    'depth) of the near lobe, alpha^1.66) and draws the skin pass\'s screen-space blur as Cycles\' Christensen-Burley '
    'subsurface at the same diffusion distance (the skin profile texel x 0.005 x (0.6 occlusion + 0.4) x (2^blur - 1) / 7 '
    'world units): Cycles blurs over the surface where the game blurs seven taps in screen space. Not drawn: the '
    'program\'s Henyey-Greenstein scatter on its second normal (at most 0.003 x 10 of the specular weight, at grazing '
    'views) and the transmission through the shadow-map depth.')
NO_PROFILE_LIMITATION = ('The skin profile index is not constant in this material: Blender draws its diffuse without the '
                         'skin pass blur (Burley radius 0).')
CLOTH_LIMITATION = (
    'Lighting class 2 (cloth): Blender draws the program\'s specular on the second normal, mix(GGX, sheen, sheen field / '
    '31) with r = 0.9725 - 0.6011 gloss: the GGX exactly, the Charlie sheen (Neubelt visibility, tinted by F0) as a fitted '
    'Ashikhmin velvet of the same mean reflectance (about 55 % relative L2 difference in shape). The diffuse, wrapped and '
    'scaled per channel by min(|N.L| + w sqrt(albedo), 1), is a Diffuse and a Translucent BSDF of the same irradiance from '
    'each hemisphere: ambient light is the program\'s, a single light falls off as Lambert. Not drawn: the scatter lobe '
    '(Henyey-Greenstein, at most 3 % of the specular).')
FOLIAGE_LIMITATION = (
    'Lighting class 5 (foliage): Blender draws the wrapped diffuse and back transmission as a Diffuse and a Translucent '
    'BSDF of the same irradiance from each hemisphere (ambient light is the program\'s, a single light falls off as '
    'Lambert), and the class-0 specular as the program does.')
CLASS_LIMITATIONS = {1: SUBSURFACE_LIMITATION, 2: CLOTH_LIMITATION, 5: FOLIAGE_LIMITATION}


def skin_profile(source, recipe):
    """The linear skin profile texel (rgb) of a class-1 recipe whose profile index (extra.4) is constant, else None:
    lighting.PROFILE_TEXTURE as the engine binds it, texel PROFILE_BASE + index (lighting.profile_texel)."""
    from nwo.games.rcra.texture import load_texture_full
    i = recipe['terms'].get('extra.%d' % lighting.PROFILE_FIELD)
    node = recipe['nodes'][i] if i is not None else None
    if node is None or node['op'] != 'const':
        return None
    texture = load_texture_full(source, lighting.PROFILE_TEXTURE)
    texels = texel_array(texture)[0]
    if texture['format'] in SRGB_FORMATS:
        texels[:, :3] = _srgb_to_linear(texels[:, :3])
    return [float(x) for x in lighting.profile_texel(texels, int(node['value']))[:3]]
# Class 4: Cycles' anisotropic GGX has the program's lobes (nwo.games.rcra.lighting.cycles_anisotropic).
ANISOTROPIC_LIMITATION = ('Lighting class 4 (anisotropic): Blender averages two anisotropic GGX lobes as the class-4 apply '
                          'program does, on the G-buffer tangent and on it turned by the rotation field, with alpha r^4 '
                          'from the gloss along the tangent and from the second gloss across it (Metallic BSDF, F82 with a '
                          'white edge: Schlick, F90 = 1). Cycles uses the exact anisotropic Smith term where the game uses '
                          'a separable approximation, makes the tangent orthogonal to the normal, and caps the ratio of '
                          'the two alphas at 10, keeping their product.')
CUBE_LIMITATION = 'Cube map %s resampled to an equirectangular image (filtering differs at face seams and poles).'
SAMPLER_NOTE = ('Texture sampling follows each texture\'s own sampler state (its header, nwo.games.rcra.texture.sampler_state): '
                'point or linear filtering and wrap, mirror or clamp addressing per axis. Cycles reads the top mip only, so the '
                'mip filtering, anisotropy and mip LOD bias have no effect; cube maps and 3D textures keep their own lookups.')
# D3D / GL cube lookup (the same table): per face +X, -X, +Y, -Y, +Z, -Z, the major axis and the (sc, tc) axes with signs.
CUBE_FACE_AXES = (((0, 1.0), (2, -1.0), (1, -1.0)), ((0, -1.0), (2, 1.0), (1, -1.0)),
                  ((1, 1.0), (0, 1.0), (2, 1.0)), ((1, -1.0), (0, 1.0), (2, -1.0)),
                  ((2, 1.0), (0, 1.0), (1, -1.0)), ((2, -1.0), (0, -1.0), (1, -1.0)))


def slice_index(texture, index):
    """``index`` clamped to the texture's slices, as D3D clamps an array index."""
    return min(max(int(index), 0), len(texture.get('slices', [texture['pixels']])) - 1)


def texel_array(texture, index=0):
    """Float32 (height, width, 4) of one slice as stored (the index clamped, slice_index), missing channels 0 and alpha 1."""
    fmt, width, height = texture['format'], texture['width'], texture['height']
    index = slice_index(texture, index)
    pixels = texture.get('slices', [texture['pixels']])[index]
    if fmt in FLOAT_FORMATS:
        dtype, channels = FLOAT_FORMATS[fmt]
        data = np.frombuffer(pixels, dtype=dtype, count=width * height * channels).astype(np.float32)
        if fmt in UNORM16:
            data /= 65535.0
        data = data.reshape(height, width, channels)
    else:
        from nwo.viewer.textures import to_image
        image = to_image(texture, index)
        channels = {'L': 1, 'LA': 2, 'RGB': 3, 'RGBA': 4}.get(image.mode)
        if channels is None:
            image = image.convert('RGBA')
            channels = 4
        data = np.asarray(image, dtype=np.float32).reshape(height, width, channels) / 255.0
        if image.mode == 'LA':   # luminance + alpha is not a D3D layout; keep it as red + alpha
            data = np.stack([data[..., 0], np.zeros_like(data[..., 0]), np.zeros_like(data[..., 0]), data[..., 1]], -1)
            channels = 4
        if fmt == 49 or (channels == 3 and fmt in (83, 84)):   # R8G8 / BC5: blue is not stored
            data[..., 2] = 0.0
    out = np.zeros((height, width, 4), np.float32)
    out[..., 3] = 1.0
    out[..., :min(channels, 4)] = data[..., :4]
    return out


def write_unit_texture(source, path, folder, index=0, cache=None):
    """Write one texture slice for Blender; returns {path, colourspace, kind, width, height}."""
    from nwo.games.rcra.texture import load_texture_full
    key = (path, index)
    if cache is not None and key in cache:
        return cache[key]
    texture = load_texture_full(source, path)
    data = texel_array(texture, index)
    row = _save(data, texture, Path(path).stem + ('_slice%d' % index if index else ''), folder, path)
    if cache is not None:
        cache[key] = row
    return row


def volume_atlas(slices, columns):
    """(rows, cols, 4) atlas of equal (h, w, 4) slices (in their dtype), each tile (h + 2, w + 2) with a wrapped one-texel
    border."""
    h, w = slices[0].shape[:2]
    rows = -(-len(slices) // columns)
    atlas = np.zeros((rows * (h + 2), columns * (w + 2), 4), slices[0].dtype)
    for k, data in enumerate(slices):
        r, c = divmod(k, columns)
        atlas[r * (h + 2):(r + 1) * (h + 2), c * (w + 2):(c + 1) * (w + 2)] = np.pad(data, ((1, 1), (1, 1), (0, 0)), 'wrap')
    return atlas


def write_volume_texture(source, path, folder, cache=None):
    """Write a 3D texture as a padded slice atlas; the row carries volume=dict(depth, columns, width, height)."""
    from nwo.games.rcra.texture import load_texture_full
    key = (path, VOLUME)
    if cache is not None and key in cache:
        return cache[key]
    texture = load_texture_full(source, path)
    count = len(texture.get('slices', [texture['pixels']]))
    columns = max(1, int(np.ceil(np.sqrt(count))))
    atlas = volume_atlas([texel_array(texture, k) for k in range(count)], columns)
    row = _save(atlas, texture, Path(path).stem + '_volume', folder, path)
    row['volume'] = dict(depth=count, columns=columns, width=int(texture['width']), height=int(texture['height']))
    if cache is not None:
        cache[key] = row
    return row


def cube_lookup(faces, direction):
    """D3D lookup of a cube map: ``faces`` (6, n, n, channels) in +X, -X, +Y, -Y, +Z, -Z order with the top row first,
    ``direction`` (..., 3) in the cube's space. The major axis picks the face, s = (sc / |ma| + 1) / 2 and t = (tc / |ma| + 1) / 2
    with t = 0 the top row (GL's table is the same), bilinear within the face (no filtering across edges)."""
    d = np.asarray(direction, np.float64)
    a = np.abs(d)
    x_major = (a[..., 0] >= a[..., 1]) & (a[..., 0] >= a[..., 2])
    y_major = ~x_major & (a[..., 1] >= a[..., 2])
    face = np.where(x_major, np.where(d[..., 0] >= 0, 0, 1), np.where(y_major, np.where(d[..., 1] >= 0, 2, 3),
                                                                      np.where(d[..., 2] >= 0, 4, 5)))
    major = np.maximum(np.where(x_major, a[..., 0], np.where(y_major, a[..., 1], a[..., 2])), 1e-30)
    axes = np.array([[[axis, sign] for axis, sign in face_axes] for face_axes in CUBE_FACE_AXES])   # (6, 3, 2)
    n = faces.shape[1]
    st = []
    for k in (1, 2):   # sc, tc
        axis = axes[face, k, 0].astype(int)
        value = np.take_along_axis(d, axis[..., None], -1)[..., 0] * axes[face, k, 1]
        st.append(np.clip(((value / major + 1.0) * 0.5) * n - 0.5, 0.0, n - 1.0))
    (x, y) = st
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, n - 1), np.minimum(y0 + 1, n - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    top = faces[face, y0, x0] * (1 - fx) + faces[face, y0, x1] * fx
    bottom = faces[face, y1, x0] * (1 - fx) + faces[face, y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def equirect_directions(width, height, rows=None):
    """(rows, width, 3) directions of the equirectangular pixel centres, rows top first. Blender's Environment Texture reads
    u = 0.5 - atan2(y, x) / 2 pi and v = 0.5 + atan2(z, hypot(x, y)) / pi (v = 1 at the top), so a pixel's direction is
    (cos phi cos theta, sin phi cos theta, sin theta) with phi = (0.5 - u) 2 pi and theta = (v - 0.5) pi."""
    rows = range(height) if rows is None else rows
    u = (np.arange(width) + 0.5) / width
    v = 1.0 - (np.asarray(list(rows)) + 0.5) / height
    phi, theta = (0.5 - u)[None, :] * 2.0 * np.pi, (v - 0.5)[:, None] * np.pi
    shape = (len(v), width)
    return np.stack([np.broadcast_to(np.cos(phi) * np.cos(theta), shape), np.broadcast_to(np.sin(phi) * np.cos(theta), shape),
                     np.broadcast_to(np.sin(theta), shape)], -1)


def cube_equirect(faces, srgb=False):
    """(2n, 4n, 4) float32 equirectangular image of a cube map (``faces`` (6, n, n, 4), top row first): each pixel is the
    cube lookup of its direction (``equirect_directions``). sRGB colour is filtered in linear light, as the hardware does."""
    n = faces.shape[1]
    width, height = 4 * n, 2 * n
    data = faces.astype(np.float64)
    if srgb:
        data[..., :3] = _srgb_to_linear(data[..., :3])
    out = np.empty((height, width, faces.shape[3]), np.float32)
    for start in range(0, height, 64):
        rows = range(start, min(height, start + 64))
        out[rows.start:rows.stop] = cube_lookup(data, equirect_directions(width, height, rows))
    if srgb:
        out[..., :3] = _linear_to_srgb(out[..., :3])
    return out


def write_cube_texture(source, path, folder, index=0, cache=None):
    """Write cube ``index`` of a cube texture as an equirectangular image; the row carries cube=dict(size, slice, alpha), alpha
    the row of a second image (alpha in rgb, Non-Color) when the texture's alpha is not 1, else None."""
    from nwo.games.rcra.texture import load_texture_full
    key = (path, CUBE, index)
    if cache is not None and key in cache:
        return cache[key]
    texture = load_texture_full(source, path)
    if texture.get('faces', 1) != 6:
        raise rsurface.Unsupported('Texture %s is sampled as a cube map but is not one' % path)
    cubes = len(texture['slices']) // 6
    index = min(max(index, 0), cubes - 1)
    faces = np.stack([texel_array(texture, index * 6 + k) for k in range(6)])
    equirect = cube_equirect(faces, texture['format'] in SRGB_FORMATS)
    stem = Path(path).stem + '_cube' + ('%d' % index if index else '')
    row = _save(equirect, texture, stem, folder, path)
    alpha = None
    if np.any(faces[..., 3] != 1.0):
        grey = np.repeat(equirect[..., 3:4], 4, -1)
        grey[..., 3] = 1.0
        alpha = _save(grey, texture, stem + '_alpha', folder, path, colourspace='Non-Color')
    row['cube'] = dict(size=int(faces.shape[1]), slice=int(index), alpha=alpha)
    if cache is not None:
        cache[key] = row
    return row


def _save(data, texture, stem, folder, path, colourspace=None):
    folder.mkdir(parents=True, exist_ok=True)
    colourspace = colourspace or ('sRGB' if texture['format'] in SRGB_FORMATS else 'Non-Color')
    if texture['format'] in FLOAT_FORMATS:
        target = folder / (stem + '.f32')
        target.write_bytes(np.ascontiguousarray(data, '<f4').tobytes())
        kind = 'float'
    elif colourspace == 'sRGB':
        # The GPU decodes each sRGB texel before it filters; Cycles filters an sRGB image (8 or 16 bit) and decodes after.
        # So the texels are decoded here and stored as 16-bit linear values ('png16', loaded as Non-Color): within 7.6e-6
        # of the decode, every byte value distinct. Alpha is stored as it is.
        texels = np.round(data * 255.0) / 255.0
        texels[..., :3] = _srgb_to_linear(texels[..., :3])
        target = folder / (stem + '.png')
        target.write_bytes(png16(np.round(texels * 65535.0).astype(np.uint16)))
        kind = 'png16'
    else:
        from PIL import Image
        target = folder / (stem + '.png')
        Image.fromarray(np.round(data * 255.0).astype(np.uint8), 'RGBA').save(target)
        kind = 'png'
    from nwo.games.rcra.texture import DEFAULT_SAMPLER
    sampler = texture.get('sampler') or DEFAULT_SAMPLER   # the texture header's sampler (texture.sampler_state)
    return dict(path=str(target), colourspace=colourspace, kind=kind, width=int(data.shape[1]), height=int(data.shape[0]),
                source=path, format=int(texture['format']),
                sampler=dict(filter=int(sampler['filter']), address=[int(a) for a in sampler['address']]))


def cube_index(recipe, node):
    """Index of the cube a sample reads: 0 for a cube map, the rounded constant 4th address lane for a cube array."""
    if node['dimension'] != rsurface.DIMENSION_CUBE_ARRAY:
        return 0
    z = recipe['nodes'][node['args'][3]]
    if z['op'] != 'const':
        raise rsurface.Unsupported('Cube array %s is sampled at a run-time index' % node['binding'])
    return max(0, int(math.floor(z['value'] + 0.5)))


def merge_layouts(a, b):
    """The layout of a binding sampled as both ``a`` and ``b``: one array of every slice, or None when a volume or a cube is
    sampled differently (they cannot share an image)."""
    if a == b:
        return a
    if VOLUME in (a, b) or isinstance(a, tuple) or isinstance(b, tuple):
        return None
    return ALL_SLICES


def _slice_index(value):
    """The array slice of a constant index: rounded; NaN reads slice 0, as D3D converts NaN to the integer 0, and an
    infinite index is left to the run-time path (ALL_SLICES), which clamps it to the array."""
    if value != value:
        return 0
    return int(round(value)) if math.isfinite(value) else ALL_SLICES


def _slices(recipe):
    """{binding: slice} of the recipe's samples, ALL_SLICES where the slice is chosen at run time or differs, VOLUME for a
    3D texture, (CUBE, index) for a cube map. Raises Unsupported when a binding is sampled as a cube and as something else."""
    slices = {}
    for n in recipe['nodes']:
        if n['op'] == 'sample':
            z = recipe['nodes'][n['args'][2]] if len(n['args']) > 2 else None
            if n.get('dimension') in rsurface.CUBE_DIMENSIONS:
                index = (CUBE, cube_index(recipe, n))
            elif n.get('dimension') == rsurface.DIMENSION_TEXTURE3D:
                index = VOLUME
            else:
                index = 0 if z is None else (_slice_index(z['value']) if z['op'] == 'const' else ALL_SLICES)
            merged = merge_layouts(slices.setdefault(n['binding'], index), index)
            if merged is None and VOLUME not in (slices[n['binding']], index):
                raise rsurface.Unsupported('Texture %s is sampled as a cube map and differently' % n['binding'])
            slices[n['binding']] = ALL_SLICES if merged is None else merged   # slice 0 is a slice, not "every slice"
    return slices


def _unit_images(source, path, folder, index, cache):
    if isinstance(index, tuple):
        return write_cube_texture(source, path, folder, index[1], cache)
    if index == VOLUME:
        return write_volume_texture(source, path, folder, cache)
    if index != ALL_SLICES:
        return write_unit_texture(source, path, folder, index, cache)
    from nwo.games.rcra.texture import texture_header
    count = max(1, int(texture_header(source, path).get('array_size') or 1))
    if count > MAX_SLICES:
        raise rsurface.Unsupported('Texture array %s has %d slices' % (path, count))
    return dict(slices=[write_unit_texture(source, path, folder, k, cache) for k in range(count)])


def _images(source, results, folder, texture_cache):
    """{binding: image row} of every texture the recipes sample (one material: a binding is the same texture in both)."""
    slices = {}
    for recipe, result in results:
        for binding, index in _slices(recipe).items():
            merged = merge_layouts(slices.setdefault(binding, index), index)
            if merged is None:
                raise rsurface.Unsupported('Texture %s is sampled as a volume or a cube map and differently' % binding)
            slices[binding] = merged
    textures = {}
    for _recipe, result in results:
        textures.update(result['textures'])
    return {binding: _unit_images(source, path, folder, slices.get(binding, 0), texture_cache)
            for binding, path in textures.items()}


def engine_default(kind):
    """The engine default texture (texture.ENGINE_DEFAULTS) of a slot type (material.SLOT_TYPE_*)."""
    from nwo.games.rcra import material as rmat
    from nwo.games.rcra import texture as rtex
    if kind == rmat.SLOT_TYPE_3D:
        return rtex.ENGINE_DEFAULT_3D
    if kind in rmat.SLOT_TYPES_CUBE:
        return rtex.ENGINE_DEFAULT_CUBE
    return rtex.ENGINE_DEFAULT_2D   # 2D, and 1D whose default holds the same value


def replace_missing(source, graph_data, result):
    """Replace in a translation result each texture the install does not hold by the engine default texture of its slot
    type, as the game binds it (texture.ENGINE_DEFAULTS); a unit no slot row names takes texture.UNBOUND (reads 0, as an
    unbound D3D resource). The result's limitations name each texture and its stand-in."""
    from nwo.games.rcra import material as rmat
    from nwo.games.rcra import shaders as rshaders
    from nwo.games.rcra.texture import CONSTANT_TEXTURES, UNBOUND

    def present(path):
        return path in CONSTANT_TEXTURES or sum(source.entries[k]['type'] == 'texture' for k in source.resolve(path)) == 1
    kinds = {}
    if graph_data is not None:
        kinds = {'textureUnit%d' % (rshaders.FIRST_MATERIAL_UNIT + row[0]): row[4]
                 for row in rmat.graph_slot_rows(graph_data)}
    notes = []
    for binding, path in sorted(result['textures'].items()):
        if present(path):
            continue
        stand_in = engine_default(kinds[binding]) if binding in kinds else UNBOUND
        result['textures'][binding] = stand_in
        notes.append('%s (%s): %s' % (path, binding, ENGINE_DEFAULT_NOTES.get(stand_in, 'reads 0, as an unbound texture')))
    if notes:
        result['limitations'].append(MISSING_NOTE % '; '.join(notes))


MISSING_NOTE = 'Textures the install does not hold, with the engine default the game binds in their place: %s.'
ENGINE_DEFAULT_NOTES = {'engine/default_2d.texture': 'Default 2D, 128/255 in every channel',
                        'engine/default_3d.texture': 'Default 3D, 1 in every channel',
                        'engine/default_cube.texture': 'Default Cube, 128/255 in every channel'}


def _reads_background(recipe):
    """True when a forward recipe's background tint is not the constant 0."""
    nodes = recipe['nodes']
    return any(nodes[recipe['terms'][k]]['op'] != 'const' or nodes[recipe['terms'][k]]['value']
               for k in ('background.r', 'background.g', 'background.b') if k in recipe['terms'])


# A lit additive material's G-buffer surface stands in for its forward program's lighting in Blender only where the two
# agree: the forward colour's key-light term (surface.forward_lighting: colour(L, LIGHT) - colour(L, 0)) equals the class-0
# apply program (lighting.direct) run on the surface's terms, at random inputs and texels, within the G-buffer's 8-bit F0
# and gloss (relative, with a floor). The unicroid wasp's hummingbird wing agrees within 0.9 %; star fields and the wine
# glass light other terms (14 and 44 times off) and stay unlit.
FORWARD_LIT_TOLERANCE = 0.02
FORWARD_LIT_FLOOR = 0.05
FORWARD_LIT_TRIALS = 24


def _unit(rng):
    v = [rng.gauss(0.0, 1.0) for _ in range(3)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def forward_light_matches_surface(forward_lit, surface, apply, trials=FORWARD_LIT_TRIALS, seed=3):
    """None when the forward key-light term (recipe ``forward_lit``, terms colour.r/g/b of L and LIGHT) equals ``apply``
    (lighting.direct of class 0) on the terms of ``surface`` (a G-buffer recipe), else the reason. Each trial draws a
    normal (TEXCOORD2), a tangent (TEXCOORD3) and UVs, a key light on the normal's side and a camera in front, and one
    texel per binding shared by both recipes; inputs neither names are uniform in [-1, 1]."""
    import random
    rng = random.Random(seed)
    names = set(forward_lit['inputs']) | set(surface['inputs'])
    compared, worst = 0, 0.0
    for _trial in range(trials):
        texels = {}

        def sample(node, _args):
            return texels.setdefault(node.get('binding'), [rng.uniform(0.1, 0.9) for _ in range(4)])
        values = {name: rng.uniform(-1.0, 1.0) for name in sorted(names)}
        n, light, view = _unit(rng), _unit(rng), _unit(rng)
        if sum(a * b for a, b in zip(n, light)) < 0.0:
            light = [-x for x in light]
        if sum(a * b for a, b in zip(n, view)) > 0.0:   # VIEW is camera to point: the camera on the normal's side
            view = [-x for x in view]
        t = _unit(rng)
        t = [a - sum(x * y for x, y in zip(t, n)) * b for a, b in zip(t, n)]
        tn = math.sqrt(sum(x * x for x in t)) or 1.0
        depth = rng.uniform(1.0, 5.0)
        values.update({'SV_IsFrontFace': True, 'TEXCOORD3.w': 1.0, 'CAMERA.x': 0.0, 'CAMERA.y': 0.0, 'CAMERA.z': 0.0})
        for k, c in enumerate('xyz'):
            values.update({'TEXCOORD2.' + c: n[k], 'TEXCOORD3.' + c: t[k] / tn, 'L.' + c: light[k],
                           'VIEW.' + c: view[k] * depth})
        for name in ('TEXCOORD.x', 'TEXCOORD.y', 'TEXCOORD1.x', 'TEXCOORD1.y'):
            values[name] = rng.random()
        off = rsurface.evaluate_recipe(forward_lit, dict(values, **{'LIGHT.' + c: 0.0 for c in 'rgb'}), sample)
        on = rsurface.evaluate_recipe(forward_lit, dict(values, **{'LIGHT.' + c: 1.0 for c in 'rgb'}), sample)
        terms = rsurface.evaluate_recipe(surface, values, sample)
        if terms.get('class', 0.0) != 0.0:
            return 'its G-buffer surface is not lighting class 0'
        inputs = {name: terms[name] for name in lighting.INPUTS if name in terms}
        inputs.update({name: values[name] for name in ('L.x', 'L.y', 'L.z', 'VIEW.x', 'VIEW.y', 'VIEW.z')})
        inputs.update({'LIGHT.r': 1.0, 'LIGHT.g': 1.0, 'LIGHT.b': 1.0, 'DEPTH': depth})
        want = lighting.evaluate(apply, inputs)
        got = [on['colour.' + c] - off['colour.' + c] for c in 'rgb']
        if not all(math.isfinite(x) for x in got + want):
            continue
        compared += 1
        worst = max([worst] + [abs(a - b) / max(FORWARD_LIT_FLOOR, abs(b)) for a, b in zip(got, want)])
    if not compared:
        return 'no finite trial'
    if worst > FORWARD_LIT_TOLERANCE:
        return 'its key-light term is not its G-buffer surface under the class-0 apply program (%.3g relative)' % worst
    return None


def _forward_lit(source, graph, data, surface):
    """True when an additive material's G-buffer surface lit by the renderer is its forward program's lighting
    (forward_light_matches_surface), else the reason."""
    try:
        lit = rsurface.fold_nan(rsurface.export_recipe(
            rsurface.forward_lighting(graph, data, texture_sizes=lambda path: texture_size(source, path))))
    except (rsurface.Unsupported, ValueError, KeyError) as error:
        return str(error)
    apply = lighting.direct(source.root, 0, disk=getattr(source, 'translation_disk_cache', None))
    return forward_light_matches_surface(lit, surface, apply) or True


def material_entry(source, material_path, folder, texture_cache=None, graph_cache=None):
    """The Blender entry of one material: dict(blend, recipe, forward, images, limitations), or dict(error)."""
    from nwo.games.rcra.material import _graphs, _material_key, double_sided, draw_flags, material_blend
    key = _material_key(source, material_path)
    if key is None:
        return dict(error='material asset not found')
    graphs = list(_graphs(source, key))
    program = None
    if not graphs:   # a fur material: the engine's fur writer (nwo.games.rcra.fur)
        from nwo.games.rcra.fur import EXECUTABLE, fur_program, fur_record
        from nwo.games.rcra.material import no_graph_reason
        if fur_record(source.asset_data(key)) is None:
            return dict(error=no_graph_reason(source.asset_data(key)))
        try:
            program = fur_program(source, source.asset_data(key))
        except (ValueError, OSError) as error:
            return dict(error='%s: %s' % (type(error).__name__, error))
        graphs = ['%s fur writer' % EXECUTABLE]
    graph_path = graphs[0]
    graph_cache = graph_cache if graph_cache is not None else {}
    if program is None and graph_path not in graph_cache:
        gkey = next((k for k in source.resolve(graph_path) if source.entries[k]['type'] == 'materialgraph'), None)
        graph_cache[graph_path] = source.asset_data(gkey) if gkey is not None else None
    graph = graph_cache[graph_path] if program is None else None
    if graph is None and program is None:
        return dict(error='graph not found: ' + graph_path)
    data = source.asset_data(key)
    try:
        blend = material_blend(data)
    except ValueError as error:
        return dict(error='ValueError: %s' % error, graph=graph_path)
    errors, results, limitations, forward_lit = {}, {}, [], None
    passes = (('forward', rsurface.forward), ('recipe', rsurface.surface)) if blend != 'opaque' else \
        (('recipe', rsurface.surface),)
    if program is not None:   # an engine program replaces the graph: its G-buffer writer only
        passes = (('recipe', lambda g, d, **k: rsurface.surface(g, d, program=program, **k)),)
        errors['forward'] = 'Unsupported: an engine program without a forward pass'
    for name, translate in passes:
        try:
            results[name] = translate(graph, data, texture_sizes=lambda path: texture_size(source, path))
        except (rsurface.Unsupported, ValueError, KeyError) as error:
            errors[name] = '%s: %s' % (type(error).__name__, error)
    if 'recipe' not in results and 'forward' not in results:
        return dict(error=errors.get('recipe') or errors.get('forward'), graph=graph_path, blend=blend)
    for result in results.values():
        replace_missing(source, graph, result)
    recipes = {name: rsurface.blender_recipe(rsurface.export_recipe(result)) for name, result in results.items()}
    for name in [n for n, r in recipes.items() if len(r['nodes']) > MAX_RECIPE_NODES]:
        errors[name] = 'Unsupported: %d operations, over the Blender limit of %d' % (len(recipes[name]['nodes']),
                                                                                     MAX_RECIPE_NODES)
        del results[name], recipes[name]
    if not recipes:
        return dict(error=errors.get('recipe') or errors.get('forward'), graph=graph_path, blend=blend)
    if 'forward' in recipes and rsurface.reads_scene(recipes['forward']):
        errors['forward'] = ('Unsupported: the forward program reads the scene behind it other than as a tint, which '
                             'Cycles has no node for (the viewer samples a copy of its frame)')
        del results['forward'], recipes['forward']
        if 'recipe' not in results:
            return dict(error=errors.get('recipe') or errors['forward'], graph=graph_path, blend=blend)
    if blend == 'modulate' and 'forward' in recipes and _reads_background(recipes['forward']):
        errors['forward'] = 'Unsupported: a modulate material that passes the scene behind through is not linear in it'
        del results['forward'], recipes['forward']
        if 'recipe' not in results:
            return dict(error=errors.get('recipe') or errors['forward'], graph=graph_path, blend=blend)
    try:
        images = _images(source, [(recipes[n], results[n]) for n in recipes], folder, texture_cache)
    except (rsurface.Unsupported, ValueError, KeyError, OSError) as error:
        return dict(error='%s: %s' % (type(error).__name__, error), graph=graph_path, blend=blend)
    if blend != 'opaque':
        if 'forward' in results:
            limitations.append(BLEND_LIMITATIONS[blend])
            limitations += results['forward']['limitations']
            if 'lit_background.r' in results['forward']['terms']:
                limitations.append(rsurface.UNIT_IRRADIANCE_NOTE)
            if 'recipe' not in results:
                limitations.append('The G-buffer surface does not translate (%s): only the forward colour is drawn, '
                                   'unlit by the renderer.' % errors['recipe'])
            elif blend == 'additive' and draw_flags(data):
                forward_lit = _forward_lit(source, graph, data, recipes['recipe'])
                if forward_lit is not True:
                    limitations.append('Additive and unlit by the renderer, though the material is lit: %s, so only its '
                                       'forward colour (every light 0) is added.' % forward_lit)
        else:
            limitations.append('The game draws this material %s in its forward pass, which does not translate (%s): '
                               'Blender gets its opaque G-buffer surface.' % (blend, errors['forward']))
    skin = None
    if 'recipe' in results:
        limitations += results['recipe']['limitations']
        cls = results['recipe']['graph'].value(results['recipe']['terms']['class'])
        cls = int(cls) if cls is not None else 0
        own = cls in CLASS_LIMITATIONS and all('extra.%d' % k in recipes['recipe']['terms']
                                               for k in range(len(lighting.EXTRA_FIELDS[cls])))
        if lighting.anisotropic(recipes['recipe']):
            limitations.append(ANISOTROPIC_LIMITATION)
        elif own:
            limitations.append(CLASS_LIMITATIONS[cls])
        elif cls:
            limitations.append(CLASS_LIMITATION % (cls, lighting.CLASS_NAMES[cls]))
        if own and cls == 1:
            profile = skin_profile(source, recipes['recipe'])
            if profile is None:
                limitations.append(NO_PROFILE_LIMITATION)
            else:
                skin = profile
    limitations += [CUBE_LIMITATION % binding for binding, row in sorted(images.items()) if 'cube' in row]
    limitations += [
        'Tangent frame: the decoded vertex tangent and TEXCOORD3.w, carried as coordinates in Blender\'s UV0 tangent '
        'frame so they turn with the armature; the draw is unmirrored (render flag 0x10000 clear).',
        SAMPLER_NOTE]
    entry = dict(graph=graph_path, blend=blend if 'forward' in results else 'opaque', game_blend=blend, images=images,
                 limitations=limitations, double_sided=double_sided(data))
    entry.update(recipes)
    if forward_lit is True:   # an additive material lit as its G-buffer surface: Blender adds that surface, lit
        entry['forward_lit'] = True
    if skin is not None:
        entry['skin_profile'] = skin
    if program is not None:
        try:
            entry.update(fur_shells(source, data, folder, texture_cache, material_path))
        except (rsurface.Unsupported, ValueError, KeyError, OSError) as error:
            entry['shell_error'] = '%s: %s' % (type(error).__name__, error)
            entry['limitations'] = limitations + ['Fur shells are not drawn: %s' % entry['shell_error']]
    return entry


def shell_height(source, control_path, folder, cache=None):
    """The control map's blue channel as a grey image row: the shell vertex program's offset scale per texel."""
    from nwo.games.rcra.texture import load_texture_full
    key = ('shell_height', control_path)
    if cache is not None and key in cache:
        return cache[key]
    texture = load_texture_full(source, control_path)
    data = texel_array(texture, 0)
    grey = np.repeat(data[..., 2:3], 3, axis=2)
    grey = np.concatenate([grey, np.ones_like(grey[..., :1])], axis=2)
    row = _save(grey, dict(texture, format=28), Path(control_path).stem + '_shell_height', folder, control_path,
                colourspace='Non-Color')
    if cache is not None:
        cache[key] = row
    return row


def shell_layer(source, data, k, folder, texture_cache=None):
    """The entry of shell layer ``k`` of a fur material: dict(layer, layers, t, displace, recipe, images, limitations)."""
    from nwo.games.rcra import fur
    program = fur.shell_program(source, data, k)
    result = rsurface.surface(None, data, program=program, texture_sizes=lambda path: texture_size(source, path))
    replace_missing(source, None, result)
    recipe = rsurface.blender_recipe(rsurface.export_recipe(result))
    if len(recipe['nodes']) > MAX_RECIPE_NODES:
        raise rsurface.Unsupported('Fur layer %d has %d operations' % (k, len(recipe['nodes'])))
    layer = program['shell']
    return dict(layer=k, layers=layer['layers'], t=layer['t'], displace=layer['displace'], recipe=recipe,
                images=_images(source, [(recipe, result)], folder, texture_cache), limitations=result['limitations'])


def fur_shells(source, data, folder, texture_cache=None, material_path=None):
    """dict(shells [layer entries], shell_height row, shell_length) of a fur material (nwo.games.rcra.fur). With
    ``material_path`` each layer is kept in the entry cache (cached_shell_layer)."""
    from nwo.games.rcra import fur
    record = fur.fur_record(data)
    layers = [cached_shell_layer(source, material_path, data, k, folder, texture_cache) if material_path
              else shell_layer(source, data, k, folder, texture_cache) for k in range(1, fur.layer_count(record))]
    control = fur_control_path(source, data)
    return dict(shells=layers, shell_height=shell_height(source, control, folder, texture_cache) if layers else None)


def fur_control_path(source, data):
    """The fur control map the shell vertex program reads, after the missing-texture rule (replace_missing)."""
    from nwo.games.rcra import fur
    result = dict(textures={fur.CONTROL_BINDING: fur.fur_record(data)[2][3]}, limitations=[])
    replace_missing(source, None, result)
    return result['textures'][fur.CONTROL_BINDING]


# The entry cache: material_entry results (and each fur shell layer's, which fur_shells reads, so the layers of one fur
# material can translate in parallel) in the source's translation disk cache (nwo.viewer.native_glsl
# TranslationDiskCache), their image files in a content-addressed BlobDiskCache beside it. The key holds the material and
# graph bytes (or the fur program), the archive's identity (the images come from its textures), the translation code
# (viewer_materials.CODE_SHA256) and the image-writing code below.
ENTRY_CACHE_VERSION = 1
IMAGE_CODE_FILES = ('formats/linear_png.py', 'viewer/textures.py')
IMAGE_CACHE_BYTES = 2 * 1024 ** 3
_UNITS = 'rcra-units:'   # a stored entry's image path, relative to the units folder


def _image_code_sha256():
    from hashlib import sha256
    package = Path(__file__).resolve().parents[2]
    try:
        return sha256(b''.join(name.encode() + (package / name).read_bytes() for name in IMAGE_CODE_FILES)).hexdigest()
    except OSError:   # frozen executable without sources: no entry cache
        return None


IMAGE_CODE_SHA256 = _image_code_sha256()


def _entry_inputs(source, material_path):
    """(material bytes, graph bytes or the fur program's key, fur) of a material, or None when it has no graph to key."""
    from nwo.games.rcra import viewer_materials as vm
    from nwo.games.rcra.material import _material_key
    try:
        key = _material_key(source, material_path)
        if key is None:
            return None
        data = source.asset_data(key)
        graph = vm._graph(source, key)
        if graph is None:
            from nwo.games.rcra.fur import fur_program
            return data, fur_program(source, data)['key'], True
        return data, bytes(graph), False
    except (vm.NotTranslated, ValueError, KeyError, OSError):
        return None


def entry_cache_key(source, material_path, part='', inputs=None):
    """cached_material_entry's key (or a fur shell layer's, ``part`` 'shell <k>'), or None when the entry is not cached
    (no disk cache or sources, or no graph). ``inputs``: the material's _entry_inputs, when the caller has them."""
    from nwo.games.rcra import viewer_materials as vm
    disk = getattr(source, 'translation_disk_cache', None)
    if disk is None or not vm.CODE_SHA256 or not IMAGE_CODE_SHA256:
        return None
    inputs = inputs or _entry_inputs(source, material_path)
    if inputs is None:
        return None
    variant = 'rcra-blender:%d:%s:%s:%r' % (ENTRY_CACHE_VERSION, vm.CODE_SHA256, IMAGE_CODE_SHA256,
                                            sorted(getattr(source, 'metadata', {}).items()))
    return disk.key(bytes(inputs[0]), [inputs[1]], variant + (':' + part if part else ''), own_code=True)


def translation_units(source, material_paths, folder):
    """The materials whose entries are not cached, as worker units in two phases: ('blender', (path, folder)) for each
    one without fur and ('blender-shell', (path, k, folder)) for each uncached shell layer of a fur material first, then
    ('blender', (path, folder)) for the fur materials, which read their layers from the cache. ``folder`` (the export's)
    scopes a worker's image folder and texture cache to this export, as blender_materials' are."""
    folder = str(folder)
    from nwo.games.rcra import fur
    disk = getattr(source, 'translation_disk_cache', None)
    first, shells, second = [], [], []
    for path in dict.fromkeys(material_paths):
        inputs = _entry_inputs(source, path)
        key = entry_cache_key(source, path, inputs=inputs)
        if not key or disk.has(key):
            continue
        if not inputs[2]:
            first.append(('blender', (path, folder)))
            continue
        for k in range(1, fur.layer_count(fur.fur_record(inputs[0]))):
            if not disk.has(entry_cache_key(source, path, 'shell %d' % k, inputs)):
                shells.append(('blender-shell', (path, k, folder)))
        second.append(('blender', (path, folder)))
    return [phase for phase in (first + shells, second) if phase]


def _image_cache(source):
    from nwo.viewer.blob_cache import BlobDiskCache
    cache = source.__dict__.get('_rcra_blender_images')
    if cache is None:
        cache = source.__dict__['_rcra_blender_images'] = BlobDiskCache(
            Path(source.translation_disk_cache.folder).parent / 'blender_images', IMAGE_CACHE_BYTES)
    return cache


def _relocate(value, old, new):
    """``value`` with every string that starts with ``old`` starting with ``new`` instead."""
    if isinstance(value, str):
        return new + value[len(old):] if value.startswith(old) else value
    if isinstance(value, dict):
        return {k: _relocate(v, old, new) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_relocate(v, old, new) for v in value)
    return value


def _image_paths(value, prefix, found):
    if isinstance(value, str):
        if value.startswith(prefix):
            found.add(value[len(prefix):])
    elif isinstance(value, dict):
        for v in value.values():
            _image_paths(v, prefix, found)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _image_paths(v, prefix, found)
    return found


def _cached(source, key, folder, compute):
    """``compute()`` (an entry whose image paths lie in ``folder``), kept across sessions under ``key``: a stored entry
    writes its images into ``folder`` from the image cache; without a key, or when a stored image is missing, it is
    computed."""
    import hashlib
    disk = getattr(source, 'translation_disk_cache', None)
    folder = Path(folder)
    prefix = str(folder) + os.sep
    stored = disk.get(key) if key else None
    if stored is not None:
        images, blobs = _image_cache(source), {}
        for name, digest in stored['files'].items():
            target = folder / name
            if not target.is_file():
                blobs[name] = images.get(digest)
                if blobs[name] is None:
                    stored = None
                    break
        if stored is not None:
            for name, blob in blobs.items():
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_bytes(blob)
            return _relocate(stored['entry'], _UNITS, prefix)
    entry = compute()
    if key:
        files = {}
        for name in sorted(_image_paths(entry, prefix, set())):
            try:
                blob = (folder / name).read_bytes()
            except OSError:
                return entry
            files[name] = hashlib.sha256(blob).hexdigest()
            _image_cache(source).put(files[name], blob)
        disk.put(key, dict(entry=_relocate(entry, prefix, _UNITS), files=files))
    return entry


def cached_material_entry(source, material_path, folder, texture_cache=None, graph_cache=None):
    """material_entry, kept across sessions (_cached)."""
    return _cached(source, entry_cache_key(source, material_path), folder,
                   lambda: material_entry(source, material_path, folder, texture_cache, graph_cache))


def cached_shell_layer(source, material_path, data, k, folder, texture_cache=None):
    """shell_layer, kept across sessions (_cached)."""
    return _cached(source, entry_cache_key(source, material_path, 'shell %d' % k), folder,
                   lambda: shell_layer(source, data, k, folder, texture_cache))


def blender_materials(source, materials, folder):
    """Write ``rcra_materials.json`` for {glTF material name: material dict}; returns (path, {name: error})."""
    folder = Path(folder)
    texture_cache, graph_cache, entries, errors = {}, {}, {}, {}
    for name, material in materials.items():
        path = material.get('path')
        entry = cached_material_entry(source, path, folder / 'textures' / 'units', texture_cache, graph_cache) if path \
            else dict(error='no material asset')
        entries[name] = dict(entry, material=path)
        if 'error' in entry:
            errors[name] = entry['error']
    target = folder / 'rcra_materials.json'
    target.write_text(json.dumps(dict(schema=ENTRY_SCHEMA, materials=entries)), encoding='utf-8')
    return target, errors
