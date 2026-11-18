"""Rift Apart materials for Blender: each material's surface recipe and its unit textures, written beside an export.

``blender_materials`` is the ``Game.blender_materials`` hook. For every material it translates the graph's G-buffer
writer (rc2016.games.rcra.surface), writes the textures the recipe samples and returns a JSON-ready entry that
rc2016/export/rcra_blender.py turns into a node tree. A material whose writer is not supported gets an entry with its
reason and keeps the glTF importer's material.

A blended material (rc2016.games.rcra.material.material_blend: additive, blended or modulate) also gets its forward program
translated (rc2016.games.rcra.surface.forward) as ``forward``: the colour and alpha the game blends, with every light source
zero. Its G-buffer surface, when it translates, stays as ``recipe`` and is lit by the renderer. A blended material whose
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
"""
import json
import math
import re
from pathlib import Path

import numpy as np

from rc2016.formats.linear_png import linear_to_srgb as _linear_to_srgb, png16, srgb_to_linear as _srgb_to_linear
from rc2016.games.rcra import surface as rsurface
from rc2016.games.rcra.texture import texture_size

SRGB_FORMATS = frozenset({29, 72, 75, 78, 91, 93, 99})
# DXGI float/integer formats as (numpy dtype, channels).
FLOAT_FORMATS = {2: ('<f4', 4), 10: ('<f2', 4), 16: ('<f4', 2), 34: ('<f2', 2), 41: ('<f4', 1), 54: ('<f2', 1),
                 11: ('<u2', 4), 35: ('<u2', 2), 56: ('<u2', 1)}
UNORM16 = frozenset({11, 35, 56})
ENTRY_SCHEMA = 'rcra-blender-materials-v2'
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
CUBE_LIMITATION = 'Cube map %s resampled to an equirectangular image (filtering differs at face seams and poles).'
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
        from rc2016.viewer.textures import to_image
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
    from rc2016.games.rcra.texture import load_texture_full
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
    from rc2016.games.rcra.texture import load_texture_full
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
    from rc2016.games.rcra.texture import load_texture_full
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
    return dict(path=str(target), colourspace=colourspace, kind=kind, width=int(data.shape[1]), height=int(data.shape[0]),
                source=path, format=int(texture['format']))


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
    from rc2016.games.rcra.texture import texture_header
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


def replace_missing(source, graph_data, result):
    """Replace in a translation result each texture the install does not hold: by the graph's default for that unit when
    the install has it, else by texture.UNBOUND (reads 0, as an unbound D3D resource). The result's limitations say which;
    the game's own fallback for a texture it cannot load is not known."""
    from rc2016.games.rcra import shaders as rshaders
    from rc2016.games.rcra.texture import UNBOUND

    def present(path):
        return path == UNBOUND or sum(source.entries[k]['type'] == 'texture' for k in source.resolve(path)) == 1
    defaults = {}
    if graph_data is not None:
        rows, _offsets, _defaults = rshaders._layout(graph_data)
        defaults = {'textureUnit%d' % (rshaders.FIRST_MATERIAL_UNIT + unit): re.sub('/+', '/', default)
                    for unit, _key, default in rows if default}
    notes = []
    for binding, path in sorted(result['textures'].items()):
        if present(path):
            continue
        default = defaults.get(binding)
        stand_in = default if default and default != path and present(default) else UNBOUND
        result['textures'][binding] = stand_in
        notes.append('%s (%s): %s' % (path, binding, 'the graph default %s' % stand_in if stand_in != UNBOUND
                                       else 'reads 0, as an unbound texture'))
    if notes:
        result['limitations'].append(MISSING_NOTE % '; '.join(notes))


MISSING_NOTE = 'Textures the install does not hold, with their stand-ins: %s.'


def _reads_background(recipe):
    """True when a forward recipe's background tint is not the constant 0."""
    nodes = recipe['nodes']
    return any(nodes[recipe['terms'][k]]['op'] != 'const' or nodes[recipe['terms'][k]]['value']
               for k in ('background.r', 'background.g', 'background.b') if k in recipe['terms'])


def material_entry(source, material_path, folder, texture_cache=None, graph_cache=None):
    """The Blender entry of one material: dict(blend, recipe, forward, images, limitations), or dict(error)."""
    from rc2016.games.rcra.material import _graphs, _material_key, material_blend
    key = _material_key(source, material_path)
    if key is None:
        return dict(error='material asset not found')
    graphs = list(_graphs(source, key))
    program = None
    if not graphs:   # a fur material: the engine's fur writer (rc2016.games.rcra.fur)
        from rc2016.games.rcra.fur import EXECUTABLE, fur_program, fur_record
        from rc2016.games.rcra.material import no_graph_reason
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
    errors, results, limitations = {}, {}, []
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
        else:
            limitations.append('The game draws this material %s in its forward pass, which does not translate (%s): '
                               'Blender gets its opaque G-buffer surface.' % (blend, errors['forward']))
    if 'recipe' in results:
        limitations += results['recipe']['limitations']
    limitations += [CUBE_LIMITATION % binding for binding, row in sorted(images.items()) if 'cube' in row]
    limitations += [
        'Tangent frame: the decoded vertex tangent and TEXCOORD3.w, carried as coordinates in Blender\'s UV0 tangent '
        'frame so they turn with the armature; the draw is unmirrored (render flag 0x10000 clear).',
        'Texture filtering is Blender\'s linear repeat; the game\'s sampler states are not in the program.']
    entry = dict(graph=graph_path, blend=blend if 'forward' in results else 'opaque', game_blend=blend, images=images,
                 limitations=limitations)
    entry.update(recipes)
    return entry


def blender_materials(source, materials, folder):
    """Write ``rcra_materials.json`` for {glTF material name: material dict}; returns (path, {name: error})."""
    folder = Path(folder)
    texture_cache, graph_cache, entries, errors = {}, {}, {}, {}
    for name, material in materials.items():
        path = material.get('path')
        entry = material_entry(source, path, folder / 'textures' / 'units', texture_cache, graph_cache) if path else \
            dict(error='no material asset')
        entries[name] = dict(entry, material=path)
        if 'error' in entry:
            errors[name] = entry['error']
    target = folder / 'rcra_materials.json'
    target.write_text(json.dumps(dict(schema=ENTRY_SCHEMA, materials=entries)), encoding='utf-8')
    return target, errors
