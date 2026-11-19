"""Rift Apart translated materials for the viewer: a receipt (rc2016.viewer.rcra_glsl) and its textures, per material.

The receipt holds the material's G-buffer surface recipe (rc2016.games.rcra.surface.surface), its forward recipe when the
game draws it blended (surface.forward, by material.material_blend), the blend mode and each texture binding's layout. The
choices follow the Blender export (rc2016.games.rcra.blender_export.material_entry): a blended material whose forward
program does not translate keeps its opaque surface; additive and modulate materials take no lit surface; a modulate
material that passes the scene behind it through is not linear in it and keeps its surface.

Textures are viewer texture dicts (rc2016.viewer.textures shape). A binding sampled at one array slice uploads that slice as
stored; a runtime-chosen slice or a 3D texture uploads a padded atlas of every slice (blender_export.volume_atlas), without
mipmaps so tiles never mix. A cube map is a texture dict whose ``cube`` holds its six face dicts (+X, -X, +Y, -Y, +Z, -Z, top
row first, as stored); the viewport uploads them as a GL cube map at the top mip. Float and 16-bit data formats, which the viewer's block uploads do not take, upload as float
RGBA (``gl_float``) with the stored values, missing channels following D3D (blender_export.texel_array).
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

import numpy as np

from rc2016.games.rcra import blender_export as bexp, surface as rsurface
from rc2016.viewer import rcra_glsl

MAX_BINDINGS = 14   # rc2016.viewer.native_glsl.TEXTURE_UNITS
CUBE_LIMITATION = ('Viewer: cube maps are the six stored faces in a GL cube map, looked up along the program\'s direction; '
                   'exact apart from filtering across face edges (GL seamless cube maps where the driver has them).')
RGBA8, RGBA8_SRGB, GL_RGBA = 28, 29, 0x1908


class NotTranslated(ValueError):
    pass


def _graph(source, material_key):
    """The material's graph bytes, or None for a fur material (no graph: rc2016.games.rcra.fur)."""
    from rc2016.games.rcra.fur import fur_record
    from rc2016.games.rcra.material import _graphs, no_graph_reason
    graphs = list(_graphs(source, material_key))
    if not graphs:
        if fur_record(source.asset_data(material_key)) is not None:
            return None
        raise NotTranslated(no_graph_reason(source.asset_data(material_key)))
    graph_key = next((k for k in source.resolve(graphs[0]) if source.entries[k]['type'] == 'materialgraph'), None)
    if graph_key is None:
        raise NotTranslated('graph not found: ' + graphs[0])
    return source.asset_data(graph_key)


def _code_sha256():
    """The sources a translated receipt depends on: the Rift Apart reader and translation, the SM5 translator and the viewer
    emitter. Any change to them invalidates every cached translation."""
    from pathlib import Path
    package = Path(__file__).resolve().parents[2]
    files = sorted((package / 'games' / 'rcra').glob('*.py')) + [package / 'shaders' / 'sm5.py', package / 'shaders' / 'dxbc.py',
                                                                 package / 'viewer' / 'rcra_glsl.py']
    try:
        return hashlib.sha256(b''.join(f.name.encode() + f.read_bytes() for f in files)).hexdigest()
    except OSError:   # frozen executable without sources: no disk cache
        return None


CODE_SHA256 = _code_sha256()


def translate(source, material_key):
    """(surface recipe or None, forward recipe or None, blend, limitations, textures {binding: path}), kept across sessions in
    the source's translation disk cache (keyed by the material and graph bytes and CODE_SHA256) when it has one."""
    graph, data = _graph(source, material_key), source.asset_data(material_key)
    program = None
    if graph is None:
        from rc2016.games.rcra.fur import fur_program
        try:
            program = fur_program(source, data)
        except (ValueError, OSError) as error:
            raise NotTranslated(str(error)) from error
    disk = getattr(source, 'translation_disk_cache', None)
    program_bytes = bytes(graph) if program is None else program['key']
    key = disk.key(bytes(data), [program_bytes], 'rcra-viewer:%s' % CODE_SHA256) if disk is not None and CODE_SHA256 else None
    stored = _floats(disk.get(key), decode=True) if key else None
    if stored is not None:
        if 'error' in stored:
            raise NotTranslated(stored['error'])
        return stored['surface'], stored['forward'], stored['blend'], stored['limitations'], stored['textures']
    try:
        out = _translate(graph, data, source, program)
    except NotTranslated as error:
        if key:
            disk.put(key, dict(error=str(error)))
        raise
    if key:
        disk.put(key, _floats(dict(zip(('surface', 'forward', 'blend', 'limitations', 'textures'), out))))
    return out


NON_FINITE = '__nonfinite__'


def _floats(value, decode=False):
    """``value`` with each inf or NaN float as {NON_FINITE: 'inf' | '-inf' | 'nan'} (recipes keep D3D's inf and NaN
    constants; the cache stores strict JSON), or back with ``decode``."""
    if not decode and isinstance(value, float) and not math.isfinite(value):
        return {NON_FINITE: repr(value)}
    if isinstance(value, dict):
        if decode and len(value) == 1 and NON_FINITE in value:
            return float(value[NON_FINITE])
        return {k: _floats(v, decode) for k, v in value.items()}
    if isinstance(value, list):
        return [_floats(v, decode) for v in value]
    if isinstance(value, tuple):
        return tuple(_floats(v, decode) for v in value)
    return value


def _translate(graph, data, source, program=None):
    from rc2016.games.rcra.material import material_blend
    from rc2016.games.rcra.texture import texture_size
    blend = material_blend(data)
    results, errors = {}, {}
    passes = (('forward', rsurface.forward), ('surface', rsurface.surface)) if blend != 'opaque' else \
        (('surface', rsurface.surface),)
    if program is not None:   # an engine program (fur) replaces the graph: its G-buffer writer only
        passes = (('surface', lambda g, d, **k: rsurface.surface(g, d, program=program, **k)),)
        errors['forward'] = 'an engine program without a forward pass'
    for name, fn in passes:
        try:
            # Programs that query a texture's size (resinfo) need it, as in the Blender export.
            results[name] = fn(graph, data, texture_sizes=lambda path: texture_size(source, path))
        except (rsurface.Unsupported, ValueError, KeyError) as error:
            errors[name] = str(error)
    for result in results.values():
        bexp.replace_missing(source, graph, result)
    # Constant NaN resolved by D3D rules (as the Blender export does): GLSL's min, max and clamp of NaN are undefined.
    recipes = {name: rsurface.fold_nan(rsurface.export_recipe(r)) for name, r in results.items()}
    if blend == 'modulate' and 'forward' in recipes and bexp._reads_background(recipes['forward']):
        errors['forward'] = 'a modulate material that passes the scene behind through is not linear in it'
        del recipes['forward'], results['forward']
    limitations = [rcra_glsl.LIMITATION]
    if blend != 'opaque' and 'forward' not in recipes:
        limitations.append('The game draws this material %s; its forward program does not translate (%s), so the viewer '
                           'draws its opaque surface.' % (blend, errors.get('forward')))
        blend = 'opaque'
    if blend in ('additive', 'modulate'):
        recipes.pop('surface', None)
        results.pop('surface', None)
    if not recipes or (blend == 'opaque' and 'surface' not in recipes):
        raise NotTranslated(errors.get('surface') or errors.get('forward') or 'nothing translates')
    if blend != 'opaque':
        limitations.append(bexp.BLEND_LIMITATIONS[blend])
    if 'forward' in recipes and blend in ('blended', 'additive') and not rsurface.reads_scene(recipes['forward']):
        try:   # the program's own key-light term, drawn once per review light (rcra_glsl)
            lit = rsurface.forward_lighting(graph, data, texture_sizes=lambda path: texture_size(source, path))
        except (rsurface.Unsupported, ValueError, KeyError):
            lit = None
        if lit is not None:
            bexp.replace_missing(source, graph, lit)
            recipes['forward']['lit'] = rsurface.fold_nan(rsurface.export_recipe(lit))
            results['forward_lit'] = lit
    textures = {}
    for result in results.values():
        textures.update(result['textures'])
        limitations += result['limitations']
    return recipes.get('surface'), recipes.get('forward'), blend, limitations, textures


def _layouts(recipes):
    """{binding: slice index | ALL_SLICES | VOLUME | (CUBE, index)} over the recipes (blender_export._slices)."""
    slices = {}
    for recipe in recipes:
        try:
            layouts = bexp._slices(recipe)
        except rsurface.Unsupported as error:
            raise NotTranslated(str(error)) from error
        for binding, index in layouts.items():
            merged = bexp.merge_layouts(slices.setdefault(binding, index), index)
            if merged is None:
                raise NotTranslated('texture %s is sampled as a volume or a cube map and differently' % binding)
            slices[binding] = merged
    return slices


def _cube(texture, path, index):
    """(cube texture dict, cube index): the six faces of cube ``index`` as face dicts, clamped to the cubes in the texture."""
    slices = texture.get('slices', [texture['pixels']])
    if texture.get('faces', 1) != 6:
        raise NotTranslated('texture %s is sampled as a cube map but is not one' % path)
    index = min(max(index, 0), len(slices) // 6 - 1)
    faces = [_slice(texture, '%s#cube%d.%d' % (path, index, k), index * 6 + k) for k in range(6)]
    cube = dict(path='%s#cube%d' % (path, index), width=texture['width'], height=texture['height'], format=texture['format'],
                srgb=bool(texture.get('srgb')), cube=faces)
    return cube, index


def _float(data, path, **extra):
    """A float RGBA viewer texture dict of (h, w, 4) stored values."""
    return dict(path=path, width=int(data.shape[1]), height=int(data.shape[0]), format=2, gl_format=None, gl_pixel_format=None,
                gl_float=True, pixels=np.ascontiguousarray(data, '<f4').tobytes(), srgb=False, **extra)


ATLAS_CACHE_BYTES = 256 << 20   # atlases kept per source: materials share their texture arrays


def _atlas(texture, path, source=None):
    """_build_atlas, memoised on ``source`` (least recently used out past ATLAS_CACHE_BYTES)."""
    if source is None:
        return _build_atlas(texture, path)
    memo = source.__dict__.setdefault('_rcra_viewer_atlases', OrderedDict())
    key = (path, texture['width'], texture['height'], texture['format'], len(texture.get('slices', [texture['pixels']])))
    if key in memo:
        memo.move_to_end(key)
        return memo[key]
    memo[key] = _build_atlas(texture, path)
    total = sum(len(out['pixels']) for out, _grid in memo.values())
    while len(memo) > 1 and total > ATLAS_CACHE_BYTES:
        _key, (out, _grid) = memo.popitem(last=False)
        total -= len(out['pixels'])
    return memo[key]


def _texel_bytes(texture, index):
    """Uint8 (height, width, 4) of an 8-bit slice: bexp.texel_array's values times 255, read without the float round trip
    (missing channels 0, alpha 255; luminance + alpha as red + alpha; no blue for R8G8 and BC5)."""
    from rc2016.viewer.textures import to_image
    width, height, fmt = texture['width'], texture['height'], texture['format']
    image = to_image(texture, index)
    channels = {'L': 1, 'LA': 2, 'RGB': 3, 'RGBA': 4}.get(image.mode)
    if channels is None:
        image, channels = image.convert('RGBA'), 4
    data = np.asarray(image, dtype=np.uint8).reshape(height, width, channels)
    out = np.zeros((height, width, 4), np.uint8)
    out[..., 3] = 255
    if image.mode == 'LA':
        out[..., 0], out[..., 3] = data[..., 0], data[..., 1]
        return out
    out[..., :channels] = data[..., :4]
    if fmt == 49 or (channels == 3 and fmt in (83, 84)):
        out[..., 2] = 0
    return out


def _build_atlas(texture, path):
    """A padded atlas texture dict of every slice of ``texture`` (RGBA8, or float for float formats), and its grid."""
    count = len(texture.get('slices', [texture['pixels']]))
    columns = max(1, int(np.ceil(np.sqrt(count))))
    if texture['format'] in bexp.FLOAT_FORMATS:
        atlas = bexp.volume_atlas([bexp.texel_array(texture, k) for k in range(count)], columns)
        out = _float(atlas, path + '#atlas', no_mipmaps=True)
    else:
        srgb = bool(texture.get('srgb'))
        atlas = bexp.volume_atlas([_texel_bytes(texture, k) for k in range(count)], columns)
        pixels = atlas.tobytes()
        out = dict(path=path + '#atlas', width=int(atlas.shape[1]), height=int(atlas.shape[0]),
                   format=RGBA8_SRGB if srgb else RGBA8, gl_format=None, gl_pixel_format=GL_RGBA, pixels=pixels, srgb=srgb,
                   no_mipmaps=True)
    grid = dict(depth=count, columns=columns, width=int(texture['width']), height=int(texture['height']))
    return out, grid


def _slice(texture, path, index):
    slices = texture.get('slices', [texture['pixels']])
    index = bexp.slice_index(texture, index)   # D3D clamps an array index to the slices
    if texture['format'] in bexp.FLOAT_FORMATS:
        return _float(bexp.texel_array(texture, index), '%s#float%d' % (path, index))
    if index == 0:
        return texture
    return dict(texture, pixels=slices[index], path='%s#slice%d' % (path, index))


def _load_bindings(source, recipes, paths, hd, limitations):
    """({binding: layout}, {binding: viewer texture dict}) of the textures ``recipes`` sample."""
    from rc2016.games.rcra.texture import load_texture, load_texture_full
    kinds = _layouts(recipes)
    if len(paths) > MAX_BINDINGS:
        raise NotTranslated('%d texture bindings, the viewer has %d units' % (len(paths), MAX_BINDINGS))
    layouts, textures = {}, {}
    for binding, path in sorted(paths.items()):
        try:
            texture = load_texture(source, path, hd=hd)
        except ValueError:
            try:   # a data format the viewer previews nowhere else: read as stored, uploaded as float
                texture = load_texture_full(source, path)
            except (ValueError, KeyError, OSError) as error:
                raise NotTranslated('texture %s: %s' % (path, error)) from error
            if texture['format'] not in bexp.FLOAT_FORMATS:
                raise NotTranslated('texture %s: format %d is not previewable' % (path, texture['format']))
        except (KeyError, OSError) as error:
            raise NotTranslated('texture %s: %s' % (path, error)) from error
        index = kinds.get(binding, 0)
        if isinstance(index, tuple):
            textures[binding], cube = _cube(texture, path, index[1])
            layouts[binding] = dict(kind='cube', slice=cube)
            if CUBE_LIMITATION not in limitations:
                limitations.append(CUBE_LIMITATION)
        elif index in (bexp.ALL_SLICES, bexp.VOLUME):
            textures[binding], grid = _atlas(texture, path, source)
            layouts[binding] = dict(grid, kind='volume' if index == bexp.VOLUME else 'array')
        else:
            textures[binding] = _slice(texture, path, index)
            layouts[binding] = dict(kind='slice')
    return layouts, textures


def material_receipt(source, material_key, hd=False):
    """(receipt, {binding: viewer texture dict}); raises NotTranslated with the reason."""
    surface, forward, blend, limitations, paths = translate(source, material_key)
    recipes = [r for r in (surface, forward, (forward or {}).get('lit')) if r]
    layouts, textures = _load_bindings(source, recipes, paths, hd, limitations)
    emission = surface['terms']['emission'] if surface else forward['terms']['colour.r']
    recipe = surface or forward
    lit = class_lighting(source, surface, limitations) if surface else None
    receipt = dict(kind='rcra', status='ready', blend=blend, surface=surface, forward=forward, layouts=layouts,
                   limitations=limitations, nodes=recipe['nodes'], outputs=dict(emission=emission), lighting=lit,
                   translation_sha256=rcra_glsl.receipt_sha256(surface, forward, blend, layouts, lit))
    return receipt, textures


def recipe_class(recipe):
    """The lighting class a surface recipe stores (its constant ``class`` term), 0 without one."""
    index = recipe['terms'].get('class')
    return int(recipe['nodes'][index]['value']) if index is not None else 0


def class_lighting(source, surface, limitations):
    """The direct-light recipe of the surface's lighting class (rc2016.games.rcra.lighting.direct), appending its
    limitation to ``limitations``; None, with a limitation, when the executable's apply program does not translate."""
    from rc2016.games.rcra import lighting
    cls = recipe_class(surface)
    try:
        recipe = lighting.direct(source.root, cls, disk=getattr(source, 'translation_disk_cache', None))
    except (OSError, ValueError, rsurface.Unsupported) as error:
        limitations.append('Viewer: the class-%d apply program is not available (%s); lit by the class-0 BRDF.' % (cls, error))
        return None
    limitations += recipe['limitations']
    return recipe


SHELL_BLEND = 'fur_shell'   # rcra_glsl: the surface lit, its ``coverage`` as alpha


def translate_shell(source, material_key):
    """A fur material's shell recipe, every layer at once (rc2016.games.rcra.fur.shell_symbolic_program): (recipe, fur
    values, limitations, textures), or None for a material without a fur record. Cached on disk as translate."""
    from rc2016.games.rcra.fur import fur_record, shell_symbolic_program
    from rc2016.games.rcra.texture import texture_size
    data = source.asset_data(material_key)
    if fur_record(data) is None:
        return None
    try:
        program = shell_symbolic_program(source, data)
    except (ValueError, OSError) as error:
        raise NotTranslated(str(error)) from error
    disk = getattr(source, 'translation_disk_cache', None)
    key = disk.key(bytes(data), [program['key']], 'rcra-viewer-shell:%s' % CODE_SHA256) if disk is not None and CODE_SHA256 \
        else None
    stored = _floats(disk.get(key), decode=True) if key else None
    if stored is not None:
        if 'error' in stored:
            raise NotTranslated(stored['error'])
        return stored['recipe'], stored['fur'], stored['limitations'], stored['textures']
    try:
        result = rsurface.surface(None, data, program=program, texture_sizes=lambda path: texture_size(source, path))
    except (rsurface.Unsupported, ValueError, KeyError) as error:
        if key:
            disk.put(key, dict(error=str(error)))
        raise NotTranslated(str(error)) from error
    bexp.replace_missing(source, None, result)
    out = rsurface.fold_nan(rsurface.export_recipe(result)), program['fur'], result['limitations'], result['textures']
    if key:
        disk.put(key, _floats(dict(zip(('recipe', 'fur', 'limitations', 'textures'), out))))
    return out


def shell_receipt(source, material_key, hd=False):
    """(receipt, textures) of a fur material's shells (blend SHELL_BLEND, ``fur`` the shell vertex program's values), or
    None without a fur record; raises NotTranslated."""
    out = translate_shell(source, material_key)
    if out is None:
        return None
    recipe, fur, limitations, paths = out
    limitations = [rcra_glsl.LIMITATION] + list(limitations)
    layouts, textures = _load_bindings(source, [recipe], paths, hd, limitations)
    lit = class_lighting(source, recipe, limitations)
    receipt = dict(kind='rcra', status='ready', blend=SHELL_BLEND, surface=recipe, forward=None, layouts=layouts,
                   limitations=limitations, nodes=recipe['nodes'], outputs=dict(emission=recipe['terms']['emission']),
                   fur=fur, lighting=lit, translation_sha256=rcra_glsl.receipt_sha256(recipe, None, SHELL_BLEND, layouts, lit))
    return receipt, textures


def cached_shell_receipt(source, material_key, hd=False):
    """shell_receipt memoised on the source: (receipt, textures), (None, reason), or None for a material without fur."""
    memo = source.__dict__.setdefault('_rcra_viewer_shells', {})
    key = (material_key, bool(hd))
    if key not in memo:
        try:
            memo[key] = shell_receipt(source, material_key, hd)
        except NotTranslated as error:
            memo[key] = (None, str(error))
    return memo[key]


def cached_receipt(source, material_key, hd=False):
    """material_receipt memoised on the source: (receipt, textures) or (None, reason)."""
    memo = source.__dict__.setdefault('_rcra_viewer_receipts', {})
    key = (material_key, bool(hd))
    if key not in memo:
        try:
            memo[key] = material_receipt(source, material_key, hd)
        except NotTranslated as error:
            memo[key] = (None, str(error))
    return memo[key]
