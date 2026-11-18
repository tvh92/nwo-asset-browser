"""Rift Apart glTF/OBJ export maps: the ``Game.material_maps`` hook.

Only what is decoded goes in. Every material gets the base colour (slot ABDAD780) and the DirectX tangent normal (slot
7AE34E7A) at full resolution, converted by the same pixel code as the R&C 2016 export (``export.models._convert_maps``).

F0, gloss and occlusion come from the material's own G-buffer writer (``decoded_maps``): its translated surface terms are
evaluated at random texels and inputs, and a term is used only when it is exactly a plain channel of the _g or _m map
sampled at UV0. The _g map then holds gloss² in red and a grey F0 in green, or F0 in rgb and gloss in alpha (picked by the
material's cb2 parameter 7107A2D8), and the _m map's blue is occlusion (docs/rift-apart.md, "_g and _m channels"). F0 and
gloss are converted to glTF metal/roughness with the Khronos spec-gloss conversion, with roughness 0.9725 - 0.7514 gloss so
that glTF's GGX alpha (roughness²) is the game's. Where a writer scales, remaps or mixes these channels, or does not
translate, the material keeps the neutral stand-in (non-metallic, fully rough) and its notes say why.

Emission, the graph-specific masks, blending and every other graph term are not carried into these formats; the Blender
formats translate them. Every material is exported opaque and single-sided.
"""
from __future__ import annotations

import random
from pathlib import Path

NOTE = ('Rift Apart portable material: base colour and normal; F0, gloss and occlusion only where the material\'s writer '
        'reads them as plain _g and _m channels. Emission, graph-specific masks and blending are not carried into this '
        'format (the Blender formats translate them), so the export is opaque.')
STAND_IN = 'no game F0 or gloss: exported non-metallic and fully rough (%s)'
CONVERSION = ('F0 and gloss from _g (%s), converted with the Khronos spec-gloss to metal-rough conversion; roughness '
              '0.9725 - 0.7514 gloss, so the GGX alpha is the game\'s. Lambert diffuse without the glTF energy split is '
              'not reproduced.')
LAYOUTS = {'grey': 'gloss² in red, grey F0 in green', 'colored': 'F0 in rgb, gloss in alpha'}
OCCLUSION_NOTE = 'occlusion from _m blue (glTF occlusion: the red channel of the metal-rough image)'
PROBES = 6
GLOSS_TOLERANCE = 0.5 / 255 + 1e-6   # the gloss term keeps the G-buffer's 8-bit rounding
F0_TOLERANCE = 0.005                 # F0 is stored as a rounded sqrt(F0) byte and read back squared
OCCLUSION_TOLERANCE = 1e-5           # occlusion is used before the target's rounding
UV_TOLERANCE = 1e-6


def exportable(fmt) -> bool:
    """Formats ``viewer.textures.to_image`` reads as stored: RGBA8, R8G8, R8 and the BC formats."""
    from rc2016.viewer.textures import GL_COMPRESSED, GL_UNCOMPRESSED
    return fmt in GL_COMPRESSED or fmt in GL_UNCOMPRESSED


def _slice_zero(args):
    """Whether a sample's address reads array slice 0: no slice lane, or one that rounds to 0 (NaN reads 0, as in D3D)."""
    if len(args) < 3 or not isinstance(args[2], float):
        return True
    return args[2] != args[2] or round(args[2]) == 0


def _probe(recipe, bindings, rng):
    """Evaluate the recipe once at random inputs and texels: (terms, {binding: texel}, {bindings read other than at UV0 and
    slice 0}). Every sample of a binding returns the same texel."""
    from rc2016.games.rcra import surface as rsurface
    inputs = {name: rng.random() for name in recipe['inputs']}
    inputs['SV_IsFrontFace.x'] = 0xFFFFFFFF
    uv = inputs.get('TEXCOORD.x'), inputs.get('TEXCOORD.y')
    texels, moved = {}, set()

    def sample(node, args):
        binding = node.get('binding')
        if binding in bindings and (None in uv or len(args) < 2 or abs(args[0] - uv[0]) > UV_TOLERANCE
                                    or abs(args[1] - uv[1]) > UV_TOLERANCE or not _slice_zero(args)):
            moved.add(binding)
        return texels.setdefault(binding, [rng.random() for _ in range(4)])
    return rsurface.evaluate_recipe(recipe, inputs, sample), texels, moved


def _plain(recipe, g_binding, m_binding, seed=7):
    """(g layout 'grey' | 'colored' | None, occlusion from _m blue, reason) of a surface recipe: a layout only when gloss and
    F0 equal the layout's _g channels at every probe, occlusion only when it equals _m blue, each sampled at UV0, slice 0."""
    usable = {b for b in (g_binding, m_binding) if b is not None}
    rng = random.Random(seed)
    grey = colored = occlusion = True
    for _ in range(PROBES):
        terms, texels, moved = _probe(recipe, usable, rng)
        g = texels.get(g_binding) if g_binding not in moved else None
        m = texels.get(m_binding) if m_binding not in moved else None
        f0 = [terms.get(k) for k in ('f0.r', 'f0.g', 'f0.b')]
        if g is None or None in f0 or terms.get('gloss') is None:
            grey = colored = False
        else:
            gloss = terms['gloss']
            grey = grey and abs(gloss - g[0] ** 0.5) <= GLOSS_TOLERANCE and all(abs(x - g[1]) <= F0_TOLERANCE for x in f0)
            colored = colored and abs(gloss - g[3]) <= GLOSS_TOLERANCE and all(abs(x - c) <= F0_TOLERANCE for x, c in zip(f0, g[:3]))
        occlusion = occlusion and m is not None and terms.get('occlusion') is not None and \
            abs(terms['occlusion'] - m[2]) <= OCCLUSION_TOLERANCE
    layout = 'grey' if grey and not colored else 'colored' if colored and not grey else None
    reason = None if layout else ('the writer has no _g slot' if g_binding is None else
                                  'the writer does not read F0 and gloss as plain _g channels at UV0')
    return layout, occlusion, reason


def decoded_maps(source, key):
    """dict(layout, occlusion, g, m, reason) for one material: the _g layout ('grey', 'colored' or None), whether occlusion is
    _m blue, the _g and _m texture paths the writer binds (graph defaults where the material binds none), and why there is
    no layout."""
    from rc2016.games.rcra import material as rmat, shaders as rshaders, viewer_materials as vm
    out = dict(layout=None, occlusion=False, g=None, m=None, reason=None)
    try:
        graph = vm._graph(source, key)
        surface, _forward, _blend, _limitations, textures = vm.translate(source, key)
    except (ValueError, KeyError, OSError) as error:   # NotTranslated and Unsupported are ValueErrors
        out['reason'] = 'the writer does not translate: %s' % error
        return out
    if surface is None:
        out['reason'] = 'the material draws no G-buffer surface'
        return out
    if graph is None:   # a fur material: the engine fur writer's gloss and control maps (rc2016.games.rcra.fur)
        g_binding, m_binding = 'g_FurGlossMap', 'g_FurControlMap'
    else:
        rows, _offsets, _defaults = rshaders._layout(graph)
        binding = {k: 'textureUnit%d' % (rshaders.FIRST_MATERIAL_UNIT + u) for u, k, _p in rows}
        g_binding, m_binding = binding.get(rmat.SLOT_G), binding.get(rmat.SLOT_MASK)
    out['layout'], out['occlusion'], out['reason'] = _plain(surface, g_binding, m_binding)
    out['g'] = textures.get(g_binding) if out['layout'] else None
    out['m'] = textures.get(m_binding) if out['occlusion'] else None
    out['occlusion'] = out['m'] is not None
    return out


def material_maps(source, material, cache, map_cache=None):
    from rc2016.export.models import _convert_maps
    from rc2016.games.rcra.material import PREVIEW_ROLES, _material_key, material_slots
    from rc2016.games.rcra.texture import load_texture_full
    name = material.get('name') or (Path(material['path']).stem if material.get('path') else f"material_{material.get('index', 0)}")
    out = dict(name=name, base=None, mr=None, normal=None, emissive=None, emissive_strength=0.0,
               alpha_mode='OPAQUE', alpha=1.0, double_sided=False, occlusion=False, notes=[], sources={}, png={})
    if not material.get('path'):
        out['notes'].append('no material file; neutral grey')
        out['base_factor'] = [0.5, 0.5, 0.5]
        return out
    paths = {}
    decoded = dict(layout=None, occlusion=False, g=None, m=None, reason='material unreadable')
    try:
        key = _material_key(source, material['path'])
        if key is None:
            raise ValueError('no single material asset for this path')
        slots = dict(material_slots(source.asset_data(key)))
        paths = {role: slots[slot] for slot, role in PREVIEW_ROLES.items() if slots.get(slot)}
        decoded = decoded_maps(source, key)
    except (ValueError, KeyError, OSError) as error:
        out['notes'].append(f'material unreadable: {error}')
    if decoded['g']:
        paths['_g'] = decoded['g']
    if decoded['m']:
        paths['_m'] = decoded['m']

    def texture(role):
        path = paths.get(role)
        if not path:
            return None
        if path not in cache:
            try:
                loaded = load_texture_full(source, path)
                if not exportable(loaded['format']):
                    # Float/16-bit data formats (e.g. the 4x4 R32G32_FLOAT default_pink_shader_n placeholder): how
                    # their values map to 8-bit is not decoded, so they are left out instead of guessed.
                    raise ValueError(f"DXGI format {loaded['format']} has no decoded 8-bit conversion")
                cache[path] = loaded
            except (ValueError, KeyError, OSError) as error:
                cache[path] = None
                out['notes'].append(f'{role} texture {path} left out: {error}')
        if cache[path] is not None:
            out['sources'][role] = path
        return cache[path]

    c_tex, n_tex = texture('_c'), texture('_n')
    g_tex = texture('_g') if decoded['layout'] else None
    m_tex = texture('_m') if decoded['occlusion'] else None
    if c_tex is None:
        out['base_factor'] = [0.8, 0.8, 0.8]
        out['notes'].append('no base colour texture: white-grey placeholder')
    options = dict(g_mode=decoded['layout'] if g_tex is not None else 'colored', k=0.0, spec=1.0, gloss=1.0,
                   roughness='rcra', occlusion='blue' if m_tex is not None else None)
    textured = any(t is not None for t in (c_tex, n_tex, g_tex, m_tex))
    cache_key = ('rcra', paths.get('_c'), paths.get('_n'), paths.get('_g') if g_tex is not None else None,
                 options['g_mode'], paths.get('_m') if m_tex is not None else None)
    entry = map_cache.get(cache_key) if map_cache is not None and textured else None
    if entry is None:
        entry = _convert_maps(c_tex, g_tex, n_tex, options, o_tex=m_tex)
        entry['png'] = {}
        if map_cache is not None and textured:
            map_cache.put(cache_key, entry)
    for field in ('base', 'mr', 'normal', 'emissive', 'emissive_strength', 'occlusion', 'png'):
        out[field] = entry[field]
    # Rift Apart notes instead of _convert_maps' 2016 wording.
    out['notes'].append(CONVERSION % LAYOUTS[decoded['layout']] if g_tex is not None else
                        STAND_IN % (decoded['reason'] or 'its _g texture is left out'))
    if out['occlusion']:
        out['notes'].append(OCCLUSION_NOTE)
    out['notes'].append(NOTE)
    return out
