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

Emission and alpha are checked the same way (``decoded_emission``, ``decoded_alpha``). The writer's emission E (self-lit
albedo x E) is carried when the albedo is plain _c and E is a constant or a constant times one channel of _c, _g or _m:
the emissive map is _c x that channel, at strength K (KHR_materials_emissive_strength). An alpha-over material (flags blend
"blended") is exported with glTF BLEND when its forward program's alpha is a constant or a constant times _c alpha.
Additive and modulate materials have no glTF blend and stay opaque. The graph-specific masks and every other graph term
are not carried into these formats; the Blender formats translate them. Every material is single-sided: the culling state
is not decoded.
"""
from __future__ import annotations

import random
from pathlib import Path

NOTE = ('Rift Apart portable material: base colour and normal; F0, gloss, occlusion, emission and alpha-over alpha only '
        'where the material\'s programs read them as plain texture channels or constants. Graph-specific masks and other '
        'terms are not carried into this format (the Blender formats translate them).')
EMISSION_NOTE = 'emission from the writer: _c x %s x %g (self-lit albedo x E)'
NO_EMISSION_NOTE = 'emission not carried: %s'
ALPHA_NOTE = 'alpha-over (glTF BLEND): alpha = %s from the forward program'
NO_ALPHA_NOTE = 'drawn %s in the game; exported opaque: %s'
RATIO_TOLERANCE = 1e-4
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


def _ratio(pairs):
    """K when every (value, channel) pair has value = K x channel (channel None: value = K), K > 0; else None."""
    k = None
    for value, channel in pairs:
        if channel is None:
            ratio = value
        elif abs(channel) < 1e-3:
            if abs(value) > RATIO_TOLERANCE:
                return None
            continue
        else:
            ratio = value / channel
        if k is None:
            k = ratio
        elif abs(ratio - k) > RATIO_TOLERANCE * max(1.0, abs(k)):
            return None
    return float('%.6g' % k) if k is not None and k > 0 else None   # six digits: the probes' float32 noise


def _channel_form(probes, term, candidates):
    """(binding, channel, K) of the first candidate with term = K x that texel channel at every probe, or (None, None, K)
    for a constant K; None when no form holds. ``probes`` are _probe results; a moved binding is not a candidate."""
    values = [p[0].get(term) for p in probes]
    if None in values:
        return None
    k = _ratio([(v, None) for v in values])
    if k is not None:
        return None, None, k
    for binding, channel in candidates:
        if binding is None or any(binding in p[2] or binding not in p[1] for p in probes):
            continue
        k = _ratio([(v, p[1][binding][channel]) for v, p in zip(values, probes)])
        if k is not None:
            return binding, channel, k
    return None


def decoded_emission(recipe, c_binding, g_binding, m_binding, seed=11):
    """(dict(binding, channel, k) or None, reason): the writer's emission E as K x one channel of _c (alpha), _g or _m, or a
    constant K, with the albedo plain _c at UV0; (None, None) when E is 0 at every probe."""
    rng = random.Random(seed)
    usable = {b for b in (c_binding, g_binding, m_binding) if b is not None}
    probes = [_probe(recipe, usable, rng) for _ in range(PROBES)]
    values = [p[0].get('emission') for p in probes]
    if None in values:
        return None, 'the emission term does not evaluate'
    if all(v == 0 for v in values):
        return None, None
    if c_binding is None or any(c_binding in p[2] or c_binding not in p[1] for p in probes) or not all(
            abs(p[0].get('albedo.' + c, -1) - p[1][c_binding][i]) <= 1e-5 for p in probes for i, c in enumerate('rgb')):
        return None, 'the self-lit albedo is not plain _c at UV0'
    candidates = [(c_binding, 3)] + [(b, i) for b in (g_binding, m_binding) for i in range(4)]
    form = _channel_form(probes, 'emission', candidates)
    if form is None:
        return None, 'the emission is not a constant times one _c, _g or _m channel at UV0'
    binding, channel, k = form
    return dict(binding=binding, channel=channel, k=k), None


def decoded_alpha(recipe, c_binding, seed=13):
    """(dict(binding, k) or None, reason): a forward program's alpha as K x _c alpha (binding c_binding) or a constant K."""
    rng = random.Random(seed)
    probes = [_probe(recipe, {c_binding} if c_binding else set(), rng) for _ in range(PROBES)]
    form = _channel_form(probes, 'alpha', [(c_binding, 3)])
    if form is None:
        return None, 'the forward alpha is not a constant times _c alpha at UV0'
    binding, _channel, k = form
    if binding is None and k >= 1.0:
        return None, 'the forward alpha is 1'
    return dict(binding=binding, k=min(k, 1.0)), None


def _label(binding, channel, names):
    return '%s.%s' % (names.get(binding, binding), 'rgba'[channel])


def decoded_maps(source, key):
    """dict(layout, occlusion, g, m, reason, emission, emission_reason, alpha, alpha_reason, blend) for one material: the _g
    layout ('grey', 'colored' or None), whether occlusion is _m blue, the _g and _m texture paths the writer binds (graph
    defaults where the material binds none), and why there is no layout; the emission form (decoded_emission, with the
    mask texture path) and the alpha-over alpha form (decoded_alpha) with their reasons."""
    from rc2016.games.rcra import material as rmat, shaders as rshaders, viewer_materials as vm
    out = dict(layout=None, occlusion=False, g=None, m=None, reason=None, emission=None, emission_reason=None,
               alpha=None, alpha_reason=None, blend='opaque')
    try:
        graph = vm._graph(source, key)
        surface, forward, blend, _limitations, textures = vm.translate(source, key)
    except (ValueError, KeyError, OSError) as error:   # NotTranslated and Unsupported are ValueErrors
        out['reason'] = 'the writer does not translate: %s' % error
        return out
    out['blend'] = blend
    if graph is None:   # a fur material: the engine fur writer's maps (rc2016.games.rcra.fur)
        c_binding, g_binding, m_binding = 'g_BaseMap', 'g_FurGlossMap', 'g_FurControlMap'
    else:
        rows, _offsets, _defaults = rshaders._layout(graph)
        binding = {k: 'textureUnit%d' % (rshaders.FIRST_MATERIAL_UNIT + u) for u, k, _p in rows}
        c_binding, g_binding = binding.get(rmat.SLOT_BASE_COLOUR), binding.get(rmat.SLOT_G)
        m_binding = binding.get(rmat.SLOT_MASK)
    names = {c_binding: '_c', g_binding: '_g', m_binding: '_m'}
    if blend == 'blended':
        if forward is None:
            out['alpha_reason'] = 'the forward program does not translate'
        else:
            out['alpha'], out['alpha_reason'] = decoded_alpha(forward, c_binding)
            if out['alpha'] is not None:
                out['alpha']['label'] = ('%g' % out['alpha']['k'] if out['alpha']['binding'] is None else
                                         '%g x _c.a' % out['alpha']['k'])
    elif blend in ('additive', 'modulate'):
        out['alpha_reason'] = 'glTF has no %s blend' % blend
    if surface is None:
        out['reason'] = 'the material draws no G-buffer surface'
        return out
    out['layout'], out['occlusion'], out['reason'] = _plain(surface, g_binding, m_binding)
    out['g'] = textures.get(g_binding) if out['layout'] else None
    out['m'] = textures.get(m_binding) if out['occlusion'] else None
    out['occlusion'] = out['m'] is not None
    emission, out['emission_reason'] = decoded_emission(surface, c_binding, g_binding, m_binding)
    if emission is not None:
        emission['path'] = textures.get(emission['binding']) if emission['binding'] else None
        emission['label'] = '1' if emission['binding'] is None else _label(emission['binding'], emission['channel'], names)
        if emission['binding'] is not None and emission['path'] is None:
            emission, out['emission_reason'] = None, 'the mask texture is not bound'
    out['emission'] = emission
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

    emission, alpha = decoded.get('emission'), decoded.get('alpha')
    if emission is not None and emission.get('path'):
        paths['_e'] = emission['path']
    c_tex, n_tex = texture('_c'), texture('_n')
    g_tex = texture('_g') if decoded['layout'] else None
    m_tex = texture('_m') if decoded['occlusion'] else None
    e_tex = texture('_e') if emission is not None and emission.get('path') else None
    if c_tex is None:
        out['base_factor'] = [0.8, 0.8, 0.8]
        out['notes'].append('no base colour texture: white-grey placeholder')
    if emission is not None and (c_tex is None or (emission['binding'] is not None and e_tex is None)):
        decoded['emission_reason'], emission = 'its textures are left out', None
    if alpha is not None and alpha['binding'] is not None and c_tex is None:
        decoded['alpha_reason'], alpha = 'its _c texture is left out', None
    options = dict(g_mode=decoded['layout'] if g_tex is not None else 'colored', k=0.0, spec=1.0, gloss=1.0,
                   roughness='rcra', occlusion='blue' if m_tex is not None else None)
    if emission is not None:
        options.update(k=emission['k'], self_lit='one' if emission['binding'] is None else 'texture',
                       self_lit_channel=emission['channel'] or 0, self_lit_label=emission['label'])
    if alpha is not None and alpha['binding'] is not None:
        options['base_alpha'] = True
    textured = any(t is not None for t in (c_tex, n_tex, g_tex, m_tex))
    cache_key = ('rcra', paths.get('_c'), paths.get('_n'), paths.get('_g') if g_tex is not None else None,
                 options['g_mode'], paths.get('_m') if m_tex is not None else None,
                 (paths.get('_e'), options['k'], options.get('self_lit'), options.get('self_lit_channel')),
                 bool(options.get('base_alpha')))
    entry = map_cache.get(cache_key) if map_cache is not None and textured else None
    if entry is None:
        entry = _convert_maps(c_tex, g_tex, n_tex, options, o_tex=m_tex, e_tex=e_tex)
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
    if out['emissive'] is not None:
        out['notes'].append(EMISSION_NOTE % (emission['label'], emission['k']))
    elif decoded.get('emission_reason'):
        out['notes'].append(NO_EMISSION_NOTE % decoded['emission_reason'])
    if alpha is not None:
        out['alpha_mode'], out['alpha'] = 'BLEND', alpha['k']   # glTF alpha: the factor times the base map's alpha
        out['notes'].append(ALPHA_NOTE % alpha['label'])
    elif decoded.get('blend', 'opaque') != 'opaque':
        out['notes'].append(NO_ALPHA_NOTE % (decoded['blend'], decoded.get('alpha_reason') or 'not decoded'))
    out['notes'].append(NOTE)
    return out
