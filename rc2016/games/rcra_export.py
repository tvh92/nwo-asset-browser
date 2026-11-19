"""Rift Apart glTF/OBJ export maps: the ``Game.material_maps`` hook.

Only what is decoded goes in. Every material gets its DirectX tangent normal (slot 7AE34E7A) at full resolution, converted
by the same pixel code as the R&C 2016 export (``export.models._convert_maps``). The other maps come from the material's own
programs, per group of terms, in this order:

1. **Baked** (``rcra.bake``): a group whose terms in the translated G-buffer writer (albedo; F0 and gloss; occlusion;
   emission, with the albedo) or forward program (alpha, for an alpha-over material) reach only UV0 and constants is
   evaluated at the texel centres of a UV0 grid, with every tint, palette, mask and blend the program applies there.
2. **Plain channels** (``decoded_maps``): the terms are evaluated at random texels and inputs, and a group is used when it
   is exactly a plain channel of the _g or _m map at UV0 (``_plain``), a constant times one _c, _g or _m channel
   (``decoded_emission``) or a constant times _c alpha (``decoded_alpha``). This keeps the few writers whose terms reach UV1
   or the normal on a path that does not change them.
3. Otherwise the stand-in: the _c map as the base colour, non-metallic and fully rough, no occlusion, no emission, opaque.

F0 and gloss are converted to glTF metal/roughness with the Khronos spec-gloss conversion, with roughness
(0.9725 - 0.7514 gloss)² so that glTF's GGX alpha (roughness²) is the game's, r⁴. The emissive map is albedo x E over its
maximum, at that maximum as strength (KHR_materials_emissive_strength). An alpha-over material (flags blend "blended") is
glTF BLEND with the forward alpha in the base map's alpha (or a constant factor). Additive and modulate materials have no
glTF blend and stay opaque. Normal-, view- and UV1-dependent terms are not carried into these formats; the Blender formats
translate them. Every material is single-sided: the culling state is not decoded. A float or 16-bit texture is used as its
sampled values rounded to 8 bits when they lie in 0..1 (``rcra.unit_texture``); otherwise it is left out with its range.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

NOTE = ('Rift Apart portable material: base colour, F0, gloss, occlusion, emission and alpha-over alpha baked from the '
        'material\'s own programs where they read only UV0, or read as plain texture channels or constants; normal-, view- and '
        'UV1-dependent terms are not carried into this format (the Blender formats translate them).')
BAKE_GROUPS = dict(albedo=('albedo.r', 'albedo.g', 'albedo.b'), f0gloss=('f0.r', 'f0.g', 'f0.b', 'gloss'),
                   occlusion=('occlusion',), emission=('emission',))
BAKED_NOTE = '%s baked from the G-buffer writer over UV0 (%dx%d)'
NOT_BAKED_NOTE = '%s not baked (%s): %s'
BAKED_CONVERSION = ('F0 and gloss baked from the G-buffer writer over UV0 (%dx%d), converted with the Khronos spec-gloss to '
                    'metal-rough conversion; roughness (0.9725 - 0.7514 gloss)², so the GGX alpha is the game\'s. Lambert '
                    'diffuse without the glTF energy split is not reproduced.')
BAKED_EMISSION_NOTE = 'emission baked from the G-buffer writer: albedo x E, strength %g (the largest E), over UV0 (%dx%d)'
BAKED_ALPHA_NOTE = ('alpha-over (glTF BLEND): the forward program\'s alpha baked over UV0 (%dx%d) into the base map\'s '
                    'alpha')
BAKE_FAILED_NOTE = 'bake failed, plain forms used: %s'
EMISSION_NOTE = 'emission from the writer: _c x %s x %g (self-lit albedo x E)'
NO_EMISSION_NOTE = 'emission not carried: %s'
ALPHA_NOTE = 'alpha-over (glTF BLEND): alpha = %s from the forward program'
NO_ALPHA_NOTE = 'drawn %s in the game; exported opaque: %s'
UNIT_NOTE = '%s texture %s: stored as DXGI %d (float or 16-bit), its sampled values rounded to 8 bits'
RATIO_TOLERANCE = 1e-4
STAND_IN = 'no game F0 or gloss: exported non-metallic and fully rough (%s)'
CONVERSION = ('F0 and gloss from _g (%s), converted with the Khronos spec-gloss to metal-rough conversion; roughness '
              '(0.9725 - 0.7514 gloss)², so the GGX alpha is the game\'s. Lambert diffuse without the glTF energy split is '
              'not reproduced.')
LAYOUTS = {'grey': 'gloss² in red, grey F0 in green', 'colored': 'F0 in rgb, gloss in alpha'}
OCCLUSION_NOTE = 'occlusion from _m blue (glTF occlusion: the red channel of the metal-rough image)'
PROBES = 6
GLOSS_TOLERANCE = 0.5 / 255 + 1e-6   # the gloss term keeps the G-buffer's 8-bit rounding
F0_TOLERANCE = 0.005                 # F0 is stored as a rounded sqrt(F0) byte and read back squared
OCCLUSION_TOLERANCE = 1e-5           # occlusion is used before the target's rounding
UV_TOLERANCE = 1e-6


def exportable(fmt) -> bool:
    """Formats ``viewer.textures.to_image`` reads as stored: RGBA8, R8G8, R8 and the BC formats. Float and 16-bit
    formats go through ``rcra.unit_texture`` first."""
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


def _bake_plan(surface, forward, blend):
    """{group: None when its terms bake over UV0 (rcra.bake.blocker), else why not}."""
    from rc2016.games.rcra import bake as rbake
    plan = {}
    if surface is not None:
        plan = {group: rbake.blocker(surface, names) for group, names in BAKE_GROUPS.items()}
        if plan['emission'] is None and plan['albedo'] is not None:
            plan['emission'] = 'the self-lit albedo does not bake: %s' % plan['albedo']
    if blend == 'blended' and forward is not None:
        plan['alpha'] = rbake.blocker(forward, ('alpha',))
    return plan


def decoded_maps(source, key):
    """dict(layout, occlusion, g, m, reason, emission, emission_reason, alpha, alpha_reason, blend, bake, bindings,
    surface, forward) for one material: the _g layout ('grey', 'colored' or None), whether occlusion is _m blue, the _g and
    _m texture paths the writer binds (graph defaults where the material binds none), and why there is no layout; the
    emission form (decoded_emission, with the mask texture path) and the alpha-over alpha form (decoded_alpha) with their
    reasons; ``bake`` {group: None when it bakes over UV0, else why not} for BAKE_GROUPS and, for an alpha-over material,
    'alpha'; the texture bindings and the recipes the bake evaluates."""
    from rc2016.games.rcra import material as rmat, shaders as rshaders, viewer_materials as vm
    out = dict(layout=None, occlusion=False, g=None, m=None, reason=None, emission=None, emission_reason=None,
               alpha=None, alpha_reason=None, blend='opaque')
    try:
        graph = vm._graph(source, key)
        surface, forward, blend, _limitations, textures = vm.translate(source, key)
    except (ValueError, KeyError, OSError) as error:   # NotTranslated and Unsupported are ValueErrors
        out['reason'] = 'the writer does not translate: %s' % error
        return out
    out.update(blend=blend, bake=_bake_plan(surface, forward, blend), bindings=textures, surface=surface, forward=forward)
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


def _linear(texture, size, channels=slice(0, 3)):
    """Float32 (h, w, n) of a fallback texture's channels resized to ``size``: sRGB colour decoded, as _convert_maps reads."""
    from rc2016.export.models import _SRGB_LUT, _rgba_u8, _unit
    from rc2016.viewer.textures import SRGB_FORMATS
    data = _rgba_u8(texture, size)
    picked = data[..., channels]
    if texture['format'] in SRGB_FORMATS and channels.stop <= 3:
        return _SRGB_LUT[picked]
    return _unit(picked)


def _finite(array, low=0.0, high=1.0):
    """A baked term as a render target stores it: NaN and infinity 0, clamped (high None: no upper bound)."""
    array = np.nan_to_num(np.asarray(array, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(array, low, high) if high is not None else np.maximum(array, low)


def _baked_terms(source, decoded, groups, size):
    """({group: float32 arrays} at ``size`` (None: the bake grid), grid size, {binding: path} read, sampler notes)."""
    from rc2016.games.rcra import bake as rbake
    sampler = rbake.Sampler(source, decoded['bindings'])
    surface_names = [n for g in sorted(groups) if g != 'alpha' for n in BAKE_GROUPS[g]]
    bindings = set(rbake.sampled_bindings(decoded['surface'], surface_names)) if surface_names else set()
    if 'alpha' in groups:
        bindings |= set(rbake.sampled_bindings(decoded['forward'], ('alpha',)))
    grid = rbake.grid_size(sampler, sorted(bindings))
    if size is not None:
        grid = (max(grid[0], size[0]), max(grid[1], size[1]))
    values = rbake.bake(decoded['surface'], surface_names, sampler, grid) if surface_names else {}
    if 'alpha' in groups:
        values.update(rbake.bake(decoded['forward'], ('alpha',), sampler, grid))
    out = {}
    if 'albedo' in groups:
        out['albedo'] = _finite(np.stack([values['albedo.' + c] for c in 'rgb'], -1))
    if 'f0gloss' in groups:
        out['f0'] = _finite(np.stack([values['f0.' + c] for c in 'rgb'], -1))
        out['gloss'] = _finite(values['gloss'])
    if 'occlusion' in groups:
        out['occlusion'] = _finite(values['occlusion'])
    if 'emission' in groups:
        out['emission'] = _finite(values['emission'], high=None)
    if 'alpha' in groups:
        out['alpha'] = _finite(values['alpha'])
    read = {b: decoded['bindings'][b] for b in sorted(bindings)}
    return out, grid, read, sorted(sampler.notes)


def _compose(size, albedo, f0, gloss, occlusion, emission, alpha):
    """8-bit base (rgb, or rgba with ``alpha``), metal-rough (occlusion, roughness, metallic) and emissive maps, and the
    emissive strength, from linear float arrays at ``size`` (None: not carried)."""
    from rc2016.export.models import _linear_to_srgb, _u8, rcra_roughness, spec_gloss_to_metal_rough
    out = dict(base=None, mr=None, emissive=None, emissive_strength=0.0, occlusion=occlusion is not None)
    width, height = size
    if albedo is None:
        albedo = np.full((height, width, 3), 0.6, np.float32)   # the placeholder grey (base factor 0.8)
    if f0 is not None:
        base, metallic = spec_gloss_to_metal_rough(albedo, np.clip(f0, 0.0, 1.0))
        rough = rcra_roughness(gloss)
    else:
        base, metallic, rough = albedo, np.zeros((height, width), np.float32), np.ones((height, width), np.float32)
    out['base'] = _u8(_linear_to_srgb(base))
    out['mr'] = _u8(np.stack([occlusion if occlusion is not None else np.ones_like(rough), rough, metallic], axis=-1))
    if alpha is not None:
        out['base'] = np.concatenate([out['base'], _u8(alpha)[..., None]], axis=-1)
    if emission is not None:
        strength = float(emission.max())
        if strength > 0.0:
            out['emissive'] = _u8(_linear_to_srgb(albedo * (emission / strength)[..., None]))
            out['emissive_strength'] = float('%.6g' % strength)
    return out


def material_maps(source, material, cache, map_cache=None):
    from rc2016.export.models import _convert_maps
    from rc2016.games.rcra import unit_texture
    from rc2016.games.rcra.material import PREVIEW_ROLES, _material_key, material_slots
    from rc2016.games.rcra.texture import load_texture_full
    name = material.get('name') or (Path(material['path']).stem if material.get('path') else f"material_{material.get('index', 0)}")
    out = dict(name=name, base=None, mr=None, normal=None, emissive=None, emissive_strength=0.0,
               alpha_mode='OPAQUE', alpha=1.0, double_sided=False, occlusion=False, notes=[], sources={}, png={})
    if not material.get('path'):
        out['notes'].append('no material file; neutral grey')
        out['base_factor'] = [0.5, 0.5, 0.5]
        return out
    paths, key = {}, None
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
                if unit_texture.is_data(loaded['format']):
                    # Float/UNORM16 (e.g. the 4x4 R32G32_FLOAT default_pink_shader_n placeholder): the sampled values
                    # rounded to 8 bits when they lie in 0..1; otherwise it raises with their range.
                    loaded = unit_texture.as_rgba8(loaded)
                elif not exportable(loaded['format']):
                    raise ValueError(f"DXGI format {loaded['format']} has no image conversion")
                cache[path] = loaded
            except (ValueError, KeyError, OSError) as error:
                cache[path] = None
                out['notes'].append(f'{role} texture {path} left out: {error}')
        if cache[path] is not None:
            out['sources'][role] = path
            if 'stored_format' in cache[path]:
                out['notes'].append(UNIT_NOTE % (role, path, cache[path]['stored_format']))
        return cache[path]

    plan = decoded.get('bake') or {}
    baking = {group for group, why in plan.items() if why is None}
    emission, alpha = decoded.get('emission'), decoded.get('alpha')
    if emission is not None and emission.get('path'):
        paths['_e'] = emission['path']
    n_tex = texture('_n')
    # A baked entry depends on the material; a plain one only on its textures and forms, which materials share.
    cache_key = ('rcra-bake', key) if baking else (
        'rcra', paths.get('_c'), paths.get('_g') if decoded['layout'] else None, decoded['layout'],
        paths.get('_m') if decoded['occlusion'] else None, repr(emission), repr(alpha), decoded.get('blend'))
    entry = map_cache.get(cache_key) if map_cache is not None else None
    if entry is None:
        first_note, before = len(out['notes']), dict(out['sources'])
        entry = _material_entry(source, decoded, baking, texture)
        entry['texture_notes'] = out['notes'][first_note:]
        del out['notes'][first_note:]
        entry['sources'] = {**{k: v for k, v in out['sources'].items() if k not in before}, **entry['sources']}
        if map_cache is not None and entry['textured']:
            map_cache.put(cache_key, entry)
    if n_tex is not None:
        normal_key = ('rcra-normal', paths.get('_n'))
        normal = map_cache.get(normal_key) if map_cache is not None else None
        if normal is None:
            normal = _convert_maps(None, None, n_tex, {})
            normal['png'] = {}
            if map_cache is not None:
                map_cache.put(normal_key, normal)
        out['normal'] = normal['normal']
    for field in ('base', 'mr', 'emissive', 'emissive_strength', 'occlusion', 'alpha_mode', 'alpha'):
        out[field] = entry[field]
    out['sources'].update(entry['sources'])
    out['png'] = entry['png']
    if 'base_factor' in entry:
        out['base_factor'] = entry['base_factor']
    out['notes'] += entry['texture_notes'] + entry['notes'] + [NOTE]
    return out


def _material_entry(source, decoded, baking, texture):
    """The material's converted maps and notes (cached by ``material_maps``): each group baked where ``baking`` names
    it, else the plain-channel form, else the stand-in."""
    notes, sources = [], {}
    plan = decoded.get('bake') or {}
    emission, alpha = decoded.get('emission'), decoded.get('alpha')
    baked, grid, read, sampler_notes = {}, None, {}, []
    for attempt in (baking, set()):
        fallback = dict(
            c=texture('_c') if 'albedo' not in attempt or (alpha is not None and alpha['binding'] is not None
                                                           and 'alpha' not in attempt) else None,
            g=texture('_g') if decoded['layout'] and 'f0gloss' not in attempt else None,
            m=texture('_m') if decoded['occlusion'] and 'occlusion' not in attempt else None,
            e=texture('_e') if emission is not None and emission.get('path') and 'emission' not in attempt else None)
        size = None
        for tex in fallback.values():
            if tex is not None:
                size = (max(size[0], tex['width']), max(size[1], tex['height'])) if size else (tex['width'], tex['height'])
        if not attempt:
            break
        try:
            baked, grid, read, sampler_notes = _baked_terms(source, decoded, attempt, size)
            break
        except (ValueError, KeyError, OSError) as error:   # surface.Unsupported is a ValueError
            notes.append(BAKE_FAILED_NOTE % error)
            baking = set()
    size = grid or size
    c_tex, g_tex, m_tex, e_tex = (fallback[k] for k in 'cgme')
    entry = dict(base=None, mr=None, normal=None, emissive=None, emissive_strength=0.0, occlusion=False, alpha_mode='OPAQUE',
                 alpha=1.0, png={}, sources=sources, notes=notes, textured=size is not None)
    if c_tex is None and 'albedo' not in baking:
        entry['base_factor'] = [0.8, 0.8, 0.8]
        notes.append('no base colour texture: white-grey placeholder')
    if size is None:   # nothing to convert: no textures and nothing baked
        notes.append(STAND_IN % (plan.get('f0gloss') or decoded['reason'] or 'its _g texture is left out'))
        if decoded.get('blend', 'opaque') != 'opaque':
            notes.append(NO_ALPHA_NOTE % (decoded['blend'], plan.get('alpha') or decoded.get('alpha_reason') or 'not decoded'))
        return entry
    sources.update(read)   # the textures the bake sampled, by binding
    # Albedo.
    if 'albedo' in baking:
        albedo = baked['albedo']
        notes.append(BAKED_NOTE % ('albedo', size[0], size[1]))
    else:
        albedo = _linear(c_tex, size) if c_tex is not None else None
        if plan.get('albedo'):
            notes.append(NOT_BAKED_NOTE % ('albedo', 'the base colour is the _c map', plan['albedo']))
    # F0 and gloss.
    f0 = gloss = None
    if 'f0gloss' in baking:
        f0, gloss = baked['f0'], baked['gloss']
        notes.append(BAKED_CONVERSION % (size[0], size[1]))
    elif g_tex is not None:
        g = _linear(g_tex, size)
        if decoded['layout'] == 'grey':   # gloss² in red, a grey F0 in green
            f0, gloss = np.repeat(g[..., 1:2], 3, axis=-1), np.sqrt(np.clip(g[..., 0], 0, 1))
        else:                             # F0 in rgb, gloss in alpha
            f0, gloss = g, _linear(g_tex, size, slice(3, 4))[..., 0]
        notes.append(CONVERSION % LAYOUTS[decoded['layout']])
    else:
        reason = plan.get('f0gloss') or decoded['reason'] or 'its _g texture is left out'
        notes.append(STAND_IN % reason)
    # Occlusion.
    occlusion = None
    if 'occlusion' in baking:
        occlusion = baked['occlusion']
        if float(occlusion.min()) < 1.0:
            notes.append(BAKED_NOTE % ('occlusion', size[0], size[1]))
        else:
            occlusion = None
    elif m_tex is not None:
        occlusion = _linear(m_tex, size, slice(2, 3))[..., 0]
        notes.append(OCCLUSION_NOTE)
    # Emission: the self-lit term albedo x E.
    e_map = None
    if 'emission' in baking:
        if float(baked['emission'].max()) > 0.0:
            e_map = baked['emission']
    elif emission is not None and (c_tex is not None or 'albedo' in baking) and (emission['binding'] is None or e_tex is not None):
        if emission['binding'] is None:
            e_map = np.full((size[1], size[0]), emission['k'], np.float32)
        else:
            channel = emission['channel'] or 0
            e_map = _linear(e_tex, size, slice(channel, channel + 1))[..., 0] * np.float32(emission['k'])
    elif emission is not None:
        notes.append(NO_EMISSION_NOTE % 'its textures are left out')
    elif decoded.get('emission_reason'):
        notes.append(NO_EMISSION_NOTE % (plan.get('emission') or decoded['emission_reason']))
    # Alpha-over alpha.
    a_map, factor = None, 1.0
    if 'alpha' in baking:
        if float(baked['alpha'].min()) < 1.0:
            a_map = baked['alpha']
            notes.append(BAKED_ALPHA_NOTE % (size[0], size[1]))
        else:
            notes.append(NO_ALPHA_NOTE % ('blended', 'the forward alpha is 1'))
    elif alpha is not None and alpha['binding'] is not None and c_tex is None:
        notes.append(NO_ALPHA_NOTE % ('blended', 'its _c texture is left out'))
    elif alpha is not None:
        factor = alpha['k']
        if alpha['binding'] is not None:
            from rc2016.export.models import _rgba_u8
            a_map = _rgba_u8(c_tex, size)[..., 3].astype(np.float32) / 255.0
        notes.append(ALPHA_NOTE % alpha['label'])
    elif decoded.get('blend', 'opaque') != 'opaque':
        notes.append(NO_ALPHA_NOTE % (decoded['blend'], plan.get('alpha') or decoded.get('alpha_reason') or 'not decoded'))
    entry.update(_compose(size, albedo, f0, gloss, occlusion, e_map, a_map))
    if entry['emissive'] is not None:
        notes.append(BAKED_EMISSION_NOTE % (entry['emissive_strength'], size[0], size[1]) if 'emission' in baking else
                     EMISSION_NOTE % (emission['label'], emission['k']))
    if a_map is not None or factor < 1.0:
        entry['alpha_mode'], entry['alpha'] = 'BLEND', factor   # glTF alpha: the factor times the base map's alpha
    notes.extend(sampler_notes)
    return entry
