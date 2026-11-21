"""Rift Apart deferred lighting classes: the engine's per-class apply programs, translated into one direct-light recipe per
class for renderers.

The G-buffer writer stores a lighting class in bits 13 to 15 of NormalSpec x; bit 12 set means class 0 whatever those bits
hold. The engine applies each class with its own compute program in RiftApart.exe (they read g_GBufDiffuseEmis and write
g_GBufferApplyOutput), found here by the class test at their start: ``ubfe(3, 13, NormalSpec.x)`` compared with the class.

    Class  Graphs (writers)                           Apply program
    0      the rest                                   GGX + Lambert (docs/rift-apart.md)
    1      subsurface: skin, heads, nymphs, dinosaurs reads g_SkinProfiles; diffuse and specular to separate targets
    2      cloth                                       sheen (Charlie) mixed with GGX on a second normal, wrap diffuse
    3      hair: the engine fur writer                strand tangent, shifted lobes, transmission
    4      anisotropic                                two GGX lobes on a rotated tangent frame
    5      foliage and transmittance                  wrapped diffuse with back transmission

Classes 1 to 5 read a fifth target, g_GBufExtra (the writer's SV_Target3, one u32), whose fields the apply program reads at
fixed bit offsets (EXTRA_FIELDS). ``rc2016.games.rcra.surface`` returns those fields as the terms ``extra.<k>`` (their integer
values) and the class as the term ``class``.

``direct(root, k)`` translates class k's apply program with:
- the G-buffer: NormalSpec packed from the terms normal, F0, gloss and the class exactly as the writers pack them (12-bit
  normal, F0 square roots and gloss as bytes, occlusion mode set), DiffuseEmis = (albedo, occlusion), Extra packed from the
  ``extra`` fields;
- one light: the key light, direction ``L`` (to the light, world, game space) and colour ``LIGHT`` (linear), unshadowed: the
  shadow atlas, cloud shadows, ray-traced shadows and gobo read 1, the point lies past the shadow shells, the sunlight
  occlusion factor is 0;
- the view: the camera-to-point offset ``VIEW`` and its view depth ``DEPTH``;
- no other light: ambient fill, light grid, probes, GI, SSR and the local light lists are 0 (empty lists), so the result is
  the direct term of one light. Emission is 0 (occlusion mode).

Per class the result is the colour the engine's later passes see for that light: classes 0 and 2 to 5 write it to
g_GBufferApplyOutput; class 1 writes the diffuse light there and the specular to g_GBufferApplyOutput2, and the skin pass
adds them as albedo x (blurred diffuse) + specular, so the recipe is albedo x diffuse + specular, the blur not applied
(SKIN_NOTE). The class-1 program loads one texel of the skin profiles (g_SkinProfiles): texel PROFILE_BASE + the profile
index (G-buffer extra bits 5 to 8, the field extra.4). The recipe reads that texel as the inputs ``PROFILE.r/g/b/a``; a
renderer looks it up in PROFILE_TEXTURE, the texture the engine binds (``profile_texel``).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from pathlib import Path

from rc2016.games.rcra import shaders as rshaders
from rc2016.shaders import dxbc, sm5

EXECUTABLE = 'RiftApart.exe'
CLASSES = (0, 1, 2, 3, 4, 5)
CLASS_NAMES = {0: 'standard', 1: 'subsurface', 2: 'cloth', 3: 'hair', 4: 'anisotropic', 5: 'foliage'}
# The g_GBufExtra fields each class's apply program reads: (bit offset, width), in the order of the terms extra.0, extra.1, ...
EXTRA_FIELDS = {
    1: ((0, 3), (9, 6), (18, 7), (25, 7), (5, 4)),    # scatter, transmission, second normal (7 + 7 bits), skin profile
    2: ((0, 8), (8, 8), (16, 5), (21, 6), (27, 5)),   # second normal (8 + 8), sheen mix, scatter lobe, wrap
    3: ((0, 9), (9, 9), (18, 1), (19, 7), (26, 6)),   # strand tangent (9 + 9, z sign), transmittance, lobe shift
    4: ((0, 9), (9, 9), (18, 1), (19, 6), (25, 7)),   # tangent (9 + 9, z sign), rotation, second gloss
    5: ((0, 6), (6, 6), (12, 6), (18, 7), (25, 7)),   # transmission colour (6 bits each), transmittance, wrap
}
TERMS = ('diffuse.r', 'diffuse.g', 'diffuse.b', 'specular.r', 'specular.g', 'specular.b')
INPUTS = ('normal.x', 'normal.y', 'normal.z', 'albedo.r', 'albedo.g', 'albedo.b', 'f0.r', 'f0.g', 'f0.b', 'gloss',
          'occlusion', 'L.x', 'L.y', 'L.z', 'LIGHT.r', 'LIGHT.g', 'LIGHT.b', 'VIEW.x', 'VIEW.y', 'VIEW.z', 'DEPTH')
WEATHER = 'g_WeatherPorosityMap'   # the rain variants of the apply programs: not used
SKIN_PROFILES = 'g_SkinProfiles'
APPLY_OUTPUT, SPECULAR_OUTPUT = 'g_GBufferApplyOutput', 'g_GBufferApplyOutput2'
CLASS_NOTE = ('Lighting class %d (%s): direct light from the engine\'s class-%d apply program (%s), translated; ambient light '
              'is the review rig\'s.')
SKIN_NOTE = (" The skin profiles are the engine's (texel 64 + the material's profile index of "
             "required/textures/char-skin-profiles-srgb_1d.texture); the program reads them only for transmission through "
             "the shadow-map depth, which the unshadowed review lights do not have, so they do not change the light. The "
             "skin pass blurs the diffuse light in screen space with the skin profiles before adding the specular; the blur "
             "is not drawn (albedo x diffuse + specular at each pixel).")
# The skin profiles: RiftApart.exe binds required/textures/char-skin-profiles-srgb_1d.texture once it has loaded, else its
# "Default Skin Profile Lookup" built from the same 128 texels embedded in the executable (getter 0x141353a90,
# TextureDefaultsInit 0x14135550a). R8G8B8A8_UNORM_SRGB, 128 x 1. The class-1 apply program and the skin pass load texel
# PROFILE_BASE + ubfe(4, 5, extra word).
PROFILE_TEXTURE = 'required/textures/char-skin-profiles-srgb_1d.texture'
PROFILE_BASE = 64
PROFILE_FIELD = 4   # the index of the profile field in EXTRA_FIELDS[1]
PROFILE_INPUTS = ('PROFILE.r', 'PROFILE.g', 'PROFILE.b', 'PROFILE.a')
UNSHADOWED_NOTE = (' Lights are unshadowed (shadow maps, cloud shadows and gobos read 1; transmission that the game takes '
                   'from the shadow map depth reads it at the lit surface).')
ONE_RESOURCES = ('g_ShadowAtlas', 'g_CloudShadowMap', 'g_RayTracedShadows', 'g_GoboAtlas', 'g_AoSsrBuffer',
                 'g_SssoHistoryTex')
_LOCK = threading.Lock()
_PROGRAMS = {}
_RECIPES = {}


# ------------------------------------------------------------------------------------------------ finding the programs

def _gate(shader):
    """The class an apply program tests at its start (ubfe(3, 13, NormalSpec.x) against a literal), 'pair' for the
    programs that test two classes at once (1 and 2), or None."""
    text = [str(i) for i in shader.program.instructions[:48]]
    for k, line in enumerate(text):
        m = re.match(r'ubfe (r\d+\.[xyzw]), l\(0x00000003\), l\(0x0000000d\), ', line)
        if not m:
            continue
        for later in text[k + 1:k + 4]:
            m2 = re.match(r'(?:ine|ieq) \S+, %s, l\((0x[0-9a-f]+)\)$' % re.escape(m.group(1)), later)
            if m2:
                return int(m2.group(1), 16)
        return 'pair'
    return None


def find_programs(data):
    """{class: (offset, DXBC bytes)} of the apply programs in the executable bytes: per class the smallest program
    (fewest instructions, then lowest offset) without the rain inputs. Of the two-class programs, the one that reads the
    skin profiles applies class 1, the other class 2."""
    best = {}
    for at, size in dxbc.find_containers(data):
        blob = bytes(data[at:at + size])
        try:
            shader = dxbc.shader(blob)
        except dxbc.DxbcError:
            continue
        r = shader.reflection
        if shader.stage != 'cs' or r is None:
            continue
        names = {b.name for b in r.bindings}
        if 'g_GBufDiffuseEmis' not in names or APPLY_OUTPUT not in names or WEATHER in names:
            continue
        cls = _gate(shader)
        if cls == 'pair':
            cls = 1 if SKIN_PROFILES in names else 2
        if cls not in CLASSES:
            continue
        rank = (len(shader.program.instructions), at)
        if cls not in best or rank < best[cls][0]:
            best[cls] = (rank, at, blob)
    missing = [c for c in CLASSES if c not in best]
    if missing:
        raise ValueError('Apply programs not found in the executable for classes %s' % missing)
    return {c: (best[c][1], best[c][2]) for c in CLASSES}


def programs(root, disk=None):
    """find_programs of the install's executable, kept in memory by path, size and time. With ``disk`` (a source's
    translation disk cache) the programs' places are kept across sessions too, keyed by the same facts and the scanning
    code, and their bytes read back from the executable; the scan parses every DXBC container of the executable."""
    path = os.path.join(str(root), EXECUTABLE)
    stat = os.stat(path)
    key = (path, stat.st_size, stat.st_mtime_ns)
    with _LOCK:
        if key in _PROGRAMS:
            return _PROGRAMS[key]
        code = _scan_code_key() if disk is not None else None
        disk_key = disk.key(repr(key).encode(), [], 'rcra-lighting-programs:%s' % code, own_code=True) if code else None
        found = _read_places(path, disk.get(disk_key)) if disk_key else None
        if found is None:
            with open(path, 'rb') as f:
                found = find_programs(f.read())
            if disk_key:
                disk.put(disk_key, {str(cls): [offset, len(blob)] for cls, (offset, blob) in found.items()})
        _PROGRAMS[key] = found
        return found


def _scan_code_key():
    """Hash of the scanning code (this module and the DXBC reader), or None without readable sources (no disk record)."""
    here = Path(__file__).resolve()
    try:
        return hashlib.sha256(b''.join(f.read_bytes() for f in (here, here.parents[2] / 'shaders' / 'dxbc.py'))).hexdigest()
    except OSError:
        return None


def _read_places(path, places):
    """{class: (offset, DXBC bytes)} read from the executable at stored places {str(class): [offset, size]}, or None when
    the record is missing or does not hold a DXBC container of every class."""
    if not isinstance(places, dict) or sorted(places) != sorted(str(c) for c in CLASSES):
        return None
    found = {}
    try:
        with open(path, 'rb') as f:
            for cls in CLASSES:
                offset, size = places[str(cls)]
                f.seek(offset)
                blob = f.read(size)
                if len(blob) != size or blob[:4] != b'DXBC':
                    return None
                found[cls] = (offset, blob)
    except (OSError, TypeError, ValueError):
        return None
    return found


# ------------------------------------------------------------------------------------------------ translation

class _ApplyBindings(sm5.Bindings):
    """The G-buffer packed from term inputs, one unshadowed key light, the view from VIEW and DEPTH, every other engine
    input 0 (recorded in ``zeroed``)."""

    def __init__(self, cls):
        self.cls = cls
        self.graph = None
        self.zeroed = set()
        self._words = None
        self.profile_addresses = []   # the texel addresses of the skin profile loads

    def _inputs(self):
        g = self.graph
        return {name: g.input(name, 'f') for name in INPUTS}

    def words(self):
        if self._words is None:
            g = self.graph
            x = self._inputs()
            c = lambda v, t='f': g.const(t, v)   # noqa: E731

            def q(v):   # the byte the writers store: ftoi(255 v + 0.5)
                return g.op('ftoi', g.op('add', g.op('mul', v, c(255.0)), c(0.5)))

            weight = g.op('rsq', g.op('add', g.op('mul', g.op('abs', x['normal.z']), c(8.0)), c(8.0)))

            def packed(component):
                return g.op('ftou', g.op('add', g.op('mul', g.op('mul', weight, component), c(NORMAL_SCALE)),
                                         c(2048.0)))
            sign = g.op('bfi', c(1, 'i'), c(15, 'i'), g.op('b2i', g.op('lt', x['normal.z'], c(0.0))), c(0, 'i'))
            # Class 0 as most class-0 writers store it, with bit 12 set (its other bits are 0: they feed only ambient terms).
            ns = [g.op('or', packed(x['normal.x']), c(self.cls << 13 if self.cls else 0x1000, 'i')),
                  g.op('or', packed(x['normal.y']), g.op('or', sign, c(0x2000, 'i'))),
                  g.op('or', g.op('ishl', q(g.op('sqrt', x['f0.r'])), c(8, 'i')), q(g.op('sqrt', x['f0.g']))),
                  g.op('or', g.op('ishl', q(g.op('sqrt', x['f0.b'])), c(8, 'i')), q(x['gloss']))]
            de = [x['albedo.r'], x['albedo.g'], x['albedo.b'], x['occlusion']]
            extra = c(0, 'i')
            for k, (offset, _width) in enumerate(EXTRA_FIELDS.get(self.cls, ())):
                field = g.op('ftou', g.input('extra.%d' % k, 'f'))
                extra = g.op('or', extra, g.op('ishl', field, c(offset, 'i')) if offset else field)
            self._words = dict(ns=ns, de=de, ex=[extra, c(0, 'i'), c(0, 'i'), c(0, 'i')])
        return self._words

    def constant(self, name):
        g = self.graph
        x = self._inputs()
        view = {'GlobalViewportCBuffer.m_ScreenToViewA+8': g.op('div', x['VIEW.x'], x['DEPTH']),
                'GlobalViewportCBuffer.m_ScreenToViewA+12': g.op('div', x['VIEW.y'], x['DEPTH']),
                'GlobalViewportCBuffer.m_ViewToWorldMat+40': g.op('div', x['VIEW.z'], x['DEPTH'])}
        if name in view:
            return view[name]
        light = {'GlobalWorldCBuffer.m_KeyLightVec': 'L.x', 'GlobalWorldCBuffer.m_KeyLightVec+4': 'L.y',
                 'GlobalWorldCBuffer.m_KeyLightVec+8': 'L.z', 'GlobalWorldCBuffer.m_KeyLightColorLinear': 'LIGHT.r',
                 'GlobalWorldCBuffer.m_KeyLightColorLinear+4': 'LIGHT.g',
                 'GlobalWorldCBuffer.m_KeyLightColorLinear+8': 'LIGHT.b'}
        if name in light:
            return x[light[name]]
        if name in rshaders.NEUTRAL_INTEGERS:
            return g.const('i', rshaders.NEUTRAL_INTEGERS[name])
        if name in rshaders.NEUTRAL_FLOATS:
            return g.const('f', rshaders.NEUTRAL_FLOATS[name])
        fixed = {'GlobalViewportCBuffer.m_ScreenToViewB': 1.0, 'GlobalViewportCBuffer.m_ViewToWorldMat': 1.0,
                 'GlobalViewportCBuffer.m_ViewToWorldMat+20': 1.0, 'GlobalViewportCBuffer.m_FarClip': 1000.0,
                 'GlobalViewportCBuffer.m_NearClip': 0.1, 'GlobalViewportCBuffer.m_Dimensions': 1920.0,
                 'GlobalViewportCBuffer.m_Dimensions+4': 1080.0,
                 # past every shadow shell: max of the shell planes is 1, sat(1 x 10 - 9) = 1, so the key light is unshadowed
                 'GlobalWorldCBuffer.m_KeyLightShellPlanesW': 1.0, 'GlobalWorldCBuffer.m_KeyLightShellPlanesW+4': 1.0,
                 'GlobalWorldCBuffer.m_KeyLightShellPlanesW+8': 1.0, 'GlobalWorldCBuffer.m_KeyLightShellPlanesW+12': 1.0,
                 'GlobalWorldCBuffer.m_KeyLightShellLodConsts+12': 10.0}
        if name in fixed:
            return g.const('f', fixed[name])
        self.zeroed.add(name)
        return g.const('f', 0.0)

    def dynamic_constant(self, buffer, offset, index):
        self.zeroed.add('%s+%d[dynamic]' % (buffer, offset))
        return self.graph.const('f', 0.0)

    def input(self, name):
        return None

    def resource(self, graph, op, binding, args):
        if binding == 'g_GBufNormalSpec':
            return list(self.words()['ns'])
        if binding == 'g_GBufDiffuseEmis':
            return list(self.words()['de'])
        if binding == 'g_GBufExtra':
            return list(self.words()['ex'])
        if binding == 'g_ViewDepthBuffer':
            return [graph.input('DEPTH', 'f')] * 4
        if binding == 'g_GBufferApplyWorkBuffer':
            return [graph.const('i', 0)] * 4
        if binding in ONE_RESOURCES:
            return [graph.const('f', 1.0)] * 4
        if binding == SKIN_PROFILES and op.startswith('ld'):   # one texel, looked up by the renderer (profile_texel)
            self.profile_addresses.append(args[0])
            return [graph.input(name, 'f') for name in PROFILE_INPUTS]
        self.zeroed.add(binding)
        return [graph.const('i' if op.startswith('ld') else 'f', 0)] * 4


NORMAL_SCALE = 5791.205   # rc2016.games.rcra.surface.NORMAL_SCALE (the writers' 12-bit normal scale)
THREAD_INPUTS = ('vThreadGroupID', 'vThreadIDInGroup', 'vThreadID', 'vThreadIDInGroupFlattened')


def _split(g, root, albedo):
    """(X, Y) of an apply output add(mul(albedo, X), Y): the diffuse factor the program multiplies by DiffuseEmis rgb, and
    the rest (specular)."""
    n = g.nodes[root]
    if n['op'] == 'add':
        for a, b in (n['args'], n['args'][::-1]):
            m = g.nodes[a]
            if m['op'] == 'mul' and albedo in m['args']:
                x = m['args'][1] if m['args'][0] == albedo else m['args'][0]
                return x, b
    return None


def translate(shader, cls):
    """dict(graph, terms {diffuse.r/g/b, specular.r/g/b}, zeroed [engine inputs bound to 0]) of one class's apply program,
    as portable float expressions of INPUTS and the class's extra fields. The light the class reflects is
    albedo x diffuse + specular."""
    from rc2016.games.rcra import surface as rsurface
    bindings = _ApplyBindings(cls)
    translator = sm5.Translator(shader, bindings)
    bindings.graph = translator.g
    tr = rsurface._run(translator)
    g = tr.graph
    stores = {}
    for store in tr.stores:
        stores.setdefault(store['binding'], store)
    if APPLY_OUTPUT not in stores:
        raise rsurface.Unsupported('Apply program writes no %s' % APPLY_OUTPUT)
    for address in bindings.profile_addresses:
        _check_profile_address(g, address, cls)
    colour = [stores[APPLY_OUTPUT]['values'][c] for c in range(3)]
    terms = {}
    for c, channel in enumerate('rgb'):
        if cls == 1:   # the skin pass: albedo x (blurred) diffuse + specular, blur radius 0
            diffuse, specular = colour[c], stores[SPECULAR_OUTPUT]['values'][c]
        else:
            parts = _split(g, colour[c], g.input('albedo.' + channel, 'f'))
            if parts is None:
                raise rsurface.Unsupported('Apply output is not albedo x diffuse + specular')
            diffuse, specular = parts
        terms['diffuse.' + channel], terms['specular.' + channel] = diffuse, specular
    # The pixel's thread position reads only multiplied by the zero inverse dimensions (dither, screen position): 0.
    replace = {i: g.const(n['type'], 0) for i, n in enumerate(g.nodes)
               if n['op'] == 'input' and n['name'].split('.')[0] in THREAD_INPUTS}
    terms = rsurface._substitute(g, terms, replace)
    unknown = g.unknowns(list(terms.values()))
    if unknown:
        raise rsurface.Unsupported('Apply program depends on unknown values: ' + ', '.join(unknown[:3]))
    terms = _portable(g, terms)
    # The light's own contribution: the terms minus their value with LIGHT = 0. Some programs add terms that no light
    # scales (the hair program's ambient transmission, from inputs bound to 1 here); those are ambient light, not this
    # light's, and a renderer adds the recipe once per light.
    dark = rsurface._substitute(g, terms, {i: g.const('f', 0.0) for i, n in enumerate(g.nodes)
                                           if n['op'] == 'input' and n['name'].startswith('LIGHT.')})
    terms = {name: (node if dark[name] == node or g.value(dark[name]) == 0.0
                    else g.op('add', node, g.op('neg', dark[name])))
             for name, node in terms.items()}
    terms = _fold_unused_profile(g, terms, cls)
    return dict(graph=g, terms=terms, zeroed=sorted(bindings.zeroed))


def _check_profile_address(g, address, cls):
    """Raise Unsupported unless a skin profile load reads texel PROFILE_BASE + the profile field (at random field values)."""
    import random
    from collections import defaultdict

    from rc2016.games.rcra import surface as rsurface
    fields = EXTRA_FIELDS.get(cls, ())
    if len(fields) <= PROFILE_FIELD:
        raise rsurface.Unsupported('Class %d reads the skin profiles without a profile field' % cls)
    rng = random.Random(1)
    for _trial in range(16):
        values = {'extra.%d' % k: float(rng.randrange(1 << w)) for k, (_o, w) in enumerate(fields)}
        got = sm5.evaluate(g, [address], defaultdict(float, values), lambda *a: (0.0,) * 4)[address]
        if got != PROFILE_BASE + values['extra.%d' % PROFILE_FIELD]:
            raise rsurface.Unsupported('The skin profile load does not read texel %d + the profile field' % PROFILE_BASE)


def profile_reach(g, terms, cls, trials=48):
    """True when the skin profile inputs change the terms at random inputs, fields and profile colours."""
    import random
    from collections import defaultdict
    names = {g.nodes[i]['name'] for i in g.reachable(list(terms.values())) if g.nodes[i]['op'] == 'input'}
    if not names & set(PROFILE_INPUTS):
        return False
    rng = random.Random(7)
    roots = list(terms.values())
    for _trial in range(trials):
        values = defaultdict(float, {name: rng.uniform(-1.0, 1.0) for name in names})
        values.update({'extra.%d' % k: float(rng.randrange(1 << w)) for k, (_o, w) in enumerate(EXTRA_FIELDS.get(cls, ()))})
        values.update(DEPTH=rng.uniform(0.2, 8.0), gloss=rng.random(), occlusion=rng.random())
        outs = []
        for colour in ((0.0, 0.0, 0.0, 0.0), tuple(rng.uniform(0.05, 1.0) for _ in range(4))):
            values.update(zip(PROFILE_INPUTS, colour))
            outs.append(sm5.evaluate(g, roots, values, lambda *a: (0.0,) * 4))
        if any(outs[0][r] != outs[1][r] and not (outs[0][r] != outs[0][r] and outs[1][r] != outs[1][r]) for r in roots):
            return True
    return False


def _fold_unused_profile(g, terms, cls):
    """The terms with the skin profile inputs at 0 when they do not change them: under these bindings the class-1 program
    reads the profile only for transmission through the shadow-map depth (gated by the key light's shadow cascade
    constants, 0 here) and the light grid's ambient transmission, both 0 (PROFILE_NOTE)."""
    from rc2016.games.rcra import surface as rsurface
    if profile_reach(g, terms, cls):
        return terms
    return rsurface._substitute(g, terms, {i: g.const('f', 0.0) for i, n in enumerate(g.nodes)
                                           if n['op'] == 'input' and n['name'] in PROFILE_INPUTS})


def profile_texel(texels, index):
    """The skin profile texel (linear RGBA) an apply program loads for profile ``index``: texels[PROFILE_BASE + index], 0
    past the texture's end as a D3D load reads."""
    k = PROFILE_BASE + int(index)
    return tuple(float(x) for x in texels[k]) if 0 <= k < len(texels) else (0.0, 0.0, 0.0, 0.0)


def _portable(g, terms, sampled=()):
    """The terms rewritten to portable operations; texture reads raise Unsupported unless their binding is in ``sampled``
    (kept as sample nodes for the renderer)."""
    from rc2016.games.rcra import surface as rsurface
    bad = {g.nodes[i]['op'] for i in g.reachable(list(terms.values()))} - rsurface.PORTABLE_OPS
    if bad:
        terms = rsurface._enumerate_cases(g, terms)
        for _round in range(rsurface.INT_FLOAT_ROUNDS):
            terms, changed, _packed = rsurface._integers_in_float(g, terms)
            if not changed:
                break
    bad = {g.nodes[i]['op'] for i in g.reachable(list(terms.values()))} - rsurface.PORTABLE_OPS
    if bad:
        raise rsurface.Unsupported('Apply program keeps non-portable operations: ' + ', '.join(sorted(bad)))
    for i in g.reachable(list(terms.values())):
        n = g.nodes[i]
        if n['op'] in ('sample', 'load') and (n['op'] == 'load' or n.get('binding') not in sampled):
            raise rsurface.Unsupported('Apply program reads an engine texture (%s)' % n.get('binding'))
    return rsurface._prune_nan(g, terms)


def _code_key():
    here = Path(__file__).resolve()
    files = [here, here.parents[2] / 'shaders' / 'sm5.py', here.parent / 'surface.py']
    return hashlib.sha256(b''.join(f.read_bytes() for f in files)).hexdigest()


def direct(root, cls, disk=None):
    """The direct-light recipe of class ``cls`` (surface.export_recipe shape: nodes, terms diffuse.r/g/b and
    specular.r/g/b, inputs, limitations, plus offset: the program's position in the executable, zeroed and lighting_class).
    Kept in memory and, with ``disk`` (a source's translation disk cache: key, get, put), across sessions, keyed by the
    program bytes and this code."""
    from rc2016.games.rcra import surface as rsurface
    offset, blob = programs(root, disk)[cls]
    tag = 'rcra-lighting:%d:%s' % (cls, _code_key())
    memo = (tag, hashlib.sha256(blob).hexdigest())
    with _LOCK:
        if memo in _RECIPES:
            return _RECIPES[memo]
    key = disk.key(blob, [], tag, own_code=True) if disk is not None else None
    recipe = _decode(disk.get(key)) if key else None
    if recipe is None:
        result = translate(dxbc.shader(blob), cls)
        g = result['graph']
        result.update(textures={}, limitations=[class_note(cls, offset)],
                      inputs=sorted({g.nodes[i]['name'] for i in g.reachable(list(result['terms'].values()))
                                     if g.nodes[i]['op'] == 'input'}))
        recipe = rsurface.fold_nan(rsurface.export_recipe(result))
        recipe.update(offset=offset, zeroed=result['zeroed'], lighting_class=cls)
        if key:
            disk.put(key, _encode(recipe))
    with _LOCK:
        _RECIPES[memo] = recipe
    return recipe


# ------------------------------------------------------------------------------------------------ ambient light
# The apply programs' own ambient light: the program with the key light off, so its output is the ambient alone. Its sources
# are bound for a scene that has only the environment: the light grid empty (lookups and cells 0), no local probes (empty
# lists), no screen-space reflections, distant GI or ambient fill, and the review environment as the default probe
# (g_EnvProbeDefault, rc2016.games.rcra.probe), with the engine's BRDF lookup (g_BrdfLookup, rc2016.games.rcra.engine_tables).
# Both stay texture samples in the recipe, for the renderer to look up. m_LightGridIntensity, which scales the probe's
# ambient and reflections, is a review value of 1.
# The light grid (camera-centred, 512 units each way) blends each point between the grid's radiance and the default probe by a
# probe weight: the eight corner cells' max(cell data.w & 0x3C0, lookup word & 0x3FC), averaged by the trilinear and facing
# weights, over 960 (outside the grid's extent the weight is 960). The same weight scales the probe's reflection by the
# grid's irradiance over the probe's. The review scene has no grid: every lookup word is GRID_PROBE_WORD (weight 960) and
# every cell is empty, so the program lights each point from the default probe alone, the grid's radiance cancels exactly,
# and the reflection scale is the probe's own (its lod-5 radiance at the normal over that at the reflection).
AMBIENT_SAMPLED = ('g_EnvProbeDefault', 'g_BrdfLookup')
GRID_LOOKUP = 'g_LightGridLookup'
GRID_PROBE_WORD = 0x3C0
AMBIENT_EMPTY = ('g_LightGridData', 'g_SSRFinalTex', 'g_GiDistantSamples', 'g_GiDistantHeight', 'g_EnvProbeArray',
                 'g_EnvProbeEnvs', 'g_LightVolumes', 'g_SsgiHistoryTex', 'g_GiPrepassOutput')
AMBIENT_REVIEW = {'GlobalWorldCBuffer.m_LightGridIntensity': 1.0}
AMBIENT_TERMS = ('ambient.r', 'ambient.g', 'ambient.b')
AMBIENT_NOTE = ("Ambient light from lighting class %d's own apply program (%s at 0x%X): the review environment as the "
                "engine's default probe, prefiltered by the engine's probe program, and the engine's BRDF lookup; no light "
                "grid (every cell hands its points to the default probe: lookup words 0x3C0, empty cells), local "
                "probes, screen-space reflections, distant GI or ambient fill, and m_LightGridIntensity 1 (review values).")


class _AmbientBindings(_ApplyBindings):
    """_ApplyBindings with the key light off and the ambient sources of AMBIENT_SAMPLED / AMBIENT_EMPTY."""

    def constant(self, name):
        if name.startswith('GlobalWorldCBuffer.m_KeyLightColorLinear'):
            return self.graph.const('f', 0.0)
        if name in AMBIENT_REVIEW:
            return self.graph.const('f', AMBIENT_REVIEW[name])
        return super().constant(name)

    def resource(self, graph, op, binding, args):
        if binding in AMBIENT_SAMPLED and op.startswith('sample'):
            return None   # a sample node, looked up by the renderer
        if binding == GRID_LOOKUP:
            return [graph.const('i', GRID_PROBE_WORD)] * 4
        if binding in AMBIENT_EMPTY:
            return [graph.const('i' if op.startswith('ld') else 'f', 0)] * 4
        return super().resource(graph, op, binding, args)


def _light_free(g, terms, trials=24):
    """True when the terms do not change with the key light's direction (the key light is off) at random inputs and texels."""
    import random
    from collections import defaultdict
    names = {g.nodes[i]['name'] for i in g.reachable(list(terms.values())) if g.nodes[i]['op'] == 'input'}
    if not names & {'L.x', 'L.y', 'L.z'}:
        return True
    rng = random.Random(11)
    roots = list(terms.values())
    texels = {}

    def sample(node, _args):
        return texels.setdefault(node.get('binding'), [rng.uniform(0.05, 0.95) for _ in range(4)])
    for _trial in range(trials):
        texels.clear()
        values = defaultdict(float, {name: rng.uniform(-1.0, 1.0) for name in names})
        values.update(DEPTH=rng.uniform(0.2, 8.0), gloss=rng.random(), occlusion=rng.random())
        outs = []
        for _light in range(2):
            values.update({'L.' + c: rng.uniform(-1.0, 1.0) for c in 'xyz'})
            outs.append(sm5.evaluate(g, roots, values, sample))
        if any(outs[0][r] != outs[1][r] and not (outs[0][r] != outs[0][r] and outs[1][r] != outs[1][r]) for r in roots):
            return False
    return True


def translate_ambient(shader, cls):
    """dict(graph, terms {ambient.r/g/b}, zeroed): class ``cls``'s apply output with the key light off (the whole colour,
    albedo included; class 1 as albedo x diffuse + specular), the default probe and BRDF lookup as sample nodes."""
    from rc2016.games.rcra import surface as rsurface
    bindings = _AmbientBindings(cls)
    translator = sm5.Translator(shader, bindings)
    bindings.graph = translator.g
    tr = rsurface._run(translator)
    g = tr.graph
    stores = {}
    for store in tr.stores:
        stores.setdefault(store['binding'], store)
    if APPLY_OUTPUT not in stores:
        raise rsurface.Unsupported('Apply program writes no %s' % APPLY_OUTPUT)
    for address in bindings.profile_addresses:
        _check_profile_address(g, address, cls)
    terms = {}
    for c, name in enumerate(AMBIENT_TERMS):
        colour = stores[APPLY_OUTPUT]['values'][c]
        if cls == 1:   # the skin pass: albedo x diffuse + specular
            colour = g.op('add', g.op('mul', g.input('albedo.' + 'rgb'[c], 'f'), colour),
                          stores[SPECULAR_OUTPUT]['values'][c])
        terms[name] = colour
    replace = {i: g.const(n['type'], 0) for i, n in enumerate(g.nodes)
               if n['op'] == 'input' and n['name'].split('.')[0] in THREAD_INPUTS}
    terms = rsurface._substitute(g, terms, replace)
    unknown = g.unknowns(list(terms.values()))
    if unknown:
        raise rsurface.Unsupported('Apply program ambient depends on unknown values: ' + ', '.join(unknown[:3]))
    terms = _portable(g, terms, AMBIENT_SAMPLED)
    if not _light_free(g, terms):
        raise rsurface.Unsupported('Apply program ambient reads the key light direction')
    terms = rsurface._substitute(g, terms, {i: g.const('f', 0.0) for i, n in enumerate(g.nodes)
                                            if n['op'] == 'input' and n['name'] in ('L.x', 'L.y', 'L.z')})
    terms = _fold_unused_profile(g, terms, cls)
    return dict(graph=g, terms=terms, zeroed=sorted(bindings.zeroed))


RECIPE_TAGS = {'direct': 'rcra-lighting', 'ambient': 'rcra-ambient'}


def cold_classes(root, disk):
    """[(kind, class)] of the direct and ambient recipes not yet in ``disk`` (in memory counts as kept): the shared work a
    model's materials wait for after this code changes, which viewer.loader.TranslationPool runs in parallel."""
    if disk is None:
        return []
    found, code = programs(root, disk), _code_key()
    cold = []
    for kind, prefix in RECIPE_TAGS.items():
        for cls in CLASSES:
            if cls not in found:
                continue
            blob = found[cls][1]
            tag = '%s:%d:%s' % (prefix, cls, code)
            with _LOCK:
                if (tag, hashlib.sha256(blob).hexdigest()) in _RECIPES:
                    continue
            if not disk.has(disk.key(blob, [], tag, own_code=True)):
                cold.append((kind, cls))
    return cold


def ambient(root, cls, disk=None):
    """The ambient-light recipe of class ``cls`` (export_recipe shape, terms ambient.r/g/b, with sample nodes on
    AMBIENT_SAMPLED), cached as ``direct``."""
    from rc2016.games.rcra import surface as rsurface
    offset, blob = programs(root, disk)[cls]
    tag = 'rcra-ambient:%d:%s' % (cls, _code_key())
    memo = (tag, hashlib.sha256(blob).hexdigest())
    with _LOCK:
        if memo in _RECIPES:
            return _RECIPES[memo]
    key = disk.key(blob, [], tag, own_code=True) if disk is not None else None
    recipe = _decode(disk.get(key)) if key else None
    if recipe is None:
        result = translate_ambient(dxbc.shader(blob), cls)
        g = result['graph']
        result.update(textures={}, limitations=[AMBIENT_NOTE % (cls, EXECUTABLE, offset)],
                      inputs=sorted({g.nodes[i]['name'] for i in g.reachable(list(result['terms'].values()))
                                     if g.nodes[i]['op'] == 'input'}))
        recipe = rsurface.fold_nan(rsurface.export_recipe(result))
        recipe.update(offset=offset, zeroed=result['zeroed'], lighting_class=cls)
        if key:
            disk.put(key, _encode(recipe))
    with _LOCK:
        _RECIPES[memo] = recipe
    return recipe


NON_FINITE = '__nonfinite__'


def _encode(value):
    """``value`` with inf and NaN floats tagged, for strict JSON (as rc2016.games.rcra.viewer_materials stores recipes)."""
    if isinstance(value, float) and (value != value or value in (float('inf'), float('-inf'))):
        return {NON_FINITE: repr(value)}
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    return value


def _decode(value):
    if isinstance(value, dict):
        if len(value) == 1 and NON_FINITE in value:
            return float(value[NON_FINITE])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def class_note(cls, offset):
    return (CLASS_NOTE % (cls, CLASS_NAMES[cls], cls, '%s at 0x%X' % (EXECUTABLE, offset))
            + (SKIN_NOTE if cls == 1 else '') + UNSHADOWED_NOTE)


def material_class(terms_graph, word):
    """The lighting class of a NormalSpec x word node of a writer: an int when its class bits are constant, else None."""
    value = terms_graph.value(word)
    if value is not None:
        return class_of(value)
    from rc2016.games.rcra import surface as rsurface
    fields = rsurface._Unpack(terms_graph).strip_or(word, dict(bits=0, parts=[]))
    if any(terms_graph.nodes[p]['op'] != 'ftou' for p in fields['parts']):
        return None   # class bits chosen at run time (a select of packed words)
    return class_of(fields['bits'])


def class_of(bits):
    """The class the apply programs read from NormalSpec x bits: 0 when bit 12 is set, else bits 13 to 15."""
    return 0 if bits & 0x1000 else (bits >> 13) & 7


def evaluate(recipe, values):
    """The light the recipe's class reflects at ``values`` ({input: float}), albedo x diffuse + specular per channel, by
    the translator's arithmetic (rc2016.games.rcra.surface)."""
    from rc2016.games.rcra import surface as rsurface
    out = rsurface.evaluate_recipe(recipe, values, lambda *a: (0.0, 0.0, 0.0, 0.0))
    return [values['albedo.' + c] * out['diffuse.' + c] + out['specular.' + c] for c in 'rgb']


def pack_extra(cls, fields):
    """The g_GBufExtra word of a class's field values (integers, in EXTRA_FIELDS order)."""
    word = 0
    for (offset, width), value in zip(EXTRA_FIELDS.get(cls, ()), fields):
        word |= (int(value) & ((1 << width) - 1)) << offset
    return word


# ------------------------------------------------------------------------------------------------ class 4 lobes
# The class-4 (anisotropic) apply program, as its translation reads (0x37E5A7C): the specular is the average of two
# anisotropic GGX lobes, D = 1 / (a_t a_b (h.n^2 + h.t^2 / a_t^2 + h.b^2 / a_b^2)^2) (no 1/pi, as every class), times a
# separable visibility and Schlick Fresnel with F90 = 1; the diffuse is Lambert, as class 0's.
# - lobe 1 on the G-buffer tangent t (extra fields 0, 1 and the z sign 2, decoded as the writers' octahedral normal) and
#   b = normalize(n x t);
# - lobe 2 on t turned about n by the rotation field x ROTATION_STEP radians (Rodrigues), and b' = normalize(n x t');
# - both with a_t = r(gloss)^4 along the tangent and a_b = r(second gloss)^4 across it, r = 0.9725 - 0.7514 g, the second
#   gloss being extra field 4 / 127;
# - visibility 0.25 / ((v.h (1 - k^2) + k^2)(n.l (1 - k^2) + k^2)), k = (0.75 min(a_t, a_b) + 0.25 max(a_t, a_b) + 1) / sqrt 8.
ANISOTROPIC = 4
TANGENT_STEP, TANGENT_BIAS = 0.00553508243, -1.41421354   # a 9-bit field to an octahedral coordinate
ROTATION_STEP = 0.0498665534                              # radians per step of the 6-bit rotation field
SECOND_GLOSS_STEP = 1.0 / 127.0                           # the 7-bit second gloss
# Cycles' anisotropic GGX (Metallic BSDF, measured in Blender 5.0 against the analytic lobe within 4e-6): with
# Roughness R and Anisotropy A in [0, 1] (negative reads 0, above 1 reads 1), alpha_x = R^2 / sqrt(1 - 0.9 A) along the Tangent input
# turned about the normal by 2 pi x Rotation (right-handed), alpha_y = R^2 sqrt(1 - 0.9 A) across it.
CYCLES_MAX_ANISOTROPY = 1.0


def anisotropic(recipe):
    """True when a surface recipe (export_recipe shape) is lit by class 4 and carries its five extra fields: the materials
    the Blender export lights with the class-4 lobes (rc2016.export.rcra_blender.anisotropic, its in-Blender twin)."""
    i = recipe['terms'].get('class')
    node = recipe['nodes'][i] if i is not None else None
    return (node is not None and node['op'] == 'const' and node['value'] == ANISOTROPIC
            and all('extra.%d' % k in recipe['terms'] for k in range(len(EXTRA_FIELDS[ANISOTROPIC]))))


def gloss_roughness(gloss):
    """r of the deferred apply programs: 0.9725 - 0.7514 gloss (GGX alpha = r^4)."""
    return 0.9725 - 0.7514 * gloss


def anisotropic_tangent(fx, fy, negative_z):
    """The class-4 tangent (game space, unit) of its extra fields 0, 1 and 2, as the apply program decodes them."""
    x, y = fx * TANGENT_STEP + TANGENT_BIAS, fy * TANGENT_STEP + TANGENT_BIAS
    q = x * x + y * y
    k = math.sqrt(max(1.0 - 0.25 * q, 0.0))
    z = 1.0 - 0.5 * q
    return x * k, y * k, -z if negative_z else z


def anisotropic_lobes(gloss, second_field, rotation_field):
    """(a_t, a_b, turn) of the class-4 lobes: alpha along and across the tangent, and the second lobe's turn in radians."""
    return (gloss_roughness(gloss) ** 4, gloss_roughness(second_field * SECOND_GLOSS_STEP) ** 4,
            rotation_field * ROTATION_STEP)


def cycles_anisotropic(gloss, second_field, rotation_field):
    """Cycles Metallic BSDF inputs (Roughness, Anisotropy, Rotation of lobe 1, Rotation of lobe 2) for the class-4 lobes.

    R = r1 r2 keeps alpha_x alpha_y = a_t a_b; A = (1 - (min r / max r)^4) / 0.9 gives their ratio. Cycles has no negative
    anisotropy, so when the second gloss is the rougher one the tangent is turned a quarter (Cycles' x axis onto b). The
    ratio is capped at 1 / (1 - 0.9) = 10 (CYCLES_MAX_ANISOTROPY), keeping the product."""
    r1, r2 = gloss_roughness(gloss), gloss_roughness(second_field * SECOND_GLOSS_STEP)
    anisotropy = min((1.0 - (min(r1, r2) / max(r1, r2)) ** 4) / 0.9, CYCLES_MAX_ANISOTROPY)
    quarter = 0.25 if r1 < r2 else 0.0
    return r1 * r2, anisotropy, quarter, quarter + rotation_field * ROTATION_STEP / (2.0 * math.pi)

