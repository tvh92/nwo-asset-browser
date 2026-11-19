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
(SKIN_NOTE). The skin profiles (g_SkinProfiles, an engine texture filled at run time) read 1.
"""
from __future__ import annotations

import hashlib
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
    1: ((0, 3), (9, 6), (18, 7), (25, 7)),            # scatter, transmission, second normal (octahedral, 7 + 7 bits)
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
SKIN_NOTE = (' The skin pass blurs the diffuse light in screen space with the skin profiles before adding the specular; '
             'the blur is not drawn (albedo x diffuse + specular at each pixel), and the skin profiles, an engine texture '
             'filled at run time, read 1.')
UNSHADOWED_NOTE = (' Lights are unshadowed (shadow maps, cloud shadows and gobos read 1; transmission that the game takes '
                   'from the shadow map depth reads it at the lit surface).')
ONE_RESOURCES = ('g_ShadowAtlas', 'g_CloudShadowMap', 'g_RayTracedShadows', 'g_GoboAtlas', 'g_AoSsrBuffer', SKIN_PROFILES,
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


def programs(root):
    """find_programs of the install's executable (cached by path, size and time)."""
    path = os.path.join(str(root), EXECUTABLE)
    stat = os.stat(path)
    key = (path, stat.st_size, stat.st_mtime_ns)
    with _LOCK:
        if key not in _PROGRAMS:
            with open(path, 'rb') as f:
                _PROGRAMS[key] = find_programs(f.read())
        return _PROGRAMS[key]


# ------------------------------------------------------------------------------------------------ translation

class _ApplyBindings(sm5.Bindings):
    """The G-buffer packed from term inputs, one unshadowed key light, the view from VIEW and DEPTH, every other engine
    input 0 (recorded in ``zeroed``)."""

    def __init__(self, cls):
        self.cls = cls
        self.graph = None
        self.zeroed = set()
        self._words = None

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
    return dict(graph=g, terms=terms, zeroed=sorted(bindings.zeroed))


def _portable(g, terms):
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
    if any(g.nodes[i]['op'] in ('sample', 'load') for i in g.reachable(list(terms.values()))):
        raise rsurface.Unsupported('Apply program reads an engine texture')
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
    offset, blob = programs(root)[cls]
    tag = 'rcra-lighting:%d:%s' % (cls, _code_key())
    memo = (tag, hashlib.sha256(blob).hexdigest())
    with _LOCK:
        if memo in _RECIPES:
            return _RECIPES[memo]
    key = disk.key(blob, [], tag) if disk is not None else None
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

