"""Rift Apart visual effects: the built ``.visualeffect`` file, its curves, and the conduit responses that attach
effects to a model's locators.

Layout sources (docs/rift-apart.md, Visual effects):

- **Section strides** come from the game's loader (``0x14136F9A0``), which derives every record count from its section
  size.
- **Render types** (node byte ``+0x17A``) are the editor enum (the exe's reflected names ``kSingleQuad`` ...
  ``kSingleDecalTrail``) with its one asset pair split three ways. Types 9 to 14 are named after the asset each node
  references, and the asset entry's type hash agrees on every node in the corpus (``ASSET_TYPES``).
- **Asset ids:** a node's model or child asset (``+0x00``), its material (``+0x08``) and a light's asset (``+0x104``)
  are path CRC64s (``toc.path_crc64_many``). The game's unload routine (``0x1413703E0``) releases them by render type.
- **The curve** (``0x14138D140``): the keys bracketing the time, linearly interpolated, times the header's |scale|. A
  slot's time rate (the half at slot byte +6) picks the time: 0 the particle's normalised age, clamped to 1; above 0
  ``frac(rate x seconds)`` (a loop); below 0 ``min(|rate| x seconds, 1)`` (played once).
- **The curve key** (``0x14138D0E0``): base (x, y, z) plus range (x, y, z) times a random vector its mode half picks
  against the fixed thresholds 1.9 and 2.9 (``0x14015A740``): mode 1 the particle's random value u, mode 2 a value
  seeded by the particle and the effect instance, mode 3 a coin, 1 when u > 0.5 else 0. Each is one value for x, y and
  z; the values come from a 255-entry table of uniform numbers in [0, 1) (``0x141393A20``).
- **Curve names:** ``EMITTER_CURVES`` from the spawn and update code that reads each slot (nwo.games.rcra.particles);
  ``NODE_CURVES`` 2 to 7 map onto the same render refs. Node slots 0 and 1 stay numbered.

Not decoded: how the renderer turns curve values into a draw's per-object values.
"""
import struct

import numpy as np

from nwo.games.rcra.toc import path_crc64_many, sections

TOC_HEADER = 36
NODES = 0xB86CF442
EMITTERS = 0xC3CC2AB5
CURVES = 0xD78E007C
KEYS = 0xB1F4C248
MATERIALS = 0xAF650E9E
ASSETS = 0xB40E134B
SETTINGS = 0x73CE6A3F
SHAPES = 0x3F03BE86   # spawn-shape records, 0x78 bytes; an emitter's u32 +0x118 indexes them (0x14139551A)
RECORD = 448
# Record sizes the loader divides each section by (0x14136F9A0).
STRIDES = {SHAPES: 120, 0xEFD8600D: 16, 0x01670690: 20, EMITTERS: RECORD, NODES: RECORD, MATERIALS: 8,
           0xECC0D7AF: 24, ASSETS: 12, KEYS: 16, SETTINGS: 32, CURVES: 16}

RENDER_TYPES = ('SingleQuad', 'MultiQuads', 'SingleLight', 'MultiLights', 'SingleSwipe', 'SingleTrail', 'MultiTrails',
                'RadialBlur', 'MultiSplines', 'SingleEffect', 'MultiEffects', 'SingleModel', 'MultiModels',
                'SingleActor', 'MultiActors', 'SunLensFlare', 'ForceField', 'SingleDecal', 'MultiDecals',
                'SingleTube', 'SingleDecalTrail')
# Asset entry type hashes (the loader's switch at 0x14137030E) and the kinds they name.
ASSET_TYPES = {0xA4070B70: 'model', 0xDABA2AEA: 'visualeffect', 0xEA7EFDD4: 'conduit', 0x37E72F50: 'actor'}
# A node's eight curve slots. Multi types carry none: their particles take the linked emitter's slots 12 to 19. Single
# types read slots 2 to 7 as the particle update reads emitter slots 14 to 19 (0x14137DC67).
NODE_CURVES = ('curve0', 'curve1', 'luminance', 'rotation', 'custom', 'scales', 'color', 'alpha')
# An emitter's twenty slots, named from the code that reads them: the spawn loop (0x14137C7B0: speed, burst, per-metre,
# rate, distance gap, life, emitter rotation and offset, shape scales) and the curve refs the particle update reads
# (built at 0x14137DC67: size at spawn, follow, acceleration, turbulence, velocity, then the render curves).
EMITTER_CURVES = ('size_at_spawn', 'speed', 'burst', 'per_metre', 'rate', 'gap', 'life', 'emitter_rotation',
                  'emitter_position', 'shape_scales', 'follow', 'acceleration', 'turbulence', 'velocity', 'luminance',
                  'rotation', 'custom', 'scales', 'color', 'alpha')
KEY_TIME = 65535.0
# Light nodes (render types 2 and 3): byte +0x17B in the reflected order kPyramidLight, kPointLight, kSpotLight
# (0x14137A1E0 maps it onto the engine's light kinds), range +0x130, spot cone +0x120 / +0x124 degrees.
LIGHT_TYPES = ('pyramid', 'point', 'spot')
# Spawn-shape samplers by the shape record's type hash (+0x0C), named from their sampling maths (the hashes are not a
# plain CRC32 or FNV of the reflected names); the spawn context picks the sampler (0x141395510), BD6A18A2 by default.
SHAPE_KINDS = {0xBD6A18A2: 'cone', 0x323F9580: 'disc', 0xD8E0E141: 'plane', 0xD6F48302: 'box_faces',
               0x22FC61F3: 'box_projected', 0xCEC6BE18: 'model_surface'}
# Emitter record fixed fields read by the spawn and update code (0x14137C7B0, 0x141395800, 0x14137181C ...): name,
# offset, struct format. Names marked inferred in the docs are labelled there.
EMITTER_FIELDS = (('accel_direction', 0x30, '3f'), ('inherit_velocity', 0x48, 'f'), ('gravity', 0x4C, 'f'),
                  ('wind', 0x50, 'f'), ('drag', 0x54, 'f'), ('lifetime', 0x64, 'f'), ('lifetime_random', 0x68, 'f'),
                  ('delay', 0x6C, 'f'), ('delay_random', 0x70, 'f'), ('rotation_range', 0x80, '3f'),
                  ('rotation_base', 0x8C, '3f'), ('flags', 0x104, 'I'), ('max_spawned', 0x108, 'i'),
                  ('death_burst', 0x10C, 'i'), ('space', 0x115, 'B'), ('shape', 0x118, 'I'), ('max_alive', 0x11C, 'I'))
# Key mode thresholds (0x146796D80 and 0x146796D70, set from 0x1444CF8F0 and 0x1444CF900): above 2.9 the coin, above
# 1.9 the particle-and-instance value, else the particle's own value.
MODE_INSTANCE, MODE_COIN = 1.9, 2.9


def _strings(data, base):
    def cstring(offset):
        start = base + offset
        return data[start:data.index(b'\0', start)].decode('latin-1')
    return cstring


def _path(raw):
    return raw.replace('\\', '/').lower()


def _records(data, base, table, tag):
    if tag not in table:
        return []
    offset, size = table[tag]
    stride = STRIDES[tag]
    if size % stride:
        raise ValueError('Section %08X size %d is not a multiple of %d' % (tag, size, stride))
    return [data[base + offset + stride * i:base + offset + stride * (i + 1)] for i in range(size // stride)]


def _floats(record, offset, count):
    return tuple(float(v) for v in struct.unpack_from('<%df' % count, record, offset))


def _slots(record, offset, count, curve_count):
    """[(curve index, override)]: the low 16 bits index the curves; the high 16 bits name a run-time override of the
    effect instance (0 none; 0x14138D350 multiplies, replaces or adds its vector by the override's operation)."""
    slots = []
    for i in range(count):
        word = struct.unpack_from('<I', record, offset + 8 * i)[0]
        index = word & 0xFFFF
        if index >= curve_count:
            raise ValueError('Curve slot %d names curve %d of %d' % (i, index, curve_count))
        slots.append((index, word >> 16))
    return slots


def _rates(record, offset, count):
    """Each slot's time rate: the half at slot byte +6 (0x14138D350 passes it to the evaluator as the time scale)."""
    return [float(np.frombuffer(record, '<f2', 1, offset + 8 * i + 6)[0]) for i in range(count)]


def decode_curves(data, base, table):
    """[{'hash', 'scale', 'keys'}]: each key is (base xyz, time 0..1, range xyz, mode). Values are multiplied by |scale|
    (curve_value); its sign picks the effect instance's times over the particle's (0x14138D4C0)."""
    headers = _records(data, base, table, CURVES)
    keys = _records(data, base, table, KEYS)
    raw = np.frombuffer(b''.join(keys), '<u2').reshape(-1, 8) if keys else np.zeros((0, 8), np.uint16)
    halves = raw.view('<f2').astype(np.float32)
    curves, expect = [], 0
    for header in headers:
        hash_, first, count, scale = struct.unpack('<IIIf', header)
        if first != expect or first + count > len(raw):
            raise ValueError('Curve keys %d..%d do not follow on from %d' % (first, first + count, expect))
        expect = first + count
        curves.append(dict(hash=hash_, scale=scale, keys=[
            dict(base=tuple(halves[k, :3].tolist()), time=raw[k, 3] / KEY_TIME if count > 1 else 0.0,
                 range=tuple(halves[k, 4:7].tolist()), mode=float(halves[k, 7]))
            for k in range(first, first + count)]))
    if expect != len(raw):
        raise ValueError('Curves leave %d of %d keys unused' % (len(raw) - expect, len(raw)))
    return curves


def curve_time(rate, seconds, normalised):
    """The key time of a slot with time rate ``rate`` (0x14138D140): 0 the normalised age clamped to 1, above 0 a loop
    ``frac(rate x seconds)``, below 0 ``min(|rate| x seconds, 1)``."""
    if rate > 0:
        value = rate * seconds
        return value - np.floor(value)
    return min(abs(rate) * seconds if rate < 0 else normalised, 1.0)


def key_random(mode, random, instance_random=None):
    """The random value a key of ``mode`` multiplies its range by (0x14138D0E0 with the vectors of 0x141393A20): the
    particle's value for mode 1, the particle-and-instance value for mode 2, and for mode 3 a coin, 1 when the
    particle's value is above 0.5."""
    if mode > MODE_COIN:
        return 1.0 if random > 0.5 else 0.0
    if mode > MODE_INSTANCE:
        return random if instance_random is None else instance_random
    return random


def curve_value(curve, t=0.0, random=0.0, instance_random=None):
    """(x, y, z) of a curve at key time t (0x14138D140): the bracketing keys' base + range x their mode's random value
    (key_random, one value for x, y and z), interpolated linearly, times the curve's |scale|.

    ``random`` is the particle's random value in [0, 1) and ``instance_random`` the mode 2 value (``random`` when
    None). Before the first key and after the last the end keys hold; a looping slot's wrap from the last key to the
    first is not interpolated here."""
    keys = curve['keys']
    if len(keys) == 1 or t <= keys[0]['time']:
        a = b = keys[0]
        w = 0.0
    elif t >= keys[-1]['time']:
        a = b = keys[-1]
        w = 0.0
    else:
        i = next(i for i in range(1, len(keys)) if keys[i]['time'] >= t)
        a, b = keys[i - 1], keys[i]
        span = b['time'] - a['time']
        w = (t - a['time']) / span if span > 0 else 0.0
    ra, rb = key_random(a['mode'], random, instance_random), key_random(b['mode'], random, instance_random)
    scale = abs(curve['scale'])
    return tuple(float(scale * ((1 - w) * (a['base'][c] + a['range'][c] * ra) + w * (b['base'][c] + b['range'][c] * rb)))
                 for c in range(3))


def decode_effect(data):
    """The effect's materials, assets, nodes, emitters and curves.

    ``data`` is the asset as stored (36-byte toc header, then the DAT1). Each node holds its render type, the paths of
    its asset and material (resolved through the effect's own lists, else None), its transform and scale, its draw
    order and its curve slots. A material is its path, or an inline material's name. Each emitter holds its name hash,
    transform, curve slots and the fixed fields of EMITTER_FIELDS; ``shapes`` are its spawn-shape records. ``settings`` is the first three floats of section 0x73CE6A3F (not traced)."""
    base = TOC_HEADER
    if data[base:base + 4] != b'1TAD':
        raise ValueError('Not a DAT1 visual effect')
    table = sections(data, base)
    cstring = _strings(data, base)
    materials = []
    for record in _records(data, base, table, MATERIALS):
        offset, inline = struct.unpack('<Ii', record)
        # An inline material is built from a material graph inside the effect: its id is the CRC64 of its node name
        # ("materialnode: material_15, <graph path>"), and the path is the graph's.
        materials.append(dict(path=_path(cstring(offset)), inline=None if inline == -1 else _path(cstring(inline))))
    assets = []
    for record in _records(data, base, table, ASSETS):
        offset, _unknown, type_hash = struct.unpack('<III', record)
        assets.append(dict(path=_path(cstring(offset)), type_hash=type_hash, kind=ASSET_TYPES.get(type_hash)))
    names = [m['inline'] or m['path'] for m in materials] + [a['path'] for a in assets]
    by_id = {}
    for name, crc in zip(names, path_crc64_many(names)):
        by_id.setdefault(int(crc), name)
    curves = decode_curves(data, base, table)
    shapes = []
    for i, record in enumerate(_records(data, base, table, SHAPES)):
        hash_ = struct.unpack_from('<I', record, 0x0C)[0]
        shapes.append(dict(index=i, hash=hash_, kind=SHAPE_KINDS.get(hash_), raw=record))
    emitters = []
    for i, record in enumerate(_records(data, base, table, EMITTERS)):
        fields = {}
        for name, offset, fmt in EMITTER_FIELDS:
            value = struct.unpack_from('<' + fmt, record, offset)
            fields[name] = tuple(float(v) for v in value) if len(value) > 1 else value[0]
        emitters.append(dict(
            fields, index=i, name_hash=struct.unpack_from('<I', record, 0x100)[0],
            rotation=_floats(record, 0x00, 9), position=_floats(record, 0x24, 3),
            curves=dict(zip(EMITTER_CURVES, _slots(record, 0x120, 20, len(curves)))),
            rates=dict(zip(EMITTER_CURVES, _rates(record, 0x120, 20))), raw=record))
    emitter_by_hash = {e['name_hash']: e['index'] for e in emitters}
    nodes = []
    for i, record in enumerate(_records(data, base, table, NODES)):
        kind = record[0x17A]
        if kind >= len(RENDER_TYPES):
            raise ValueError('Node %d has render type %d' % (i, kind))
        asset_id, material_id = struct.unpack_from('<QQ', record, 0)
        emitter_hash = struct.unpack_from('<I', record, 0x10)[0]
        light_id = struct.unpack_from('<Q', record, 0x104)[0] if kind in (2, 3) else 0
        multi = RENDER_TYPES[kind].startswith('Multi')
        light = None
        if kind in (2, 3):
            inner, outer = struct.unpack_from('<2f', record, 0x120)
            light = dict(type=LIGHT_TYPES[record[0x17B]] if record[0x17B] < len(LIGHT_TYPES) else record[0x17B],
                         range=struct.unpack_from('<f', record, 0x130)[0], inner_degrees=inner, outer_degrees=outer)
        nodes.append(dict(
            index=i, render_type=kind, render_type_name=RENDER_TYPES[kind],
            asset_id=asset_id, asset=by_id.get(asset_id), material_id=material_id, material=by_id.get(material_id),
            light_id=light_id, light=light, emitter=emitter_by_hash.get(emitter_hash) if emitter_hash else None,
            rotation=_floats(record, 0x20, 9), position=_floats(record, 0x44, 3), scale=_floats(record, 0xC4, 3),
            sort_order=record[0x17F],
            curves=None if multi else dict(zip(NODE_CURVES, _slots(record, 0x180, 8, len(curves)))),
            rates=None if multi else dict(zip(NODE_CURVES, _rates(record, 0x180, 8))), raw=record))
    settings = _records(data, base, table, SETTINGS)
    return dict(materials=materials, assets=assets, nodes=nodes, emitters=emitters, curves=curves, shapes=shapes,
                settings=_floats(settings[0], 0, 3) if settings else None)


def _serialized(data, tag):
    """The ALERT dat1lib reading of a serialized section (the guarded generic reader)."""
    from nwo.research import cross_reference as xr
    xr.load_library()
    from dat1lib.types.sections import SerializedSection
    base = TOC_HEADER
    table = sections(data, base)
    if tag not in table:
        return None
    offset, size = table[tag]
    cstring = _strings(data, base)

    class Strings:
        version = None

        def get_string(self, at):
            return cstring(at)

    reader = xr.guarded_serialized(SerializedSection)(data[base + offset:base + offset + size], Strings())
    return reader.root


CONDUIT_BUILT = 0xCEB30E68


def _items(value):
    """The dicts of a serialized array, which stores a one-element array as the bare element (an event map as a
    dict keyed by hash)."""
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        if value and all(isinstance(k, int) or str(k).isdigit() for k in value):
            return [v for v in value.values() if isinstance(v, dict)]
        return [value]
    return []


def conduit_effects(data):
    """[{'event', 'label', 'response', 'effect', 'locator', 'start'}] of a conduit's kVFX responses.

    ``start`` is False for a response that stops its effect (a ``StartEffect`` field). ``label`` is the event label the
    filter selects on (``start_fx``), or None for an unfiltered event such as ``ActivatedEvent``."""
    root = _serialized(data, CONDUIT_BUILT)
    if not isinstance(root, dict):
        return []
    found = []
    for event in _items(root.get('Events')):
        for flt in _items(event.get('Filters')):
            label = None
            selector = flt.get('Selector')
            for value in _items(selector.get('Value') if isinstance(selector, dict) else None):
                if value.get('Type') == 'kFieldConst':
                    label = value.get('Value')
            for response in _items(flt.get('Response')):
                if response.get('Type') != 'kVFX':
                    continue
                fields = {}
                for field in _items(response.get('FieldMap')):
                    dest, source = field.get('Dest'), field.get('Source')
                    if isinstance(dest, dict) and isinstance(source, dict):
                        fields[dest.get('Value')] = source.get('Value')
                if not fields.get('FXPackage'):
                    continue
                found.append(dict(event=event.get('Name'), label=label, response=response.get('Name'),
                                  effect=_path(fields['FXPackage']), locator=fields.get('AttachLocJoint'),
                                  start='StartEffect' not in fields))
    return found
