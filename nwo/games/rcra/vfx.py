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
- **The curve key** (``0x14138D0E0``): base (x, y, z) plus range (x, y, z) times one of three random vectors that a
  mode half picks.
- **Curve names** (``NODE_CURVES``, ``EMITTER_CURVES``) are read from corpus value distributions (colour (0.69, 0.21,
  1), rotation (0, 0, 90), gravity 9.8 ...), not from code. Slots without a confident reading stay numbered.

Not decoded: the random vectors the mode half selects (two thresholds live in run-time data), key-to-key
interpolation, the emitter spawn fields, and how the renderer turns curve values into a draw's per-object values.
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
GROUPS = 0x3F03BE86
RECORD = 448
# Record sizes the loader divides each section by (0x14136F9A0).
STRIDES = {GROUPS: 120, 0xEFD8600D: 16, 0x01670690: 20, EMITTERS: RECORD, NODES: RECORD, MATERIALS: 8,
           0xECC0D7AF: 24, ASSETS: 12, KEYS: 16, SETTINGS: 32, CURVES: 16}

RENDER_TYPES = ('SingleQuad', 'MultiQuads', 'SingleLight', 'MultiLights', 'SingleSwipe', 'SingleTrail', 'MultiTrails',
                'RadialBlur', 'MultiSplines', 'SingleEffect', 'MultiEffects', 'SingleModel', 'MultiModels',
                'SingleActor', 'MultiActors', 'SunLensFlare', 'ForceField', 'SingleDecal', 'MultiDecals',
                'SingleTube', 'SingleDecalTrail')
# Asset entry type hashes (the loader's switch at 0x14137030E) and the kinds they name.
ASSET_TYPES = {0xA4070B70: 'model', 0xDABA2AEA: 'visualeffect', 0xEA7EFDD4: 'conduit', 0x37E72F50: 'actor'}
# A node's eight curve slots. Multi types carry none: their particles take the linked emitter's slots 12 to 19.
NODE_CURVES = ('curve0', 'curve1', 'luminance', 'rotation', 'custom', 'scales', 'color', 'alpha')
EMITTER_CURVES = tuple('curve%d' % i for i in range(12)) + NODE_CURVES
EMITTER_CURVES = EMITTER_CURVES[:7] + ('emitter_rotation', 'emitter_position') + EMITTER_CURVES[9:]
KEY_TIME = 65535.0


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
    """[(curve index, flags)]: the low 16 bits index the curves, the high 16 bits are a flag or modifier number."""
    slots = []
    for i in range(count):
        word = struct.unpack_from('<I', record, offset + 8 * i)[0]
        index = word & 0xFFFF
        if index >= curve_count:
            raise ValueError('Curve slot %d names curve %d of %d' % (i, index, curve_count))
        slots.append((index, word >> 16))
    return slots


def decode_curves(data, base, table):
    """[{'hash', 'scale', 'keys'}]: each key is (base xyz, time 0..1, range xyz, mode)."""
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


def curve_value(curve, t=0.0, random=(0.0, 0.0, 0.0)):
    """(x, y, z) of a curve at key time t: base + range * random, the game's key form (0x14138D0E0).

    ``random`` stands for the vector the key's mode selects. Between keys the value is interpolated linearly, which is
    a viewer choice: the game's key-to-key interpolation is not traced."""
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
    return tuple((1 - w) * (a['base'][c] + a['range'][c] * random[c]) + w * (b['base'][c] + b['range'][c] * random[c])
                 for c in range(3))


def decode_effect(data):
    """The effect's materials, assets, nodes, emitters and curves.

    ``data`` is the asset as stored (36-byte toc header, then the DAT1). Each node holds its render type, the paths of
    its asset and material (resolved through the effect's own lists, else None), its transform and scale, its draw
    order and its curve slots. A material is its path, or an inline material's name. Each emitter holds its name hash,
    transform and curve slots. ``settings`` is the first three floats of section 0x73CE6A3F (not traced)."""
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
    emitters = []
    for i, record in enumerate(_records(data, base, table, EMITTERS)):
        emitters.append(dict(
            index=i, name_hash=struct.unpack_from('<I', record, 0x100)[0],
            rotation=_floats(record, 0x00, 9), position=_floats(record, 0x24, 3),
            curves=dict(zip(EMITTER_CURVES, _slots(record, 0x120, 20, len(curves)))), raw=record))
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
        nodes.append(dict(
            index=i, render_type=kind, render_type_name=RENDER_TYPES[kind],
            asset_id=asset_id, asset=by_id.get(asset_id), material_id=material_id, material=by_id.get(material_id),
            light_id=light_id, emitter=emitter_by_hash.get(emitter_hash) if emitter_hash else None,
            rotation=_floats(record, 0x20, 9), position=_floats(record, 0x44, 3), scale=_floats(record, 0xC4, 3),
            sort_order=record[0x17F],
            curves=None if multi else dict(zip(NODE_CURVES, _slots(record, 0x180, 8, len(curves)))), raw=record))
    settings = _records(data, base, table, SETTINGS)
    return dict(materials=materials, assets=assets, nodes=nodes, emitters=emitters, curves=curves,
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
