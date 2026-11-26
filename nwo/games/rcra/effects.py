"""Visual effects a model's actors start when they activate, drawn as extra meshes on the model in the viewer
(docs/rift-apart.md, Effects in the viewer).

The chain is decoded (nwo.games.rcra.vfx): the model's actors -> their conduits -> the ``ActivatedEvent`` responses that
start an effect at a locator (or joint) -> the effect's nodes. Each drawn node becomes a mesh weighted to the locator's
joint, so it follows the animation:

- **SingleModel / MultiModels:** the node's model with its material, at the node's position, rotation and scale
  relative to the locator; a MultiModels node draws it once per live particle, at the particle's position, size and
  rotation.
- **SingleQuad / MultiQuads:** one quad per node or per live particle, of full size node scale x the size.

A Multi node's particles come from nwo.games.rcra.particles: a replay of the traced spawn and update code to a still
emitter age. The node's facing (byte +0x178) places them: kLocal and kWorld in the effect's XY plane, kCamera and
kBillboard toward the view in the vertex program (rcra_glsl.BILLBOARD_VERTEX), as the facing switch (0x14139E020) does
each frame.

Lights, trails, swipes, decals, lens flares, child effects and the other render types are left out and listed.

A conduit counts as the model's when every locator its activation responses name is a locator or joint of the model: a
shared conduit (the gallery's ``hero_gallery_vfx``, which names the locators of many models) is skipped.

Values a draw binds (the particle update, 0x141391F00, traced): colour = the colour x luminance curves, alpha = the alpha
curve's x. A model node binds them as its per-object colour and alpha; a quad node as the TEXCOORD4 its program reads.
One draw state stands for a node, so it takes the values of the particle nearest mid-life.

Viewer choices, in the material notes: curves with every random value at 0.5; a Single node at mid-life (t 0.5) and its
rated slots at SINGLE_SECONDS; a node's 3x3 applied with its nine floats as rows (not traced; the vendor's logo plane
has the identity); particle angles turned about X, then Y, then Z.
"""
import numpy as np

from nwo.games.rcra import vfx

NODE_QUAD = vfx.RENDER_TYPES.index('SingleQuad')
NODE_QUADS = vfx.RENDER_TYPES.index('MultiQuads')
NODE_MODEL = vfx.RENDER_TYPES.index('SingleModel')
NODE_MODELS = vfx.RENDER_TYPES.index('MultiModels')
DRAWN = (NODE_QUAD, NODE_QUADS, NODE_MODEL, NODE_MODELS)
MULTI = (NODE_QUADS, NODE_MODELS)
FACINGS = ('camera', 'local', 'world', 'billboard')   # node byte +0x178, in the reflected kCamera ... kBillboard order
# rcra_glsl.BILLBOARD_VERTEX's facing codes: the view's axes (kCamera) or turned toward the eye, native Y up (kBillboard).
FACING_CODES = {'camera': 1.0, 'billboard': 2.0}
CURVE_TIME = 0.5
CURVE_RANDOM = 0.5
# TEXCOORD6.z of a particle with random byte b: frac(round(b x 171.6) x 0.0022883), for b = 128 (u 0.5).
RANDOM_TERM = float((round(128 * 171.6) * 0.0022883) % 1.0)
SINGLE_SECONDS = 1.0   # the still's time for a Single node's slots with a time rate (a viewer choice)
MODEL_NOTE = ('Visual effect mesh: node %d (%s, %s facing) of %s, started by %s at %s. Drawn as %s. Per-object values '
              'from the curves through the traced particle update and object writer: %s (colour x luminance, alpha x); '
              'object Misc and Rand 0.%s')
QUAD_NOTE = ('Visual effect quads: node %d (%s, %s facing) of %s, started by %s at %s. Drawn as %s. Interpolants as the '
             'engine particle vertex program writes them (TEXCOORD4 colour, TEXCOORD.w alpha, TEXCOORD6.y age, TEXCOORD7 '
             'custom): %s. The object custom and Misc factors the program applies are not traced and read 1; no near '
             'fade.%s')
LEFT_NOTE = ' Not drawn from this effect: %s.'


def _kind(path):
    name = path.rsplit('/', 1)[-1]
    return name.rsplit('.', 1)[-1] if '.' in name else ''


def activated_effects(source, model_path, attach_names):
    """[{'effect', 'locator', 'conduit', 'actor'}]: the effects the model's actors start on activation, deduplicated by
    (effect, locator), from the conduits whose activation responses only name ``attach_names`` (locators and joints).
    Responses behind a game-state filter (``condition``, such as a mission name) are left out: they belong to a
    look the model only has in those states."""
    from nwo.games.rcra.clip_library import dag_index
    index = dag_index(source)
    model = index.by_path.get(model_path.replace('\\', '/').lower())
    if model is None:
        return []
    found, seen = [], set()
    for actor in index.users.get(model, []):
        if _kind(index.paths[actor]) != 'actor':
            continue
        for conduit in index.deps(actor):
            path = index.paths[conduit]
            if _kind(path) != 'conduit':
                continue
            keys = source.resolve(path)
            if not keys:
                continue
            try:
                responses = vfx.conduit_effects(source.asset_data(keys[0]))
            except (ValueError, KeyError, OSError):
                continue
            started = [r for r in responses if r['event'] == 'ActivatedEvent' and r['start'] and not r['condition']]
            if not started or any(r['locator'] and r['locator'] not in attach_names for r in started):
                continue
            for r in started:
                if (r['effect'], r['locator']) not in seen:
                    seen.add((r['effect'], r['locator']))
                    found.append(dict(effect=r['effect'], locator=r['locator'], conduit=path, actor=index.paths[actor]))
    return found


def joint_rest_worlds(data, joint_count):
    """The joints' rest model-space matrices: the inverses of the inverse binds in section 0xDCC88A19 (after the rest
    SQT rows, at the next 64-byte boundary; formats.skeleton.inspect_model), or None when they are not stored."""
    from nwo.games.rcra.model import JOINT_TRANSFORMS
    from nwo.games.rcra.toc import sections
    dat1 = bytes(data)[36:]
    table = sections(dat1)
    if JOINT_TRANSFORMS not in table or not joint_count:
        return None
    offset, size = table[JOINT_TRANSFORMS]
    start = (joint_count * 48 + 63) // 64 * 64
    if size < start + joint_count * 64:
        return None
    inverse = np.frombuffer(dat1, '<f4', joint_count * 16, offset + start).reshape(-1, 4, 4).transpose(0, 2, 1)
    return np.linalg.inv(inverse.astype(np.float64))


def _attach_matrix(decoded, worlds, name):
    """(model-space matrix, joint or None) of an attach point: a locator (relative to its joint) or a joint."""
    for locator in decoded.get('locators') or []:
        if locator['name'] == name:
            joint = locator['joint']
            local = np.asarray(locator['matrix'], np.float64)
            if joint is None:
                return local, None
            if worlds is None:
                return None, None
            return worlds[joint] @ local, joint
    for i, joint in enumerate(decoded.get('joints') or []):
        if joint['name'] == name and worlds is not None:
            return worlds[i], i
    return None, None


def node_facing(node):
    """The node's facing (byte +0x178: kCamera, kLocal, kWorld, kBillboard), 'unknown' past the enum."""
    value = node['raw'][0x178]
    return FACINGS[value] if value < len(FACINGS) else 'unknown'


def facing_mesh(source_mesh, centre, local, facing):
    """A mesh of ``source_mesh`` facing the view (rcra_glsl.BILLBOARD_VERTEX): every vertex at ``centre`` (model space),
    its offset ``local`` (3x3, the node's own frame) x position in 'billboard' with the facing code, and its normal and
    tangent in the same frame."""
    positions = np.asarray(source_mesh['positions'], np.float64).reshape(-1, 3)
    count = len(positions)
    linear = np.asarray(local, np.float64)
    offsets = positions @ linear.T
    normals = np.asarray(source_mesh['normals'], np.float64).reshape(-1, 3) @ np.linalg.inv(linear).T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    tangents = np.asarray(source_mesh['tangents'], np.float64).reshape(-1, 3) @ linear.T
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    mesh = dict(source_mesh)
    mesh.update(positions=np.tile(np.asarray(centre, np.float64), (count, 1)), normals=normals, tangents=tangents,
                billboard=np.column_stack((offsets, np.full(count, FACING_CODES[facing]))))
    return mesh


def node_matrix(node):
    """The node's transform: position, then the 3x3 (rows as stored), then the scale."""
    m = np.eye(4)
    m[:3, :3] = np.asarray(node['rotation'], np.float64).reshape(3, 3) @ np.diag(node['scale'])
    m[:3, 3] = node['position']
    return m


def _mesh(source_mesh, matrix, index, material, joint, as_arrays):
    from nwo.games.rcra.model import SkinLists
    positions = np.asarray(source_mesh['positions'], np.float64).reshape(-1, 3)
    count = len(positions)
    linear = matrix[:3, :3]
    normal_matrix = np.linalg.inv(linear).T
    normals = np.asarray(source_mesh['normals'], np.float64).reshape(-1, 3) @ normal_matrix.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    tangents = np.asarray(source_mesh['tangents'], np.float64).reshape(-1, 3) @ linear.T
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    placed = positions @ linear.T + matrix[:3, 3]
    mesh = dict(source_mesh)
    mesh.update(index=index, material=material, material_index=material['index'], memberships=[],
                packed_normals=None, raw_vectors=None, static_vector_eligible=False)
    if joint is None:
        mesh.update(weights=[None] * count, skinned=False, unweighted_vertices=count)
        mesh.pop('skin_joint_array', None)
        mesh.pop('skin_weight_array', None)
    else:
        joints = np.zeros((count, 8), np.int64)
        joints[:, 0] = joint
        values = np.zeros((count, 8))
        values[:, 0] = 1.0
        mesh.update(skinned=True, unweighted_vertices=0, skin_joint_array=joints if as_arrays else None,
                    skin_weight_array=values if as_arrays else None,
                    weights=SkinLists(joints, values) if as_arrays else [[(joint, 1.0)]] * count)
        if not as_arrays:
            mesh.pop('skin_joint_array')
            mesh.pop('skin_weight_array')
    if as_arrays:
        mesh.update(positions=placed, normals=normals, tangents=tangents)
    else:
        mesh.update(positions=[tuple(p) for p in placed.tolist()], normals=[tuple(n) for n in normals.tolist()],
                    tangents=[tuple(t) for t in tangents.tolist()])
    return mesh


def single_values(effect, node, seconds):
    """A Single node's render values at mid-life (t 0.5; a slot with a time rate at ``seconds``): colour (colour x
    luminance), alpha, custom, scales and rotation (radians), as the particle update reads them through node slots 2 to
    7 (0x14137DC67, 0x141391F00)."""
    def value(name):
        index = node['curves'][name][0]
        time = vfx.curve_time(node['rates'][name], seconds, CURVE_TIME)
        return np.asarray(vfx.curve_value(effect['curves'][index], time, CURVE_RANDOM), np.float64)
    alpha = value('alpha')
    return dict(colour=value('color') * value('luminance')[0], alpha=float(alpha[0]), alpha_test=float(alpha[1]),
                custom=value('custom'), size=value('scales'), rotation=np.radians(value('rotation')))


def object_values(values):
    """Per-object record members of an effect model draw (shaders.NEUTRAL_OBJECT form), as the object writer
    (0x1410FDC00) fills them from the draw the model-node routine (0x141377EC0) sets: colour RGB in m_ColorRGFp16 and
    the low half of m_ColorBMiscFp16 (Misc 0, the draw's aux index left at 0xFFFF), alpha, custom X, Y and Z (the Rand
    half 0: its table index is not set from the particle)."""
    from nwo.games.rcra.shaders import half_pair
    rgb, custom = values['colour'], values['custom']
    return {'m_ColorRGFp16': (half_pair(rgb[0], rgb[1]),), 'm_ColorBMiscFp16': (half_pair(rgb[2], 0.0),),
            'm_Alpha': (float(values['alpha']),), 'm_CustomXYFp16': (half_pair(custom[0], custom[1]),),
            'm_CustomZRandFp16': (half_pair(custom[2], 0.0),)}


def quad_interpolants(values):
    """A quad draw's interpolants as the engine particle vertex program writes them from a particle (the program in the
    exe reading g_VisualEffectParticlesU4): TEXCOORD4 = (RGB, 1), TEXCOORD.z = the alpha-test threshold, TEXCOORD.w =
    alpha, TEXCOORD6.y = the normalised age, TEXCOORD6.z = the particle random term, TEXCOORD7.xyz = custom. The program
    also multiplies RGB by the object's custom X, Y, Z and alpha by its Misc half and a near fade; the object record a
    quad draw binds is not traced, so those are 1 here."""
    out = {'TEXCOORD4.%s' % c: float(v) for c, v in zip('xyz', values['colour'][:3])}
    out.update({'TEXCOORD4.w': 1.0, 'TEXCOORD.z': float(values['alpha_test']), 'TEXCOORD.w': float(values['alpha']),
                'TEXCOORD6.y': float(values['t']), 'TEXCOORD6.z': RANDOM_TERM})
    out.update({'TEXCOORD7.%s' % c: float(v) for c, v in zip('xyz', values['custom'][:3])})
    return out


def _corners(half_x, half_y, angle):
    c, s = np.cos(angle), np.sin(angle)
    return [(c * x * half_x - s * y * half_y, s * x * half_x + c * y * half_y) for x, y in
            ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def quad_mesh(quads, facing):
    """An effect-space mesh of quads, each dict(centre, size (full x, y), angle (radians, in its plane)): in the effect's
    local XY plane facing +Z for a fixed facing, else at their centres with BILLBOARD_VERTEX offsets (UV0 in the
    decoder's flipped-V form)."""
    positions, offsets, triangles, uvs = [], [], [], []
    for n, quad in enumerate(quads):
        centre = np.asarray(quad['centre'], np.float64)
        for x, y in _corners(quad['size'][0] / 2.0, quad['size'][1] / 2.0, quad['angle']):
            if facing in FACING_CODES:
                positions.append(centre)
                offsets.append((x, y, 0.0, FACING_CODES[facing]))
            else:
                positions.append(centre + (x, y, 0.0))
        uvs += [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
        triangles += [(4 * n, 4 * n + 1, 4 * n + 2), (4 * n, 4 * n + 2, 4 * n + 3)]
    count = len(positions)
    mesh = dict(positions=np.asarray(positions, np.float64).reshape(-1, 3), normals=np.tile([0.0, 0.0, 1.0], (count, 1)),
                tangents=np.tile([1.0, 0.0, 0.0], (count, 1)), tangent_signs=np.ones(count), tangent_w=np.ones(count),
                uv0=np.asarray(uvs, np.float64), uv1=None, colors=None, triangles=np.asarray(triangles, np.int64),
                flags=0, skin_batch_count=0, mesh_id='effect', vertex_count=count, index_count=3 * len(triangles))
    if offsets:
        mesh['billboard'] = np.asarray(offsets, np.float64)
    return mesh


def light_values(values, flags):
    """A light node's light from its render values, as 0x14137A1E0 builds it: colour = the hue (RGB / max) at luminance 1,
    intensity = max(|size x|, |size y|) x |RGB| x alpha, 0 unless node flag bit 0 is set."""
    rgb = np.asarray(values['colour'][:3], np.float64)
    peak = float(rgb.max()) if len(rgb) else 0.0
    hue = rgb / peak if peak > 0 else rgb
    luminance = float(hue @ (0.2126, 0.7152, 0.0722))
    intensity = (max(abs(values['size'][0]), abs(values['size'][1])) * float(np.linalg.norm(rgb)) * values['alpha']
                 if flags & 1 else 0.0)
    return dict(colour=tuple(float(c) for c in (hue / luminance if luminance > 0 else hue)), intensity=float(intensity))


def _middle(particles):
    """The particle nearest mid-life, whose values a node's one draw state binds."""
    return min(particles, key=lambda p: abs(p['t'] - 0.5))


def _entry(decoded, path, draw):
    material = dict(index=len(decoded['materials']), path=path, name=path.rsplit('/', 1)[-1].rsplit('.', 1)[0], draw=draw)
    decoded['materials'].append(material)
    return material


def _values_text(values):
    return 'colour (%s), alpha %.4g, custom (%s)' % (', '.join('%.4g' % c for c in values['colour'][:3]), values['alpha'],
                                                     ', '.join('%.4g' % c for c in values['custom'][:3]))


def attach(source, decoded, data, model_path, as_arrays=True):
    """Add the activated effects' drawable nodes (DRAWN) to ``decoded`` as meshes in every look and LOD, each with its
    own material entry (``draw``: the per-draw values its translation binds) and a note in ``material_notes``. Returns
    ``decoded['effects']``: one row per attached effect with the nodes drawn and left out."""
    from nwo.formats.native_model import _selected_meshes
    from nwo.games.rcra import particles as replay
    from nwo.games.rcra.model import decode_model
    names = {loc['name'] for loc in decoded.get('locators') or []} | {j['name'] for j in decoded.get('joints') or []}
    rows = decoded.setdefault('effects', [])
    attachments = activated_effects(source, model_path, names)
    if not attachments:
        return rows
    worlds = joint_rest_worlds(data, len(decoded.get('joints') or []))
    next_index = max([m['index'] for m in decoded['meshes']] + [decoded.get('mesh_record_count', 0) - 1]) + 1
    added = []
    models = {}

    def add(mesh, placement, material, joint):
        nonlocal next_index
        decoded['meshes'].append(_mesh(mesh, placement, next_index, material, joint, as_arrays))
        added.append(next_index)
        next_index += 1

    def model_meshes(path):
        if path not in models:
            keys = source.resolve(path) if path else []
            if not keys:
                models[path] = 'model not resolved'
            else:
                try:
                    part = decode_model(path, data=source.asset_data(keys[0]), as_arrays=as_arrays)
                    selected = set(_selected_meshes(part['looks'], part['mesh_record_count'], 'Highest', 0))
                    models[path] = [m for m in part['meshes'] if m['index'] in selected]
                except (ValueError, KeyError, OSError) as error:
                    models[path] = str(error)
        return models[path]

    for attachment in attachments:
        row = dict(attachment, drawn=[], skipped=[])
        rows.append(row)
        keys = source.resolve(attachment['effect'])
        if not keys:
            row['skipped'].append('effect not in the archive')
            continue
        effect = vfx.decode_effect(source.asset_data(keys[0]))
        matrix, joint = _attach_matrix(decoded, worlds, attachment['locator'])
        if matrix is None:
            row['skipped'].append('attach point %s not placed' % attachment['locator'])
            continue
        notes = []
        where = (attachment['effect'], attachment['conduit'], attachment['locator'])
        for node in effect['nodes']:
            kind = node['render_type']
            label = '%d %s' % (node['index'], node['render_type_name'])
            material_path = node['material']
            if node['light'] is not None and node['curves'] is not None:   # listed with its decoded values, not drawn
                flags = int(np.frombuffer(node['raw'], '<u8', 1, 0x18)[0])
                values = light_values(single_values(effect, node, SINGLE_SECONDS), flags)
                row.setdefault('lights', []).append(dict(node['light'], node=node['index'], **values))
                row['skipped'].append('%s (%s light, intensity %.3g, range %.3g)' % (
                    label, node['light']['type'], values['intensity'], node['light']['range']))
                continue
            if kind not in DRAWN or (kind in MULTI and node['emitter'] is None):
                row['skipped'].append(label)
                continue
            if not material_path or not material_path.endswith('.material'):
                row['skipped'].append(label + ' (material not resolved)')
                continue
            facing = node_facing(node)
            if facing == 'unknown':
                row['skipped'].append('%s (facing %d)' % (label, node['raw'][0x178]))
                continue
            place = np.eye(4)
            place[:3, :3] = np.asarray(node['rotation'], np.float64).reshape(3, 3)
            place[:3, 3] = node['position']
            placed = matrix @ place
            flags = int(np.frombuffer(node['raw'], '<u8', 1, 0x18)[0])
            if kind in MULTI:
                emitter = effect['emitters'][node['emitter']]
                world_to_effect = np.linalg.inv(placed[:3, :3])
                particles, left = replay.simulate(effect, emitter, world_to_effect, node_flags=flags)
                if not particles:
                    row['skipped'].append(label + ' (no particle alive)')
                    continue
                still = 'a still of %d particles at emitter age %.3g s' % (len(particles),
                                                                          replay.still_age(effect, emitter))
                if left:
                    still += ' (not followed: %s)' % ', '.join(left)
            else:
                values = single_values(effect, node, SINGLE_SECONDS)
                particles = [dict(values, position=np.zeros(3), t=CURVE_TIME)]
                still = 'its curves at mid-life (t 0.5, rated slots at %.3g s)' % SINGLE_SECONDS
            shown = _middle(particles)
            if kind in (NODE_MODEL, NODE_MODELS):
                meshes = model_meshes(node['asset'])
                if isinstance(meshes, str):
                    row['skipped'].append('%s (%s)' % (label, meshes))
                    continue
                material = _entry(decoded, material_path, dict(object=object_values(shown)))
                for particle in particles:
                    local = np.diag(node['scale'])
                    if kind == NODE_MODELS:
                        from nwo.games.rcra.particles import euler
                        local = euler(*particle['rotation']) @ np.diag(np.asarray(node['scale']) * particle['size'])
                    centre = (placed @ np.append(particle['position'], 1.0))[:3]
                    for mesh in meshes:
                        if facing in FACING_CODES:   # turned toward the view in the vertex program, about its centre
                            add(facing_mesh(mesh, centre, place[:3, :3] @ local, facing), np.eye(4), material, joint)
                        else:
                            at = np.eye(4)
                            at[:3, :3] = placed[:3, :3] @ local
                            at[:3, 3] = centre
                            add(mesh, at, material, joint)
                template = MODEL_NOTE
            else:
                quads = [dict(centre=p['position'], size=np.asarray(node['scale'][:2]) * np.asarray(p['size'][:2]),
                              angle=float(p['rotation'][2])) for p in particles]
                material = _entry(decoded, material_path, dict(interpolants=quad_interpolants(shown)))
                add(quad_mesh(quads, facing), placed, material, joint)
                template = QUAD_NOTE
            notes.append((material_path, template, (node['index'], node['render_type_name'], facing) + where
                          + (still, _values_text(shown))))
            row['drawn'].append(label)
        left = LEFT_NOTE % ', '.join(row['skipped']) if row['skipped'] else ''
        for material_path, template, values in notes:
            decoded.setdefault('material_notes', {}).setdefault(material_path, []).append(template % (values + (left,)))
    if added:
        for look in decoded['looks']:
            for level in look['lods']:
                if level['count']:
                    level['meshes'] = list(level['meshes']) + added
    return rows
