"""Visual effects a model's actors start when they activate, drawn as extra meshes on the model in the viewer
(docs/rift-apart.md, Effects in the viewer).

The chain is decoded (nwo.games.rcra.vfx): the model's actors -> their conduits -> the ``ActivatedEvent`` responses that
start an effect at a locator (or joint) -> the effect's nodes. Only SingleModel nodes are drawn: one model with its
material, placed at the node's position, rotation and scale relative to the locator, and weighted to the locator's joint
so it follows the animation. Quads, particles, lights, trails and the other render types are left out and listed.

A conduit counts as the model's when every locator its activation responses name is a locator or joint of the model: a
shared conduit (the gallery's ``hero_gallery_vfx``, which names the locators of many models) is skipped.

Viewer choices, in the material notes:

- **Curves:** evaluated at the middle of the particle's life (t = 0.5) with the random vector at 0.5, the middle of each
  random range.
- **Object colour:** the node's colour curve times its luminance curve. This follows the decoded R&C 2016 path, where a
  particle's colour (the colour slot times the intensity slot) reaches a model draw as its per-instance colour
  (nwo.levels.effects); the Rift Apart draw routine (0x141377EC0) was not traced that far.
- **Rotation:** the node's 3x3 is applied with its nine floats as rows; the storage order is not traced (the vendor's
  logo plane has the identity).
"""
import numpy as np

from nwo.games.rcra import vfx

NODE_MODEL = vfx.RENDER_TYPES.index('SingleModel')
CURVE_TIME = 0.5
CURVE_RANDOM = (0.5, 0.5, 0.5)
NOTE = ('Visual effect mesh: node %d (%s) of %s, started by %s at %s. Drawn at rest with its curves at mid-life (t 0.5, '
        'random ranges at their middle). Object colour = the node colour x luminance curves %s, inferred from the R&C '
        '2016 decode (particle colour reaches the instance colour); not traced in Rift Apart. Other render types of the '
        'effect are not drawn%s.')


def _kind(path):
    name = path.rsplit('/', 1)[-1]
    return name.rsplit('.', 1)[-1] if '.' in name else ''


def activated_effects(source, model_path, attach_names):
    """[{'effect', 'locator', 'conduit', 'actor'}]: the effects the model's actors start on activation, deduplicated by
    (effect, locator), from the conduits whose activation responses only name ``attach_names`` (locators and joints)."""
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
            started = [r for r in responses if r['event'] == 'ActivatedEvent' and r['start']]
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


def node_matrix(node):
    """The node's transform: position, then the 3x3 (rows as stored), then the scale."""
    m = np.eye(4)
    m[:3, :3] = np.asarray(node['rotation'], np.float64).reshape(3, 3) @ np.diag(node['scale'])
    m[:3, 3] = node['position']
    return m


def object_values(effect, node):
    """The node's per-object record members: colour x luminance as the fp16 colour, alpha (shaders.NEUTRAL_OBJECT form)."""
    from nwo.games.rcra.shaders import half_pair

    def value(name):
        return vfx.curve_value(effect['curves'][node['curves'][name][0]], CURVE_TIME, CURVE_RANDOM)
    colour, luminance, alpha = value('color'), value('luminance'), value('alpha')
    rgb = [c * luminance[0] for c in colour]
    return {'m_ColorRGFp16': (half_pair(rgb[0], rgb[1]),), 'm_ColorBMiscFp16': (half_pair(rgb[2], 0.0),),
            'm_Alpha': (float(alpha[0]),)}


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


def attach(source, decoded, data, model_path, as_arrays=True):
    """Add the activated effects' SingleModel nodes to ``decoded`` as meshes in every look and LOD, each with its own
    material entry (``object``: object_values) and a note in ``material_notes``. Returns ``decoded['effects']``: one row per
    attached effect with what was drawn and what was left out."""
    from nwo.formats.native_model import _selected_meshes
    from nwo.games.rcra.model import decode_model
    from nwo.games.rcra.shaders import object_text
    names = {loc['name'] for loc in decoded.get('locators') or []} | {j['name'] for j in decoded.get('joints') or []}
    rows = decoded.setdefault('effects', [])
    attachments = activated_effects(source, model_path, names)
    if not attachments:
        return rows
    worlds = joint_rest_worlds(data, len(decoded.get('joints') or []))
    next_index = max([m['index'] for m in decoded['meshes']] + [decoded.get('mesh_record_count', 0) - 1]) + 1
    added = []
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
        for node in effect['nodes']:
            label = '%d %s' % (node['index'], node['render_type_name'])
            if node['render_type'] != NODE_MODEL:
                row['skipped'].append(label)
                continue
            model_keys = source.resolve(node['asset']) if node['asset'] else []
            material_path = node['material']
            if not model_keys or not material_path or not material_path.endswith('.material'):
                row['skipped'].append(label + ' (model or material not resolved)')
                continue
            try:
                part = decode_model(node['asset'], data=source.asset_data(model_keys[0]), as_arrays=as_arrays)
            except (ValueError, KeyError, OSError) as error:
                row['skipped'].append('%s (%s)' % (label, error))
                continue
            selected = set(_selected_meshes(part['looks'], part['mesh_record_count'], 'Highest', 0))
            values = object_values(effect, node)
            material = dict(index=len(decoded['materials']), path=material_path,
                            name=material_path.rsplit('/', 1)[-1].rsplit('.', 1)[0], object=values)
            decoded['materials'].append(material)
            placement = matrix @ node_matrix(node)
            for mesh in part['meshes']:
                if mesh['index'] in selected:
                    decoded['meshes'].append(_mesh(mesh, placement, next_index, material, joint, as_arrays))
                    added.append(next_index)
                    next_index += 1
            row['drawn'].append(label)
            notes.append((material_path, node, values))
        for material_path, node, values in notes:
            colour = '%s %s' % (object_text('m_ColorRGFp16', values['m_ColorRGFp16']),
                                object_text('m_ColorBMiscFp16', values['m_ColorBMiscFp16']))
            left = ' (%s)' % ', '.join(row['skipped']) if row['skipped'] else ''
            decoded.setdefault('material_notes', {}).setdefault(material_path, []).append(NOTE % (
                node['index'], node['render_type_name'], attachment['effect'], attachment['conduit'],
                attachment['locator'], colour, left))
    if added:
        for look in decoded['looks']:
            for level in look['lods']:
                if level['count']:
                    level['meshes'] = list(level['meshes']) + added
    return rows
