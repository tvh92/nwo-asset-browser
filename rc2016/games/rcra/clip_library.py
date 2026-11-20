"""Rift Apart clip library for the 3D preview and the portable export (the ``Game.animation_library`` hook).

``RcraAnimationLibrary`` is ``viewer.animation.AnimationLibrary`` with Rift Apart sources: the same rows,
inverse binds, inheritance rule, classification, additive preview and skin matrices, but each clip is its
own ``.animclip`` asset decoded by ``rcra.animation.decode_clip``. Clips are found through the dag, never
by joint coverage alone:

* **Animsets** the model's actors use: an actor or config that depends on the model, and the configs it
  depends on, name the animsets; each animset depends on its clips.
* **Same-folder animsets**: the animsets in the model's own folder, or whose first clip is in it or in its
  ``animations``/``animation`` subfolder, whose first clip drives the model's skeleton (most of its joints,
  by joint hash). Outfit models such as ``hero_ratchet_pirate.model`` are assembled at runtime and no actor
  names them; their character folder holds the animset. Zone-placed crowd sets (the thug's cheering) and
  gallery sets keep their clips beside the model. A set that only repeats clip assets already offered is
  left out.
* **Cutscene shots** (``with_cinematic``): clips under ``cinematics/`` whose name ends in
  ``_shot<N>_<performer>`` where the performer is the model's file stem or one of its folder names holding
  an underscore (optionally followed by ``_master<N>`` or a copy number), as for R&C 2016. The clip's
  ``.cinematic2`` (the asset that depends on it) is its source.
* **Face performances** (``with_performances``): the voice-line lip sync of the speaker whose face pose clip
  the model plays (``performance_sources``), blended from that pose clip (``rcra.animation.performance_frames``).

Each found clip is still classified on its first frame; one that does not drive the skeleton is listed as
unplayable. ``root_displacement`` gives a clip's root motion since its first frame, which ``clip_frames``
applies on request.
"""
from __future__ import annotations

import copy
import re
import struct
import tempfile
from pathlib import Path

import numpy as np

from rc2016.common import sha
from rc2016.formats.animation_transforms import sample_transforms
from rc2016.games.rcra.animation import (clip_dat1, decode_clip, header, motion, performance_clip, performance_frames,
                                         pose_table)
from rc2016.games.rcra.morph import clip_weights, shape_hashes
from rc2016.games.rcra.toc import sections as dat1_sections
from rc2016.viewer.animation import AnimationLibrary, _quat_mul, _quat_rotate, on_rest

CUTSCENE_SOURCE = 'cinematic2'
PERFORMANCE = 'performance'   # a face performance clip's source
FACE_KIND = 'face performance'
PERFORMER = re.compile(r'_shot\d+[a-z]?_(.+)$')
COPY_SUFFIX = re.compile(r'(_master\d*|\d+)$')
# A same-folder animset is kept when its first clip names at least this share of the model's joints.
FOLDER_FIT = 0.5
# Where an animset's clips may live for it to count as the model folder's own ('' is the folder itself).
CLIP_FOLDERS = ('', 'animations/', 'animation/')
FOLLOW = ('actor', 'config')


def _kind(path):
    name = path.rsplit('/', 1)[-1]
    return name.rsplit('.', 1)[-1] if '.' in name else ''


class _DagIndex:
    """Reverse dag dependencies and the cutscene performer index of one RcraSource, built once."""

    def __init__(self, source):
        dag = source.dag
        self.paths = [p.replace('\\', '/').lower() for p in dag.paths]
        self.by_path = {p: i for i, p in enumerate(self.paths)}
        users = {}
        starts, deps = dag.dep_starts, dag.deps
        stops = np.nonzero(deps == -1)[0]
        for index, start in enumerate(starts.tolist()):
            if start < 0:
                continue
            end = int(stops[np.searchsorted(stops, start)]) if len(stops) else len(deps)
            for target in deps[start:end].tolist():
                users.setdefault(int(target), []).append(index)
        self.users = users
        self.dag = dag
        self.performers = {}
        for index, path in enumerate(self.paths):
            if path.startswith('cinematics/') and path.endswith('.animclip'):
                found = PERFORMER.search(path.rsplit('/', 1)[-1][:-len('.animclip')])
                if found:   # listed as written and without a _master<N> or copy-number suffix
                    for name in {found.group(1), COPY_SUFFIX.sub('', found.group(1))}:
                        self.performers.setdefault(name, []).append(index)
        self.animsets = [i for i, p in enumerate(self.paths) if p.endswith('.animset')]
        self.performance_sets = [i for i, p in enumerate(self.paths) if p.endswith('.performanceset')]
        self.performances = {}   # folder -> performance clip indices
        for i, p in enumerate(self.paths):
            if p.endswith('.performanceclip'):
                self.performances.setdefault(_folder(p), []).append(i)

    def deps(self, index):
        return self.dag.dependencies(index)


def dag_index(source):
    index = getattr(source, '_rcra_clip_index', None)
    if index is None:
        index = source._rcra_clip_index = _DagIndex(source)
    return index


def _key(source, path):
    keys = source.resolve(path)
    return keys[0] if len(keys) == 1 else None


def model_animsets(source, model_path):
    """Animsets the model's actors and configs name, in dag order (paths with '/')."""
    index = dag_index(source)
    model = index.by_path.get(model_path.replace('\\', '/').lower())
    if model is None:
        return []
    queue = [u for u in index.users.get(model, []) if _kind(index.paths[u]) in FOLLOW]
    seen, found = set(), []
    while queue:
        node = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        for dep in index.deps(node):
            kind = _kind(index.paths[dep])
            if kind == 'animset':
                found.append(index.paths[dep])
            elif kind == 'config':
                queue.append(dep)
    return list(dict.fromkeys(found))


def animset_clips(source, animset):
    """Clip paths an animset depends on, in its dependency order."""
    index = dag_index(source)
    node = index.by_path.get(animset)
    if node is None:
        return []
    return [index.paths[d] for d in index.deps(node) if index.paths[d].endswith('.animclip')]


def _folder(path):
    return path.rsplit('/', 1)[0] + '/'


def folder_animsets(source, model_path, joint_hashes, exclude=()):
    """Animsets of the model's folder whose first clip names at least FOLDER_FIT of the model's joints.

    An animset belongs to the folder when it is in it, or when its first clip is in the folder or in the
    folder's ``animations``/``animation`` subfolder (zone-placed crowd sets such as the thug's cheering, and
    sets kept in a folder of their own such as ``enm_pool_shark_underwater``)."""
    index = dag_index(source)
    folder = _folder(model_path.replace('\\', '/').lower())
    homes = {folder + sub for sub in CLIP_FOLDERS}
    found = []
    for node in index.animsets:
        path = index.paths[node]
        if path in exclude:
            continue
        clips = animset_clips(source, path)
        if _folder(path) != folder and not (clips and _folder(clips[0]) in homes):
            continue
        for clip in clips:
            key = _key(source, clip)
            if key is None:
                continue
            try:
                decoded = decode_clip(source.bytes(key)[0], first_frames=1)
            except ValueError:
                continue
            clip_hashes = {int(j['hash'], 16) for j in decoded['joints']}
            if joint_hashes and len(clip_hashes & joint_hashes) >= FOLDER_FIT * len(joint_hashes):
                found.append(path)
            break
    return found


def cutscene_clips(source, model_path):
    """[(cinematic2 path, clip path)] of the model's performances (see the module notes)."""
    index = dag_index(source)
    parts = model_path.replace('\\', '/').lower().split('/')
    names = {Path(parts[-1]).stem} | {p for p in parts[:-1] if '_' in p}
    found = []
    for node in sorted({n for name in names for n in index.performers.get(name, [])}):
        owners = [index.paths[u] for u in index.users.get(node, []) if index.paths[u].endswith('.cinematic2')]
        owner = owners[0] if owners else index.paths[node].rsplit('/', 1)[0] + '.cinematic2'
        found.append((owner, index.paths[node]))
    return sorted(set(found))


def performance_sources(source, model_path, offered, joint_hashes):
    """[(pose clip path, [(owner animset, performance clip path)])] of the model's face performances.

    A performance set (``localization/<speaker>/<speaker>.performanceset``) depends on its speaker's face pose
    clip (the dependency with a phoneme table) and plays the speaker's ``.performanceclip`` assets, which sit in
    the set's folder and belong to its ``<speaker>_<group>.animset`` files. A set is the model's when that pose
    clip lies in the model's folder (or its animation subfolder) or, only when no such pose clip exists, is one
    of the model's offered clips (``offered``); either way the model must have at least FOLDER_FIT of the
    joints it drives (crowd rigs lack a few accessory joints, which are then not posed). Heroes share animsets
    (Rivet's actor offers Ratchet's clips), so the folder comes first."""
    index = dag_index(source)
    folder = _folder(model_path.replace('\\', '/').lower())
    homes = {folder + sub for sub in CLIP_FOLDERS}
    poses, found = {}, {}
    for node in index.performance_sets:
        for dep in index.deps(node):
            pose = index.paths[dep]
            if not pose.endswith('.animclip') or not (pose in offered or _folder(pose) in homes):
                continue
            if pose not in poses:
                key = _key(source, pose)
                poses[pose] = False
                if key is not None:
                    try:
                        data = source.bytes(key)[0]
                        drives = {int(j['hash'], 16) for j in decode_clip(data, first_frames=1)['joints']}
                        poses[pose] = (pose_table(data) is not None and bool(drives)
                                       and len(drives & joint_hashes) >= FOLDER_FIT * len(drives))
                    except ValueError:
                        pass
            if poses[pose]:
                clips = found.setdefault(pose, [])
                for clip in index.performances.get(_folder(index.paths[node]), []):
                    owners = [index.paths[u] for u in index.users.get(clip, []) if index.paths[u].endswith('.animset')]
                    clips.append((owners[0] if owners else index.paths[node], index.paths[clip]))
                break
    own = {pose for pose in found if _folder(pose) in homes}
    return [(pose, sorted(set(clips))) for pose, clips in found.items() if pose in own or not own]


class RcraAnimationLibrary(AnimationLibrary):
    """``AnimationLibrary`` over a Rift Apart model and its ``.animclip`` assets."""

    def __init__(self, source, model_bytes, model_path, clips=True):
        from rc2016.formats.skeleton import inheritance_rule, inspect_model, rest_inheritance_rule
        self.source, self.model_path = source, model_path
        self.rows, self.inverse, self.clips, self.animset_data = [], [], [], {}
        self._kinds, self.retarget, self._cutscenes, self._last = {}, {}, None, (None, None)
        self._performances = None
        self.inheritance = self.rest_inheritance = 'full'
        self.names = {}
        with tempfile.TemporaryDirectory(prefix='rcra-anim-') as tmp:
            model_file = Path(tmp) / 'selected.model'
            model_file.write_bytes(bytes(model_bytes)[36:])   # the DAT1 after the toc header
            inspected = inspect_model(model_file)
        self.rows = inspected['skeleton']
        self.inverse = [np.array(m) for m in inspected['inverse_bind_matrices']]
        self.inheritance = inheritance_rule(inspected['profile'])
        self.rest_inheritance = rest_inheritance_rule(inspected['profile'])
        self.names = {int(r['hash'], 16): r['name'] for r in self.rows}
        dat1 = bytes(model_bytes)[36:]
        try:   # the model's blend shapes, in its order: a clip's morph channels name them (rcra.morph)
            self.shape_hashes = shape_hashes(dat1, dat1_sections(dat1))
        except (ValueError, struct.error):
            self.shape_hashes = []
        if not clips or not self.rows or len(self.inverse) != len(self.rows):
            return
        animsets = model_animsets(source, model_path)
        offered = {clip for animset in animsets for clip in animset_clips(source, animset)}
        for animset in folder_animsets(source, model_path, set(self.names), exclude=animsets):
            clips = set(animset_clips(source, animset))
            if clips - offered:   # a same-folder set that only repeats clip assets already offered is left out
                animsets.append(animset)
                offered |= clips
        for animset in animsets:
            for clip in animset_clips(source, animset):
                self._add(animset, clip, 'animset')

    def _add(self, owner, clip_path, kind, **extra):
        key = _key(self.source, clip_path)
        if key is None:
            return
        entry = dict(animset=owner, offset=0, size=None, key=key, path=clip_path, name=Path(clip_path).stem,
                     seconds=None, source=kind, **extra)
        try:
            facts = header(self._bytes(entry))
        except ValueError:
            facts = {}
        entry.update(name=facts.get('name') or entry['name'], seconds=facts.get('duration_seconds'))
        self.animset_data.setdefault(owner, None)
        self.clips.append(entry)

    def _bytes(self, clip):
        key, data = self._last
        if key != clip['key']:
            data = self.source.bytes(clip['key'])[0]
            self._last = (clip['key'], data)
        return data

    def _clip_digest(self, clip):
        return sha(self._bytes(clip))

    def _decode_clip(self, clip, first_frames=None):
        return decode_clip(self._bytes(clip), self.names, first_frames=first_frames, label=clip['name'])

    def with_performances(self):
        """Copy that also offers the model's face performances (``performance_sources``): one clip per voice line,
        listed under the speaker's ``<speaker>_<group>`` animsets. They play on the rest pose (``clip_frames``)."""
        other = copy.copy(self)
        other.clips, other.animset_data = list(self.clips), dict(self.animset_data)
        if self._performances is None and self.rows:
            offered = {c['path'] for c in self.clips if c.get('source') == 'animset'}
            self._performances = performance_sources(self.source, self.model_path, offered, set(self.names))
        for pose, clips in self._performances or ():
            for owner, clip in clips:
                other._add(owner, clip, PERFORMANCE, pose=pose)
        return other

    def clip_notes(self):
        """``AnimationLibrary.clip_notes``; a face performance plays on its own (None) and is not decoded here."""
        clips = self.clips
        try:
            self.clips = [c for c in clips if c.get('source') != PERFORMANCE]
            notes = iter(super().clip_notes())
        finally:
            self.clips = clips
        return [None if c.get('source') == PERFORMANCE else next(notes) for c in clips]

    def morph_weights(self, index, count):
        """Blend-shape weights of clip ``index`` for its ``count`` frames: (count, shapes) in the model's shape order
        (``rcra.morph.clip_weights``), or None when the model has no shapes or the clip names none of them. A face
        performance blends its pose clip's frames by the performance's weights, as its joints do."""
        if not self.shape_hashes or count < 1:
            return None
        entry = self.clips[index]
        if entry.get('source') == PERFORMANCE:
            key = _key(self.source, entry['pose'])
            if key is None:
                return None
            pose = self.source.bytes(key)[0]
            poses = self._channel_weights(pose)
            if poses is None:
                return None
            performance = performance_clip(self._bytes(entry))
            table = np.asarray(pose_table(pose), np.intp)
            weight = performance['weight'][:count]
            weight = weight / np.maximum(1.0, weight.sum(1, keepdims=True))
            frame_of = np.minimum(table[np.minimum(performance['index'][:count], len(table) - 1)], len(poses) - 1)
            return np.einsum('fs,fsk->fk', weight, poses[frame_of]).astype(np.float32)
        weights = self._channel_weights(self._bytes(entry))
        if weights is None:
            return None
        if len(weights) < count:   # a frame past the clip's morph frames keeps the last
            weights = np.concatenate([weights, np.repeat(weights[-1:], count - len(weights), 0)])
        return weights[:count]

    def _channel_weights(self, data):
        """(frames, shapes) weights of a clip asset's morph channels that name this model's shapes, or None."""
        dat1, table = clip_dat1(data)
        try:
            found = clip_weights(dat1, table, header(data)['original_frame_count'])
        except (ValueError, struct.error):
            return None
        if found is None:
            return None
        order = {h: i for i, h in enumerate(self.shape_hashes)}
        columns = [(c, order[h]) for c, h in enumerate(found['hashes']) if h in order]
        if not columns:
            return None
        weights = np.zeros((len(found['weights']), len(self.shape_hashes)), np.float32)
        weights[:, [s for _c, s in columns]] = found['weights'][:, [c for c, _s in columns]]
        return weights

    def pose(self, index, root_motion=False):
        """``AnimationLibrary.pose`` plus ``morph_weights``: (frames, shapes) or None (``morph_weights``)."""
        result = super().pose(index, root_motion)
        result['morph_weights'] = self.morph_weights(index, len(result['frames']))
        return result

    def clip_frames(self, index, root_motion=False):
        """``AnimationLibrary.clip_frames``; a face performance blends its pose clip's frames by the performance's
        weights (``performance_frames``) and shows them on the rest pose, as additive layers are. Root motion does
        not apply to it."""
        entry = self.clips[index]
        if entry.get('source') != PERFORMANCE:
            return super().clip_frames(index, root_motion)
        performance = performance_clip(self._bytes(entry))
        key = _key(self.source, entry['pose'])
        if key is None:
            raise ValueError(f"Face pose clip {entry['pose']} is not in the install")
        pose = self.source.bytes(key)[0]
        frames = performance_frames(sample_transforms(decode_clip(pose, self.names)), pose_table(pose), performance)
        clip = dict(name=performance['name'] or entry['name'], pose=entry['pose'],
                    metadata=dict(sample_rate_fps=performance['sample_rate_fps'],
                                  duration_seconds=performance['duration_seconds'], original_frame_count=len(frames)))
        return clip, FACE_KIND, on_rest(self.rows, frames)

    def root_displacement(self, index, count):
        """The clip's root motion (``rcra.animation.motion``) since its first frame: record f relative to record 0,
        for ``count`` frames (a closing record past them is not played; a frame past the records keeps the last).
        None when the clip has no root motion records or they do not read."""
        try:
            found = motion(self._bytes(self.clips[index]))
        except ValueError:
            return None
        if found is None or not len(found['rotation_xyzw']) or count < 1:
            return None
        q, t = found['rotation_xyzw'], found['translation']
        take = np.minimum(np.arange(count), len(q) - 1)
        q, t = q[take], t[take]
        x, y, z, w = q[0]
        inverse = np.array([-x, -y, -z, w])
        rotation = np.array([_quat_mul(inverse, row) for row in q])
        translation = np.array([_quat_rotate(inverse, row - t[0]) for row in t])
        return rotation, translation

    def with_cinematic(self, archive=None, model_path=None, replace=False, sources=None):
        """Copy that also offers the model's cutscene shots (``cutscene_clips``); with ``replace`` only them."""
        other = copy.copy(self)
        other.clips = [] if replace else list(self.clips)
        other.animset_data = {} if replace else dict(self.animset_data)
        other._kinds = self._kinds
        if self._cutscenes is None and self.rows:
            self._cutscenes = cutscene_clips(self.source, model_path or self.model_path)
        for owner, clip in self._cutscenes or ():
            other._add(owner, clip, CUTSCENE_SOURCE)
        return other


def animation_library(source, model_bytes, model_path, clips=True):
    """The ``Game.animation_library`` hook: the model's clip library (``clips=False``: skeleton only)."""
    return RcraAnimationLibrary(source, model_bytes, model_path, clips=clips)
