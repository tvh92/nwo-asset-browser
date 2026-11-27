"""Clips in a portable .blend export, laid out as the research pipeline lays out its own (docs/animation.md, Overview
and controls): every clip keyed on its own samples (sample k at frame k + 1), one sequential NLA track of every clip on
a common clock with a scene marker at each boundary, the clip catalog and the Animations panel (nwo.export.blend_controls)
embedded as text blocks. The file opens on the overview: Play runs every clip in order. Loading one clip in the panel
sets the scene's rate and range to that clip, so playback loops it.

Runs in Blender (blender_convert.py --clips), after the FBX and USD are written: those keep the glTF importer's timing.
The helpers without ``bpy`` are tested outside Blender.
"""
from __future__ import annotations

import json
from pathlib import Path

from nwo.pipeline.timeline import timeline

CATALOG = 'Animation Catalog.json'
CONTROLS = 'Animation Controls.py'
NOTE = 'START HERE'
OVERVIEW = 'All clips | sequential timeline'   # blend_controls.OVERVIEW
SNAP = 1e-3   # a retimed key this close to a whole frame sits on it (float32 sample times)


def blender_name(name):
    """``name`` as Blender stores an ID or NLA track name: at most 63 bytes of UTF-8, cut on a character boundary."""
    return name.encode('utf-8')[:63].decode('utf-8', 'ignore')


def strip_label(index, name):
    return blender_name(f'{index + 1:04d} | {name}')


def retime(frames, fps, import_fps):
    """Frames the glTF importer placed at ``time * import_fps`` moved to the clip's own samples: sample k (time k / fps) at
    frame k + 1, snapped to the whole frame within SNAP."""
    import numpy as np
    out = np.asarray(frames, np.float64) * (fps / import_fps) + 1.0
    whole = np.round(out)
    return np.where(np.abs(out - whole) <= SNAP, whole, out)


def track_clips(track_names, clip_count):
    """{NLA track name: clip index} for the importer's tracks: glTF animation i is track ``track_names[i]`` (the
    importer's unique names in animation order); clip i is animation i. Names that collide once cut to Blender's length
    are left out."""
    seen, out = set(), {}
    for index, name in enumerate(track_names[:clip_count]):
        key = blender_name(name)
        if key in seen:
            out.pop(key, None)
        else:
            out[key] = index
        seen.add(key)
    return out


def _fcurves(action):
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                yield from bag.fcurves


def _retime_action(action, fps, import_fps, frames):
    import numpy as np
    last = 1.0
    for curve in _fcurves(action):
        points = curve.keyframe_points
        if not len(points):
            continue
        values = np.empty(2 * len(points))
        for prop in ('co', 'handle_left', 'handle_right'):
            points.foreach_get(prop, values)
            values[0::2] = retime(values[0::2], fps, import_fps)
            points.foreach_set(prop, values)
            if prop == 'co':
                last = max(last, float(values[0::2].max()))
        curve.update()
    if last > frames + SNAP:
        raise RuntimeError(f'Clip "{action.name}" has keys up to frame {last:g} but {frames} samples')
    action.use_frame_range = True
    action.frame_start, action.frame_end = 1, frames


def setup(scene, clips, import_fps):
    """Lay out the imported clips. ``clips``: one dict per glTF animation, in order, with name, kind, fps and frames
    (sample count). ``import_fps``: the rate the importer converted times with (render.fps * fps_base at import).
    Returns the catalog, or None when no clip was imported."""
    import bpy
    tracks = getattr(bpy.data.scenes[0], 'gltf2_animation_tracks', None)
    names = [t.name for t in tracks] if tracks is not None else []
    if len(names) < len(clips):
        names = [c['name'] for c in clips]   # the importer's own list is missing: the names as written
    wanted = track_clips(names, len(clips))
    targets, owners = {}, []
    for obj in bpy.data.objects:
        data = obj.data
        for kind, item in (('OBJECT', obj), ('SHAPE_KEYS', getattr(data, 'shape_keys', None) if data is not None else None)):
            animation = getattr(item, 'animation_data', None) if item is not None else None
            if animation is None:
                continue
            used = False
            for track in list(animation.nla_tracks):
                index = wanted.get(track.name)
                if index is None or not len(track.strips):
                    continue   # not a clip (the baked vertex-program animation keeps its importer track)
                strip = track.strips[0]
                targets.setdefault(index, []).append([kind, obj.name, strip.action.name, strip.action_slot.identifier])
                animation.nla_tracks.remove(track)
                used = True
            if used:
                animation.action = None
                owners.append([kind, obj.name])
    if not targets:
        return None
    if tracks is not None:
        tracks.clear()   # the importer's own animation switcher would point at the tracks replaced here
    entries, retimed = [], set()
    for index in sorted(targets):
        clip = clips[index]
        frames = max(1, int(clip['frames']))
        for _kind, _name, action_name, _slot in targets[index]:
            if action_name not in retimed:
                action = bpy.data.actions[action_name]
                _retime_action(action, float(clip['fps']), import_fps, frames)
                action.use_fake_user = True
                retimed.add(action_name)
        # The armature's Action names the clip (an importer older than slotted Actions makes one per object).
        action = next((t[2] for t in targets[index] if t[0] == 'OBJECT'), targets[index][0][2])
        entries.append(dict(name=clip['name'], kind=clip['kind'], fps=float(clip['fps']), frames=frames, action=action,
                            targets=targets[index]))
    return layout(scene, entries, owners, import_fps=import_fps,
                  left_out=[c['name'] for i, c in enumerate(clips) if i not in targets])


def layout(scene, entries, owners, **extra):
    """The overview track, markers, catalog, panel and note for ``entries`` (name, kind, fps, frames, action, and
    targets: [owner kind, object name, Action name, slot identifier] per animated owner), replacing any earlier layout
    on ``owners``. A file that combines models calls it again with both models' entries and owners. ``extra`` goes
    into the catalog. Leaves the file on the overview at frame 1; returns the catalog."""
    import bpy
    intervals, overview_fps = timeline(entries)
    for marker in list(scene.timeline_markers):
        scene.timeline_markers.remove(marker)
    for entry, interval in zip(entries, intervals):
        scene.timeline_markers.new(strip_label(interval['index'], entry['name']), frame=interval['start'])
    for kind, name in owners:
        obj = bpy.data.objects[name]
        item = obj if kind == 'OBJECT' else obj.data.shape_keys
        for old in [t for t in item.animation_data.nla_tracks if t.name == OVERVIEW]:
            item.animation_data.nla_tracks.remove(old)
        track = item.animation_data.nla_tracks.new()
        track.name = OVERVIEW
        for entry, interval in zip(entries, intervals):
            target = next((t for t in entry['targets'] if t[:2] == [kind, name]), None)
            if target is None:
                continue   # this owner does not move in the clip
            action = bpy.data.actions[target[2]]
            label = strip_label(interval['index'], entry['name'])
            strip = track.strips.new(label, interval['start'], action)
            strip.name = label
            strip.action_slot = next(s for s in action.slots if s.identifier == target[3])
            strip.use_sync_length = False
            strip.action_frame_start, strip.action_frame_end = 1, interval['nla_action_end']
            strip.frame_start, strip.frame_end = interval['start'], interval['nla_end']
            strip.scale = interval['sample_step']
            strip.repeat, strip.blend_type, strip.blend_in, strip.blend_out = 1, 'REPLACE', 0, 0
            strip.use_auto_blend = False
            # A face that does not move in the next clip must not keep this one's last shape.
            strip.extrapolation = 'HOLD_FORWARD' if kind == 'OBJECT' else 'NOTHING'
    catalog = dict(schema='nwo-blend-clips-v1', owners=owners, overview_fps=overview_fps, clips=entries,
                   timeline=intervals, **extra)
    _text(CATALOG, json.dumps(catalog, indent=1))
    controls = _text(CONTROLS, Path(__file__).with_name('blend_controls.py').read_text(encoding='utf-8'))
    controls.use_module = True   # registers the panel when the file opens
    _text(NOTE, 'Press Space / Play: every clip plays in order on the sequential timeline (one scene marker per clip).\n\n'
                'To play one clip on a loop, or the rest pose: N > Animations in the 3D View. The panel registers when the '
                'file opens; if Blender asks about automatic script execution for this file, allow it once.\n\n'
                'Each clip is keyed on its own samples at its own rate (sample k at frame k + 1). Loading one clip sets the '
                "scene's frame rate and range to that clip, so playback loops it.\n")
    namespace = {}
    exec(controls.as_string(), namespace)   # leave the file on the overview, as the panel's "Load all clips in order"
    namespace['play_all']()
    return catalog


def _text(name, body):
    import bpy
    text = bpy.data.texts.get(name) or bpy.data.texts.new(name)
    text.clear()
    text.write(body)
    return text


def check(scene):
    """Problems with a reopened file's clip layout (blender_convert.py --validate); empty when it holds."""
    import bpy
    problems = []
    if CONTROLS not in bpy.data.texts or not bpy.data.texts[CONTROLS].use_module:
        problems.append('Animation Controls.py is missing or does not register on load')
    catalog = json.loads(bpy.data.texts[CATALOG].as_string())
    starts = {}
    for entry, interval in zip(catalog['clips'], catalog['timeline']):
        for kind, name, action_name, identifier in entry['targets']:
            starts.setdefault((kind, name), []).append(interval['start'])
            action = bpy.data.actions.get(action_name)
            if action is None or not action.use_fake_user:
                problems.append(f'clip {entry["name"]}: Action {action_name} missing or without a fake user')
                continue
            if not any(s.identifier == identifier for s in action.slots):
                problems.append(f'clip {entry["name"]}: Action {action_name} has no slot {identifier}')
            if tuple(action.frame_range) != (1.0, float(entry['frames'])) and entry['frames'] > 1:
                problems.append(f'clip {entry["name"]}: Action range {tuple(action.frame_range)}, {entry["frames"]} samples')
    for kind, name in catalog['owners']:
        obj = bpy.data.objects.get(name)
        item = obj if kind == 'OBJECT' or obj is None else getattr(obj.data, 'shape_keys', None)
        animation = getattr(item, 'animation_data', None)
        track = next((t for t in animation.nla_tracks if t.name == OVERVIEW), None) if animation else None
        if track is None:
            problems.append(f'{name}: no overview track')
            continue
        found = sorted(int(s.frame_start) for s in track.strips)
        if found != sorted(starts.get((kind, name), [])):
            problems.append(f'{name}: overview strips start at {found[:5]}..., the catalog at {starts.get((kind, name), [])[:5]}...')
    if len(scene.timeline_markers) != len(catalog['timeline']):
        problems.append(f'{len(scene.timeline_markers)} scene markers for {len(catalog["timeline"])} clips')
    return problems
