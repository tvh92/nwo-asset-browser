"""Animation panel embedded in a portable .blend export (nwo.export.blend_animation): N > Animations in the 3D View.

Embedded verbatim as the text block 'Animation Controls.py', registered when the file opens (a module text block), so no
add-on, script path or preference change is needed. It plays the clips listed in 'Animation Catalog.json' as the
research pipeline's controls (nwo.pipeline.controls) do: one clip on its own, looping over its frame range at its own
rate; every clip in order on the sequential NLA track; or the rest pose.
"""
import json

import bpy

CATALOG = 'Animation Catalog.json'
OVERVIEW = 'All clips | sequential timeline'
RANGE_TRACK = 'Clip range | temporary overview'
RANGE_CLIPS = 24


def catalog():
    return json.loads(bpy.data.texts[CATALOG].as_string())


def owner(kind, name):
    """The animated ID of a catalog owner: the object itself, or its mesh's shape keys."""
    obj = bpy.data.objects.get(name)
    if obj is None:
        return None
    if kind == 'OBJECT':
        return obj
    return getattr(obj.data, 'shape_keys', None) if obj.data is not None else None


def owners(data):
    found = [(kind, owner(kind, name)) for kind, name in data['owners']]
    return [(kind, item) for kind, item in found if item is not None and item.animation_data is not None]


def slot_of(action, identifier):
    return next((slot for slot in action.slots if slot.identifier == identifier), None)


def set_fps(fps):
    scene = bpy.context.scene
    scene.render.fps = max(1, round(fps))
    scene.render.fps_base = scene.render.fps / fps


def drop_range(data):
    # Blender evaluates every unmuted strip's channels each frame, so a long overview plays slowly; the range track
    # holds a window of the overview strips and is removed by every other control.
    for _kind, item in owners(data):
        for track in list(item.animation_data.nla_tracks):
            if track.name == RANGE_TRACK:
                item.animation_data.nla_tracks.remove(track)


def clear(data):
    drop_range(data)
    for _kind, item in owners(data):
        item.animation_data.action = None
        for track in item.animation_data.nla_tracks:
            track.mute = True


def rest(data):
    """Rest pose: bones at their rest transform, shape keys at 0."""
    for kind, item in owners(data):
        if kind == 'OBJECT' and item.type == 'ARMATURE':
            for bone in item.pose.bones:
                bone.location = (0.0, 0.0, 0.0)
                bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
                bone.rotation_euler = (0.0, 0.0, 0.0)
                bone.scale = (1.0, 1.0, 1.0)
        elif kind == 'SHAPE_KEYS':
            for block in item.key_blocks[1:]:
                block.value = 0.0


def select_clip(index):
    data = catalog()
    entry = data['clips'][index]
    scene = bpy.context.scene
    clear(data)
    rest(data)
    for kind, name, action_name, identifier in entry['targets']:
        item = owner(kind, name)
        action = bpy.data.actions[action_name]
        item.animation_data.action = action
        item.animation_data.action_slot = slot_of(action, identifier)
    # The scene range is the clip's samples, so playback loops the clip (the closing sample of a loop is not stored).
    set_fps(entry['fps'])
    scene.frame_start, scene.frame_end = 1, entry['frames']
    scene.frame_set(1)
    scene['active_clip'] = entry['name']
    bpy.context.view_layer.update()


def play_all():
    data = catalog()
    scene = bpy.context.scene
    clear(data)
    rest(data)
    for _kind, item in owners(data):
        for track in item.animation_data.nla_tracks:
            track.mute = track.name != OVERVIEW
    set_fps(data['overview_fps'])
    scene.frame_start = 1
    scene.frame_end = data['timeline'][-1]['end_inclusive'] if data['timeline'] else 1
    scene.frame_set(1)
    scene['active_clip'] = 'All clips in order'
    bpy.context.view_layer.update()


def play_range(first, count=RANGE_CLIPS):
    data = catalog()
    scene = bpy.context.scene
    if not data['timeline']:
        return restore_rest()
    first = max(0, min(int(first), len(data['timeline']) - 1))
    last = min(len(data['timeline']), first + max(1, int(count)))
    clear(data)
    rest(data)
    starts = {data['timeline'][i]['start'] for i in range(first, last)}
    for _kind, item in owners(data):
        overview = next((t for t in item.animation_data.nla_tracks if t.name == OVERVIEW), None)
        if overview is None:
            continue
        track = item.animation_data.nla_tracks.new()
        track.name = RANGE_TRACK
        for source in overview.strips:
            if int(source.frame_start) not in starts:
                continue
            strip = track.strips.new(source.name, int(source.frame_start), source.action)
            strip.action_slot = source.action_slot
            for attr in ('use_sync_length', 'action_frame_start', 'action_frame_end', 'frame_start', 'frame_end', 'scale',
                         'repeat', 'blend_type', 'extrapolation', 'blend_in', 'blend_out', 'use_auto_blend'):
                setattr(strip, attr, getattr(source, attr))
    set_fps(data['overview_fps'])
    scene.frame_start = data['timeline'][first]['start']
    scene.frame_end = data['timeline'][last - 1]['end_inclusive']
    scene.frame_set(scene.frame_start)
    scene['active_clip'] = f'Clips {first + 1}-{last} in order'
    bpy.context.view_layer.update()


def jump_to(index):
    play_all()
    bpy.context.scene.frame_set(catalog()['timeline'][index]['start'])


def restore_rest():
    data = catalog()
    scene = bpy.context.scene
    clear(data)
    rest(data)
    scene.frame_start = scene.frame_end = 1
    scene.frame_set(1)
    scene['active_clip'] = 'Rest pose'
    bpy.context.view_layer.update()


class NWOCLIPS_OT_control(bpy.types.Operator):
    bl_idname = 'nwoclips.control'
    bl_label = 'Animation control'
    operation: bpy.props.StringProperty()

    def execute(self, context):
        try:
            index = int(context.scene.nwo_clip_choice)
            if self.operation == 'select':
                select_clip(index)
            elif self.operation == 'all':
                play_all()
            elif self.operation == 'jump':
                jump_to(index)
            elif self.operation == 'range':
                play_range(index)
            elif self.operation == 'rest':
                restore_rest()
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class NWOCLIPS_PT_controls(bpy.types.Panel):
    bl_label = 'Model animations'
    bl_idname = 'NWOCLIPS_PT_controls'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Animations'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        data = catalog()
        if data['clips']:
            layout.prop(scene, 'nwo_clip_choice', text='Clip')
            for operation, label in (('select', 'Load selected clip (loops)'), ('all', 'Load all clips in order'),
                                     ('range', f'Load {RANGE_CLIPS} clips from selected in order (fast)'),
                                     ('jump', 'Jump to selected boundary')):
                layout.operator('nwoclips.control', text=label).operation = operation
        layout.operator('nwoclips.control', text='Restore rest pose').operation = 'rest'
        active = scene.get('active_clip', '')
        if active == 'All clips in order':
            active = next((e['name'] for e in data['timeline']
                           if e['start'] <= scene.frame_current < e['end_exclusive']), active)
        layout.label(text='Active: ' + active)
        layout.prop(scene, 'frame_current', text='Frame')
        layout.operator('screen.animation_play', text='Play / pause', icon='PLAY')
        layout.operator('render.render', text='Render current frame')


def register():
    global _choices
    for cls in (NWOCLIPS_PT_controls, NWOCLIPS_OT_control):
        old = getattr(bpy.types, cls.__name__, None)
        if old:
            bpy.utils.unregister_class(old)
    # Kept in a global: Blender does not hold the strings of an EnumProperty's item list.
    _choices = [(str(i), e['name'], e['kind']) for i, e in enumerate(catalog()['clips'])] or [('0', 'No clips', 'Rest pose only')]
    bpy.types.Scene.nwo_clip_choice = bpy.props.EnumProperty(items=_choices)
    bpy.utils.register_class(NWOCLIPS_OT_control)
    bpy.utils.register_class(NWOCLIPS_PT_controls)


if __name__ == '__main__':
    register()
