"""Rift Apart (PC) materialgraph shader programs: the pass table, the generic material binding and a neutral engine profile.

A ``.materialgraph`` carries its compiled programs in DAT1 section 0xBBFC8900: entries on 16-byte boundaries (zero bytes
may sit between them), each a 12-byte header (u32 magic 'IGSH', u32 SM5 size, u32 DXIL size) followed by the SM5 DXBC
container and the DXIL one. Raytracing libraries use magic 'IGRS' and are DXIL only. Section 0x8C049CCA is the pass
table: 18 or 36 records of (u32 flags, u32 hash, u32 entry offset from the section start, u32 entry size), the record
index being the pass slot; section 0xF9C35F30 is one such record for the raytracing library.

Pass slots as their programs show them (outputs and bindings, over all 1,474 graphs): 1 forward pixel program (one
target, does its own lighting), 4 and 5 G-buffer pixel programs (4 and 3 targets: packed normal/specular, albedo and
occlusion or emission, view depth, and velocity in 4), 8 depth-only pixel program, 11 a two-target pixel program,
15/16 the GI voxelization pixel/geometry programs, 3/7/10/13/17 vertex programs of vertex-deforming graphs.

Material binding is generic and needs no per-graph decoding: constant buffer cb2 (``MultiCustomMaterialCB``) holds the
graph's float defaults overwritten by the material's parameter values at the graph's byte offsets, and texture unit
``t(5 + u)`` (``textureUnit<5+u>``) binds the graph's slot-table row ``u``: the material's texture for that slot key, else
the row's default texture.
"""
import contextlib
import contextvars
import re
import struct

from nwo.games.rcra import material as rmat
from nwo.games.rcra.toc import sections
from nwo.shaders import dxbc, sm5

SHADERS, PASSES, RAYTRACING = 0xBBFC8900, 0x8C049CCA, 0xF9C35F30
ENTRY_MAGIC, RAYTRACING_MAGIC = 0x48534749, 0x53524749    # 'IGSH', 'IGRS'
SLOT_FORWARD, SLOT_GBUFFER, SLOT_GBUFFER_SHORT, SLOT_DEPTH = 1, 4, 5, 8
MATERIAL_BUFFER = 'MultiCustomMaterialCB.g_CustomMaterial_Buffer'
FIRST_MATERIAL_UNIT = 5

# Engine inputs that switch whole engine features off. With them the forward and G-buffer programs of almost every graph
# translate without unbounded loops: no local lights or probes (empty lookup lists), no debug view, no optional features.
# These are viewer choices, not game data; every other engine input stays a named input.
NEUTRAL_INTEGERS = {
    'GlobalWorldCBuffer.m_LightLookupWordCount': 0,
    'GlobalWorldCBuffer.m_EnvProbeLookupWordCount': 0,
    'GlobalWorldCBuffer.m_FeatureFlags': 0,
    'GlobalWorldCBuffer.m_RTFlags': 0,
    'GlobalViewportCBuffer.m_MiscFlags': 0,
}
NEUTRAL_FLOATS = {'GlobalWorldCBuffer.m_KeyLightTapScale': 1.0}    # sun shadow PCF: 8 taps
# Engine flag volumes: the per-tile bitset of flag volumes covering a pixel (g_VolumeFlagBitsetView, whose set bits index
# GlobalVolumeFlagBuffer boxes; rain and snow shelters, puddle areas) is empty in every pass: no flag volume covers the
# material. A viewer choice like the empty light lists.
VOLUME_FLAGS = 'g_VolumeFlagBitsetView'
# Material LOD distances in the engine's per-material and per-subset records (structured buffers, member names from the
# reflection). Writers take the first positive of the object override (NEUTRAL_OBJECT: 0), the subset's and the
# material's distance; with none set the LOD fade term is 0: full detail. The material's alpha is 1, opaque, as the
# neutral object's. Viewer choices, not game data. Other members of these records stay unbound.
NEUTRAL_RECORDS = {'g_MaterialView': {'m_LodDist': 0.0, 'm_Alpha': 1.0}, 'g_ModelSubsetView': {'m_MaterialLodDist': 0.0}}

# Forward profile (pass slot 1), for exporting blended materials. The forward programs light themselves (key light, light
# grid, probes, GI, SSR); an export lights the material's G-buffer surface with the renderer's own lights instead. So every
# light source a forward program reads is zero (every engine constant not bound here is zero), and the forward colour is
# the material's own unlit contribution: emission, self-lit terms and effects. These are export choices, not game data.
# - Draw flags: MatDrawCB.m_RenderFlags holds the material's lit bit (material.draw_flags, decoded); everything else in
#   it (lighting class, per-object state) is 0.
# - Engine textures and buffers: the stored view depth at the far clip (no soft depth fade). The lighting inputs (light
#   grid and lists, shadow atlas, gobos, probes, SSR, AO, the BRDF lookup) read 0, as the light sources they belong to are
#   zero. G-buffer writers that read them (soft-faded effects, a default probe in the emission) get the same values. The scene colour behind is split out by surface.forward; the scene G-buffer and per-object records are bound below.
FORWARD_FAR_CLIP = 1000.0
FORWARD_LIGHTING = ('g_LightGridLookup', 'g_LightLookupView', 'g_LightZBinView', 'g_LightGridData', 'g_LightVolumes',
                    'g_Lights', 'g_ShadowAtlas', 'g_GoboAtlas', 'g_EnvProbeEnvs', 'g_EnvProbeArray', 'g_EnvProbeDefault',
                    'g_SSRFinalTex', 'g_AoSsrBuffer', 'g_BrdfLookup', 'g_RayTracedShadows')
# - The deformable-snow map (snow writers read it at the world position once the view is bound): empty, 0, as the
#   deformable-snow constants are 0 under review values. Not decoded.
# - Screen-space reflections a forward program traces itself (eyes: a march through the half-resolution depth, then the
#   previous frame's reflection reprojected by the velocity buffer): the empty scene, so the half-resolution linear depth
#   is the far clip as the view depth, the reflection history 0 (a light source) and the velocity 0 (no motion). Not decoded.
SCREEN_REFLECTION = {'g_HalfResLinearDepth': FORWARD_FAR_CLIP, 'g_SSRHistory': 0.0, 'g_SSRVelocityTex': 0.0}
FLOAT_RESOURCES = ('g_ViewDepthBuffer', 'g_HalfResLinearDepth')   # depths: an integer-addressed read (ld) is a float
FORWARD_RESOURCES = dict({name: 0.0 for name in FORWARD_LIGHTING}, g_ViewDepthBuffer=FORWARD_FAR_CLIP, g_DeformablesView=0.0,
                         **SCREEN_REFLECTION)
# - The scene's G-buffer (screen effects, eye normal smoothing, decals read it around the pixel): cleared, every word 0,
#   as the empty scene the far-clip depth stands for; its size is the review viewport's (resinfo). Not decoded.
SCENE_GBUFFER = {'g_GBufNormalSpec': 0}
SCENE_SIZE = (1920, 1080, 1, 1)
# - View: the programs rebuild the view-space position from SV_POSITION (screen xy, view depth w) and turn it into world
#   space with m_ViewToWorldMat. With these values the view-space position is SV_POSITION.(x, y, w) and view space has the
#   world's axes, so a renderer supplies SV_POSITION.xyw as the world offset from the camera to the point (input
#   'VIEW.xyz') and the matrix translation as the camera's world position ('CAMERA.xyz'), both in game space.
FORWARD_VIEW_FLOATS = {
    'GlobalViewportCBuffer.m_InvDimensions': 1.0, 'GlobalViewportCBuffer.m_InvDimensions+4': 1.0,
    'GlobalViewportCBuffer.m_ScreenToViewA': 1.0, 'GlobalViewportCBuffer.m_ScreenToViewA+4': 1.0,
    'GlobalViewportCBuffer.m_ScreenToViewA+8': 0.0, 'GlobalViewportCBuffer.m_ScreenToViewA+12': 0.0,
    'GlobalViewportCBuffer.m_ScreenToViewB': 0.0, 'GlobalViewportCBuffer.m_ScreenToViewB+4': 1.0,
    'GlobalViewportCBuffer.m_ViewToWorldMat': 1.0, 'GlobalViewportCBuffer.m_ViewToWorldMat+20': 1.0,
    'GlobalViewportCBuffer.m_ViewToWorldMat+40': 1.0,
    'GlobalViewportCBuffer.m_FarClip': FORWARD_FAR_CLIP,
    'GlobalViewportCBuffer.m_CustomClipPlane+12': 1.0,   # the custom clip plane (0, 0, 0, 1) clips nothing
}
FORWARD_VIEW_INPUTS = {'SV_POSITION.x': 'VIEW.x', 'SV_POSITION.y': 'VIEW.y', 'SV_POSITION.w': 'VIEW.z',
                       'GlobalViewportCBuffer.m_ViewToWorldMat+48': 'CAMERA.x',
                       'GlobalViewportCBuffer.m_ViewToWorldMat+52': 'CAMERA.y',
                       'GlobalViewportCBuffer.m_ViewToWorldMat+56': 'CAMERA.z'}


# The engine's per-object record (structured buffer g_SceneObjectView, element "$Element", 128 bytes; member names and
# offsets from the programs' reflection). The engine model vertex program reads it: the transform is row-major 3x4 (the
# world normal is the object normal dotted with rows 0, 1 and 2 at bytes 0, 16 and 32, translation in dword 3), and it
# passes the object's custom values on as TEXCOORD6 = (1, Misc, Rand, LOD factor) and TEXCOORD7 = (custom X, Y, Z, FX
# lighting index). The values are set per object at run time; NEUTRAL_OBJECT is a plain object: identity transform, alpha
# 1, white colour, every custom value, time and bias 0. Viewer choices, not game data. Fp16 pairs pack low half first.
HALF_ONE = 0x3C00
NEUTRAL_OBJECT = {
    'm_ObjToWorldMatTrans': (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    'm_ObjToWorldNormScale': (1.0, 1.0, 1.0),
    'm_TransformScale': (1.0,),
    'm_Alpha': (1.0,),
    'm_ColorRGFp16': (HALF_ONE | HALF_ONE << 16,),
    'm_ColorBMiscFp16': (HALF_ONE,),
    'm_Radius': (1.0,),
}
OBJECT_BUFFER = 'g_SceneObjectView'
NEUTRAL_OBJECT_INTERPOLANTS = {'TEXCOORD6.x': 1.0, 'TEXCOORD6.y': 0.0, 'TEXCOORD6.z': 0.0, 'TEXCOORD6.w': 0.0,
                               'TEXCOORD7.x': 0.0, 'TEXCOORD7.y': 0.0, 'TEXCOORD7.z': 0.0, 'TEXCOORD7.w': 0.0}


# Members of NEUTRAL_OBJECT a caller replaces for one translation (object_override): the visual-effect meshes the viewer
# draws on a model take their node's colour (nwo.games.rcra.effects). A context variable, so concurrent translations on
# other threads keep the neutral object.
OBJECT_OVERRIDE = contextvars.ContextVar('rcra_object_override', default=None)


@contextlib.contextmanager
def object_override(values):
    """Translate with ``values`` ({member: tuple, NEUTRAL_OBJECT's form}) over the neutral object; None keeps it."""
    token = OBJECT_OVERRIDE.set(dict(values) if values else None)
    try:
        yield
    finally:
        OBJECT_OVERRIDE.reset(token)


def half_pair(low, high):
    """Two fp16 values packed in one u32, low half first (m_ColorRGFp16, m_ColorBMiscFp16)."""
    import numpy as np
    return int(np.float16(low).view(np.uint16)) | int(np.float16(high).view(np.uint16)) << 16


def object_text(name, value):
    """An object member's value for a limitation: fp16 pairs unpacked, floats as written."""
    import numpy as np
    if name.endswith('Fp16'):
        halves = [float(np.uint16(w & 0xFFFF).view(np.float16)) for word in value for w in (word, word >> 16)]
        return '(%s)' % ', '.join('%.4g' % h for h in halves)
    return '(%s)' % ', '.join('%.4g' % v for v in value)


def object_words(reflection):
    """{byte offset: u32 bits} of the neutral object (with any object_override) in the layout the program's reflection
    gives; None without one."""
    import numpy as np
    if reflection is None:
        return None
    buffer = reflection.constant_buffer(OBJECT_BUFFER)
    if buffer is None or not buffer.variables or not buffer.variables[0].members:
        return None
    values = dict(NEUTRAL_OBJECT, **(OBJECT_OVERRIDE.get() or {}))
    words = {at: 0 for at in range(0, buffer.size, 4)}
    for name, at, _member in buffer.variables[0].members:
        for i, value in enumerate(values.get(name, ())):
            words[at + 4 * i] = int(np.float32(value).view(np.uint32)) if isinstance(value, float) else value
    return words


def pass_table(graph_data):
    """{slot: (flags, hash, entry offset, entry size)} of the filled pass slots."""
    dat = graph_data[36:]
    table = sections(dat)
    if PASSES not in table or SHADERS not in table:
        raise ValueError('Material graph has no shader or pass section')
    offset, size = table[PASSES]
    rows = {}
    for slot in range(size // 16):
        record = struct.unpack_from('<4I', dat, offset + 16 * slot)
        if record[3]:
            rows[slot] = record
    return rows


def pass_blobs(graph_data, slot):
    """(SM5 DXBC bytes or None, DXIL DXBC bytes or None) of one pass slot."""
    dat = graph_data[36:]
    base, size = sections(dat)[SHADERS]
    _flags, _hash, offset, length = pass_table(graph_data)[slot]
    if offset + length > size:
        raise ValueError('Pass entry outside the shader section')
    magic, sm5_size, dxil_size = struct.unpack_from('<3I', dat, base + offset)
    if magic != ENTRY_MAGIC or 12 + sm5_size + dxil_size != length:
        raise ValueError('Pass entry header does not match its record')
    start = base + offset + 12
    return (dat[start:start + sm5_size] or None), (dat[start + sm5_size:start + sm5_size + dxil_size] or None)


def pass_shader(graph_data, slot):
    """The decoded SM5 program of a pass slot."""
    blob = pass_blobs(graph_data, slot)[0]
    if blob is None:
        raise ValueError('Pass slot %d has no SM5 program' % slot)
    return dxbc.shader(blob)


def _layout(graph_data):
    """graph_layout, or no slots, parameters and defaults for a graph without those sections (template graphs)."""
    try:
        return rmat.graph_layout(graph_data)
    except ValueError as error:
        if 'no slot, parameter or default section' not in str(error):
            raise
        return [], {}, ()


def material_constants(graph_data, material_data):
    """cb2 as floats: the graph's defaults overwritten by the material's values at the graph's byte offsets."""
    _rows, offsets, defaults = _layout(graph_data)
    cb2 = list(defaults)
    for key, values in rmat.material_parameters(material_data).items():
        if key in offsets:
            for i, value in enumerate(values):
                at = offsets[key] // 4 + i
                cb2 += [0.0] * (at + 1 - len(cb2))
                cb2[at] = float(value)
    return cb2


def constant_name(index):
    """The translator's input name of cb2 dword ``index``."""
    return MATERIAL_BUFFER + ('+%d' % (4 * index) if index else '')


def material_units(graph_data, material_data):
    """{'textureUnit<n>': texture path}: what the engine binds for each slot-table row (material.slot_texture): the
    material's texture for an exposed row, else the graph default."""
    slots = dict(rmat.material_slots(material_data))
    return {'textureUnit%d' % (FIRST_MATERIAL_UNIT + row[0]): rmat.slot_texture(slots, row)
            for row in rmat.graph_slot_rows(graph_data)}


class NeutralEngine(sm5.Bindings):
    """NEUTRAL_INTEGERS/NEUTRAL_FLOATS bound as constants; ``material`` (cb2 floats) bound when given."""

    def __init__(self, material=None):
        self.graph = None
        self.material = {constant_name(i): v for i, v in enumerate(material or ())}

    def constant(self, name):
        if name in NEUTRAL_INTEGERS:
            return self.graph.const('i', NEUTRAL_INTEGERS[name])
        if name in NEUTRAL_FLOATS:
            return self.graph.const('f', NEUTRAL_FLOATS[name])
        if name in self.material:
            return self.graph.const('f', self.material[name])
        return None


def translate(shader, material=None):
    """Translate a pass program under the neutral engine profile (and the material's cb2 when given)."""
    bindings = NeutralEngine(material)
    translator = sm5.Translator(shader, bindings)
    bindings.graph = translator.g
    return translator.run()
