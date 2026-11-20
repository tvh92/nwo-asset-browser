"""Rift Apart translated materials in the viewer: recipe DAGs (rc2016.games.rcra.surface) -> GLSL 1.20 fragment programs.

A receipt (rc2016.games.rcra.viewer_materials) carries a material's G-buffer surface recipe, its forward recipe when the game
draws it blended, the blend mode and the layout of each texture binding. Every recipe node becomes one GLSL statement, so
the viewer runs the game's own arithmetic, as the Blender emitter does (rc2016/export/rcra_blender.py).

Inputs, in the viewer's varyings:
- TEXCOORD.xy / TEXCOORD1.xy: vUV / vUV1, game UVs (the viewer uploads V unflipped and image rows top first).
- TEXCOORD2.xyz: the world normal vTintWorldNrm (native Y-up), not turned on back faces, as the game's interpolant.
- TEXCOORD3.xyz and .w: the decoded vertex tangent (rc2016.games.rcra.model.unpack_tangents, the engine model vertex
  program's decode) as vTintWorldTan, skinned with the vertex, and the whole stored TEXCOORD3.w (vTangentSign: the
  bitangent sign over a squared scale), both interpolated as the game's interpolant. The draw's render flag 0x10000, which
  negates the sign, is clear (an unmirrored draw). A mesh without a tangent stream gets the UV-derivative frame of
  native_glsl.FRAME_HELPERS, bitangent along the game's -v, the orientation the stored tangents have.
- TEXCOORD4: vRcraColour, the vertex colour squared per channel in the vertex stage, as the engine model vertex program
  does (``mul r13, r14, r14`` after its typed load of the colour: g_MainPaintVBView for a plain instance). vertex_source(), the materials' vertex program,
  reads the model's colour section as normalized bytes in R, G, B, A order (attribute aRcraColour at
  shading.RCRA_COLOUR_ATTRIBUTE); the game sets its view's format at run time, so that order is the reader's, not read
  from the game. A model without the section reads 1, the engine's value for a subset without a colour stream when render
  flag 0x800 is clear. Per-object paint, which the engine reads instead when an object has it, is not drawn.
- SV_IsFrontFace: gl_FrontFacing. VIEW.xyz: vTintViewOffset (camera to point); CAMERA.xyz: vTintWorldPos - vTintViewOffset.

Textures bind as uNativeTex<binding> (the native path's uniforms, so the viewport's binding code serves both games). A
binding's layout is 'slice' (one 2D texture), 'array' (a padded atlas of every slice, picked by round(z), clamped) or
'volume' (a padded atlas, trilinear in depth with wrap), as rc2016.games.rcra.blender_export writes them for Blender, or
'cube' (a samplerCube of the six stored faces, looked up with the program's direction unchanged: D3D and GL share the face
table and the top-row-first face images, so the lookup returns the D3D texel; the top mip, the program's explicit mip level
is not applied).

Lighting: each light of the viewer's review rig (key, fill, preview) is applied by the material's lighting class's own
apply program, translated (rc2016.games.rcra.lighting.direct, the receipt's ``lighting`` recipe), emitted as the function
rcraLight and called once per light. Ambient light is the rig's, with the class-0 BRDF (GGX alpha = r^4, r = 0.9725 -
0.7514 gloss) and occlusion on ambient light only. A receipt without a lighting recipe takes the class-0 BRDF for its
lights too: Smith joint visibility approximation, Schlick with F90 = 1, Lambert.

Ambient light under the Shared HDRI lighting: a receipt with an ``ambient`` recipe (rc2016.games.rcra.lighting.ambient, the
class's apply program with the key light off) draws it as rcraGameAmbient once the viewport has the review probe
(uRcraProbeOn): its default-probe reads look up the probe cube (uRcraProbe, the review environment prefiltered by the
engine's probe program, rc2016.games.rcra.probe) at the program's own lod, along the program's direction turned into the
environment's world frame (uViewToWorld . vNativeToView); its BRDF lookup reads slice 0 of the engine's table (uRcraBrdf,
rc2016.games.rcra.engine_tables) with a linear clamp sampler, as the game's g_LinearClampSampler. Otherwise the rig's ambient.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math

EMITTER_VERSION = 5
PASS_UNIFORM = 'uRcraPass'   # forward programs: 0 the multiply pass (background factor), 1 the colour pass
COLOUR_VARYING = 'vRcraColour'
GEOMETRY = {'TEXCOORD.x': 'vUV.x', 'TEXCOORD.y': 'vUV.y', 'TEXCOORD1.x': 'vUV1.x', 'TEXCOORD1.y': 'vUV1.y',
            'TEXCOORD2.x': 'vTintWorldNrm.x', 'TEXCOORD2.y': 'vTintWorldNrm.y', 'TEXCOORD2.z': 'vTintWorldNrm.z',
            'TEXCOORD3.x': 'rcraTangent.x', 'TEXCOORD3.y': 'rcraTangent.y', 'TEXCOORD3.z': 'rcraTangent.z',
            'TEXCOORD3.w': 'rcraHandedness', 'TEXCOORD4.x': COLOUR_VARYING + '.x', 'TEXCOORD4.y': COLOUR_VARYING + '.y',
            'TEXCOORD4.z': COLOUR_VARYING + '.z', 'TEXCOORD4.w': COLOUR_VARYING + '.w',
            'SV_IsFrontFace': '(gl_FrontFacing ? 1.0 : 0.0)',
            'VIEW.x': 'vTintViewOffset.x', 'VIEW.y': 'vTintViewOffset.y', 'VIEW.z': 'vTintViewOffset.z',
            'CAMERA.x': 'rcraCamera.x', 'CAMERA.y': 'rcraCamera.y', 'CAMERA.z': 'rcraCamera.z',
            # The scene behind (surface.forward's scene mode): the frame copy at the pixel (viewport._scene_copies).
            'BACKGROUND.r': 'rcraScene.r', 'BACKGROUND.g': 'rcraScene.g', 'BACKGROUND.b': 'rcraScene.b',
            'BACKGROUND.a': 'rcraScene.a',
            # Fur shells (rc2016.games.rcra.fur.SHELL_INPUTS): the shell vertex program's per-vertex values (shell_vertex_source).
            'FURSHELL.z': 'vFurShell.x', 'FURSHELL.offset': 'vFurShell.y', 'FURSHELL.base': 'vFurShell.z'}
SHELL_BLEND = 'fur_shell'
SHELL_UNIFORMS = ('uFurLayer', 'uFurInverse', 'uFurLength', 'uFurOffsetScale')
SHELL_CONTROL = 'g_FurControlMap'   # the shell vertex program's offset scale (blue), read in the vertex stage
SCENE_UNIFORMS = ('uSceneColour', 'uHasSceneColour', 'uSceneDepth', 'uHasSceneDepth', 'uSceneDepthInfo')
AMBIENT_UNIFORMS = ('uRcraProbe', 'uRcraBrdf', 'uRcraProbeOn')   # the game's ambient light: probe cube, BRDF lookup, ready
PROBE_BINDING, BRDF_BINDING = 'g_EnvProbeDefault', 'g_BrdfLookup'
BRDF_SIZE = 64   # rc2016.games.rcra.engine_tables.SIZE
AMBIENT_LAYOUTS = {PROBE_BINDING: dict(kind='probe'), BRDF_BINDING: dict(kind='lut')}
LIMITATION = ('Viewer: the draw is unmirrored (render flag 0x10000 clear); vertex colour is the '
              'model\'s colour section squared, as the engine squares it, with its bytes read in R, G, B, A order (the '
              'game\'s view format is not read), 1 without one, and no per-object paint; the review rig and environment '
              'stand in for the game\'s lights.')


# D3D's inf and NaN constants (a division by a zero profile value; NaN the receipt keeps after fold_nan): GLSL 1.20 has
# no literal for them, so they are made at run time from a uniform that is never set (0 from the link), which the
# compiler cannot fold: rcraInf = 1 / 0, rcraNan = 0 / 0.
NON_FINITE_DECLARATION = 'uniform float uRcraZero;   // never set: 0, for inf and NaN at run time\n'
NON_FINITE_LINE = '    float rcraInf = 1.0 / uRcraZero; float rcraNan = uRcraZero / uRcraZero;'


def _literal(value):
    value = float(value)
    if value != value:
        return 'rcraNan'
    if math.isinf(value):
        return 'rcraInf' if value > 0 else '(-rcraInf)'
    text = format(value, '.9g')
    return text if '.' in text or 'e' in text or 'n' in text else text + '.0'


def uniform(binding):
    return 'uNativeTex' + binding


class _Emitter:
    def __init__(self, recipe, layouts, prefix, inputs=None):
        self.recipe, self.layouts, self.prefix = recipe, layouts, prefix
        self.inputs = GEOMETRY if inputs is None else inputs
        self.lines, self.names, self.samples = [], {}, {}

    def value(self, i):
        if i in self.names:
            return self.names[i]
        n = self.recipe['nodes'][i]
        op, args = n['op'], n['args']
        if op == 'const':
            expr = ('1.0' if n['value'] else '0.0') if n.get('type') == 'b' else _literal(n['value'])
            self.names[i] = expr
            return expr
        if op == 'input':
            if n['name'] not in self.inputs:
                raise ValueError('Recipe input not available in the viewer: ' + n['name'])
            self.names[i] = self.inputs[n['name']]
            return self.names[i]
        if op == 'channel':
            expr = self.sample(args[0]) + '.' + 'rgba'[n['index']]
        else:
            a = [self.value(x) for x in args]
            expr = self.operation(op, a)
        name = '%s%d' % (self.prefix, i)
        self.lines.append('    float %s = %s;' % (name, expr))
        self.names[i] = name
        return name

    @staticmethod
    def operation(op, a):
        binary = {'add': '+', 'mul': '*', 'div': '/'}
        if op in binary:
            return '(%s %s %s)' % (a[0], binary[op], a[1])
        unary = {'abs': 'abs', 'sqrt': 'sqrt', 'rsq': 'inversesqrt', 'log2': 'log2', 'exp2': 'exp2', 'sin': 'sin',
                 'cos': 'cos', 'round_ni': 'floor', 'round_pi': 'ceil', 'round_ne': 'rcraRoundEven', 'round_z': 'rcraTrunc'}
        if op in unary:
            return '%s(%s)' % (unary[op], a[0])
        if op in ('min', 'max'):
            return '%s(%s, %s)' % (op, a[0], a[1])
        if op == 'neg':
            return '(-%s)' % a[0]
        if op == 'sat':
            return 'clamp(%s, 0.0, 1.0)' % a[0]
        if op == 'rcp':
            return '(1.0 / %s)' % a[0]
        if op == 'frc':
            return '(%s - floor(%s))' % (a[0], a[0])
        if op in ('eq', 'ne', 'lt', 'ge'):
            return '(%s %s %s ? 1.0 : 0.0)' % (a[0], {'eq': '==', 'ne': '!=', 'lt': '<', 'ge': '>='}[op], a[1])
        if op == 'b_and':
            return '(%s * %s)' % (a[0], a[1])
        if op == 'b_or':
            return 'max(%s, %s)' % (a[0], a[1])
        if op == 'b_not':
            return '(1.0 - %s)' % a[0]
        if op == 'select':
            return '(%s != 0.0 ? %s : %s)' % (a[0], a[1], a[2])
        if op == 'ddx':
            return 'dFdx(%s)' % a[0]
        if op == 'ddy':
            return '(-dFdy(%s))' % a[0]   # D3D screen y runs down, GL's up
        raise ValueError('Recipe operation not emitted: ' + op)

    def sample(self, i):
        if i in self.samples:
            return self.samples[i]
        n = self.recipe['nodes'][i]
        binding = n['binding']
        layout = self.layouts.get(binding)
        if layout is None:
            raise ValueError('No texture for ' + binding)
        u, v = self.value(n['args'][0]), self.value(n['args'][1])
        name = '%ss%d' % (self.prefix, i)
        sampler = uniform(binding)
        if layout['kind'] == 'probe':   # the review probe at the program's own lod (sample_l: args 0 to 2, lod arg 4)
            expr = 'rcraProbe(vec3(%s, %s, %s), %s)' % (u, v, self.value(n['args'][2]), self.value(n['args'][4]))
            self.lines.append('    vec4 %s = %s;' % (name, expr))
            self.samples[i] = name
            return name
        if layout['kind'] == 'lut':   # the BRDF lookup's slice 0
            if self.recipe['nodes'][n['args'][2]].get('value') != 0.0:
                raise ValueError('The BRDF lookup is read at a slice other than 0')
            self.lines.append('    vec4 %s = rcraBrdf(%s, %s);' % (name, u, v))
            self.samples[i] = name
            return name
        if layout['kind'] == 'cube':
            expr = 'textureCube(%s, vec3(%s, %s, %s))' % (sampler, u, v, self.value(n['args'][2]))
        elif layout['kind'] == 'slice':
            expr = 'texture2D(%s, vec2(%s, %s))' % (sampler, u, v)
        else:
            z = self.value(n['args'][2])
            grid = 'vec4(%s, %s, %s, %s)' % tuple(_literal(x) for x in (layout['columns'], layout['width'], layout['height'],
                                                                        -(-layout['depth'] // layout['columns'])))
            helper = 'rcraVolume' if layout['kind'] == 'volume' else 'rcraArray'
            expr = '%s(%s, vec2(%s, %s), %s, %s, %s)' % (helper, sampler, u, v, z, _literal(layout['depth']), grid)
        self.lines.append('    vec4 %s = %s;' % (name, expr))
        self.samples[i] = name
        return name

    def term(self, name, default='0.0'):
        terms = self.recipe['terms']
        return self.value(terms[name]) if name in terms else default


HELPERS = """
float rcraRoundEven(float x) { float r = floor(x + 0.5); return (r - x == 0.5 && mod(r, 2.0) != 0.0) ? r - 1.0 : r; }
float rcraTrunc(float x) { return x < 0.0 ? ceil(x) : floor(x); }
// Padded slice atlas (rc2016.games.rcra.blender_export.volume_atlas): g = (columns, tile width, tile height, rows); each tile
// carries a one-texel border copied from its opposite edge, so a bilinear lookup inside it wraps as the texture does.
vec4 rcraTile(sampler2D t, vec2 uv, float k, vec4 g) {
    float row = floor(k / g.x), col = k - row * g.x;
    vec2 f = uv - floor(uv);
    vec2 p = vec2(col * (g.y + 2.0) + 1.0 + f.x * g.y, row * (g.z + 2.0) + 1.0 + f.y * g.z);
    return texture2D(t, p / vec2(g.x * (g.y + 2.0), g.w * (g.z + 2.0)));
}
vec4 rcraArray(sampler2D t, vec2 uv, float z, float n, vec4 g) { return rcraTile(t, uv, clamp(floor(z + 0.5), 0.0, n - 1.0), g); }
vec4 rcraVolume(sampler2D t, vec2 uv, float z, float n, vec4 g) {
    float s = z * n - 0.5, k = floor(s);
    return mix(rcraTile(t, uv, mod(k, n), g), rcraTile(t, uv, mod(k + 1.0, n), g), s - k);
}
// Rift Apart deferred BRDF (class-0 apply program) on the review rig: GGX alpha = r^4, r = 0.9725 - 0.7514 gloss.
float rcraVisibility(float NdL, float NdV, float a) {
    return 0.5 / max(NdL * (NdV * (1.0 - a) + a) + NdV * (NdL * (1.0 - a) + a), 1e-6);
}
vec3 rcraLobe(vec3 N, vec3 V, vec3 L, vec3 F0, float a) {
    vec3 H = normalize(L + V);
    float NdL = max(dot(N, L), 0.0), NdV = max(dot(N, V), 1e-4), NdH = max(dot(N, H), 0.0);
    float a2 = a * a, d = NdH * NdH * (a2 - 1.0) + 1.0;
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(V, H), 0.0), 5.0);
    return a2 / (PI * d * d) * rcraVisibility(NdL, NdV, a) * F * NdL;
}
vec3 rcraDiffuse(vec3 N, vec3 albedo, float occlusion) {
    if (!uLightingEnabled) return albedo;
    vec3 previewL; float previewW = previewLightWeight(previewL);
    vec3 preview = previewW > 0.0 ? uPreviewLightColour * previewW * max(dot(N, previewL), 0.0) : vec3(0.0);
    return albedo * ((max(dot(N, rigKey()), 0.0) * keyColour() + max(dot(N, rigFill()), 0.0) * fillColour()
        + ambientLight(N) * occlusion) / PI * diffuseScale() + preview);
}
float rcraAlpha(float gloss) { float r = 0.9725 - 0.7514 * clamp(gloss, 0.0, 1.0); r *= r; return r * r; }
vec3 rcraSpecular(vec3 N, vec3 V, vec3 F0, float gloss) {
    if (!uLightingEnabled) return vec3(0.0);
    float a = rcraAlpha(gloss);
    vec3 previewL; float previewW = previewLightWeight(previewL);
    vec3 S = rcraLobe(N, V, rigKey(), F0, a) * keyColour() + rcraLobe(N, V, rigFill(), F0, a) * fillColour()
        + envSpecular(N, V, F0, vec3(1.0), a, ambientLight(N));
    if (previewW > 0.0) S += rcraLobe(N, V, previewL, F0, a) * uPreviewLightColour * previewW;
    return S;
}
// The rig's ambient light alone: diffuse with occlusion, and the environment's specular (class-0 BRDF).
vec3 rcraAmbient(vec3 N, vec3 V, vec3 albedo, vec3 F0, float gloss, float occlusion) {
    vec3 ambient = ambientLight(N);
    return albedo * ambient * occlusion / PI * diffuseScale() + envSpecular(N, V, F0, vec3(1.0), rcraAlpha(gloss), ambient);
}
"""

# Inputs of a lighting recipe (rc2016.games.rcra.lighting.INPUTS and the class's extra fields) -> rcraLight parameters, in
# game space: N the world normal, VIEW the camera-to-point offset, DEPTH the view depth, L the direction to the light and
# LIGHT its colour (the game's convention: diffuse = albedo x N.L x LIGHT, so LIGHT = the rig colour / pi).
LIGHT_INPUTS = {'normal.x': 'rN.x', 'normal.y': 'rN.y', 'normal.z': 'rN.z', 'VIEW.x': 'rView.x', 'VIEW.y': 'rView.y',
                'VIEW.z': 'rView.z', 'DEPTH': 'rDepth', 'albedo.r': 'rA.r', 'albedo.g': 'rA.g', 'albedo.b': 'rA.b',
                'f0.r': 'rF.r', 'f0.g': 'rF.g', 'f0.b': 'rF.b', 'gloss': 'rG', 'occlusion': 'rO', 'extra.0': 'rE.x',
                'extra.1': 'rE.y', 'extra.2': 'rE.z', 'extra.3': 'rE.w', 'extra.4': 'rE4', 'L.x': 'rL.x', 'L.y': 'rL.y',
                'L.z': 'rL.z', 'LIGHT.r': 'rC.r', 'LIGHT.g': 'rC.g', 'LIGHT.b': 'rC.b'}
LIGHT_SIGNATURE = ('vec3 rcraLight(vec3 rN, vec3 rView, float rDepth, vec3 rA, vec3 rF, float rG, float rO, vec4 rE, '
                   'float rE4, vec3 rL, vec3 rC, float rKd)')
# Each rig light through rcraLight: L and the normal turned from view space into the game's world frame (vNativeToView
# maps native vectors to view space; its transpose turns back, up to scale), LIGHT = the rig colour / pi. The diffuse factor
# rKd keeps the rig's balance: diffuseScale() for key and fill, pi for the preview light, whose rig colour is not over pi.
CLASS_LIT = """
vec3 rcraClassLit(vec3 Nv, vec3 Vv, vec3 albedo, vec3 F0, float gloss, float occlusion, vec3 Nw, vec4 E, float E4) {
    if (!uLightingEnabled) return albedo;
    vec3 n = dot(Nw, Nw) > 0.0 ? normalize(Nw) : normalize(Nv * vNativeToView);
    vec3 view = vTintViewOffset; float depth = max(-vPos.z, 1e-4);
    vec3 lit = %(ambient)s;
    lit += rcraLight(n, view, depth, albedo, F0, gloss, occlusion, E, E4, normalize(rigKey() * vNativeToView),
                     keyColour() / PI, diffuseScale());
    if (dot(fillColour(), vec3(1.0)) > 0.0)
        lit += rcraLight(n, view, depth, albedo, F0, gloss, occlusion, E, E4, normalize(rigFill() * vNativeToView),
                         fillColour() / PI, diffuseScale());
    vec3 previewL; float previewW = previewLightWeight(previewL);
    if (previewW > 0.0)
        lit += rcraLight(n, view, depth, albedo, F0, gloss, occlusion, E, E4, normalize(previewL * vNativeToView),
                         uPreviewLightColour * previewW / PI, PI);
    return lit;
}
"""


RIG_AMBIENT = 'rcraAmbient(Nv, Vv, albedo, F0, gloss, occlusion)'
GAME_AMBIENT = ('(uRcraProbeOn ? rcraGameAmbient(n, view, depth, albedo, F0, gloss, occlusion, E, E4) : '
                'rcraAmbient(Nv, Vv, albedo, F0, gloss, occlusion))')
AMBIENT_SIGNATURE = ('vec3 rcraGameAmbient(vec3 rN, vec3 rView, float rDepth, vec3 rA, vec3 rF, float rG, float rO, vec4 rE, '
                     'float rE4)')
AMBIENT_HELPERS = """
uniform samplerCube uRcraProbe; uniform sampler2D uRcraBrdf; uniform bool uRcraProbeOn;
// The review probe along a game-space direction, at an explicit lod (sample_l): the cube is in the environment's world frame.
vec4 rcraProbe(vec3 d, float lod) {
#ifdef GL_ARB_shader_texture_lod
    return textureCubeLod(uRcraProbe, uViewToWorld * (vNativeToView * d), lod);
#else
    return textureCube(uRcraProbe, uViewToWorld * (vNativeToView * d));
#endif
}
// Slice 0 of the engine's BRDF lookup, linear and clamped to the edge texels' centres as a clamp sampler reads it.
vec4 rcraBrdf(float u, float v) {
    return texture2D(uRcraBrdf, clamp(vec2(u, v), vec2(0.5 / %(size)s.0), vec2(1.0 - 0.5 / %(size)s.0)));
}
""" % dict(size=BRDF_SIZE)


def ambient_source(recipe):
    """GLSL of rcraGameAmbient for an ambient recipe: the class's apply program with the key light off, one statement per
    node, returning the ambient colour (albedo included)."""
    emitter = _Emitter(recipe, AMBIENT_LAYOUTS, 'a', inputs=LIGHT_INPUTS)
    t = emitter.term
    colour = 'vec3(%s, %s, %s)' % (t('ambient.r'), t('ambient.g'), t('ambient.b'))
    lines = emitter.lines
    if any('rcraInf' in line or 'rcraNan' in line for line in lines):
        lines.insert(0, NON_FINITE_LINE)
    body = '\n'.join(lines + ['    return %s;' % colour])
    return '\n// Ambient light: lighting class %d apply program at 0x%X, key light off (rc2016.games.rcra.lighting.ambient)\n%s {\n%s\n}\n' % (
        recipe.get('lighting_class', 0), recipe.get('offset', 0), AMBIENT_SIGNATURE, body)


def light_source(recipe, ambient=None):
    """GLSL of rcraLight and rcraClassLit for a lighting recipe: the class's apply program, one statement per node,
    returning albedo x diffuse x rKd + specular; with an ``ambient`` recipe, rcraGameAmbient too, which rcraClassLit uses
    while the review probe is ready."""
    emitter = _Emitter(recipe, {}, 'l', inputs=LIGHT_INPUTS)
    t = emitter.term
    diffuse = 'vec3(%s, %s, %s)' % (t('diffuse.r'), t('diffuse.g'), t('diffuse.b'))
    specular = 'vec3(%s, %s, %s)' % (t('specular.r'), t('specular.g'), t('specular.b'))
    lines = emitter.lines
    if any('rcraInf' in line or 'rcraNan' in line for line in lines):
        lines.insert(0, NON_FINITE_LINE)
    body = '\n'.join(lines + ['    return rA * %s * rKd + %s;' % (diffuse, specular)])
    extra = AMBIENT_HELPERS + ambient_source(ambient) if ambient else ''
    return extra + '\n// Lighting class %d: apply program at 0x%X, translated (rc2016.games.rcra.lighting)\n%s {\n%s\n}\n%s' % (
        recipe.get('lighting_class', 0), recipe.get('offset', 0), LIGHT_SIGNATURE, body,
        CLASS_LIT % dict(ambient=GAME_AMBIENT if ambient else RIG_AMBIENT))


class _PerLightEmitter(_Emitter):
    """An _Emitter for a recipe of the light inputs L.xyz and LIGHT.rgb, emitted once per light: statements that read no
    light input are emitted once and shared; the others again for each light (``light``), all at the top level."""

    def __init__(self, recipe, layouts, prefix):
        super().__init__(recipe, layouts, prefix, inputs=dict(GEOMETRY))
        nodes = recipe['nodes']
        self.dependent = set()
        for i, n in enumerate(nodes):   # arguments precede their users
            if (n['op'] == 'input' and n['name'].startswith(('L.', 'LIGHT.'))) or any(a in self.dependent for a in n['args']):
                self.dependent.add(i)
        self.base = prefix

    def light(self, k, direction, colour):
        for cache in (self.names, self.samples):
            for i in [i for i in cache if i in self.dependent]:
                del cache[i]
        self.prefix = '%s%d_' % (self.base, k)
        self.inputs = dict(GEOMETRY, **{'L.' + c: '%s.%s' % (direction, c) for c in 'xyz'},
                           **{'LIGHT.' + c: '%s.%s' % (colour, c) for c in 'rgb'})


# The review rig's lights for a forward program's own key-light term: direction to the light in the game's world frame and
# its colour in the game's convention (diffuse = albedo x N.L x LIGHT): the rig colour / pi.
FORWARD_LIGHTS = (('rcraKeyL', 'normalize(rigKey() * vNativeToView)', 'rcraKeyC', 'keyColour() / PI'),
                  ('rcraFillL', 'normalize(rigFill() * vNativeToView)', 'rcraFillC', 'fillColour() / PI'),
                  ('rcraPreviewL', 'normalize(rcraPreviewDir * vNativeToView)', 'rcraPreviewC',
                   'uPreviewLightColour * rcraPreviewW / PI'))


def forward_light_lines(recipe, layouts):
    """Statements adding the forward program's own key-light term for each review light to rcraOwn: colour(L, LIGHT) -
    colour, the forward colour rcraColour having LIGHT = 0."""
    emitter = _PerLightEmitter(recipe, layouts, 'k')
    lines = ['    vec3 rcraOwn = vec3(0.0);',
             '    vec3 rcraPreviewDir; float rcraPreviewW = uLightingEnabled ? previewLightWeight(rcraPreviewDir) : 0.0;',
             '    if (rcraPreviewW <= 0.0) rcraPreviewDir = vec3(0.0, 0.0, 1.0);']
    for k, (direction, d_expr, colour, c_expr) in enumerate(FORWARD_LIGHTS):
        lines.append('    vec3 %s = %s; vec3 %s = %s;' % (direction, d_expr, colour, c_expr))
        emitter.light(k, direction, colour)
        start = len(emitter.lines)
        t = emitter.term
        value = 'vec3(%s, %s, %s)' % (t('colour.r'), t('colour.g'), t('colour.b'))
        lines += emitter.lines[start:]
        lines.append('    rcraOwn += %s - rcraColour;' % value)
    return lines


def _extra_arguments(emitter):
    """The vec4 and float of a surface's extra fields (terms extra.0 to extra.4) for rcraClassLit."""
    t = emitter.term
    return 'vec4(%s, %s, %s, %s)' % (t('extra.0'), t('extra.1'), t('extra.2'), t('extra.3')), t('extra.4')


@functools.lru_cache(maxsize=1)
def vertex_source():
    """The vertex program of the translated materials: shading.VERTEX, plus the vertex colour squared into COLOUR_VARYING
    (the engine squares its colour load in its vertex program, so the square is interpolated, not the colour)."""
    from rc2016.viewer.shading import VERTEX
    head, body = VERTEX.split('void main() {', 1)
    return (head + 'attribute vec4 aRcraColour; varying vec4 %s;   // Rift Apart vertex colour (rcra_glsl)\n'
            'void main() {\n    %s = aRcraColour * aRcraColour;' % (COLOUR_VARYING, COLOUR_VARYING) + body)


SHELL_VERTEX = '''
    { // Fur shell layer uFurLayer (RiftApart.exe VS_ModelFurShellGBufferDeferred, rc2016.games.rcra.fur)
        vec4 rcraEye = gl_ModelViewMatrix * p;
        float rcraT = uFurLayer * uFurInverse;
        float rcraFacing = dot(normalize(gl_NormalMatrix * n), normalize(-rcraEye.xyz));
        float rcraS = 1.0 - clamp(rcraFacing - 0.0065 * abs(aTangent.w) - 0.1, 0.0, 1.0);
        rcraS = min(rcraS * rcraS + 0.1, 1.0);
        float rcraF = rcraT / rcraS, rcraTp = clamp(rcraF, 0.0, 1.0);
        float rcraControl = texture2DLod(%s, gl_MultiTexCoord0.xy, 0.0).b;
        p.xyz += normalize(n) * (rcraControl * rcraTp * uFurLength);
        vFurShell = vec4(32.0 * rcraTp, (rcraTp * rcraTp + 0.4 * rcraTp) * uFurLength * uFurOffsetScale,
                         uFurLayer == 0.0 ? 1.0 : 0.0, rcraTp < rcraF ? 1.0 : 0.0);
    }
'''


@functools.lru_cache(maxsize=1)
def shell_vertex_source():
    """The fur shell vertex program: vertex_source() with the game's shell stage after skinning. Layer uFurLayer of
    1 / uFurInverse sits at t = k / L, spread by the view as the game does (t' = sat(t / s), the layer culled where t / s
    passes 1), moved along the skinned normal by the control map's blue x t' x uFurLength (metres, as the payload), and
    passes the writer its TEXCOORD.z, TEXCOORD2.w and TEXCOORD3.w in vFurShell."""
    head, body = vertex_source().split('    vTangent = gl_NormalMatrix * tn;', 1)
    declarations = ('uniform float %s; uniform sampler2D %s; varying vec4 vFurShell;   // fur shells (rcra_glsl)\n'
                    % (', '.join(SHELL_UNIFORMS), uniform(SHELL_CONTROL)))
    head = head.replace('void main() {', declarations + 'void main() {', 1)
    return head + SHELL_VERTEX % uniform(SHELL_CONTROL) + '    vTangent = gl_NormalMatrix * tn;' + body


def _prefix():
    from rc2016.viewer.shading import FRAGMENT
    prefix = FRAGMENT.split('void main() {', 1)[0]
    extra = []
    for declaration in ('varying vec2 vUV1;', 'varying mat3 vNativeToView;', 'varying vec4 %s;' % COLOUR_VARYING):
        if declaration.split()[-1].rstrip(';') not in prefix:
            extra.append(declaration)
    return prefix + '\n'.join(extra) + '\n'


def _frame_lines():
    return ['    vec3 rcraTangent = nativeVertexTangent();',
            '    float rcraHandedness = dot(vTintWorldTan, vTintWorldTan) > 0.0 ? vTangentSign'
            ' : nativeFrameHandedness(rcraTangent);',
            '    vec3 rcraCamera = vTintWorldPos - vTintViewOffset;']


def _surface(emitter, discard=True):
    """Lines computing the G-buffer surface terms into rcraAlbedo, rcraF0, rcraGloss, rcraNormal (view), rcraOcclusion,
    rcraEmission, rcraUnlit (a forward technique's unlit colour, surface.forward_albedo; 0 without one) and the discard."""
    t = emitter.term
    lines = emitter.lines
    albedo = 'vec3(%s, %s, %s)' % (t('albedo.r'), t('albedo.g'), t('albedo.b'))
    unlit = 'vec3(%s, %s, %s)' % (t('unlit.r'), t('unlit.g'), t('unlit.b'))
    f0 = 'vec3(%s, %s, %s)' % (t('f0.r'), t('f0.g'), t('f0.b'))
    normal = 'vec3(%s, %s, %s)' % (t('normal.x'), t('normal.y'), t('normal.z'))
    gloss, occlusion, emission, clip = t('gloss'), t('occlusion', '1.0'), t('emission'), t('clip')
    lines += (['    if (%s != 0.0) discard;' % clip] if discard else []) + [
              '    vec3 rcraAlbedo = clamp(%s, 0.0, 1.0); vec3 rcraF0 = %s; float rcraGloss = %s;' % (albedo, f0, gloss),
              '    float rcraOcclusion = %s; float rcraEmission = %s;' % (occlusion, emission),
              '    vec3 rcraUnlit = %s;' % unlit,
              '    vec3 rcraWorldN = %s;' % normal,
              '    vec3 rcraNormal = dot(rcraWorldN, rcraWorldN) > 0.0 ? normalize(vNativeToView * rcraWorldN) : N;']


def fragment_source(receipt):
    """The fragment program of a receipt: {'source', 'samplers' [uniform names], 'key'}."""
    layouts = receipt['layouts']
    body = _frame_lines()
    scene = bool(receipt.get('forward')) and reads_scene(receipt['forward'])
    if scene:
        body.append('    vec4 rcraScene = texture2D(uSceneColour, gl_FragCoord.xy / uSceneDepthInfo.xy);')
    surface = forward = coverage = None
    blend = receipt.get('blend', 'opaque')
    if receipt.get('surface'):
        surface = _Emitter(receipt['surface'], layouts, 'g')
        _surface(surface, discard=not receipt.get('forward'))
        if blend == SHELL_BLEND:   # emitted here so its declarations join surface.lines
            coverage = surface.term('coverage')
        body += surface.lines
    if receipt.get('forward'):
        forward = _Emitter(receipt['forward'], layouts, 'f')
        t = forward.term
        forward.lines += ['    if (%s != 0.0) discard;' % t('clip'),
                          '    vec3 rcraColour = vec3(%s, %s, %s);' % (t('colour.r'), t('colour.g'), t('colour.b')),
                          '    vec3 rcraBackground = vec3(%s, %s, %s);' % (t('background.r'), t('background.g'),
                                                                          t('background.b')),
                          '    float rcraAlpha = clamp(%s, 0.0, 1.0);' % t('alpha')]
        if 'lit_background.r' in receipt['forward']['terms']:   # the scene behind lit as albedo (surface.forward)
            lit_tint = 'vec3(%s, %s, %s)' % (t('lit_background.r'), t('lit_background.g'), t('lit_background.b'))
            forward.lines.append('    rcraBackground += %s * rcraDiffuse(%s, vec3(1.0), 1.0);'
                                 % (lit_tint, 'rcraNormal' if surface else 'N'))
        body += forward.lines
        own = receipt['forward'].get('lit') if blend in ('blended', 'additive') else None
        if own:   # the program's own key-light term per review light (surface.forward_lighting)
            body += forward_light_lines(own, layouts)
    if blend == SHELL_BLEND:   # the layer is culled past the strand (vFurShell.w), its coverage is its alpha
        body.insert(0, '    if (vFurShell.w > 0.5) discard;')
    lighting = receipt.get('lighting') if surface else None
    if lighting:
        start = len(surface.lines)
        extra, extra4 = _extra_arguments(surface)
        body += surface.lines[start:]   # the extra terms' statements
        lit = ('rcraClassLit(rcraNormal, V, rcraAlbedo, rcraF0, rcraGloss, rcraOcclusion, rcraWorldN, %s, %s)'
               % (extra, extra4))
    else:
        lit = ('rcraDiffuse(rcraNormal, rcraAlbedo, rcraOcclusion) + rcraSpecular(rcraNormal, V, rcraF0, rcraGloss)'
               if surface else 'vec3(0.0)')
    if forward is not None and receipt['forward'].get('lit') and blend in ('blended', 'additive'):
        # the program lights itself: its own term per light, plus the surface's ambient light (the program's is 0)
        lit = ('rcraOwn + (uLightingEnabled ? rcraAmbient(rcraNormal, V, rcraAlbedo, rcraF0, rcraGloss, rcraOcclusion) '
               ': vec3(0.0))' if surface else 'rcraOwn')
    if blend == 'opaque':
        if surface is None:
            raise ValueError('An opaque receipt needs a surface recipe')
        own = receipt['surface'].get('own_specular')
        if own:   # a forward technique's own specular term per review light (surface.forward_specular), 0 at LIGHT = 0
            body += ['    vec3 rcraColour = vec3(0.0);'] + forward_light_lines(own, layouts)
            lit = '(%s) + (uLightingEnabled ? rcraOwn : vec3(0.0))' % lit
        body.append('    sceneOutput(vec4(displayFog(%s + rcraAlbedo * rcraEmission + rcraUnlit), 1.0));' % lit)
    elif blend == SHELL_BLEND:
        body.append('    sceneOutput(vec4(displayFog(%s + rcraAlbedo * rcraEmission), clamp(%s, 0.0, 1.0)));'
                    % (lit, coverage))
    elif blend == 'modulate':
        body.append('    sceneOutput(vec4(mix(vec3(1.0), rcraColour, rcraAlpha), 1.0));')
    else:
        factor = ('mix(vec3(1.0), rcraBackground, rcraAlpha)' if blend == 'blended'
                  else 'vec3(1.0) + rcraAlpha * rcraBackground')
        colour = ('(%s + rcraColour) * rcraAlpha' % lit if blend == 'blended' or 'rcraOwn' in lit
                  else 'rcraColour * rcraAlpha')
        body.append('    if (%s == 0) { sceneOutput(vec4(%s, 1.0)); return; }' % (PASS_UNIFORM, factor))
        body.append('    sceneOutput(vec4(display(%s), rcraAlpha));' % colour)
    declarations = '\n'.join('uniform %s %s;' % ('samplerCube' if layouts[b]['kind'] == 'cube' else 'sampler2D', uniform(b))
                             for b in sorted(layouts))
    declarations += '\nuniform int %s;\n' % PASS_UNIFORM
    if blend == SHELL_BLEND:
        declarations += 'varying vec4 vFurShell;   // shell_vertex_source: 32 t\', offset term, base flag, culled\n'
    if scene:
        declarations += 'uniform sampler2D uSceneColour; uniform vec4 uSceneDepthInfo;   // frame copy, viewport size\n'
    light = light_source(lighting, receipt.get('ambient')) if lighting else ''
    if any('rcraInf' in line or 'rcraNan' in line for line in body) or 'rcraInf' in light or 'rcraNan' in light:
        declarations += NON_FINITE_DECLARATION
    if any('rcraInf' in line or 'rcraNan' in line for line in body):
        body.insert(0, NON_FINITE_LINE)
    from rc2016.viewer.native_glsl import FRAME_HELPERS, HELPERS as NATIVE_HELPERS
    source = (_prefix() + declarations + NATIVE_HELPERS + FRAME_HELPERS + HELPERS + light + '\nvoid main() {\n'
              '    vec3 N = normalize(vNrm) * (gl_FrontFacing ? 1.0 : -1.0); sceneCaptureNormal = N;\n'
              '    vec3 V = normalize(-vPos);\n' + '\n'.join(body) + '\n}\n')
    key = (EMITTER_VERSION, receipt['translation_sha256'], hashlib.sha256(source.encode()).hexdigest())
    return dict(source=source, samplers=[uniform(b) for b in sorted(layouts)], key=key)


def reads_scene(recipe):
    """True when a forward recipe samples the scene behind (inputs BACKGROUND.*; rc2016.games.rcra.surface.reads_scene)."""
    return any(n['op'] == 'input' and n.get('name', '').startswith('BACKGROUND.') for n in recipe['nodes'])


def receipt_sha256(surface, forward, blend, layouts, lighting=None, ambient=None):
    payload = dict(version=EMITTER_VERSION, surface=surface, forward=forward, blend=blend, layouts=layouts)
    if lighting:
        payload['lighting'] = lighting
    if ambient:
        payload['ambient'] = ambient
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
