"""Rift Apart's bloom, decoded from RiftApart.exe, for the model viewer's "Game (Rift Apart)" tonemap.

Programs (SM5, named by the exe's sorted program-name table, anchored on PS_BloomDownsampleLatLongThresholded and
PS_ToneMapApplyBasic): PS_BloomDownsampleThresholded (exe offset 65846156), PS_BloomDownsample (65820300) and
PS_BloomUpsample (65855228). The pass function (0x1411da720) fills their constants from the atmosphere's BloomSettings
{Intensity, Threshold, GaussianStrength0..4}, and runs over the targets kBloom0..kBloom6, at 1/2 to 1/128 of the screen
(the render-target table's size shifts -1 to -7):

1. B1 = down13(B0) x sat(luma x (1 / L) / max(Threshold, 1/16) - 1) / (G0 + ... + G4), L the adapted luminance
   (g_AdaptedLumBuffer[1] = 1 / L). down13 is the 13-tap downsample at +-1 and +-2 source texels (weights 1/8 for the centre
   and the four +-1 diagonals, 1/16 for the four +-2 edges, 1/32 for the four +-2 corners); the gate reads the luma of the
   filtered colour.
2. Bk = down13(Bk-1) for k = 2 to 6.
3. Up: tent(Bk) is the 3 x 3 tent at +-1.5 source texels (1/4 centre, 1/8 edges, 1/16 corners), scaled by the pass's colour
   scale and drawn with blend state 6 (colour = src + srcAlpha x dst) at the pass's alpha:
       B5 = G4 tent(B6) + G3 B5,  B4 = tent(B5) + G2 B4,  B3 = tent(B4) + G1 B3,  B2 = tent(B3) + G0 B2,
       B1 = tent(B2) (alpha 0),  B0 = Intensity x tent(B1) (blend state 0, overwrite).
   So the five strengths weight the 1/8 to 1/128 levels and the 1/4 bright-pass level adds nothing of its own.
4. CS_ToneMap (55136892) adds B0 to the scene radiance (m_DoBloom) before the exposure and the ACES lookup.

Viewer values: the scene is the model alone; B0 is a 2 x 2 box half-resolution copy of the float scene target (how the game
fills kBloom0 is not traced; four taps at texel centres, so the target's nearest filtering does not matter); 1 / L is the viewer exposure / 0.18 (the display exposure is m_AcesMiddleGray / L); the
settings are those of required/atmospheres/default_atmosphere.atmosphere (DEFAULT_SETTINGS), since a model has no atmosphere.
Level sizes halve with a floor; the programs' uv scale for odd sizes (2 x dst / src) is taken as 1.
"""
from __future__ import annotations

import numpy as np

LEVELS = 6                     # downsamples: kBloom0 (half resolution) to kBloom6 (1/128)
THRESHOLD_FLOOR = 0.0625       # 1 / max(Threshold, 0.0625)
MIDDLE_GRAY = 0.18             # m_AcesMiddleGray: the display exposure is 0.18 / L
LUMA = (0.2126, 0.7152, 0.0722)
# BloomSettings of required/atmospheres/default_atmosphere.atmosphere (built block offsets 656 to 680): Intensity, Threshold,
# GaussianStrength0..4.
DEFAULT_SETTINGS = dict(intensity=0.1, threshold=3.0, strengths=(0.4, 0.5, 0.6, 0.8, 1.0))
BLOCK_OFFSETS = dict(intensity=656, threshold=660, strengths=664)   # in the atmosphere's 0x02F06D4E section
HALF_TAPS = (((-0.5, -0.5), 0.25), ((0.5, -0.5), 0.25), ((-0.5, 0.5), 0.25), ((0.5, 0.5), 0.25))   # 2 x 2 box
DOWN_TAPS = (((0, 0), 0.125), ((-1, -1), 0.125), ((1, -1), 0.125), ((-1, 1), 0.125), ((1, 1), 0.125),
             ((0, -2), 0.0625), ((-2, 0), 0.0625), ((2, 0), 0.0625), ((0, 2), 0.0625),
             ((-2, -2), 0.03125), ((2, -2), 0.03125), ((-2, 2), 0.03125), ((2, 2), 0.03125))
UP_TAPS = (((0, 0), 0.25), ((-1.5, 0), 0.125), ((1.5, 0), 0.125), ((0, -1.5), 0.125), ((0, 1.5), 0.125),
           ((-1.5, -1.5), 0.0625), ((1.5, -1.5), 0.0625), ((-1.5, 1.5), 0.0625), ((1.5, 1.5), 0.0625))


def settings_from_block(block):
    """BloomSettings from an atmosphere's built 0x02F06D4E section (bytes)."""
    import struct
    f = lambda offset: struct.unpack_from('<f', block, offset)[0]  # noqa: E731
    return dict(intensity=f(BLOCK_OFFSETS['intensity']), threshold=f(BLOCK_OFFSETS['threshold']),
                strengths=tuple(f(BLOCK_OFFSETS['strengths'] + 4 * i) for i in range(5)))


def level_sizes(width, height):
    """Sizes of B0..B6 for a (width, height) frame: 1/2 to 1/128, halved with a floor, at least 1."""
    return [(max(1, width >> k), max(1, height >> k)) for k in range(1, LEVELS + 2)]


def up_constants(level, settings):
    """(colour scale, alpha, blend) of the up pass that reads B`level` and writes B`level - 1` (level 6 to 1)."""
    g, intensity = settings['strengths'], settings['intensity']
    if level == 1:
        return intensity, 0.0, None
    scale = g[4] if level == LEVELS else 1.0
    alpha = g[level - 3] if level >= 3 else 0.0   # settings word level - 1: level 6: G3, 5: G2, 4: G1, 3: G0; level 2: 0
    return scale, alpha, 'one_src_alpha'


def passes(width, height, inverse_luminance, settings=None):
    """The pass list for one frame size as dicts: kind (rcra_half, rcra_down, rcra_up), src and dst image names (scene,
    B0..B6), dst size, the source texel size (1 / source size; the first downsample's is 0.5 / destination size), and the
    gate, scale, alpha and blend. B0 holds the bloom texture at the end."""
    settings = settings or DEFAULT_SETTINGS
    sizes = level_sizes(width, height)
    gate = dict(inverse_luminance=float(inverse_luminance),
                threshold_scale=1.0 / max(float(settings['threshold']), THRESHOLD_FLOOR),
                normalise=1.0 / max(float(sum(settings['strengths'])), 1e-30))
    out = [dict(kind='rcra_half', src='scene', dst='B0', size=sizes[0], texel=(1.0 / width, 1.0 / height))]
    for k in range(1, LEVELS + 1):
        src, dst = sizes[k - 1], sizes[k]
        texel = (0.5 / dst[0], 0.5 / dst[1]) if k == 1 else (1.0 / src[0], 1.0 / src[1])
        out.append(dict(kind='rcra_down', src='B%d' % (k - 1), dst='B%d' % k, size=dst, texel=texel,
                        gate=gate if k == 1 else None))
    for k in range(LEVELS, 0, -1):
        scale, alpha, blend = up_constants(k, settings)
        src = sizes[k]
        out.append(dict(kind='rcra_up', src='B%d' % k, dst='B%d' % (k - 1), size=sizes[k - 1],
                        texel=(1.0 / src[0], 1.0 / src[1]), scale=scale, alpha=alpha, blend=blend))
    return out


# CPU reference (numpy): the same arithmetic as the fragment programs below, for the tests. -----------------------------

def _bilinear(image, u, v):
    """Clamp-to-edge bilinear at normalised coordinates."""
    h, w = image.shape[:2]
    x = np.clip(u * w - 0.5, 0, w - 1)
    y = np.clip(v * h - 0.5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    return (image[y0, x0] * (1 - fx) + image[y0, x1] * fx) * (1 - fy) + (image[y1, x0] * (1 - fx) + image[y1, x1] * fx) * fy


def _grid(size):
    ys, xs = np.mgrid[0:size[1], 0:size[0]]
    return (xs + 0.5) / size[0], (ys + 0.5) / size[1]


def _filter(image, size, texel, taps):
    u, v = _grid(size)
    return sum(w * _bilinear(image, u + dx * texel[0], v + dy * texel[1]) for (dx, dy), w in taps)


def reference_chain(scene, inverse_luminance, settings=None):
    """The bloom texture (B0, half resolution) of a (h, w, 3) scene-linear frame."""
    height, width = scene.shape[:2]
    images = {'scene': np.asarray(scene, np.float64)}
    for p in passes(width, height, inverse_luminance, settings):
        src = images[p['src']]
        if p['kind'] == 'rcra_half':
            out = _filter(src, p['size'], p['texel'], HALF_TAPS)
        elif p['kind'] == 'rcra_down':
            out = _filter(src, p['size'], p['texel'], DOWN_TAPS)
            if p['gate']:
                g = p['gate']
                luma = out @ np.array(LUMA)
                out = out * (np.clip(luma * g['inverse_luminance'] * g['threshold_scale'] - 1.0, 0.0, 1.0)
                             * g['normalise'])[..., None]
        else:
            out = p['scale'] * _filter(src, p['size'], p['texel'], UP_TAPS)
            if p['blend']:
                out = out + p['alpha'] * images[p['dst']]
        images[p['dst']] = out
    return images['B0']


# Fragment programs (full-screen quad over the destination; uSize the destination size, uTexel the source texel step). -------

VERTEX = """
#version 120
void main() { gl_Position = gl_Vertex; }
"""


def _taps_glsl(taps):
    return ' + '.join('%r * texture2D(uSrc, uv + vec2(%r, %r) * uTexel).rgb' % (w, float(dx), float(dy)) for (dx, dy), w in taps)


HALF_FRAGMENT = """
#version 120
uniform sampler2D uSrc; uniform vec2 uSize; uniform vec2 uTexel;
void main() {
    vec2 uv = gl_FragCoord.xy / uSize;
    gl_FragColor = vec4(%(taps)s, 1.0);
}
""" % dict(taps=_taps_glsl(HALF_TAPS))

DOWN_FRAGMENT = """
#version 120
uniform sampler2D uSrc; uniform vec2 uSize; uniform vec2 uTexel;
uniform bool uGate; uniform float uInvLum, uThresholdScale, uNormalise;
void main() {
    vec2 uv = gl_FragCoord.xy / uSize;
    vec3 c = %(taps)s;
    if (uGate) c *= clamp(dot(c, vec3(%(luma)s)) * uInvLum * uThresholdScale - 1.0, 0.0, 1.0) * uNormalise;
    gl_FragColor = vec4(c, 1.0);
}
""" % dict(taps=_taps_glsl(DOWN_TAPS), luma=', '.join(repr(x) for x in LUMA))

UP_FRAGMENT = """
#version 120
uniform sampler2D uSrc; uniform vec2 uSize; uniform vec2 uTexel; uniform float uScale, uAlpha;
void main() {
    vec2 uv = gl_FragCoord.xy / uSize;
    gl_FragColor = vec4(uScale * (%(taps)s), uAlpha);
}
""" % dict(taps=_taps_glsl(UP_TAPS))

PROGRAMS = {
    'rcra_half': (HALF_FRAGMENT, ('uSrc', 'uSize', 'uTexel')),
    'rcra_down': (DOWN_FRAGMENT, ('uSrc', 'uSize', 'uTexel', 'uGate', 'uInvLum', 'uThresholdScale', 'uNormalise')),
    'rcra_up': (UP_FRAGMENT, ('uSrc', 'uSize', 'uTexel', 'uScale', 'uAlpha')),
}
