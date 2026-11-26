"""Rift Apart lighting classes 1, 2 and 5 as Cycles closures.

The closed forms of the light terms that the engine's class apply programs compute (as their translations,
nwo.games.rcra.lighting.direct, read), and the Cycles inputs that stand for them in the Blender export
(nwo.export.rcra_blender). Plain Python with no package imports, so it loads inside Blender as well.

Conventions: game space (Y up); c = N.L; light terms in the game's units (no 1/pi: diffuse = albedo x N.L x LIGHT, and a
Blender light's strength is pi x the game's colour). r(gloss) = 0.9725 - slope x gloss, GGX alpha = r^4.

Class 1 (subsurface: skin, heads, nymphs, dinosaurs), under one unshadowed light:
- diffuse: Lambert, which the skin pass then blurs in screen space with Burley's normalised diffusion,
  0.125 / d (e^(-r/d) + e^(-r/3d)), d = the skin profile colour (linear) x 0.005 x (0.6 o + 0.4) x (2^f - 1) / 7 in world
  units, f the extra word's bits 15 to 17 and o the G-buffer occlusion (``skin_blur_distance``). Its seven taps sit at 0 to
  about 0.32 of that kernel unit, a few profile distances. Cycles' Christensen-Burley subsurface has the same profile with
  d = radius x scale / (4 pi), whatever the albedo (measured in Blender 5.0: a shadow edge on a plane, fitted d within 1 %
  at albedos 0.2, 0.5 and 0.9), so ``burley_radius`` gives the radius whose profile is the game's;
- specular: (1 - m) GGX(alpha) + m GGX(alpha^1.66), m = sat(0.15 - 0.03 x view depth): a sharper lobe close to the camera;
- plus a Henyey-Greenstein scatter (g = 0.65) on the second normal, weighted by 0.003 sqrt(scatter / 7) /
  max(N2.V - 0.05, 0.1), which no Cycles closure has (left out, labelled).

Class 2 (cloth):
- diffuse per channel: [wrap(c) + w Y back(c)] x min(|c| + w sqrt(albedo), 1) with wrap(c) = sat((c + w) / (1 + w)^2),
  back(c) = (1 - sat(c + 0.9)^2) sat((w - c) / (1 + w)^2), w = wrap field / 31 and Y the albedo's luminance
  (``cloth_diffuse``);
- specular on the second normal N2 (``cloth_normal``): mix(GGX, sheen, sheen field / 31), both with r = 0.9725 - 0.6011
  gloss, the sheen Charlie's (2 + 1/alpha) sin^(1/alpha)(theta_h) / 2 with Neubelt's visibility 1 / (4 (NL + NV - NL NV)),
  tinted by F0 (``charlie_sheen``);
- plus a scatter lobe (Henyey-Greenstein, g = 0.65) weighted by scatter field / 21000 / max(N2.V, 0.1), at most 0.03
  (left out, labelled).

Class 5 (foliage): diffuse wrap(c) + T back(c), w = wrap field / 127 and T = transmittance field / 127
(``foliage_diffuse``); the specular is class 0's.

Cycles has no closure with these diffuse shapes. ``hemisphere_weights`` projects a diffuse shape f(c) onto a Diffuse BSDF
(front) and a Translucent BSDF (back) with the weights 2 integral(f) over each hemisphere: the light the shape reflects from
a uniform sky over that side, which a Lambert lobe of that weight reflects too. Ambient light is then the class's; a single
light's falloff with c is Lambert's. The weights are exact integrals of the piecewise polynomial forms (``CLOTH_TABLE``,
``foliage_weights``).

The sheen is fitted: Cycles' Ashikhmin velvet (``velvet``, its bsdf_ashikhmin_velvet eval) at sigma(r) and a colour scale
k(r) x F0. sigma is the least-squares fit of the velvet's shape to the Charlie lobe over uniform light and view directions
(about 55 % relative L2 error: the two lobes differ in shape), and k keeps the lobe's mean reflectance over those directions.
Both are cubic fits over r in [0.3714, 0.9725] (sigma within 0.007, k within 2.2 %).
"""
import math

GGX_SLOPE = 0.7514          # r = 0.9725 - 0.7514 gloss: the deferred apply programs (classes 0, 1, 4, 5)
CLOTH_SLOPE = 0.6011        # class 2's r = 0.9725 - 0.6011 gloss
ROUGHNESS_BASE = 0.9725
SUBSURFACE, CLOTH, FOLIAGE = 1, 2, 5

# Class 1
NEAR_MIX = (0.15, -0.03)    # m = sat(0.15 - 0.03 x view depth)
NEAR_LOBE_POWER = 1.66      # the near lobe's alpha^2 is (alpha^2)^1.66
SKIN_KERNEL = 0.005         # world units per kernel unit at full scatter scale 1
SKIN_OCCLUSION = (0.6, 0.4)
BURLEY_RADIUS = 4.0 * math.pi   # Cycles' Christensen-Burley radius per unit of Burley's d (Blender 5.0, measured)
SCATTER_PHASE_G = 0.65

# Class 2
CLOTH_WRAP_STEPS = 32       # the 5-bit wrap field
SHEEN_STEPS = 31
SHEEN_SIGMA = (-2.2859289726409995, 4.712107213606554, -2.0500907911686506, 0.37980410887696975)   # cubic in r
SHEEN_SCALE = (-0.08843780612824684, 0.7857910927954392, -0.9133564099447461, 0.8253100656698396)  # cubic in 1 / r
LUMINANCE = (0.2125999927520752, 0.7152000069618225, 0.0722000002861023)
CLOTH_TABLE_K = 129         # samples of k = w sqrt(albedo) over [0, 1]

# Class 5
FOLIAGE_STEPS = 127         # the 7-bit wrap and transmittance fields


def sat(x):
    return min(max(x, 0.0), 1.0)


def roughness(gloss, slope=GGX_SLOPE):
    return ROUGHNESS_BASE - slope * gloss


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit(v):
    n = math.sqrt(_dot(v, v))
    return [x / n for x in v]


# ------------------------------------------------------------------------------------------------ class 1

def near_lobe_mix(depth):
    """The weight of class 1's near lobe at a view depth (world units)."""
    return sat(NEAR_MIX[0] + NEAR_MIX[1] * depth)


def near_lobe_roughness(r):
    """The Cycles roughness (sqrt of alpha) of class 1's near lobe for r: alpha = (r^4)^1.66."""
    return r ** (2.0 * NEAR_LOBE_POWER)


def skin_blur_distance(profile, field, occlusion):
    """Burley's d per channel (world units) of the skin pass for a skin profile colour (linear rgb), the blur field (extra
    bits 15 to 17) and the G-buffer occlusion."""
    scale = SKIN_KERNEL * (SKIN_OCCLUSION[0] * occlusion + SKIN_OCCLUSION[1]) * (2.0 ** field - 1.0) / 7.0
    return [p * scale for p in profile]


def burley_radius(d):
    """The Cycles Christensen-Burley radius (with scale 1) whose diffusion profile has distance d."""
    return d * BURLEY_RADIUS


# ------------------------------------------------------------------------------------------------ class 2

def cloth_normal(fx, fy):
    """Class 2's second normal (game space) from extra fields 0 and 1: 8-bit octahedral about the Y axis, x from field 0
    and z from field 1."""
    x, z = fx / 127.5 - 1.0, fy / 127.5 - 1.0
    y = 1.0 - abs(x) - abs(z)
    t = max(-y, 0.0)
    x += -t if x >= 0.0 else t
    z += -t if z >= 0.0 else t
    return _unit([x, y, z])


def wrap(c, w):
    return sat((c + w) / (1.0 + w) ** 2)


def back(c, w):
    return (1.0 - sat(c + 0.9) ** 2) * sat((w - c) / (1.0 + w) ** 2)


def cloth_diffuse(c, w, a, lum):
    """Class 2's diffuse coefficient of one channel (the light is albedo x this): w = wrap field / 31, a the channel's
    albedo, lum the albedo's luminance."""
    return (wrap(c, w) + w * lum * back(c, w)) * min(abs(c) + w * math.sqrt(a), 1.0)


def ggx(n, l, v, f0, alpha):
    """The deferred GGX term x N.L (game units): D without 1/pi, the Smith joint approximation, Schlick with F90 = 1."""
    h = _unit([a + b for a, b in zip(l, v)])
    nl, nv = sat(_dot(n, l)), min(abs(_dot(n, v)) + 1e-5, 1.0)
    nh, vh = sat(_dot(n, h)), sat(_dot(v, h))
    a2 = alpha * alpha
    d = nh * nh * (a2 - 1.0) + 1.0
    vis = 0.5 / (nl * (nv * (1.0 - alpha) + alpha) + nv * (nl * (1.0 - alpha) + alpha))
    return a2 / (d * d) * vis * (f0 + (1.0 - f0) * (1.0 - vh) ** 5) * nl


def charlie_sheen(n, l, v, f0, alpha):
    """Class 2's sheen x N.L (game units): Charlie's D (without 1/pi) with Neubelt's visibility, tinted by F0."""
    h = _unit([a + b for a, b in zip(l, v)])
    nl, nv, nh = sat(_dot(n, l)), min(abs(_dot(n, v)) + 1e-5, 1.0), sat(_dot(n, h))
    s2 = max(1.0 - nh * nh, 0.00390625)
    return (2.0 + 1.0 / alpha) * s2 ** (0.5 / alpha) * 0.5 / (4.0 * (nl + nv - nl * nv)) * f0 * nl


def cloth_specular(n2, l, v, f0, gloss, sheen):
    """Class 2's specular x N.L on the second normal: mix(GGX, Charlie sheen, sheen field / 31)."""
    alpha = roughness(gloss, CLOTH_SLOPE) ** 4
    return (1.0 - sheen) * ggx(n2, l, v, f0, alpha) + sheen * charlie_sheen(n2, l, v, f0, alpha)


def _poly(coefficients, x):
    out = 0.0
    for c in coefficients:
        out = out * x + c
    return out


def sheen_sigma(r):
    """Cycles' Ashikhmin velvet sigma fitted to class 2's Charlie sheen at r (cloth slope)."""
    return _poly(SHEEN_SIGMA, r)


def sheen_scale(r):
    """The colour scale (x F0) of that velvet: the Charlie lobe's mean reflectance over the fitted one's."""
    return _poly(SHEEN_SCALE, 1.0 / r)


def velvet(n, l, v, sigma):
    """Cycles' Ashikhmin velvet eval (f x N.L) in the game's units (x pi), view v and light l."""
    sigma = max(sigma, 0.01)
    inv = 1.0 / (sigma * sigma)
    nl, nv = _dot(n, l), _dot(n, v)
    if nl <= 0.0 or nv <= 0.0:
        return 0.0
    h = _unit([a + b for a, b in zip(l, v)])
    nh, hv = _dot(n, h), abs(_dot(v, h))
    if not (abs(nh) < 1.0 - 1e-5 and hv > 1e-5):
        return 0.0
    q = max(nh / hv, 1e-5)
    g = min(1.0, 2.0 * abs(q * nv), 2.0 * abs(q * nl))
    sin2 = 1.0 - nh * nh
    d = math.exp(-nh * nh / sin2 * inv) * inv / math.pi / (sin2 * sin2)
    return 0.25 * d * g / nv * math.pi


# ------------------------------------------------------------------------------------------------ class 5

def foliage_diffuse(c, w, t):
    """Class 5's diffuse coefficient (the light is albedo x this): w = wrap field / 127, t = transmittance field / 127."""
    return wrap(c, w) + t * back(c, w)


# ------------------------------------------------------------------------------------------------ projections

_GAUSS3 = ((-math.sqrt(0.6), 5.0 / 9.0), (0.0, 8.0 / 9.0), (math.sqrt(0.6), 5.0 / 9.0))


def hemisphere_weights(f, breaks=()):
    """(front, back): 2 x the integral of f(c) over [0, 1] and over [-1, 0], the weights of a Diffuse and a Translucent
    BSDF that reflect what f does under a uniform sky over each side. Exact for f polynomial of degree 5 or less between
    the points ``breaks`` (3-point Gauss-Legendre per piece)."""
    out = []
    for lo, hi in ((0.0, 1.0), (-1.0, 0.0)):
        points = sorted({lo, hi} | {b for b in breaks if lo < b < hi})
        total = 0.0
        for a, b in zip(points, points[1:]):
            mid, half = 0.5 * (a + b), 0.5 * (b - a)
            total += half * sum(wt * f(mid + half * x) for x, wt in _GAUSS3)
        out.append(2.0 * total)
    return tuple(out)


def foliage_weights(w):
    """(front wrap, front transmission, back wrap, back transmission) for w: the class-5 Diffuse weight is
    albedo x (front wrap + T front transmission), the Translucent one albedo x (back wrap + T back transmission)."""
    breaks = (0.1, -0.9, w, -w)
    f1, b1 = hemisphere_weights(lambda c: wrap(c, w), breaks)
    f2, b2 = hemisphere_weights(lambda c: back(c, w), breaks)
    return f1, f2, b1, b2


def cloth_weights(w, k):
    """(F1, F2, B1, B2) for w and k = w sqrt(albedo) of one channel: the class-2 Diffuse weight of that channel is
    albedo x (F1 + w Y F2), the Translucent one albedo x (B1 + w Y B2), Y the albedo's luminance."""
    breaks = (0.1, -0.9, w, -w, 1.0 - k, k - 1.0)
    m = lambda c: min(abs(c) + k, 1.0)   # noqa: E731
    f1, b1 = hemisphere_weights(lambda c: wrap(c, w) * m(c), breaks)
    f2, b2 = hemisphere_weights(lambda c: back(c, w) * m(c), breaks)
    return f1, f2, b1, b2


def cloth_table():
    """Rows w field 0 to 31, columns k = j / (CLOTH_TABLE_K - 1): cloth_weights at each, for a bilinear lookup in k (exact
    in w, which the lookup reads at texel centres)."""
    return [[cloth_weights(f / float(CLOTH_WRAP_STEPS - 1), j / float(CLOTH_TABLE_K - 1)) for j in range(CLOTH_TABLE_K)]
            for f in range(CLOTH_WRAP_STEPS)]


def foliage_table():
    """foliage_weights at each wrap field 0 to 127."""
    return [foliage_weights(f / float(FOLIAGE_STEPS)) for f in range(FOLIAGE_STEPS + 1)]
