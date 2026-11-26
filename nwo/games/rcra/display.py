"""Rift Apart's display transform, ported from the game's compute programs in RiftApart.exe.

- ``CS_ToneMapCreateLookup`` (exe offset 55161948) builds a 32^3 lookup over linear Rec.709 at
  2^(t * 16/31 - 10.5), t = 0..31 per axis, through ACES 1.0: Rec.709 -> XYZ -> AP0 (no D65 -> D60 adaptation), glow,
  red modifier, AP1, RRT saturation 0.96, segmented spline C5, ODT spline C9 at 48 nits, dim surround 0.9811, ODT
  saturation 0.93, D60 -> D65 only when ``m_A.x`` is set, XYZ -> Rec.709 and the sRGB encode. For SDR output the CPU
  side (0x1411db740) writes ``m_A.x`` = 0 into the program's MiscCBuffer (m_A.z is set only by a debug view), so it skips the D60 -> D65
  step (greys stay neutral, white is the ACES D60 white). Its CC variants then mix up to four colour-correction LUTs
  on the encoded value, weighted by ``m_B``: the active atmospheres' LUTs, each weighted by how far its atmosphere is
  blended in (``colour_corrected``); one fully active atmosphere gives its LUT weight 1.
- ``CS_ToneMap`` (55136892) multiplies the scene by m_AcesMiddleGray / adapted luminance and samples that lookup at
  (log2(x) / 16 + 0.65625) * 31/32 + 1/64, trilinear.

``aces`` matches the lookup program, translated by ``nwo.shaders.sm5`` and evaluated at random thread ids, to about 1e-5.
The viewer samples ``lookup_texture`` in ``shading.LIGHTING`` (``riftApartDisplay``, uTonemap 3). Bloom, lens flares,
vignette, film grain and auto-exposure are not part of it.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

LOOKUP_SIZE = 32
# The display() coordinate of the lookup: c = log2(x) * LOG2_SCALE + LOG2_OFFSET in [0, 1] spans the lookup's 32 entries.
LOG2_SCALE, LOG2_OFFSET = 0.0625, 0.65625

ICB = np.array([
    [-4.0, -0.718548, -1.69897, 0.515439], [-4.0, 2.081031, -1.69897, 0.847044], [-3.157377, 3.668124, -1.4779, 1.1358],
    [-0.48525, 4.0, -1.2291, 1.3802], [1.847732, 4.0, -0.8648, 1.5197], [1.847732, 4.0, -0.448, 1.5985],
    [0.0, 0.0, 0.00518, 1.6467], [0.0, 0.0, 0.451108, 1.674609], [0.0, 0.0, 0.911374, 1.687873],
    [0.0, 0.0, 0.911374, 1.687873]], np.float64)   # the programs' spline coefficients: C5 low, high; C9 low, high
REC709_TO_XYZ = np.array([[0.412391, 0.357584, 0.180481], [0.212639, 0.715169, 0.0721923], [0.0193308, 0.119195, 0.950532]])
AP0_TO_AP1 = np.array([[1.45144, -0.236511, -0.214929], [-0.0765538, 1.17623, -0.0996759], [0.00831615, -0.00603245, 0.997716]])
RRT_SAT = np.array([[0.970889, 0.0269633, 0.00214758], [0.0108892, 0.986963, 0.00214758], [0.0108892, 0.0269633, 0.962148]])
AP1_TO_XYZ = np.array([[0.662454, 0.134004, 0.156188], [0.272229, 0.674082, 0.0536895], [-0.00557466, 0.00406073, 1.01034]])
XYZ_TO_AP1 = np.array([[1.64102, -0.324803, -0.236425], [-0.663663, 1.61533, 0.0167564], [0.0117219, -0.00828444, 0.988395]])
ODT_SAT = np.array([[0.949056, 0.0471857, 0.00375827], [0.019056, 0.977186, 0.00375827], [0.019056, 0.0471857, 0.933758]])
D60_TO_D65 = np.array([[0.987224, -0.00611327, 0.0159533], [-0.00759836, 1.00186, 0.00533002], [0.00307257, -0.00509595, 1.08168]])
XYZ_TO_REC709 = np.array([[3.24097, -1.53738, -0.498611], [-0.969243, 1.87597, 0.0415551], [0.0556301, -0.203977, 1.05697]])
LOG10_2 = 0.30103


def _spline(x, low, high, col_low, col_high, k_low, b_low, k_high, b_high, top):
    """The programs' segmented-spline pattern in log10 (C5: ``top`` 4.0 above ``high``; C9's ends are the caller's)."""
    l2 = np.log2(x)
    out = np.full_like(x, top)
    for mask, col, b, k in ((l2 < low, col_low, b_low, k_low), ((l2 >= low) & (l2 < high), col_high, b_high, k_high)):
        if not mask.any():
            continue
        kc = (l2[mask] * LOG10_2 + b) * k
        j = np.floor(kc).astype(int)
        t = kc - j
        c0, c1, c2 = ICB[j, col], ICB[j + 1, col], ICB[j + 2, col]
        out[mask] = t * t * (0.5 * c0 + 0.5 * c2 - c1) + t * (c1 - c0) + 0.5 * (c0 + c1)
    return out


def c5(x):
    """RRT segmented spline C5."""
    x = np.maximum(x, 3.05176e-05)
    return 10.0 ** _spline(x, -2.47393, 15.5261, 0, 1, 0.664386, 5.26018, 0.553655, 0.744727, 4.0)


def c9(x):
    """ODT segmented spline C9 (48 nits)."""
    x = np.maximum(x, 3.05176e-05)
    l2 = np.log2(x)
    y = _spline(x, 2.26303, 9.97401, 2, 3, 2.17266, 2.54062, 3.01564, -0.681241, 0.0)
    y = np.where(l2 <= -8.43977, -1.69897, y)
    y = np.where(l2 >= 9.97401, l2 * 0.0120412 + 1.56114, y)
    return 10.0 ** y


def aces(rgb, cat=False):
    """Encoded sRGB display value of linear Rec.709 ``rgb`` (..., 3), as the lookup program computes one entry.
    ``cat``: the D60 -> D65 adaptation (the program's m_A.x), off in the game's SDR display."""
    rgb = np.asarray(rgb, np.float64)
    xyz = rgb @ REC709_TO_XYZ.T
    R = 1.04981 * xyz[..., 0] - 9.7485e-05 * xyz[..., 2]   # XYZ -> AP0
    G = xyz @ np.array([-0.495903, 1.37331, 0.09824])
    B = 0.991252 * xyz[..., 2]
    mx, mn = np.maximum(np.maximum(R, G), B), np.minimum(np.minimum(R, G), B)
    sat = (np.maximum(mx, 1e-10) - np.maximum(mn, 1e-10)) / np.maximum(mx, 0.01)
    chroma = np.sqrt(np.maximum(B * (B - G) + G * (G - R) + R * (R - B), 0.0))
    yc = (B + G + R + 1.75 * chroma) / 3.0
    s = sat - 0.4
    tt = np.maximum(1 - np.abs(s * 2.5), 0)
    sigm = (np.sign(s) * (1 - tt * tt) + 1) * 0.025
    # glow: the program tests 3 yc against 0.16 and 0.48 (ACES: yc <= 2/3 glowMid, yc >= 2 glowMid, glowMid 0.08)
    glow = np.where(3 * yc <= 0.16, sigm, np.where(3 * yc >= 0.48, 0.0, sigm * (0.08 / np.maximum(yc, 1e-10) - 0.5))) + 1
    R, G, B = R * glow, G * glow, B * glow
    hue = np.degrees(np.arctan2(np.sqrt(3) * (G - B), 2 * R - G - B))   # red modifier
    hue = np.where((G == R) & (B == G), 0.0, hue)
    hue = np.where(hue < 0, hue + 360, hue)
    hue = np.where(hue > 180, hue - 360, np.where(hue < -180, hue + 360, hue))
    inside = (hue > -67.5) & (hue < 67.5)
    kc = (hue + 67.5) * 0.0296296
    j = np.floor(kc)
    t = kc - j
    t2, t3 = t * t, t * t * t
    w = np.select([j == 0, j == 1, j == 2, j == 3],
                  [t3 / 6, -t3 / 2 + t2 / 2 + t / 2 + 1 / 6, t3 / 2 - t2 + 2 / 3, -t3 / 6 + t2 / 2 - t / 2 + 1 / 6], 0.0)
    w = np.where(inside, w, 0.0)
    R = R + sat * w * 1.5 * (0.03 - R) * 0.18
    ap0 = np.clip(np.stack([R, G, B], -1), 0, 65535)
    ap1 = np.clip(ap0 @ AP0_TO_AP1.T, 0, 65535)
    pre = ap1 @ RRT_SAT.T
    post = np.stack([c5(pre[..., 0]), c5(pre[..., 1]), c5(pre[..., 2])], -1)
    odt = np.stack([c9(post[..., 0]), c9(post[..., 1]), c9(post[..., 2])], -1)
    lin = (odt - 0.02) * 0.020842   # cinema black 0.02, white 48 nits
    XYZ = lin @ AP1_TO_XYZ.T
    total = XYZ.sum(-1)
    total = np.where(total == 0, 1e-10, total)
    x, y = XYZ[..., 0] / total, XYZ[..., 1] / total
    Y = np.clip(XYZ[..., 1], 0, 65535) ** 0.9811   # dim surround
    yy = np.maximum(y, 1e-10)
    XYZ = np.stack([x * Y / yy, Y, (1 - x - y) * Y / yy], -1)
    XYZ = ((XYZ @ XYZ_TO_AP1.T) @ ODT_SAT.T) @ AP1_TO_XYZ.T
    if cat:
        XYZ = XYZ @ D60_TO_D65.T
    lin709 = np.clip(XYZ @ XYZ_TO_REC709.T, 0, 1)
    return np.where(lin709 >= 0.00303993, 1.055 * lin709 ** (1 / 2.4) - 0.055, lin709 * 12.9232)


def lookup_grid():
    """The scene-linear value of each lookup entry along one axis."""
    return 2.0 ** (np.arange(LOOKUP_SIZE) * (16.0 / 31.0) - 10.5)


@lru_cache(maxsize=1)
def build_lookup():
    """(32, 32, 32, 3) encoded display values indexed [r, g, b], read only."""
    g = lookup_grid()
    table = aces(np.stack(np.meshgrid(g, g, g, indexing='ij'), -1))
    table.setflags(write=False)
    return table


def sample_lut(lut, encoded):
    """A colour-correction LUT (n, n, n, 3) [r, g, b] sampled as the CC programs do: trilinear (clamped) at the
    normalised coordinate encoded * 31/32 + 1/64, whatever the LUT's size (the install has 32^3 and 33^3 LUTs)."""
    n = lut.shape[0]
    u = np.asarray(encoded, np.float64) * (31.0 / 32.0) + 1.0 / 64.0
    return sample(lut, np.clip((u * n - 0.5) / (n - 1), 0, 1))


def colour_corrected(luts, table=None):
    """The lookup with colour-correction LUTs mixed in, as CS_ToneMapCreateLookupCC1 to CC4: ``luts`` is
    [(lut, weight)], at most four, each lut (n, n, n, 3) [r, g, b] in [0, 1]. The program adds every LUT sample times its
    weight (m_B) and keeps the uncorrected value times sat(1 - sum of the weights)."""
    table = build_lookup() if table is None else table
    if not luts:
        return table
    if len(luts) > 4:
        raise ValueError('the game mixes at most four colour-correction LUTs')
    out = table * min(max(1.0 - sum(w for _, w in luts), 0.0), 1.0)
    for lut, weight in luts:
        out = out + sample_lut(lut, table) * weight
    out.setflags(write=False)
    return out


def lookup_texture(table=None):
    """The float32 (33, 1024, 3) texture the viewer samples with ``gameGrade`` (the layout of
    ``nwo.levels.display.display_table``): rows are green, blue slab b at columns 32 b to 32 b + 31 with red along x; row 32
    (the 2016 tone curve's row) is unused. ``table``: a lookup other than ``build_lookup`` (``colour_corrected``)."""
    n = LOOKUP_SIZE
    table = build_lookup() if table is None else table
    texture = np.zeros((n + 1, n * n, 3), np.float32)
    texture[:n] = table.transpose(1, 2, 0, 3).reshape(n, n * n, 3)   # [r, g, b] -> [g, b, r]
    return texture


def sample(table, coord):
    """Trilinear, clamped lookup of ``table`` (n, n, n, 3) [r, g, b] at entry coordinates ``coord`` (..., 3) in [0, 1]
    (0 the first entry's centre, 1 the last one's), as the game's sampler at (coord * (n - 1) + 0.5) / n."""
    n = table.shape[0]
    p = np.clip(np.asarray(coord, np.float64), 0, 1) * (n - 1)
    i0 = np.minimum(np.floor(p).astype(int), n - 2)
    f = p - i0
    out = 0
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                weight = ((f[..., 0] if dr else 1 - f[..., 0]) * (f[..., 1] if dg else 1 - f[..., 1])
                          * (f[..., 2] if db else 1 - f[..., 2]))[..., None]
                out = out + weight * table[i0[..., 0] + dr, i0[..., 1] + dg, i0[..., 2] + db]
    return out


def display(scene, exposure=1.0, table=None):
    """Encoded sRGB display of scene-linear ``scene`` (..., 3) at ``exposure`` (m_AcesMiddleGray / adapted luminance),
    the CPU reference of the viewer's riftApartDisplay; ``table`` another lookup (``colour_corrected``)."""
    x = np.maximum(np.asarray(scene, np.float64) * exposure, 1e-30)
    return sample(build_lookup() if table is None else table, np.log2(x) * LOG2_SCALE + LOG2_OFFSET)
