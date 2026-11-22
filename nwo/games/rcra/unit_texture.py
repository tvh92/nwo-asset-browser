"""Rift Apart float and 16-bit textures as images: the values a shader samples, where they fit an image's 0..1.

D3D samples a UNORM16 texel as its integer / 65535 and a float texel as the float itself, and fills the channels a format
lacks with 0 for green and blue and 1 for alpha. These are the same values an 8-bit UNORM image stores as
round(255 x value) / 255, so a texture whose sampled values all lie in 0..1 converts by rounding alone. Over the install
that is 6 of the 156 listed float and 16-bit textures (three R16G16: two normals and a flowmap; the foam and wetness
lookups; a coretest copy of the pink normal) and the unlisted default_pink_shader_n placeholder (flat, R32G32_FLOAT) that 386 materials bind. The other 150
hold non-finite values (140 vertex-animation tables, whose floats are packed bits) or values outside 0..1 (vertex
animation and two HDR skies), which no 0..1 image holds without a remap the files do not define; they raise with their
range or count, and the DDS export keeps their stored values.
"""
from __future__ import annotations

import numpy as np

# DXGI float and UNORM16 formats as (numpy dtype, channels); the same table the Blender export writes as float32.
FLOAT_FORMATS = {2: ('<f4', 4), 10: ('<f2', 4), 16: ('<f4', 2), 34: ('<f2', 2), 41: ('<f4', 1), 54: ('<f2', 1),
                 11: ('<u2', 4), 35: ('<u2', 2), 56: ('<u2', 1)}
UNORM16 = frozenset({11, 35, 56})
RGBA8 = 28


def is_data(fmt) -> bool:
    """True for the float and UNORM16 formats this module converts."""
    return fmt in FLOAT_FORMATS


def unit_rgba(texture, index=0):
    """Float64 (h, w, 4) of one slice: the values D3D samples (D3D defaults for missing channels). Raises ValueError
    when a value is not finite or lies outside 0..1."""
    fmt = texture['format']
    dtype, channels = FLOAT_FORMATS[fmt]
    width, height = texture['width'], texture['height']
    pixels = texture.get('slices', [texture['pixels']])[index]
    values = np.frombuffer(pixels, dtype, width * height * channels).reshape(height, width, channels)
    with np.errstate(invalid='ignore', over='ignore'):   # NaN and infinity are reported below
        values = values.astype(np.float64) / 65535.0 if fmt in UNORM16 else values.astype(np.float64)
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError(f'DXGI format {fmt} holds {int((~finite).sum())} non-finite values; no image holds them '
                         '(the DDS export keeps them)')
    low, high = float(values.min()), float(values.max())
    if low < 0.0 or high > 1.0:
        raise ValueError(f'DXGI format {fmt} holds values from {low:.4g} to {high:.4g}, outside the 0..1 an image '
                         'stores (the DDS export keeps them)')
    out = np.zeros((height, width, 4), np.float64)
    out[..., 3] = 1.0
    out[..., :channels] = values
    return out


def as_rgba8(texture):
    """A copy of a loaded float/UNORM16 texture as RGBA8 (DXGI 28, linear), every slice rounded to 8 bits, with
    ``stored_format`` naming the original. Raises ValueError like ``unit_rgba``."""
    slices = [np.rint(unit_rgba(texture, i) * 255.0).astype(np.uint8).tobytes()
              for i in range(len(texture.get('slices', [texture['pixels']])))]
    return dict(texture, format=RGBA8, pixels=slices[0], slices=slices, srgb=False, stored_format=texture['format'],
                gl_format=None, gl_pixel_format=0x1908)


def png16(texture, index=0, flip=False):
    """A 16-bit RGBA PNG (bytes) of one slice: UNORM16 values exactly, floats rounded to 1/65535."""
    from nwo.formats.linear_png import png16 as encode
    values = unit_rgba(texture, index)
    if flip:
        values = values[::-1]
    return encode(np.rint(values * 65535.0).astype(np.uint16))
