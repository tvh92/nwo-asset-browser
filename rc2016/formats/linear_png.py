"""sRGB texels for Cycles: decoded to linear light and stored as a 16-bit PNG that Blender loads as Non-Color.

The GPU decodes each sRGB texel to linear before it filters. Cycles filters an sRGB image (8 or 16 bit) first and decodes
the filtered value, which darkens lookups between contrasting texels (a corner of 0.1 and 0.9 texels reads 0.216 instead of
0.401). An image whose texels are already linear, marked Non-Color, filters as the GPU does. 16 bits keep every 8-bit sRGB
value distinct and within 7.6e-6 of its decode. Alpha is linear data and is stored as it is.

Both games use this: Rift Apart unit textures (rc2016.games.rcra.blender_export) and the R&C 2016 research pipeline
(rc2016.pipeline.geometry, rc2016.pipeline.holocard_effect).
"""
import struct
import zlib

import numpy as np

# The Blender image conversion of an sRGB DDS in the 2016 pipeline (image property 'blender_conversion').
SRGB_LINEAR16_CONVERSION = 'SRGB_DDS_TO_LINEAR16_PNG_V1'
SRGB_LINEAR16_CONTAINER = '16-bit PNG of the sRGB texels decoded to linear, alpha as stored'


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


# 8-bit sRGB code -> 16-bit linear code, the same values as round(srgb_to_linear(code / 255) * 65535).
SRGB8_TO_LINEAR16 = np.round(srgb_to_linear(np.arange(256) / 255.0) * 65535.0).astype(np.uint16)


def png16(rgba):
    """A 16-bit RGBA PNG of ``rgba`` (h, w, 4) uint16, rows top first."""
    h, w = rgba.shape[:2]
    rows = np.empty((h, 1 + w * 8), np.uint8)
    rows[:, 0] = 0
    rows[:, 1:] = np.ascontiguousarray(rgba, '>u2').view(np.uint8).reshape(h, w * 8)

    def chunk(tag, payload):
        return struct.pack('>I', len(payload)) + tag + payload + struct.pack('>I', zlib.crc32(tag + payload) & 0xFFFFFFFF)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 16, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows.tobytes(), 6)) + chunk(b'IEND', b''))


def srgb8_linear16(rgba8):
    """(h, w, 4) uint16 of 8-bit sRGB texels ``rgba8`` (h, w, 4) uint8: colour decoded to linear, alpha scaled as stored."""
    rgba8 = np.asarray(rgba8, np.uint8)
    out = np.empty(rgba8.shape, np.uint16)
    out[..., :3] = SRGB8_TO_LINEAR16[rgba8[..., :3]]
    out[..., 3] = rgba8[..., 3].astype(np.uint16) * 257
    return out


def srgb8_linear16_png(rgba8):
    """The 16-bit linear PNG (bytes) of 8-bit sRGB texels ``rgba8`` (h, w, 4) uint8, rows top first."""
    return png16(srgb8_linear16(rgba8))
