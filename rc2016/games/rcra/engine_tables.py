"""Rift Apart engine lookup tables that ship inside RiftApart.exe rather than as assets.

``TextureDefaultsInit`` (0x141353e10) builds the engine's default textures. One is the BRDF lookup the deferred apply and
forward programs sample (g_BrdfLookup, "Default Brdf Lookup", 0x1413553ef): R16G16B16A16_FLOAT, 64 x 64, 4 slices.
- Slice 0: red and green from 4,096 half-float pairs copied by the loop at 0x1413553a0 (source from 0x1444c6cc0); blue and
  alpha 0.
- Slices 1 and 2: 0x8000 bytes each, copied from 0x14548e340 and 0x145496340.
- Slice 3: 0.
No texture asset replaces it (the texture's only other reference is the teardown).

``brdf_lookup`` reads the tables through the instructions that load them: it decodes the ``lea reg, [rip + disp]``
instructions at the addresses above and checks that they still point where this build's do, so a different build of the
executable raises instead of returning other bytes.
"""
from __future__ import annotations

import os
import struct
import threading

import numpy as np

EXECUTABLE = 'RiftApart.exe'
SIZE, SLICES = 64, 4
# (instruction VA, register byte, expected target VA): the slice-0 loop's source and end (both + 2: it reads the word before
# the pointer), the two slice copies' sources, and the texture's name.
BRDF_LEAS = ((0x14135538a, 0x15, 0x1444c6cc2), (0x141355391, 0x05, 0x1444cacc2), (0x1413553be, 0x15, 0x14548e340),
             (0x1413553db, 0x15, 0x145496340), (0x1413553ef, 0x05, 0x1444cb158))
BRDF_NAME = b'Default Brdf Lookup'
_LOCK = threading.Lock()
_TABLES = {}


class _Image:
    """Section mapping of a PE file's bytes."""

    def __init__(self, data):
        self.data = data
        pe = struct.unpack_from('<I', data, 0x3C)[0]
        count = struct.unpack_from('<H', data, pe + 6)[0]
        optional = struct.unpack_from('<H', data, pe + 20)[0]
        self.base = struct.unpack_from('<Q', data, pe + 24 + 24)[0]
        self.sections = []
        for k in range(count):
            at = pe + 24 + optional + 40 * k
            vsize, va, rsize, raw = struct.unpack_from('<4I', data, at + 8)
            self.sections.append((va, max(vsize, rsize), raw, rsize))

    def read(self, va, size):
        rva = va - self.base
        for start, vsize, raw, rsize in self.sections:
            if start <= rva < start + rsize:
                return self.data[raw + rva - start:raw + rva - start + size]
        raise ValueError('Address 0x%X is not in the executable' % va)

    def lea_target(self, va, register):
        """The target of ``lea r64, [rip + disp32]`` at ``va`` (REX.W 8D, ModRM ``register``), or None."""
        code = self.read(va, 7)
        if len(code) != 7 or code[0] not in (0x48, 0x4C) or code[1] != 0x8D or code[2] != register:
            return None
        return va + 7 + struct.unpack_from('<i', code, 3)[0]


def brdf_tables(data):
    """(SLICES, SIZE, SIZE, 4) float32 of the BRDF lookup, from the executable's bytes."""
    image = _Image(data)
    for va, register, want in BRDF_LEAS:
        if image.lea_target(va, register) != want:
            raise ValueError('BRDF lookup not found: this RiftApart.exe build differs from the traced one')
    if image.read(0x1444cb158, len(BRDF_NAME)) != BRDF_NAME:
        raise ValueError('BRDF lookup not found: its name is not at the traced address')
    lut = np.zeros((SLICES, SIZE, SIZE, 4), np.float16)
    pairs = np.frombuffer(image.read(0x1444c6cc0, 0x4000), '<f2').reshape(SIZE, SIZE, 2)
    lut[0, :, :, :2] = pairs
    lut[1] = np.frombuffer(image.read(0x14548e340, 0x8000), '<f2').reshape(SIZE, SIZE, 4)
    lut[2] = np.frombuffer(image.read(0x145496340, 0x8000), '<f2').reshape(SIZE, SIZE, 4)
    return lut.astype(np.float32)


def brdf_lookup(root):
    """brdf_tables of the install's executable (cached by path, size and time)."""
    path = os.path.join(str(root), EXECUTABLE)
    stat = os.stat(path)
    key = (path, stat.st_size, stat.st_mtime_ns)
    with _LOCK:
        if key not in _TABLES:
            with open(path, 'rb') as f:
                _TABLES[key] = brdf_tables(f.read())
        return _TABLES[key]
