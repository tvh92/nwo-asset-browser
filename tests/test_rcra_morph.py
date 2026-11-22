"""Rift Apart blend shapes: synthetic model and clip morph sections, the viewer and export plumbing, and real-install
checks (RCRA_GAME, skipped without it)."""
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nwo.games.rcra import morph
from nwo.games.rcra.toc import sections

GAME = os.environ.get('RCRA_GAME')


def pack_bits(values, width):
    """MSB-first bit string of unsigned ``values`` at ``width`` bits each, zero-padded to 4 bytes."""
    bits = ''.join(format(v, f'0{width}b') for v in values)
    bits += '0' * (-len(bits) % 32)
    return int(bits, 2).to_bytes(len(bits) // 8, 'big') if bits else b''


def dat1(section_map, strings=b''):
    """A DAT1 of the given {tag: bytes}; ``strings`` sit in the string pool (offsets from the DAT1 start)."""
    head = 16 + 12 * len(section_map)
    offset = head + len(strings)
    offset += -offset % 16
    table, body = b'', b''
    for tag, data in section_map.items():
        body += bytes(-len(body) % 16)
        table += struct.pack('<III', tag, offset + len(body), len(data))
        body += data
    pool = strings + bytes(offset - head - len(strings))
    return struct.pack('<4sIIHH', b'1TAD', 0x12345678, offset + len(body), len(section_map), 0) + table + pool + body


def model_morph_sections(name_offset, bits=5, positions_only=False):
    """Morph info, data and index sections of one shape 'jaw' on mesh 3, in two chunks: vertices 2, 3, 4 of the first
    2560-vertex window and vertices 2560 + 40 .. 2560 + 71 (a run of 32) of the second. Fields are q = 1..6 per vertex."""
    fields = 1 if positions_only else 2
    per_vertex = 3 * fields
    chunks = [[(2, 3)], [(40, 0)]]                       # (skip, run) words; run 0 is 32
    counts = [3, 32]
    data = b''
    for c, count in enumerate(counts):
        data += pack_bits([(v % 7) + 1 + k for v in range(count) for k in range(per_vertex)], bits)
    index = b''.join(struct.pack('<I', skip | (run << 16)) for chunk in chunks for skip, run in chunk)
    shape = struct.pack('<4I', 0xAB, name_offset, 0, 0) + struct.pack('<4B', fields, 3 * bits, bits, 0)
    shape += struct.pack('<4f', 0.5, -1.0, 0.25, -0.5) + struct.pack('<HHII', 1, 0, len(data), len(index))
    shape += struct.pack('<b', 3) + bytes(3)                                      # subset mesh, aligned to 4
    shape += struct.pack('<II', 0, 0) + struct.pack('<H', 35) + bytes(2)          # data/index offsets, vertex count
    shape += struct.pack('<H', 0) + bytes(2)                                      # chunk-table position
    shape += struct.pack('<HI', 3, 1) + struct.pack('<HI', 32, 1)                 # chunk rows
    info = struct.pack('<IIHHIII', 0, len(data) + len(index), 1, 1, 24, 32, 0) + struct.pack('<II', 0xAB, 40)
    info += struct.pack('<II', 0xAB, 0xCD) + shape
    return {morph.MORPH_INFO: info, morph.MORPH_DATA: data, morph.MORPH_INDICES: index}


def clip_morph_sections(frames, hashes, first_bits=2, step_bits=3):
    """Clip morph info and frame data: frames of (channel, value) lists."""
    count_bits, value_bits = 3, 10
    stream, rows = b'', []
    for entries in frames:
        rows.append((len(entries), len(stream) // 4))
        bits = ''
        previous = 0
        for i, (channel, value) in enumerate(entries):
            width = first_bits if i == 0 else step_bits
            bits += format(channel if i == 0 else channel - previous, f'0{width}b') + format(value, f'0{value_bits}b')
            previous = channel
        bits += '0' * (-len(bits) % 32)
        stream += int(bits, 2).to_bytes(len(bits) // 8, 'big') if bits else b''
    offset_bits = max(1, (len(stream) // 4).bit_length())
    bits = ''.join(format(count, f'0{count_bits}b') + format(word, f'0{offset_bits}b') for count, word in rows)
    bits += '0' * (-len(bits) % 32)
    table = int(bits, 2).to_bytes(len(bits) // 8, 'big')
    info = struct.pack('<H6BI', len(hashes), count_bits, offset_bits, first_bits, step_bits, value_bits, 0, len(table))
    info += struct.pack('<I', 0) + struct.pack(f'<{len(hashes)}I', *hashes) + bytes(4 * len(hashes)) + table
    return {morph.CLIP_MORPH_INFO: info, morph.CLIP_MORPH_FRAMES: stream}


class ModelShapeTests(unittest.TestCase):
    def test_runs_windows_and_chunk_padding(self):
        name = b'jaw\0'
        raw = dat1(model_morph_sections(16 + 12 * 3), name)
        shapes = morph.model_shapes(raw, sections(raw), {3: 3000})
        self.assertEqual(len(shapes), 1)
        shape = shapes[0]
        self.assertEqual((shape['name'], shape['hash'], shape['mirror']), ('jaw', 0xAB, 0xCD))
        target = shape['targets'][0]
        self.assertEqual(target['mesh'], 3)
        self.assertEqual(target['vertices'].tolist(), [2, 3, 4] + list(range(2600, 2632)))
        # vertex 0 of each chunk: fields 1..6 -> positions q*0.5-1, normals q*0.25-0.5
        np.testing.assert_allclose(target['positions'][0], [-0.5, 0.0, 0.5])
        np.testing.assert_allclose(target['normals'][0], [0.5, 0.75, 1.0])
        np.testing.assert_allclose(target['positions'][3], [-0.5, 0.0, 0.5])   # the second chunk starts on 4 bytes

    def test_positions_only_and_rejections(self):
        raw = dat1(model_morph_sections(16 + 12 * 3, bits=14, positions_only=True), b'jaw\0')
        target = morph.model_shapes(raw, sections(raw))[0]['targets'][0]
        np.testing.assert_allclose(target['positions'][1], [0.0, 0.5, 1.0])
        self.assertFalse(target['normals'].any())
        with self.assertRaisesRegex(ValueError, 'outside their mesh'):
            morph.model_shapes(raw, sections(raw), {3: 100})
        self.assertEqual(morph.model_shapes(dat1({0x1234: b'x'}), sections(dat1({0x1234: b'x'}))), [])
        self.assertEqual(morph.shape_hashes(raw, sections(raw)), [0xAB])


class ClipWeightTests(unittest.TestCase):
    FRAMES = [[(1, 1023), (4, 512)], [], [(0, 10), (2, 20), (5, 30)]]

    def test_first_channel_steps_and_values(self):
        raw = dat1(clip_morph_sections(self.FRAMES, [11, 22, 33, 44, 55, 66]))
        found = morph.clip_weights(raw, sections(raw), 3)
        self.assertEqual(found['hashes'], [11, 22, 33, 44, 55, 66])
        expected = np.zeros((3, 6))
        for frame, entries in enumerate(self.FRAMES):
            for channel, value in entries:
                expected[frame, channel] = value / 1023
        np.testing.assert_allclose(found['weights'], expected, atol=1e-7)
        self.assertIsNone(morph.clip_weights(dat1({0x1234: b'x'}), sections(dat1({0x1234: b'x'})), 3))
        with self.assertRaisesRegex(ValueError, 'frame table'):
            morph.clip_weights(raw, sections(raw), 40)

    def test_library_maps_channels_to_the_model_shapes(self):
        from nwo.games.rcra.clip_library import RcraAnimationLibrary
        from tests.test_rcra_animation import build_clip
        clip = build_clip([(0, 0)] * 3, extra=clip_morph_sections(self.FRAMES, [11, 22, 33, 44, 55, 66]))
        library = RcraAnimationLibrary.__new__(RcraAnimationLibrary)
        library.shape_hashes = [66, 999, 22]          # the model's own order; 999 has no channel
        library.clips, library._last = [dict(key=1, source='animset')], (1, clip)
        weights = library.morph_weights(0, 4)         # a frame past the clip's keeps the last
        self.assertEqual(weights.shape, (4, 3))
        np.testing.assert_allclose(weights[:, 0], [0, 0, 30 / 1023, 30 / 1023], atol=1e-7)   # channel 5 (hash 66)
        np.testing.assert_allclose(weights[:, 1], 0)
        np.testing.assert_allclose(weights[:, 2], [1, 0, 0, 0], atol=1e-7)                   # channel 1 (hash 22)
        library.shape_hashes = [999]
        self.assertIsNone(library.morph_weights(0, 3))


class PlumbingTests(unittest.TestCase):
    def test_payload_offsets_and_basis(self):
        from nwo.viewer.loader import _morph_payload
        meshes = [dict(positions=np.zeros((4, 3))),
                  dict(positions=np.zeros((3, 3)), morphs=[(1, np.array([0, 2]), np.array([[1.0, 2.0, 3.0], [0, 0, 1.0]]),
                                                            np.zeros((2, 3)))])]
        decoded = dict(shapes=[dict(name='a', hash=1), dict(name='b', hash=2)])
        out = _morph_payload(decoded, meshes)['morphs']
        self.assertEqual(out['names'], ['a', 'b'])
        self.assertEqual(len(out['targets'][0][0]), 0)
        index, positions, _normals = out['targets'][1]
        self.assertEqual(index.tolist(), [4, 6])                          # after the first mesh's 4 vertices
        np.testing.assert_allclose(positions[0], [1.0, -3.0, 2.0])        # display basis (x, -z, y)
        self.assertEqual(_morph_payload(dict(shapes=[]), meshes), {})

    def test_viewport_blends_and_restores(self):
        import ctypes
        from nwo.viewer.viewport import ModelViewport
        vertices = np.zeros((3, 3))
        normals = np.tile([0.0, 0.0, 1.0], (3, 1))
        arrays = ((ctypes.c_float * 9)(*vertices.ravel()), (ctypes.c_float * 9)(*normals.ravel()))
        view = SimpleNamespace(_morphs=[(np.array([1]), np.array([[0.0, 0.0, 2.0]], np.float32), np.array([[1.0, 0.0, 0.0]], np.float32))],
                               _mesh=(vertices, None, normals), _arrays=arrays, _stale=set(), _morphed=False,
                               _morph_weights=None, _motion=None)
        view._refresh_vertices = lambda: ModelViewport._refresh_vertices(view)
        ModelViewport.set_morph(view, [0.5])
        moved = np.frombuffer(arrays[0], np.float32).reshape(3, 3)
        np.testing.assert_allclose(moved[1], [0, 0, 1.0])
        np.testing.assert_allclose(np.frombuffer(arrays[1], np.float32).reshape(3, 3)[1], np.array([0.5, 0, 1]) / np.hypot(0.5, 1), rtol=1e-6)
        self.assertEqual(view._stale, {id(arrays[0]), id(arrays[1])})
        ModelViewport.set_morph(view, None)
        np.testing.assert_allclose(np.frombuffer(arrays[0], np.float32), 0)
        self.assertFalse(view._morphed)

    def test_sparse_accessor(self):
        from nwo.export.models import GLTF
        g = GLTF('test')
        sparse = g.doc['accessors'][g.sparse(10, np.array([7, 2]), np.array([[1, 2, 3], [-1, 0, 4]], np.float32), 'VEC3', minmax=True)]
        self.assertEqual((sparse['count'], sparse['sparse']['count'], sparse['sparse']['indices']['componentType']), (10, 2, 5123))
        self.assertEqual((sparse['min'], sparse['max']), ([-1.0, 0.0, 0.0], [1.0, 2.0, 4.0]))
        view = g.doc['bufferViews'][sparse['sparse']['indices']['bufferView']]
        self.assertEqual(np.frombuffer(bytes(g.blob), '<u2', 2, view['byteOffset']).tolist(), [2, 7])   # sorted
        zero = g.doc['accessors'][g.sparse(5, np.zeros(0, np.int64), np.zeros((0, 3)), 'VEC3', minmax=True)]
        self.assertNotIn('bufferView', zero)
        self.assertNotIn('sparse', zero)
        self.assertEqual(zero['min'], [0.0, 0.0, 0.0])


@unittest.skipUnless(GAME, 'RCRA_GAME is not set')
class RealMorphTests(unittest.TestCase):
    RATCHET = 'characters/hero/hero_ratchet/hero_ratchet.model'

    @classmethod
    def setUpClass(cls):
        from nwo.games.rcra.source import RcraSource
        cls.temp = tempfile.TemporaryDirectory()
        cls.source = RcraSource(GAME, Path(cls.temp.name) / 'out')

    @classmethod
    def tearDownClass(cls):
        cls.source.close()
        cls.temp.cleanup()

    def data(self, path):
        return self.source.bytes(self.source.resolve(path)[0])[0]

    def test_ratchet_shapes_mirror(self):
        from nwo.games.rcra import model as rm
        data = self.data(self.RATCHET)
        decoded = rm.decode_model(self.RATCHET, data=data, as_arrays=True)
        raw = data[36:]
        shapes = {s['name']: s for s in morph.model_shapes(raw, sections(raw), {m['index']: m['vertex_count'] for m in decoded['meshes']})}
        self.assertEqual(len(shapes), 294)
        meshes = {m['index']: m for m in decoded['meshes']}
        left, right = shapes['LF_brow_up'], shapes['RT_brow_up']
        for lt in left['targets']:
            rt = next(t for t in right['targets'] if t['mesh'] == lt['mesh'])
            positions = meshes[lt['mesh']]['positions']
            mirrored = positions[rt['vertices']] * [-1, 1, 1]
            match = np.linalg.norm(positions[lt['vertices']][:, None] - mirrored[None], axis=2).argmin(1)
            error = np.linalg.norm(lt['positions'] - rt['positions'][match] * [-1, 1, 1], axis=1)
            self.assertLess(float(np.median(error)), 0.01 * float(np.abs(lt['positions']).max()))   # measured 0.003
            self.assertGreater(float(lt['positions'][:, 1].max()), 0.01)                          # the brow goes up

    def test_phoneme_silence_and_a_line(self):
        from nwo.games.rcra.animation import clip_dat1, header
        from nwo.games.rcra.clip_library import animation_library
        data = self.data('characters/hero/hero_ratchet/animations/ratchet_phonemes.animclip')
        raw, table = clip_dat1(data)
        found = morph.clip_weights(raw, table, header(data)['original_frame_count'])
        self.assertEqual(len(found['hashes']), 55)
        self.assertFalse(found['weights'][20].any())   # the silence pose
        self.assertTrue(found['weights'][0].any())
        library = animation_library(self.source, self.data(self.RATCHET), self.RATCHET).with_performances()
        index = library.labels().index('ratc_boards: RATC_BOARDS_SLICE1_017')
        weights = library.pose(index)['morph_weights']
        self.assertEqual(weights.shape, (497, 294))
        self.assertGreater(int((weights.max(0) > 0.01).sum()), 30)   # measured 53 shapes
        self.assertFalse(weights[0].any())                           # the line starts silent


if __name__ == '__main__':
    unittest.main()
