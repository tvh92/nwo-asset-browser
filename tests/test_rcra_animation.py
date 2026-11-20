"""Rift Apart animation: synthetic clips and skin batches, plus real-install checks (RCRA_GAME, skipped without it)."""
import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rc2016.formats.animation_transforms import sample_transforms
from rc2016.formats.container import game_crc32
from rc2016.games.rcra import animation as ra
from rc2016.games.rcra import model as rm

GAME = os.environ.get('RCRA_GAME')
ROOT, CHILD = game_crc32('root'), game_crc32('child')


def _pack(values, widths):
    """Frame-major LSB-first bit packing of one stored sample, padded to 4 bytes."""
    word, bit = 0, 0
    for value, width in zip(values, widths):
        word |= value << bit
        bit += width
    return word.to_bytes((bit + 31) // 32 * 4, 'little')


def build_clip(samples, table=None, frames=None, name='Walk_test', fps=30.0, extra=None, flags=0, tail=b'',
               seconds=None, stored=None):
    """A two-joint clip: root translation x (4 bits, 1/1024 units) and child quaternion z (8 bits) are sampled.

    ``samples``: stored (root tx, child qz) integer pairs. ``table``: the frame table (stored-sample index per entry,
    round(seconds * fps) + 1 entries) or None for one stored sample per frame. ``tail``: the base scale tail (32-byte
    blocks). ``seconds``/``stored`` override the timing and the declared stored-sample count."""
    frames = frames or (len(table) if table else len(samples))
    seconds = (frames - 1) / fps if seconds is None else seconds
    widths = (4, 8)
    elements = struct.pack('<2I', 32, 48 + 24) + bytes(w - 1 for w in widths)
    elements += bytes(-len(elements) % 4)
    stride = 4
    payload = b''.join(_pack(s, widths) for s in samples)
    if table is not None:
        flags |= ra.FLAG_FRAME_TABLE
        head = struct.pack(f'<{len(table)}H', *table)
        payload = head + bytes(-len(head) % 16) + payload
    base = (struct.pack('<7h', 0, 0, 0, 1024, 0, 0, 0) + bytes((10, 10))
            + struct.pack('<7h', 0, 0, 0, 1024, 1024, 0, 0) + bytes((10, 10)) + tail)
    hashes = struct.pack('<3I', ROOT, CHILD, 0xFFFFFFFF)
    built = bytearray(96)
    struct.pack_into('<IIIff', built, 0, 0, game_crc32(name), flags, fps, seconds)
    struct.pack_into('<4H', built, 32, 2, frames, 2, 0)
    struct.pack_into('<IHHH', built, 40, stride, len(tail) // 32, 0, len(samples) if stored is None else stored)
    sections = {ra.BASE: base, ra.ELEMENTS: elements, ra.SAMPLES: payload, ra.JOINT_HASHES: hashes, ra.BUILT: bytes(built)}
    sections.update(extra or {})
    strings = b'animclip built\0' + name.encode() + b'\0'
    offset = 16 + 12 * len(sections) + len(strings)
    table_bytes, body = b'', b''
    for tag, data in sections.items():
        table_bytes += struct.pack('<III', tag, offset + len(body), len(data))
        body += data
    dat1 = struct.pack('<4sIIHH', b'1TAD', ra.CLIP_MAGIC, offset + len(body), len(sections), 0) + table_bytes + strings + body
    return struct.pack('<I', ra.CLIP_MAGIC) + bytes(32) + dat1


class SyntheticClipTests(unittest.TestCase):
    def test_header_and_samples(self):
        data = build_clip([(0, 0), (3, 64), (15, 255)])
        facts = ra.header(data)
        self.assertEqual((facts['name'], facts['bone_count'], facts['original_frame_count'], facts['stored_sample_count']),
                         ('Walk_test', 2, 3, 3))
        clip = ra.decode_clip(data, {ROOT: 'root', CHILD: 'child'})
        self.assertEqual([c['unsigned_deltas'] for c in clip['channels']], [[0, 3, 15], [0, 64, 255]])
        frames = sample_transforms(clip)
        self.assertEqual(len(frames), 3)
        self.assertEqual([f[0]['translation'][0] for f in frames], [0.0, 3 / 1024, 15 / 1024])
        self.assertEqual(frames[0][1]['translation'], [1.0, 0.0, 0.0])
        q = np.array(frames[2][1]['rotation_xyzw'])
        np.testing.assert_allclose(q, np.array([0, 0, 255, 1024]) / np.hypot(255, 1024))
        self.assertEqual(frames[1][0]['scale'], [1.0, 1.0, 1.0])

    def test_frame_table_repeats_stored_samples(self):
        # Five frames, two stored samples: the table plays 0 1 0 1 0 (and a closing 0 in its sixth entry).
        data = build_clip([(1, 10), (2, 20)], table=[0, 1, 0, 1, 0, 0], frames=5)
        clip = ra.decode_clip(data)
        self.assertEqual(clip['metadata']['original_frame_count'], 5)
        self.assertEqual(clip['channels'][0]['unsigned_deltas'], [1, 2, 1, 2, 1])
        self.assertEqual(len(ra.decode_clip(data, first_frames=2)['channels'][1]['unsigned_deltas']), 2)

    def test_frame_table_without_a_closing_entry_when_the_end_pose_is_counted(self):
        # frames == seconds * fps + 1: the table has exactly one entry per frame.
        data = build_clip([(1, 10), (2, 20)], table=[0, 1, 1], frames=3)
        self.assertEqual(ra.decode_clip(data)['channels'][0]['unsigned_deltas'], [1, 2, 2])

    def test_uncounted_closing_pose_without_a_frame_table(self):
        # frames == seconds * fps == stored, and the payload holds one more sample (the first pose again).
        data = build_clip([(1, 10), (2, 20), (1, 10)], frames=2, seconds=2 / 30, stored=2)
        clip = ra.decode_clip(data)
        self.assertEqual(clip['channels'][0]['unsigned_deltas'], [1, 2])
        with self.assertRaisesRegex(ValueError, 'payload size'):   # with the end pose counted it is not allowed
            ra.decode_clip(build_clip([(1, 10), (2, 20), (1, 10)], frames=2, stored=2))

    def test_additive_scale_sentinel_is_the_identity(self):
        # Root: 0x8000 on every axis (the sentinel); child: 512 (0.5 at exponent 10). The block repeats its record.
        record = struct.pack('<8H', 0x8000, 512, 0x8000, 512, 0x8000, 512, 0, 48)
        tail = record * 2
        clip = ra.decode_clip(build_clip([(0, 0), (1, 1)], tail=tail, flags=ra.FLAG_ADDITIVE))
        frames = sample_transforms(clip)
        self.assertEqual([j['scale'] for j in frames[1]], [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
        with self.assertRaisesRegex(ValueError, 'high-bit'):   # an absolute clip keeps refusing it
            sample_transforms(ra.decode_clip(build_clip([(0, 0)], tail=tail)))
        animated = bytearray(build_clip([(0, 0)], tail=tail, flags=ra.FLAG_ADDITIVE))
        _dat1, table = ra.clip_dat1(bytes(animated))
        struct.pack_into('<I', animated, 36 + table[ra.ELEMENTS][0], 0)   # the root channel becomes scale x
        with self.assertRaisesRegex(ValueError, 'animated scale'):
            sample_transforms(ra.decode_clip(bytes(animated)))

    def test_rejections(self):
        with self.assertRaises(ra.UnsupportedClipError):
            ra.decode_clip(b'\0' * 64)
        short = bytearray(build_clip([(1, 1), (2, 2)]))
        _dat1, table = ra.clip_dat1(bytes(short))
        struct.pack_into('<H', short, 36 + table[ra.BUILT][0] + 48, 1)   # one stored sample declared, two stored
        with self.assertRaisesRegex(ValueError, 'stored sample count|payload size'):
            ra.decode_clip(bytes(short))
        with self.assertRaisesRegex(ValueError, 'frame table points past'):
            ra.decode_clip(build_clip([(1, 1)], table=[0, 3, 0], frames=3))
        padded = build_clip([(0, 0)])
        # Set a padding bit above the 12 used bits of the only stored sample.
        _dat1, table = ra.clip_dat1(padded)
        offset, _size = table[ra.SAMPLES]
        corrupt = bytearray(padded)
        corrupt[36 + offset + 2] |= 0x80
        with self.assertRaisesRegex(ValueError, 'padding'):
            ra.decode_clip(bytes(corrupt))

    def test_custom_track_clip_is_not_skeletal(self):
        data = build_clip([(0, 0)])
        dat1, table = ra.clip_dat1(data)
        # Rename the base-state section: a clip with only custom float tracks has none.
        offset = 16 + 12 * list(table).index(ra.BASE)
        corrupt = bytearray(data)
        struct.pack_into('<I', corrupt, 36 + offset, ra.CUSTOM_TRACKS)
        with self.assertRaisesRegex(ra.UnsupportedClipError, 'custom float tracks'):
            ra.decode_clip(bytes(corrupt))


def motion_section(layout, kind, records, count, frames, start_q=(0, 0, 0, 1), start_t=(0, 0, 0)):
    """A 0xE08AA35F section: the 124-byte header, then ``records``, zero-padded so the section is a multiple of 16."""
    body = records + bytes(-(124 + len(records)) % 16)
    head = bytearray(124)
    struct.pack_into('<BBH', head, 0, layout, kind, count)
    struct.pack_into('<4f', head, 4, *start_q)
    struct.pack_into('<4f', head, 32, 1.0, *start_t)
    struct.pack_into('<II', head, 64, len(body), frames)
    return bytes(head) + body


def smallest_three(q):
    """The u64 smallest-three word of unit quaternion ``q`` (xyzw)."""
    q = np.asarray(q, float)
    largest = int(np.argmax(np.abs(q)))
    q = -q if q[largest] < 0 else q
    fields = [int(round(v / ra.SMALLEST_THREE)) & 0xFFFFF for v in np.delete(q, largest)]
    return fields[0] | fields[1] << 20 | fields[2] << 40 | largest << 60


def yaw(degrees):
    half = np.radians(degrees) / 2
    return np.array([0.0, np.sin(half), 0.0, np.cos(half)])


class RootMotionTests(unittest.TestCase):
    def test_sixteen_byte_records(self):
        records = b''.join(struct.pack('<7hBB', *[round(v * 32767) for v in yaw(a)], tx, 0, tz, exponent, 165)
                           for a, tx, tz, exponent in ((0, 0, 0, 12), (90, 4096, -2048, 12), (180, 3, 0, 1)))
        found = ra.motion(build_clip([(0, 0)] * 3, extra={ra.MOTION: motion_section(0, 3, records, 3, 3, start_t=(1, 2, 3))}))
        np.testing.assert_allclose(found['start_translation'], [1, 2, 3])
        np.testing.assert_allclose(found['translation'], [[0, 0, 0], [1, 0, -0.5], [1.5, 0, 0]])
        np.testing.assert_allclose(found['rotation_xyzw'][1], yaw(90), atol=1e-4)
        self.assertEqual((found['layout'], found['kind']), (0, 3))

    def test_twenty_four_byte_records(self):
        rotations = [yaw(0), yaw(-170), np.array([0.5, -0.5, 0.5, 0.5])]
        records = b''.join(struct.pack('<Q3fI', smallest_three(q), *t, 0x40)
                           for q, t in zip(rotations, ([0, 0, 0], [1.5, -2.0, 300.25], [0, 1, 0])))
        found = ra.motion(build_clip([(0, 0)] * 3, extra={ra.MOTION: motion_section(1, 2, records, 3, 3)}))
        for got, want in zip(found['rotation_xyzw'], rotations):
            self.assertAlmostEqual(abs(float(np.dot(got, want))), 1.0, places=9)   # q and -q are one rotation
        np.testing.assert_allclose(found['translation'][1], [1.5, -2.0, 300.25])

    def test_no_records_and_no_section(self):
        found = ra.motion(build_clip([(0, 0)] * 3, extra={ra.MOTION: motion_section(1, 0, b'', 0, 3)}))
        self.assertEqual((found['rotation_xyzw'].shape, found['translation'].shape), ((0, 4), (0, 3)))
        self.assertIsNone(ra.motion(build_clip([(0, 0)] * 3)))

    def test_rejections(self):
        good = motion_section(0, 2, struct.pack('<7hBB', 0, 0, 0, 32767, 0, 0, 0, 12, 0), 1, 1)
        for section, message in (
                (good[:-1], 'record size'),
                (good[:-1] + b'\x01', 'padding'),
                (bytes([0, 2]) + good[2:124] + struct.pack('<7hBB', 0, 0, 0, 16000, 0, 0, 0, 12, 0) + good[140:], 'unit'),
                (bytes([2]) + good[1:], 'record format')):
            with self.assertRaisesRegex(ValueError, message):
                ra.motion_section(section)

    def test_library_displacement_is_relative_to_the_first_record(self):
        from rc2016.games.rcra.clip_library import RcraAnimationLibrary
        records = b''.join(struct.pack('<Q3fI', smallest_three(q), *t, 0)
                           for q, t in ((yaw(90), [1, 0, 0]), (yaw(180), [1, 0, -2]), (yaw(180), [1, 0, -4])))
        data = build_clip([(0, 0)] * 2, extra={ra.MOTION: motion_section(1, 2, records, 3, 2)})
        library = RcraAnimationLibrary.__new__(RcraAnimationLibrary)
        library.clips, library._last = [dict(key=7)], (7, data)
        rotation, translation = library.root_displacement(0, 2)   # the closing record is not played
        np.testing.assert_allclose(translation, [[0, 0, 0], [2, 0, 0]], atol=1e-6)   # (0, 0, -2) in the first record's frame
        self.assertAlmostEqual(abs(float(np.dot(rotation[1], yaw(90)))), 1.0, places=9)
        rotation, translation = library.root_displacement(0, 5)   # frames past the records keep the last one
        np.testing.assert_allclose(translation[4], [4, 0, 0], atol=1e-6)

    def test_apply_root_motion_moves_the_whole_skeleton(self):
        from rc2016.formats.skeleton import worlds
        from rc2016.viewer.animation import apply_root_motion
        rows = [dict(hash='0x1', name='root', parent=-1, flags=0, scale=[2.0, 2.0, 2.0], rotation_xyzw=[0, 0, 0, 1],
                     translation=[0.0, 1.0, 0.0]),
                dict(hash='0x2', name='child', parent=0, flags=0, scale=[1.0, 1.0, 1.0], rotation_xyzw=list(yaw(30)),
                     translation=[0.0, 0.0, 1.0]),
                dict(hash='0x3', name='other_top', parent=-1, flags=0, scale=[1.0, 1.0, 1.0], rotation_xyzw=[0, 0, 0, 1],
                     translation=[3.0, 0.0, 0.0])]
        frame = [dict(hash='0x1', scale=[2.0, 2.0, 2.0], rotation_xyzw=list(yaw(10)), translation=[0.5, 1.0, 0.0]),
                 dict(hash='0x2', scale=[1.0, 1.0, 1.0], rotation_xyzw=None, translation=[0.0, 0.0, 1.0])]
        d_q, d_t = yaw(90), np.array([1.0, 0.0, 2.0])
        moved = apply_root_motion(rows, [frame], d_q[None], d_t[None])[0]
        x, y, z, w = d_q
        displacement = np.eye(4)
        displacement[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        displacement[:3, 3] = d_t
        expected = [displacement @ m for m in worlds(rows, frame)]
        for got, want in zip(worlds(rows, moved), expected):   # the unsampled parentless joint is added and moved too
            np.testing.assert_allclose(got, want, atol=1e-12)


def build_performance(rows, frames, name='LINE_001', fps=30.0, padding=0):
    """A performance clip: built header (no joints) and one 16-byte row per frame, padded to 8 rows (+ ``padding``)."""
    count = -(-len(rows) // 8) * 8 + padding
    samples = b''.join(bytes(index) + bytes(8 - len(index)) + bytes(weight) + bytes(8 - len(weight))
                       for index, weight in rows) + bytes(16 * (count - len(rows)))
    built = bytearray(96)
    struct.pack_into('<IIIff', built, 0, 0, game_crc32(name), 8, fps, (frames - 1) / fps)
    struct.pack_into('<4H', built, 32, 0, frames, 0, 0)
    struct.pack_into('<IHHH', built, 40, 0, 0, 0, frames)
    sections = {ra.SAMPLES: samples, ra.BUILT: bytes(built)}
    strings = b'animclip built\0' + name.encode() + b'\0'
    offset = 16 + 12 * len(sections) + len(strings)
    table_bytes, body = b'', b''
    for tag, data in sections.items():
        table_bytes += struct.pack('<III', tag, offset + len(body), len(data))
        body += data
    dat1 = struct.pack('<4sIIHH', b'1TAD', ra.CLIP_MAGIC, offset + len(body), len(sections), 0) + table_bytes + strings + body
    return struct.pack('<I', ra.CLIP_MAGIC) + bytes(32) + dat1


class FacePerformanceTests(unittest.TestCase):
    def test_rows_weights_and_padding(self):
        data = build_performance([((3,), (255,)), ((1, 2), (128, 127)), ((0,), (255,))], 3, padding=8)
        found = ra.performance_clip(data)
        self.assertEqual((found['name'], found['frames'], found['sample_rate_fps']), ('LINE_001', 3, 30.0))
        self.assertEqual(found['index'].shape, (3, 8))
        self.assertEqual(found['index'][1, :2].tolist(), [1, 2])
        np.testing.assert_allclose(found['weight'][1, :2], [128 / 255, 127 / 255])
        with self.assertRaisesRegex(ValueError, 'cover'):
            ra.performance_clip(build_performance([((0,), (255,))] * 8, 9))   # 9 frames, 8 rows

    def test_pose_table(self):
        table = struct.pack('<4H', 20, 8, 0, 2)
        self.assertEqual(ra.pose_table(build_clip([(0, 0)] * 2, extra={ra.PHONEME_TABLE: table})), [20, 8, 0, 2])
        self.assertIsNone(ra.pose_table(build_clip([(0, 0)] * 2)))

    def test_blend_of_pose_frames(self):
        def row(angle, tx, scale=1.0):
            return dict(name='jaw', hash='0x9', scale=[scale] * 3, rotation_xyzw=list(yaw(angle)), translation=[tx, 0.0, 0.0])
        poses = [[row(0, 0.0)], [row(40, 1.0, 2.0)], [row(-20, -1.0)]]
        table = [0, 1, 2]
        index = np.zeros((4, 8), np.intp)
        weight = np.zeros((4, 8))
        index[0, 0], weight[0, 0] = 1, 1.0                                 # one pose
        index[1, :2], weight[1, :2] = (1, 2), (0.5, 0.5)                   # two poses
        index[2, 0], weight[2, 0] = 1, 0.5                                 # half left to the identity
        index[3, :2], weight[3, :2] = (1, 0), (128 / 255, 128 / 255)       # 256/255 in total, scaled to 1
        frames = ra.performance_frames(poses, table, dict(index=index, weight=weight))
        np.testing.assert_allclose(frames[0][0]['rotation_xyzw'], yaw(40), atol=1e-12)
        np.testing.assert_allclose(frames[0][0]['scale'], [2.0] * 3)
        np.testing.assert_allclose(frames[1][0]['rotation_xyzw'], yaw(10), atol=1e-12)   # symmetric pair: the mid angle
        np.testing.assert_allclose(frames[1][0]['translation'], [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(frames[2][0]['rotation_xyzw'], yaw(20), atol=1e-12)
        np.testing.assert_allclose(frames[2][0]['translation'], [0.5, 0, 0])
        np.testing.assert_allclose(frames[2][0]['scale'], [1.5] * 3)
        np.testing.assert_allclose(frames[3][0]['translation'], [0.5, 0, 0])
        with self.assertRaisesRegex(ValueError, 'past the phoneme table'):
            ra.performance_frames(poses, [0, 1], dict(index=np.full((1, 8), 2), weight=np.full((1, 8), 0.1)))

    def test_face_section_of_the_clip_list(self):
        from rc2016.app.browser import face_clips, group_clips
        labels = ['hero_ratchet: Walk', 'ratc_corv: RATC_CORV_001', 'ratc_corv: RATC_CORV_002', 'shot_1']
        animsets = ['characters/hero/hero_ratchet/hero_ratchet.animset', 'localization/ratc/ratc_corv.animset',
                    'cinematics/x/x.cinematic2']
        face = face_clips(labels, animsets)
        self.assertEqual(face, [1, 2])
        self.assertEqual([text for text, _nodes in group_clips(labels, [3], face)],
                         ['Animset animations', 'Cutscene animations', 'Face performances'])


class SkinBatchTests(unittest.TestCase):
    def test_groups_of_one_and_several_influences(self):
        # 18 vertices: a full one-influence group (joint 2), then a two-vertex group of (joint, weight) pairs.
        skin = bytes([0]) + bytes([2] * 16) + bytes([1]) + bytes([0, 200, 1, 56, 1, 128, 0, 128])
        batches = [(0, 0, 0, 0, 18, 0)]
        lists, joints, weights = rm._mesh_weights(skin, b'', batches, 0, 1, 18, 3, 0, True)
        self.assertEqual(lists[0], [(2, 1.0)])
        self.assertEqual(lists[16], [(0, 200 / 256), (1, 56 / 256)])
        self.assertEqual(joints[17, :2].tolist(), [1, 0])
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)

    def test_palette_and_joint_base(self):
        remap = struct.pack('<2H', 300, 7)
        skin = bytes([0, 1, 0]) + bytes([0, 0, 1])
        batches = [(0, 0, 2, 0, 2, 0), (3, 5, 0, 0, 2, 2)]   # palette of 2 at byte 0; then base 5
        lists, _j, _w = rm._mesh_weights(skin, remap, batches, 0, 2, 4, 400, 0, False)
        self.assertEqual([row[0][0] for row in lists], [7, 300, 5, 6])

    def test_rejections(self):
        with self.assertRaisesRegex(ValueError, 'without weights'):
            rm._mesh_weights(bytes([0, 0, 0]), b'', [(0, 0, 0, 0, 2, 0)], 0, 1, 3, 1, 0, False)
        with self.assertRaisesRegex(ValueError, 'outside the skeleton'):
            rm._mesh_weights(bytes([0, 5]), b'', [(0, 0, 0, 0, 1, 0)], 0, 1, 1, 2, 0, False)
        with self.assertRaisesRegex(ValueError, 'runs past'):
            rm._mesh_weights(bytes([1, 0, 255]), b'', [(0, 0, 0, 0, 1, 0)], 0, 1, 1, 2, 0, False)


@unittest.skipUnless(GAME, 'RCRA_GAME is not set')
class RealAnimationTests(unittest.TestCase):
    FISH = 'characters/ambient/amb_fish_01/amb_fish_01.model'
    RATCHET = 'characters/hero/hero_ratchet/hero_ratchet.model'

    @classmethod
    def setUpClass(cls):
        from rc2016.games.rcra.source import RcraSource
        cls.temp = tempfile.TemporaryDirectory()
        cls.source = RcraSource(GAME, Path(cls.temp.name) / 'out')

    @classmethod
    def tearDownClass(cls):
        cls.source.close()
        cls.temp.cleanup()

    def data(self, path):
        return self.source.bytes(self.source.resolve(path)[0])[0]

    def library(self, path):
        from rc2016.games.rcra.clip_library import animation_library
        return animation_library(self.source, self.data(path), path)

    def test_fish_clips_rest_pose_and_first_frame(self):
        from rc2016.formats.skeleton import worlds
        library = self.library(self.FISH)
        self.assertEqual(list(library.animset_data), ['characters/ambient/amb_fish_01/amb_fish_01.animset'])
        self.assertEqual(len(library.clips), 5)
        rest = np.stack(worlds(library.rows, (), library.inheritance)) @ np.stack(library.inverse)
        self.assertLess(float(np.abs(rest - np.eye(4)).max()), 1e-5)
        index = library.labels().index('Swimming_Idle_amb_fish_01')
        _clip, kind, frames = library.clip_frames(index)
        self.assertEqual((kind, len(frames)), ('absolute skeletal', 76))
        rows = {r['hash']: r for r in library.rows}
        moved = [np.linalg.norm(np.subtract(j['translation'], rows[j['hash']]['translation']))
                 for j in frames[0] if j['hash'] in rows]
        self.assertLess(float(np.median(moved)), 1e-3)     # frame 0 sits on the rest pose within quantization
        shots = library.with_cinematic()
        self.assertEqual(len(shots.clips), 35)
        self.assertTrue(all(c['animset'].endswith('.cinematic2') for c in shots.clips[5:]))
        self.assertEqual(shots.clip_notes(), [None] * 35)

    def test_animsets_whose_clips_live_in_the_model_folder(self):
        # A zone-placed crowd set no actor names; its standing twin repeats the same clip assets and is left out.
        thug = self.library('characters/enemy/enm_thug_brawler/enm_thug_brawler.model')
        folder = 'characters/enemy/enm_thug_brawler/animations/'
        self.assertIn(folder + 'enm_thug_cheering_siting.animset', thug.animset_data)
        self.assertNotIn(folder + 'enm_thug_cheering.animset', thug.animset_data)
        self.assertIn(folder + 'cheer_idle_enm_thug_brawler_master.animclip', [c['path'] for c in thug.clips])
        # The shark's underwater set (a folder of its own) plays only clips its main set already offers.
        shark = self.library('characters/enemy/enm_pool_shark_ps4/enm_pool_shark_ps4.model')
        self.assertNotIn('characters/enemy/enm_pool_shark_underwater/enm_pool_shark_underwater.animset', shark.animset_data)

    def test_fish_batches_match_the_fixed_weight_records(self):
        data = self.data(self.FISH)
        decoded = rm.decode_model(self.FISH, data=data, as_arrays=True)
        dat1 = data[36:]
        offset, size = rm.dat1_sections(dat1)[rm.WEIGHTS]
        records = np.frombuffer(dat1[offset:offset + size], np.uint8).reshape(-1, 8)
        for mesh in decoded['meshes']:
            ref = records[mesh['first_weight_index']:mesh['first_weight_index'] + mesh['vertex_count']]
            expected = ref[:, 4:] / ref[:, 4:].sum(axis=1, keepdims=True)
            got = np.zeros((mesh['vertex_count'], 115))
            want = np.zeros_like(got)
            rows = np.arange(mesh['vertex_count'])
            np.add.at(got, (np.repeat(rows, 8), mesh['skin_joint_array'].ravel()), mesh['skin_weight_array'].ravel())
            np.add.at(want, (np.repeat(rows, 4), ref[:, :4].ravel().astype(int)), expected.ravel())
            np.testing.assert_allclose(got, want, atol=1e-12)

    def test_ratchet_animates_without_tearing(self):
        from rc2016.viewer.animation import BASIS
        data = self.data(self.RATCHET)
        decoded = rm.decode_model(self.RATCHET, data=data, as_arrays=True, lod='Highest')
        self.assertTrue(all(m['skinned'] for m in decoded['meshes']))
        library = self.library(self.RATCHET)
        self.assertIn('characters/hero/hero_ratchet/hero_ratchet.animset', library.animset_data)
        index = library.labels().index('hero_ratchet: relaxed_neutral_idle_hero_ratchet')
        pose = library.pose(index)['frames'][0].astype(float)
        native = np.einsum('ij,fjk,kl->fil', BASIS.T, pose, BASIS)
        lengths, rest = [], []
        for mesh in decoded['meshes']:
            p = np.c_[mesh['positions'], np.ones(len(mesh['positions']))]
            moved = sum(mesh['skin_weight_array'][:, s, None] * np.einsum('nij,nj->ni', native[mesh['skin_joint_array'][:, s]], p)
                        for s in range(8))[:, :3]
            edges = np.r_[mesh['triangles'][:, :2], mesh['triangles'][:, 1:]]
            lengths.append(np.linalg.norm(moved[edges[:, 0]] - moved[edges[:, 1]], axis=1))
            rest.append(np.linalg.norm(mesh['positions'][edges[:, 0]] - mesh['positions'][edges[:, 1]], axis=1))
        lengths, rest = np.concatenate(lengths), np.concatenate(rest)
        ratio = lengths[rest > 1e-5] / rest[rest > 1e-5]
        self.assertAlmostEqual(float(np.median(ratio)), 1.0, delta=1e-3)
        # The far tail is the armpit pinch of linear blend skinning (sub-millimetre shirt edges on the upper-arm
        # twist joints as the T-pose arms come down), so the bound is on the 99th percentile.
        self.assertLess(float(np.percentile(ratio, 99)), 1.5)   # measured 1.37

    def test_root_motion_plants_ratchets_feet(self):
        # The joint tracks walk and turn on the spot, so a planted foot slides; the root motion moves the body
        # under it. Speed of the slower foot (the planted one), median over the clip, in units per second.
        from rc2016.formats.clips import worlds_batch
        library = self.library(self.RATCHET)
        names = [r['name'] for r in library.rows]
        feet = [names.index('LF_ball'), names.index('RT_ball')]

        def planted(index, root_motion):
            clip, _kind, frames = library.clip_frames(index, root_motion)
            spots = worlds_batch(library.rows, frames, library.inheritance)[:, feet][..., :3, 3]   # (frames, feet, xyz)
            speed = np.linalg.norm(np.diff(spots, axis=0), axis=2) * clip['metadata']['sample_rate_fps']
            return float(np.median(speed.min(axis=1)))

        for name, still in (('walk_forward_wrench_hero_ratchet', 0.15), ('idle_turn_left_90_right_foot_hero_ratchet', 0.02)):
            index = next(i for i, c in enumerate(library.clips) if c['path'].endswith(f'/{name}.animclip'))
            in_place, moved = planted(index, False), planted(index, True)
            self.assertGreater(in_place, 0.3, name)    # measured 1.01 (walk) and 0.47 (turn)
            self.assertLess(moved, still, name)        # measured 0.10 and 0.005
        # A clip without root motion records plays as before (the face lines are checked separately).
        index = library.labels().index('hero_ratchet: relaxed_neutral_idle_hero_ratchet')
        self.assertEqual(library.clip_frames(index, True)[2][0], library.clip_frames(index)[2][0])

    def test_face_performances_open_ratchets_jaw(self):
        from rc2016.games.rcra.clip_library import PERFORMANCE
        library = self.library(self.RATCHET).with_performances()
        faces = [i for i, c in enumerate(library.clips) if c.get('source') == PERFORMANCE]
        self.assertEqual(len(faces), 1535)   # every line of the 'ratc' speaker
        self.assertEqual({library.clips[i]['pose'] for i in faces},
                         {'characters/hero/hero_ratchet/animations/ratchet_phonemes.animclip'})
        self.assertTrue(all(c['animset'].startswith('localization/ratc/') for c in (library.clips[i] for i in faces)))
        index = library.labels().index('ratc_boards: RATC_BOARDS_SLICE1_017')
        clip, kind, frames = library.clip_frames(index)
        self.assertEqual((kind, len(frames), clip['metadata']['sample_rate_fps']), ('face performance', 497, 30.0))
        jaw = next(r for r in library.rows if r['name'] == 'CTR_jaw')
        angles = np.array([np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(row['rotation_xyzw'], jaw['rotation_xyzw']))))))
                           for frame in frames for row in frame if row['hash'] == jaw['hash']])
        self.assertAlmostEqual(float(angles[0]), 0.0, places=3)   # silence: the closed pose
        self.assertGreater(float(angles.max()), 8.0)                # measured 9.4
        self.assertGreater(int((angles > 3).sum()), 50)             # measured 96 of 497 frames
        self.assertIsNone(library.clip_notes()[index])
        # Rivet's actor also offers Ratchet's clips, but her own pose clip is in her folder: only her lines.
        rivet = self.library('characters/hero/hero_rivet/hero_rivet.model').with_performances()
        self.assertEqual({c['pose'] for c in rivet.clips if c.get('source') == PERFORMANCE},
                         {'characters/hero/hero_rivet/animations/rivet_phonemes.animclip'})


if __name__ == '__main__':
    unittest.main()
