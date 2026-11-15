"""Clip playback shared by the browser's viewport and the picker's: speed, loop or play once, frame steps and keys."""
import unittest

from rc2016.app.clip_playback import DEFAULT_SPEED, SPEEDS, ClipPlayback, playback_options, sequence_lengths


class Recorder:
    """Stands in for the viewport, the buttons, the slider and the Tk variables."""

    def __init__(self):
        self.poses, self.value, self.text, self.config = [], 0, '', {}

    def set_pose(self, frame):
        self.poses.append(frame)

    def clear_pose(self):
        pass

    def set_effects_running(self, _running, reset=False):
        pass

    def configure(self, **options):
        self.config.update(options)

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Clock:
    """after() that runs nothing by itself: the test fires the pending tick."""

    def __init__(self):
        self.delays, self.pending = [], None

    def after(self, delay, callback):
        self.delays.append(delay)
        self.pending = callback
        return f'job{len(self.delays)}'

    def after_cancel(self, _job):
        self.pending = None

    def fire(self):
        callback, self.pending = self.pending, None
        callback()


def clip(frames=4, fps=30.0):
    return dict(frames=list(range(frames)), fps=fps)


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.readout = Clock(), Recorder()
        self.playback = ClipPlayback(Recorder(), self.clock, Recorder(), Recorder(), Recorder(), self.readout)

    def test_speed_scales_the_frame_clock_and_is_named_in_the_readout(self):
        self.playback.start(clip())
        self.playback.set_speed(0.5)
        self.assertEqual(self.clock.delays, [33, 67])   # the running tick is rescheduled at the new rate
        self.clock.fire()
        self.assertEqual(self.playback.frame, 1)
        self.assertEqual(self.readout.value, '2/4 @ 30 fps  ·  0.5×')
        self.playback.set_speed(2.0)
        self.assertEqual(self.clock.delays[-1], 17)

    def test_looping_wraps_and_play_once_stops_on_the_last_frame(self):
        self.playback.start(clip(3))
        for _ in range(3):
            self.clock.fire()
        self.assertEqual((self.playback.frame, self.playback.playing), (0, True))
        self.playback.set_loop(False)
        self.clock.fire()
        self.clock.fire()
        self.assertEqual(self.playback.frame, 2)
        self.clock.fire()   # the tick on the last frame stops the clock
        self.assertEqual((self.playback.frame, self.playback.playing, self.clock.pending), (2, False, None))
        self.playback.toggle()   # Play after the end starts the clip again
        self.assertEqual((self.playback.frame, self.playback.playing), (0, True))

    def test_a_step_pauses_and_wraps_round_the_clip(self):
        self.playback.start(clip(3))
        self.playback.step(-1)
        self.assertEqual((self.playback.frame, self.playback.playing), (2, False))
        self.playback.step(1)
        self.assertEqual(self.playback.frame, 0)

    def test_without_a_clip_the_keys_do_nothing(self):
        self.playback.step(1)
        self.playback.toggle()
        self.assertFalse(self.playback.playing)


class PlayAllTests(unittest.TestCase):
    """Play all: every clip of the list's order back to back, decoded on request, on one seconds timeline."""

    def setUp(self):
        self.clock, self.readout, self.position, self.slider = Clock(), Recorder(), Recorder(), Recorder()
        self.requests, self.named = [], []
        self.playback = ClipPlayback(Recorder(), self.clock, Recorder(), self.slider, self.position, self.readout,
                                     request=self.requests.append, on_clip=self.named.append)
        # Clips 7, 3 and 5 in the list's order; 3 has no header length and takes the median (2 s).
        self.playback.offer_all([7, 3, 5], [1.0, None, 2.0], ['a', 'b', 'c'])
        self.poses = {7: clip(3), 3: clip(5), 5: clip(2)}

    def deliver(self):
        """The worker answers the last request (LatestWork keeps only the newest)."""
        index = self.requests[-1]
        self.playback.receive(index, self.poses[index])

    def test_lengths_fall_back_to_the_median_header_length(self):
        self.assertEqual(sequence_lengths([1.0, None, 2.0, 0, 3.0]), [1.0, 2.0, 2.0, 2.0, 3.0])
        self.assertEqual(sequence_lengths([None]), [1.0])
        self.assertEqual(self.slider.config['to'], 5.0)

    def test_play_runs_every_clip_in_order_and_names_each(self):
        self.playback.toggle()
        self.assertEqual((self.requests, self.named), ([7], [7]))
        self.assertIsNone(self.clock.pending)   # the clock waits for the decode
        self.deliver()
        self.assertEqual(self.requests, [7, 3])   # the next clip decodes ahead
        self.assertEqual(self.readout.value, 'Clip 1/3  ·  1/3 @ 30 fps')
        self.deliver()   # arrives early: kept until it is reached
        self.clock.fire()
        self.clock.fire()
        self.assertEqual(self.position.value, 1.0)   # the last frame of clip 1 ends its second
        self.clock.fire()   # on to clip 2, already decoded: no wait
        self.assertEqual((self.named[-1], self.playback.frame, self.requests[-1]), (3, 0, 5))
        self.assertEqual(self.readout.value, 'Clip 2/3  ·  1/5 @ 30 fps')
        self.assertIsNotNone(self.clock.pending)

    def test_loop_wraps_the_whole_list_and_play_once_stops_at_its_end(self):
        self.playback.toggle()
        for _ in range(3):
            self.deliver()
            while self.playback.frame < len(self.playback.animation['frames']) - 1:
                self.clock.fire()
            if self.playback.sequence.position < 2:
                self.clock.fire()
        self.clock.fire()   # past the last clip: loop on, back to the first (still cached)
        self.assertEqual((self.playback.sequence.position, self.playback.frame, self.named[-1]), (0, 0, 7))
        self.playback.set_loop(False)
        self.playback.jump_clip(-1)   # to the last clip
        self.clock.fire()
        self.clock.fire()
        self.assertEqual((self.playback.playing, self.clock.pending), (False, None))
        self.playback.toggle()   # Play after the end starts at the first clip again
        self.assertEqual((self.playback.sequence.position, self.playback.playing), (0, True))

    def test_scrubbing_finds_the_clip_and_frame_and_pauses(self):
        self.position.value = 2.0   # 1 s into clip 2 (3 s long... 2 s by median): its middle frame
        self.playback.scrub()
        self.assertEqual((self.requests, self.named, self.playback.playing), ([3], [3], False))
        self.deliver()
        self.assertEqual((self.playback.sequence.position, self.playback.frame), (1, 2))
        self.position.value = 2.5
        self.playback.scrub()   # the same clip: shown at once, nothing decoded
        self.assertEqual((self.playback.frame, self.requests[-1]), (3, 5))

    def test_a_clip_that_does_not_decode_is_skipped(self):
        self.playback.toggle()
        self.playback.fail(7, 'broken')
        self.assertEqual(self.requests, [7, 3])
        self.deliver()
        self.assertEqual((self.playback.sequence.position, self.named), (1, [7, 3]))

    def test_choosing_one_clip_leaves_play_all(self):
        self.playback.toggle()
        self.deliver()
        self.playback.leave_all()
        self.assertFalse(self.playback.receive(3, self.poses[3]))   # the caller plays it as the chosen clip
        self.playback.start(self.poses[3])
        self.assertEqual(self.readout.value, '1/5 @ 30 fps')

    def test_a_new_order_keeps_the_clip_shown(self):
        self.playback.toggle()
        self.deliver()
        self.assertTrue(self.playback.reorder([5, 7], [2.0, 1.0], ['c', 'a']))
        self.assertEqual((self.playback.sequence.position, self.slider.config['to']), (1, 3.0))
        self.assertFalse(self.playback.reorder([5], [2.0], ['c']))


class PlaybackWidgets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tkinter as tk
        from tkinter import ttk
        try:
            cls.root = tk.Tk()
        except tk.TclError:
            raise unittest.SkipTest("no display for Tk")
        cls.tk, cls.ttk = tk, ttk

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.playback = ClipPlayback(Recorder(), Clock(), Recorder(), Recorder(), Recorder(), Recorder())
        self.playback.start(clip())

    def test_space_and_the_arrow_keys_play_and_step_over_the_bound_widget(self):
        widget = self.tk.Frame(self.root, width=20, height=20, takefocus=1)
        widget.pack()
        self.addCleanup(widget.destroy)
        self.playback.bind_keys(widget)
        widget.focus_force()
        self.root.update()
        widget.event_generate('<space>')
        self.assertFalse(self.playback.playing)
        widget.event_generate('<Right>')
        widget.event_generate('<Right>')
        widget.event_generate('<Left>')
        self.assertEqual(self.playback.frame, 1)

    def test_the_options_apply_their_variables_at_once_and_on_change(self):
        speed = self.tk.StringVar(master=self.root, value='0.25×')
        loop = self.tk.BooleanVar(master=self.root, value=False)
        frame = playback_options(self.tk, self.ttk, self.root, self.playback, speed, loop)
        self.addCleanup(frame.destroy)
        self.assertEqual((self.playback.speed, self.playback.loop), (0.25, False))
        speed.set(DEFAULT_SPEED)
        loop.set(True)
        self.assertEqual((self.playback.speed, self.playback.loop), (1.0, True))
        self.assertEqual(tuple(frame.speed_box.cget('values')), tuple(SPEEDS))


if __name__ == '__main__':
    unittest.main()
