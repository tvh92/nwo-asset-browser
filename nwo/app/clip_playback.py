"""Clip playback on a ModelViewport: Play/Pause, a frame slider and a frame readout. The browser's Animation row and the
clip picker's viewport both play through it, so the frame clock and the effect clock behave the same in both.

The browser also plays every clip in order (``offer_all``): the slider then spans the whole sequence in seconds, from
the clip headers' lengths, and each clip is decoded when it is reached, with the next one decoded ahead."""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict

# Playback speed choices (label -> rate). The rate scales the frame clock only; every source frame is still shown.
SPEEDS = {'0.25×': 0.25, '0.5×': 0.5, '1×': 1.0, '2×': 2.0}
DEFAULT_SPEED = '1×'


def sequence_lengths(seconds) -> list[float]:
    """Timeline length of each clip: its header length, or the median of the known ones (1 s if none) where the
    header gives none or zero, so every clip has room on the slider."""
    known = sorted(s for s in seconds if isinstance(s, (int, float)) and s > 0)
    fallback = float(known[len(known) // 2]) if known else 1.0
    return [float(s) if isinstance(s, (int, float)) and s > 0 else fallback for s in seconds]


class Sequence:
    """Every clip of ``order`` (clip indices) back to back on one timeline. ``position`` is the place in ``order`` of
    the clip shown (-1 before the first); ``want`` is (place, seconds into it) while that clip decodes."""

    def __init__(self, order, seconds, names):
        self.order, self.names = list(order), list(names)
        self.lengths = sequence_lengths(seconds)
        self.starts, total = [], 0.0
        for length in self.lengths:
            self.starts.append(total)
            total += length
        self.total = total
        self.position = -1
        self.want = None
        self.failed = 0   # clips that failed to decode in a row; play-all stops once every clip has
        self.cache = OrderedDict()   # clip index -> pose: the clip shown, the one wanted and the one ahead

    def at(self, seconds: float) -> tuple[int, float]:
        """The place in ``order`` at ``seconds`` on the timeline and how far into that clip."""
        place = min(max(bisect_right(self.starts, seconds) - 1, 0), len(self.order) - 1)
        return place, min(max(seconds - self.starts[place], 0.0), self.lengths[place])

    def keep(self, clip, pose) -> None:
        self.cache[clip] = pose
        self.cache.move_to_end(clip)
        while len(self.cache) > 3:
            self.cache.popitem(last=False)


class ClipPlayback:
    """``animation`` is a pose from ``SourcePreviewLoader.pose``: {'frames': [skin matrices], 'fps': float, ...}.

    ``speed`` scales the frame clock; with ``loop`` off a clip stops on its last frame, and Play starts it again."""

    def __init__(self, viewport, clock, play_button, slider, position, readout, request=None, on_clip=None):
        self.viewport = viewport
        # Play all: ``request(clip)`` decodes a clip, whose pose comes back through ``receive``; ``on_clip(clip)``
        # names the clip play-all moved to.
        self.request, self.on_clip = request, on_clip
        self.sequence = None
        self.shown_time = None
        self.clock = clock   # the Tk widget whose after() drives the frames
        self.play_button, self.slider, self.position, self.readout = play_button, slider, position, readout
        self.animation = None
        self.frame = 0
        self.playing = False
        self.job = None
        self.closed = False
        self.speed = 1.0
        self.loop = True

    def start(self, animation) -> None:
        """Show ``animation`` from its first frame and play it; a one-frame clip is shown still."""
        self._cancel()
        self.sequence = None
        self.animation = animation
        count = len(animation['frames'])
        self.slider.configure(from_=0, to=max(1, count - 1), state='normal')
        self.play_button.configure(state='normal')
        self.show(0)
        self.playing = count > 1
        self.play_button.configure(text='Pause' if self.playing else 'Play')
        self._sync()

    def clear(self) -> None:
        """No clip: the rest pose, with the controls disabled and the readout empty."""
        self.playing = False
        self._cancel()
        self.animation = None
        self.sequence = None
        self.viewport.clear_pose()
        self.play_button.configure(state='disabled', text='Pause')
        self.slider.configure(state='disabled')
        self.readout.set('')
        self._sync()

    def offer_all(self, order, seconds, names) -> None:
        """The rest pose with clips to play: Play plays every clip of ``order`` (clip indices, in the list's order;
        ``seconds`` and ``names`` per clip of ``order``) back to back, and the slider scrubs across all of them."""
        self.clear()
        if not order or self.request is None:
            return
        self.sequence = Sequence(order, seconds, names)
        self.slider.configure(from_=0, to=self.sequence.total, state='normal')
        self.position.set(0)
        self.shown_time = 0.0
        self.play_button.configure(state='normal', text='Play all')

    def reorder(self, order, seconds, names) -> bool:
        """The list's order changed during play-all: keep the clip shown when the new order still has it. False when
        it does not (the caller goes back to the rest pose)."""
        old = self.sequence
        if old is None or old.position < 0 or old.order[old.position] not in order:
            return False
        sequence = Sequence(order, seconds, names)
        sequence.position, sequence.cache = order.index(old.order[old.position]), old.cache
        if old.want is not None and old.order[old.want[0]] in order:
            sequence.want = (order.index(old.order[old.want[0]]), old.want[1])
        self.sequence = sequence
        self.slider.configure(from_=0, to=sequence.total)
        if self.animation:
            self.show(self.frame)
        return True

    def receive(self, clip, pose) -> bool:
        """A decoded clip from ``request``. False outside play-all: the caller plays it as the chosen clip."""
        sequence = self.sequence
        if sequence is None:
            return False
        sequence.keep(clip, pose)
        if sequence.want is not None and sequence.order[sequence.want[0]] == clip:
            self._enter(*sequence.want, pose)
        return True

    def fail(self, clip, error) -> bool:
        """``clip`` did not decode. In play-all, a wanted clip is skipped; False outside play-all."""
        sequence = self.sequence
        if sequence is None:
            return False
        if sequence.want is None or sequence.order[sequence.want[0]] != clip:
            return True   # the clip decoded ahead: tried again when it is reached
        sequence.failed += 1
        if sequence.failed >= len(sequence.order):
            self.pause()
            sequence.want = None
            self.readout.set(f'No clip plays: {error}')
            return True
        self._seek(sequence.starts[(sequence.want[0] + 1) % len(sequence.order)])
        return True

    def leave_all(self) -> None:
        """A single clip was chosen: play-all stops, and the clip shown stays until the chosen one arrives."""
        self.pause()
        self.sequence = None

    def jump_clip(self, offset: int) -> bool:
        """In play-all, move to the start of the clip ``offset`` places away, wrapping; playing stays as it was."""
        sequence = self.sequence
        if sequence is None:
            return False
        place = sequence.want[0] if sequence.want is not None else sequence.position
        place = (0 if offset > 0 else -1) if place < 0 else place + offset
        self._seek(sequence.starts[place % len(sequence.order)])
        return True

    def pause(self) -> None:
        """Stop the frame clock; the clip stays on its current frame."""
        self.playing = False
        self.play_button.configure(text='Play')
        self._sync()

    def toggle(self) -> None:
        sequence = self.sequence
        if sequence is not None and not self.playing:
            self.playing = True
            self.play_button.configure(text='Pause')
            if sequence.position < 0 and sequence.want is None:
                self._seek(0.0)   # Play all from the first clip
            elif not self.loop and self._at_end():
                self._seek(0.0)   # played once to the end: start again
            else:
                self._sync()
            return
        if not self.animation and sequence is None:
            return
        self.playing = not self.playing
        if self.playing and not self.loop and self.frame == len(self.animation['frames']) - 1:
            self.show(0)   # a clip that played once and stopped starts again
        self.play_button.configure(text='Pause' if self.playing else 'Play')
        self._sync()

    def step(self, offset: int) -> None:
        """Pause and move ``offset`` frames, wrapping round the clip."""
        if not self.animation:
            return
        self.pause()
        self.show(self.frame + offset)

    def set_speed(self, speed: float) -> None:
        """A new clock rate; a playing clip keeps its frame and continues at the new rate."""
        self.speed = speed
        if self.job is not None:
            self._cancel()
            self._sync()

    def set_loop(self, loop: bool) -> None:
        self.loop = loop

    def scrub(self, _value=None) -> None:
        """The slider moved: show that frame and pause."""
        if self.sequence is not None:
            seconds = float(self.position.get())
            if self.shown_time is None or abs(seconds - self.shown_time) > 1e-6:
                self.pause()
                self._seek(seconds)
            return
        if not self.animation:
            return
        frame = int(round(self.position.get()))
        if frame != self.frame:
            self.pause()
            self.show(frame)

    def show(self, frame: int) -> None:
        count = len(self.animation['frames'])
        self.frame = frame % count
        morph = getattr(self.viewport, 'set_morph', None)
        if morph is not None:   # blend-shape weights of the frame (Rift Apart face clips), else the rest shapes
            weights = self.animation.get('morph_weights')
            morph(None if weights is None else weights[self.frame])
        self.viewport.set_pose(self.animation['frames'][self.frame])
        rate = '' if self.speed == 1.0 else f'  ·  {self.speed:g}×'
        text = f'{self.frame + 1}/{count} @ {self.animation["fps"]:g} fps{rate}'
        sequence = self.sequence
        if sequence is None or sequence.position < 0:
            self.position.set(self.frame)
            self.readout.set(text)
            return
        place = sequence.position
        share = self.frame / (count - 1) if count > 1 else 0.0
        self.shown_time = sequence.starts[place] + share * sequence.lengths[place]
        self.position.set(self.shown_time)
        self.readout.set(f'Clip {place + 1}/{len(sequence.order)}  ·  {text}')

    def bind_keys(self, widget) -> None:
        """Space plays or pauses, Left and Right step one frame, while ``widget`` has the keyboard focus."""
        def key(action):
            def handler(_event):
                action()
                return 'break'
            return handler
        widget.bind('<space>', key(self.toggle))
        widget.bind('<Left>', key(lambda: self.step(-1)))
        widget.bind('<Right>', key(lambda: self.step(1)))

    def close(self) -> None:
        self.closed = True
        self.pause()

    def _seek(self, seconds: float) -> None:
        """Play-all: show the clip and frame at ``seconds`` on the timeline, asking for the clip when it is not
        decoded yet; the frame clock waits for it."""
        sequence = self.sequence
        place, offset = sequence.at(seconds)
        clip = sequence.order[place]
        if place == sequence.position and self.animation is not None:
            sequence.want = None
            self._enter(place, offset, self.animation)
            return
        if self.on_clip is not None and (sequence.want is None or sequence.want[0] != place):
            self.on_clip(clip)   # the Animation button names the clip at once, before it decodes
        sequence.want = (place, offset)
        self._cancel()
        if clip in sequence.cache:
            self._enter(place, offset, sequence.cache[clip])
            return
        self.readout.set(f'Clip {place + 1}/{len(sequence.order)}  ·  decoding…')
        self.request(clip)

    def _enter(self, place: int, offset: float, pose) -> None:
        sequence = self.sequence
        sequence.want, sequence.position, sequence.failed = None, place, 0
        sequence.keep(sequence.order[place], pose)
        self.animation = pose
        count = len(pose['frames'])
        share = offset / sequence.lengths[place] if sequence.lengths[place] else 0.0
        self.show(min(count - 1, int(round(share * (count - 1)))))
        self._sync()
        ahead = place + 1
        if ahead < len(sequence.order) or self.loop:   # decode the next clip while this one plays
            following = sequence.order[ahead % len(sequence.order)]
            if following not in sequence.cache:
                self.request(following)

    def _at_end(self) -> bool:
        sequence = self.sequence
        return (sequence is not None and sequence.position == len(sequence.order) - 1 and self.animation is not None
                and self.frame == len(self.animation['frames']) - 1)

    def _sync(self) -> None:
        """Time-driven materials move only while a clip plays; the frame clock runs only while playing (in play-all,
        not while the next clip decodes)."""
        waiting = self.sequence is not None and self.sequence.want is not None
        self.viewport.set_effects_running(bool(self.playing and self.animation), reset=not self.animation)
        if self.playing and self.animation and not waiting and self.job is None and not self.closed:
            self._schedule()
        elif not self.playing or waiting:
            self._cancel()

    def _schedule(self) -> None:
        delay = max(10, int(round(1000 / max(1.0, self.animation['fps'] * self.speed))))
        self.job = self.clock.after(delay, self._tick)

    def _tick(self) -> None:
        self.job = None
        if self.playing and self.animation and not self.closed:
            sequence = self.sequence
            if sequence is not None and self.frame == len(self.animation['frames']) - 1:
                if self._at_end() and not self.loop:
                    self.pause()   # played every clip once: stay on the last frame
                    return
                self._seek(sequence.starts[(sequence.position + 1) % len(sequence.order)])   # the next clip
                return
            if not self.loop and self.frame == len(self.animation['frames']) - 1:
                self.pause()   # played once: stay on the last frame
                return
            self.show(self.frame + 1)
            self._schedule()

    def _cancel(self) -> None:
        if self.job is not None:
            self.clock.after_cancel(self.job)
            self.job = None


def playback_options(tk, ttk, parent, playback, speed_var=None, loop_var=None):
    """A frame with the Speed choice and the Loop box for ``playback``; the caller places it. ``speed_var`` (a speed
    label) and ``loop_var`` may be remembered variables; their values apply at once and on every change."""
    speed_var = speed_var or tk.StringVar(master=parent, value=DEFAULT_SPEED)
    loop_var = loop_var if loop_var is not None else tk.BooleanVar(master=parent, value=True)
    frame = ttk.Frame(parent)
    ttk.Label(frame, text='Speed').pack(side='left')
    box = ttk.Combobox(frame, textvariable=speed_var, values=tuple(SPEEDS), state='readonly', width=6)
    box.pack(side='left', padx=(4, 8))
    check = ttk.Checkbutton(frame, text='Loop', variable=loop_var)
    check.pack(side='left')

    def apply(*_args):
        playback.set_speed(SPEEDS.get(speed_var.get(), 1.0))
        playback.set_loop(bool(loop_var.get()))
        if playback.animation:
            playback.show(playback.frame)   # the readout names the rate

    speed_var.trace_add('write', apply)
    loop_var.trace_add('write', apply)
    apply()
    frame.speed_box, frame.loop_check = box, check
    return frame
