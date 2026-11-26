"""Rift Apart visual-effect particles at one moment: a deterministic replay of the game's spawn and update code for an
emitter, for the viewer's still (docs/rift-apart.md, Particles).

Traced (RiftApart.exe):

- **Spawn count** (0x14137C7B0): per frame ``carry + rate x dt`` new particles, the fraction carried over, rate = emitter
  slot 4 (per second); slot 2 bursts once when the emitter time passes its keys. Nothing spawns while the emitter's age
  is below 0: it starts at minus the spawn delay (+0x6C, random part +0x70). Rate spawning stops when the emitter's
  lifetime (+0x64, random part +0x68; 0 infinite) ends. +0x11C caps the particles alive.
- **New particle** (0x141395800): the shape sampler (the record the emitter's +0x118 indexes) gives a local position
  and direction, both taken into the emitter frame (rows of the emitter's 3x3, its position, slot 8's offset); velocity
  = direction x slot 1's x (speed); life = slot 6 (seconds); initial rotation = random x +0x80 + +0x8C (radians).
  Spawn-time curves read the emitter's age in seconds and its normalised age (0 for an infinite emitter).
- **Update** (0x141391F00): t = min(age / life, 0.999999); velocity += dt x accel direction (+0x30) x slot 11, then
  times the drag factor exp(-dt x +0x54), plus a gravity step (0, -dt x +0x4C, 0) in world axes; position += dt x
  velocity x slot 13. Size = slot 17 x slot 0 (slot 0 read at the spawn time on the emitter's timeline). Rotation
  integrates slot 15 as degrees per second (absolute with node flag 0x800). Colour = slot 18 x slot 14, alpha = slot 19
  x (its y the alpha-test threshold), custom = slot 16.

Viewer choices (labelled in the material notes):

- **Random values:** the game draws from a 255-entry table with a per-particle LCG; the replay uses a fixed
  low-discrepancy sequence per particle and draw (``uniform``), so spreads look even, not as in a given game frame.
  Curve random values (key modes) use RANDOM.
- **Time step:** STEP seconds; the game steps by its frame time.
- **Not applied:** wind (+0x50 x a run-time vector), turbulence (slot 12), following (slot 10), inherited emitter velocity,
  per-metre spawning (a still emitter), random sign flips of rotation and size, and slot 7's emitter rotation.
"""
import math

import numpy as np

from nwo.games.rcra import vfx

STEP = 1.0 / 60.0
RANDOM = 0.5
MAX_PARTICLES = 256
INFINITE = 2678400.0   # an emitter lifetime of 0 or below (0x1413718CA)
# Fractional parts of these step a particle's random draws (a Weyl sequence per draw index).
_STEPS = (0.6180339887, 0.7548776662, 0.5698402910, 0.3247179572, 0.4142135624, 0.7320508076, 0.2360679775,
          0.6457513111, 0.1622776602, 0.4721359550)


def uniform(particle, draw):
    """The replay's random value in [0, 1) for a particle's draw (a fixed sequence standing in for the game's table)."""
    return (0.5 + (particle + 1) * _STEPS[draw % len(_STEPS)]) % 1.0


def _floats(shape):
    """{byte offset: f32} of a shape record from +0x30 to +0x74."""
    values = np.frombuffer(shape['raw'], '<f4', 18, 0x30)
    return {0x30 + 4 * i: float(v) for i, v in enumerate(values)}


def euler(x, y, z):
    """Rotation about X, then Y, then Z (radians); the game's spread order is not traced."""
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def sample_shape(shape, scales, particle):
    """(local position, local direction) of a new particle from its shape record (docs/rift-apart.md, Spawn shapes), or
    None for a sampler the replay does not follow (box and model-surface shapes)."""
    f = _floats(shape)
    sx, sy, sz = scales
    rand = [uniform(particle, d) for d in range(6)]
    dm = f[0x48] + (f[0x4C] if rand[5] > 0.5 else 0.0)   # outward, inward or a random sign
    off = np.array([f[0x6C], f[0x70], f[0x74]])
    kind = shape['kind'] or 'cone'
    if kind == 'cone':
        theta = math.radians(f[0x38] + rand[0] * f[0x3C])
        phi = math.radians(f[0x40] + rand[1] * f[0x44] + 180.0)
        d = np.array([math.cos(phi) * math.sin(theta), math.sin(phi) * math.sin(theta), math.cos(theta)])
        r = f[0x50] + rand[2] * f[0x54]
        return (d * r + off) * f[0x34] * np.array(scales), d * dm
    if kind == 'disc':
        phi = math.radians(f[0x40] + rand[1] * f[0x44])
        r = f[0x50] + rand[2] * f[0x54]
        position = np.array([(f[0x6C] - math.sin(phi) * r) * f[0x34] * sx, (math.cos(phi) * r + f[0x70]) * f[0x34] * sz,
                             (rand[3] + f[0x74]) * f[0x30] * sy])
        a = math.radians(f[0x38] + rand[0] * f[0x3C]) * dm
        return position, np.array([-math.sin(a) * math.sin(phi), math.sin(a) * math.cos(phi), math.cos(a)])
    if kind == 'plane':
        u, v = 2.0 * rand[0] - 1.0, 2.0 * rand[1] - 1.0
        position = np.array([(u + f[0x6C]) * f[0x58] * sx, (v + f[0x70]) * f[0x5C] * sy, 0.0])
        return position, np.array([0.0, 0.0, dm])
    return None


def direction_spread(shape, direction, particle):
    """The shape's direction spread (+0x68 degrees, each Euler angle uniform in +/- the spread) and its uniform
    direction (+0x28 set: the vector at +0x10)."""
    if struct_u32(shape['raw'], 0x28):
        direction = np.frombuffer(shape['raw'], '<f4', 3, 0x10).astype(np.float64)
    spread = float(np.frombuffer(shape['raw'], '<f4', 1, 0x68)[0])
    if spread:
        angles = [math.radians((2.0 * uniform(particle, 6 + k) - 1.0) * spread) for k in range(3)]
        direction = euler(*angles) @ direction
    return direction


def struct_u32(raw, offset):
    return int(np.frombuffer(raw, '<u4', 1, offset)[0])


def emitter_value(effect, emitter, name, normalised, seconds, random=RANDOM):
    """An emitter slot's curve value at a normalised time and seconds (the slot's time rate picks between them)."""
    index = emitter['curves'][name][0]
    time = vfx.curve_time(emitter['rates'][name], seconds, normalised)
    return vfx.curve_value(effect['curves'][index], time, random)


def _spawn_times(effect, emitter, end):
    """Emitter ages (seconds) at which particles spawn up to ``end`` (an emitter age), as the spawn loop counts them."""
    duration = emitter['lifetime'] if emitter['lifetime'] > 0 else INFINITE
    times, carry, age = [], 0.0, 0.0
    normal = 0.0 if duration >= INFINITE else age / duration
    burst = int(emitter_value(effect, emitter, 'burst', normal, age)[0])
    times += [0.0] * max(burst, 0)
    while age < min(end, duration) and len(times) < 4 * MAX_PARTICLES:
        normal = 0.0 if duration >= INFINITE else age / duration
        carry += max(emitter_value(effect, emitter, 'rate', normal, age)[0], 0.0) * STEP
        count = int(carry)
        carry -= count
        times += [age] * count
        age += STEP
    return times, duration


def still_age(effect, emitter):
    """The emitter age the viewer shows: one particle life after the first spawn (a steady state for a continuous
    emitter), within its lifetime; half a life for an emitter that only bursts. A viewer choice."""
    life = max(emitter_value(effect, emitter, 'life', 0.0, 0.0)[0], STEP)
    rate = emitter_value(effect, emitter, 'rate', 0.0, 0.0)[0]
    duration = emitter['lifetime'] if emitter['lifetime'] > 0 else INFINITE
    return min(life if rate > 0 else 0.5 * life, duration)


def simulate(effect, emitter, world_to_effect, age=None, node_flags=0):
    """The emitter's live particles at emitter age ``age`` (still_age when None), in effect space:
    [dict(age, life, t, position, velocity, size (xyz), rotation (xyz radians), colour (rgb x luminance), alpha,
    alpha_test, custom)].

    ``world_to_effect`` (3x3) takes the native world axes (gravity, acceleration) into effect space. Returns (particles,
    notes): notes name the parts of the emitter the replay leaves out."""
    notes = []
    age = still_age(effect, emitter) if age is None else age
    times, duration = _spawn_times(effect, emitter, age)
    shapes = effect.get('shapes') or []
    shape = shapes[emitter['shape']] if emitter['shape'] < len(shapes) else None
    if shape is not None and shape['kind'] not in (None, 'cone', 'disc', 'plane'):
        notes.append('%s spawn shape at the emitter origin' % shape['kind'])
    rows = np.asarray(emitter['rotation'], np.float64).reshape(3, 3)
    origin = np.asarray(emitter['position'], np.float64)
    accel_dir = world_to_effect @ np.asarray(emitter['accel_direction'], np.float64)
    gravity = world_to_effect @ np.array([0.0, -emitter['gravity'], 0.0])
    drag = math.exp(-STEP * emitter['drag'])
    absolute_rotation = bool(node_flags & 0x800)
    if any(emitter_value(effect, emitter, 'emitter_rotation', 0.0, 0.0)):
        notes.append('emitter rotation curve')
    particles = []
    for k, born in enumerate(times):
        lived = age - born
        normal = 0.0 if duration >= INFINITE else born / duration
        life = emitter_value(effect, emitter, 'life', normal, born, uniform(k, 7))[0]
        if not (life > 0 and 0.0 <= lived < life):
            continue
        speed = emitter_value(effect, emitter, 'speed', normal, born, uniform(k, 8))[0]
        scales = emitter_value(effect, emitter, 'shape_scales', normal, born)
        sampled = sample_shape(shape, scales, k) if shape is not None else None
        local, direction = sampled if sampled is not None else (np.zeros(3), np.array([0.0, 0.0, 1.0]))
        if shape is not None:
            direction = direction_spread(shape, direction, k)
        offset = np.asarray(emitter_value(effect, emitter, 'emitter_position', normal, born), np.float64)
        position = local @ rows + origin + offset
        velocity = (direction @ rows) * speed
        size_at_spawn = np.asarray(emitter_value(effect, emitter, 'size_at_spawn', normal, born), np.float64)
        rotation = np.array([uniform(k, 9) * r + b for r, b in zip(emitter['rotation_range'], emitter['rotation_base'])])
        spun = np.zeros(3)
        steps = max(int(round(lived / STEP)), 0)
        for step in range(steps):
            t = min(step * STEP / life, 0.999999)
            seconds = step * STEP
            accel = emitter_value(effect, emitter, 'acceleration', t, seconds)[0]
            multiplier = emitter_value(effect, emitter, 'velocity', t, seconds)[0]
            position = position + STEP * velocity * multiplier
            velocity = (velocity + STEP * accel_dir * accel) * drag + STEP * gravity
            if not absolute_rotation:
                spun += np.radians(emitter_value(effect, emitter, 'rotation', t, seconds))[:3] * STEP
        t = min(lived / life, 0.999999)

        def now(name):
            return np.asarray(emitter_value(effect, emitter, name, t, lived), np.float64)
        rotation = rotation + (np.radians(now('rotation')) if absolute_rotation else spun)
        colour = now('color') * now('luminance')[0]
        alpha = now('alpha')
        particles.append(dict(age=lived, life=life, t=t, position=position, velocity=velocity,
                              size=now('scales') * size_at_spawn, rotation=rotation, colour=colour,
                              alpha=float(alpha[0]), alpha_test=float(alpha[1]), custom=now('custom')))
        if len(particles) >= min(MAX_PARTICLES, max(emitter['max_alive'], 1)):
            break
    return particles, notes
