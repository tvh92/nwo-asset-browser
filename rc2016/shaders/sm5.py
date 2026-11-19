"""Shader Model 5 program -> typed expression DAG, with engine inputs named from the program's reflection.

Every SM5 register component becomes a node of a hash-consed DAG. Node types are 'f' (32-bit float), 'i' (32-bit
integer bits) and 'b' (a comparison result; SM5 stores it as 0xFFFFFFFF/0). A register keeps whatever node was written;
reading it with another type inserts a bitcast, which folds for constants. Constant folding uses float32 arithmetic.

Structured control flow is linearised by predication: if/else, switch and return become select(), discard an accumulated
discard predicate. A discarded pixel's registers are don't-care (its target is dropped), so a discard does not predicate
later register writes: lanes keep computing as if it had not happened, which keeps a register one type when a program
reuses it for a mask before the discard and a float after it. UAV stores stay gated by the discard.

A loop is unrolled while its continuation predicate folds to a constant (so a light loop over an empty light list
disappears when its count is bound to 0); a loop whose trip count stays unknown poisons every register it writes with an
'unknown' node, so outputs that do not depend on it still translate. Branch, break, return and discard conditions only
act on active lanes, so they are folded under the facts the lane's active predicate implies (_given_active): a counter
written as select(active, i + 1, i) is exactly the iteration count there, and a loop with an early break still ends at
its count. Facts carry over iterations (a lane active now passed the previous iteration's exit test), and a test that
one select condition decides splits on it, so a parallax counter that jumps past its step count on a hit ends at the
count's bound; extra predicated iterations are allowed only up to that bound (taken from the unspecialised exit test
while the specialised one has decided the counter test). A counter that differs between a continue path and the
fall-through path does not fold this way.

Constant buffer reads become 'input' nodes named '<cbuffer>.<variable>+<byte offset>' from the RDEF reflection;
interpolants become 'input' nodes named '<semantic>.<component>'; texture samples and buffer loads become 'sample'/'load'
nodes (four 'channel' nodes each) carrying the binding name. A ``Bindings`` object can replace any of these with values
during translation; that is how neutral engine inputs (no lights, default probe) are applied.
"""
from __future__ import annotations

import math
import struct

import numpy as np

from rc2016.shaders import dxbc

U32 = 0xFFFFFFFF
MAX_LOOP_ITERATIONS = 65           # unrolled iterations while the loop's continuation folds (64 steps + exit test)
MAX_PREDICATED_ITERATIONS = 8      # further iterations allowed while it does not (data-dependent trip count)
TRUE_BITS = U32


def f32(x):
    with np.errstate(all='ignore'):
        return float(np.float32(x))


def float_bits(x):
    return struct.unpack('<I', struct.pack('<f', np.float32(x)))[0]


def bits_float(b):
    return float(np.frombuffer(struct.pack('<I', b & U32), dtype=np.float32)[0])


def _signed(b):
    b &= U32
    return b - (1 << 32) if b & 0x80000000 else b


def _round_ne(x):
    return float(np.rint(np.float32(x)))


def _half_bits(x):
    return int(np.array(np.float32(x), dtype=np.float16).view(np.uint16))


def _half_float(b):
    return float(np.array(b & 0xFFFF, dtype=np.uint16).view(np.float16))


def _ftoi(x, low, high):
    if math.isnan(x):
        return 0
    if math.isinf(x):
        return high if x > 0 else low
    return int(max(low, min(high, math.trunc(x))))


def _bfe(width, offset, value, signed):
    width &= 31
    offset &= 31
    if width == 0:
        return 0
    if width + offset < 32:
        field = (value << (32 - width - offset)) & U32
    else:
        field = value & U32
    shift = 32 - width
    return (_signed(field) >> shift) & U32 if signed else field >> shift


def _f(op):
    def run(a):
        with np.errstate(all='ignore'):
            return float(op(*[np.float32(x) for x in a]))
    return run


# op: (result type, argument types, fold function on Python values)
OPS = {
    'add': ('f', 'ff', _f(lambda a, b: a + b)),
    'mul': ('f', 'ff', _f(lambda a, b: a * b)),
    'div': ('f', 'ff', _f(lambda a, b: a / b)),
    'min': ('f', 'ff', _f(np.fmin)),
    'max': ('f', 'ff', _f(np.fmax)),
    'neg': ('f', 'f', _f(lambda a: -a)),
    'abs': ('f', 'f', _f(abs)),
    'sat': ('f', 'f', _f(lambda a: np.float32(0) if np.isnan(a) else np.clip(a, 0, 1))),
    'sqrt': ('f', 'f', _f(np.sqrt)),
    'rsq': ('f', 'f', _f(lambda a: np.float32(1) / np.sqrt(a))),
    'rcp': ('f', 'f', _f(lambda a: np.float32(1) / a)),
    'log2': ('f', 'f', _f(np.log2)),
    'exp2': ('f', 'f', _f(np.exp2)),
    'frc': ('f', 'f', _f(lambda a: a - np.floor(a))),
    'round_ne': ('f', 'f', _f(np.rint)),
    'round_ni': ('f', 'f', _f(np.floor)),
    'round_pi': ('f', 'f', _f(np.ceil)),
    'round_z': ('f', 'f', _f(np.trunc)),
    'sin': ('f', 'f', _f(np.sin)),
    'cos': ('f', 'f', _f(np.cos)),
    'ddx': ('f', 'f', None), 'ddy': ('f', 'f', None),
    'eq': ('b', 'ff', lambda a: a[0] == a[1]),
    'ne': ('b', 'ff', lambda a: not (a[0] == a[1])),
    'lt': ('b', 'ff', lambda a: a[0] < a[1]),
    'ge': ('b', 'ff', lambda a: a[0] >= a[1]),
    'b_and': ('b', 'bb', lambda a: a[0] and a[1]),
    'b_or': ('b', 'bb', lambda a: a[0] or a[1]),
    'b_not': ('b', 'b', lambda a: not a[0]),
    'iadd': ('i', 'ii', lambda a: (a[0] + a[1]) & U32),
    'imul': ('i', 'ii', lambda a: (_signed(a[0]) * _signed(a[1])) & U32),
    'umul_hi': ('i', 'ii', lambda a: (a[0] * a[1]) >> 32),
    'imul_hi': ('i', 'ii', lambda a: ((_signed(a[0]) * _signed(a[1])) >> 32) & U32),
    'udiv': ('i', 'ii', lambda a: U32 if a[1] == 0 else a[0] // a[1]),
    'urem': ('i', 'ii', lambda a: U32 if a[1] == 0 else a[0] % a[1]),
    'ineg': ('i', 'i', lambda a: (-a[0]) & U32),
    'iabs': ('i', 'i', lambda a: abs(_signed(a[0])) & U32),
    'and': ('i', 'ii', lambda a: a[0] & a[1]),
    'or': ('i', 'ii', lambda a: a[0] | a[1]),
    'xor': ('i', 'ii', lambda a: a[0] ^ a[1]),
    'not': ('i', 'i', lambda a: ~a[0] & U32),
    'ishl': ('i', 'ii', lambda a: (a[0] << (a[1] & 31)) & U32),
    'ishr': ('i', 'ii', lambda a: (_signed(a[0]) >> (a[1] & 31)) & U32),
    'ushr': ('i', 'ii', lambda a: a[0] >> (a[1] & 31)),
    'imin': ('i', 'ii', lambda a: min(a, key=_signed)),
    'imax': ('i', 'ii', lambda a: max(a, key=_signed)),
    'umin': ('i', 'ii', min), 'umax': ('i', 'ii', max),
    'ieq': ('b', 'ii', lambda a: a[0] == a[1]),
    'ine': ('b', 'ii', lambda a: a[0] != a[1]),
    'ilt': ('b', 'ii', lambda a: _signed(a[0]) < _signed(a[1])),
    'ige': ('b', 'ii', lambda a: _signed(a[0]) >= _signed(a[1])),
    'ult': ('b', 'ii', lambda a: a[0] < a[1]),
    'uge': ('b', 'ii', lambda a: a[0] >= a[1]),
    'itof': ('f', 'i', lambda a: f32(_signed(a[0]))),
    'utof': ('f', 'i', lambda a: f32(a[0])),
    'ftoi': ('i', 'f', lambda a: _ftoi(a[0], -2**31, 2**31 - 1) & U32),
    'ftou': ('i', 'f', lambda a: _ftoi(a[0], 0, U32)),
    'f32tof16': ('i', 'f', lambda a: _half_bits(a[0])),
    'f16tof32': ('f', 'i', lambda a: f32(_half_float(a[0]))),
    'ubfe': ('i', 'iii', lambda a: _bfe(a[0], a[1], a[2], False)),
    'ibfe': ('i', 'iii', lambda a: _bfe(a[0], a[1], a[2], True)),
    'bfi': ('i', 'iiii', lambda a: (lambda m: ((a[2] << (a[1] & 31)) & m) | (a[3] & ~m & U32))(
        (((1 << (a[0] & 31)) - 1) << (a[1] & 31)) & U32)),
    'bfrev': ('i', 'i', lambda a: int('{:032b}'.format(a[0])[::-1], 2)),
    'countbits': ('i', 'i', lambda a: bin(a[0]).count('1')),
    'firstbit_hi': ('i', 'i', lambda a: U32 if a[0] == 0 else 31 - (a[0].bit_length() - 1)),
    'firstbit_lo': ('i', 'i', lambda a: U32 if a[0] == 0 else (a[0] & -a[0]).bit_length() - 1),
    'firstbit_shi': ('i', 'i', lambda a: (lambda v: U32 if v in (0, U32) else 31 - ((v if not v & 0x80000000 else ~v & U32).bit_length() - 1))(a[0])),
    'f2bits': ('i', 'f', lambda a: float_bits(a[0])),
    'bits2f': ('f', 'i', lambda a: bits_float(a[0])),
    'b2i': ('i', 'b', lambda a: U32 if a[0] else 0),
}
# select(cond: b, a, b) has the type of its operands; 'channel' takes its sample/load node's channel type.


class Unknown(Exception):
    pass


class Graph:
    """Hash-consed typed DAG. Nodes are dicts: op, type, args, and op-specific fields (value, name, reason...)."""

    def __init__(self):
        self.nodes = []
        self._cache = {}
        self.predicated = []   # per node: a select or a truth value in its subtree (only those change under facts)

    def _add(self, op, type_, args=(), **meta):
        key = (op, type_, tuple(args), tuple(sorted(meta.items())))
        index = self._cache.get(key)
        if index is None:
            index = self._cache[key] = len(self.nodes)
            self.nodes.append(dict(op=op, type=type_, args=list(args), **meta))
            self.predicated.append(op == 'select' or type_ == 'b' or any(self.predicated[a] for a in args))
        return index

    # -- leaves
    def const(self, type_, value):
        if type_ == 'f':
            value = f32(value)
            index = self._add('const', 'f', bits=float_bits(value))
            self.nodes[index]['value'] = value
            return index
        if type_ == 'b':
            return self._add('const', 'b', value=bool(value))
        return self._add('const', 'i', value=int(value) & U32)

    def input(self, name, type_='f'):
        return self._add('input', type_, name=name)

    def unknown(self, reason, type_='f'):
        return self._add('unknown', type_, reason=reason)

    def value(self, index):
        n = self.nodes[index]
        return n['value'] if n['op'] == 'const' else None

    def is_const(self, index):
        return self.nodes[index]['op'] == 'const'

    def type(self, index):
        return self.nodes[index]['type']

    # -- conversions
    def cast(self, index, type_):
        have = self.type(index)
        if have == type_:
            return index
        memo = self.__dict__.setdefault('_casts', {})   # pure in (node, type); selects share branches, so cast once
        key = (index, type_)
        if key in memo:
            return memo[key]
        if self.nodes[index]['op'] == 'select' and 'b' not in (have, type_):
            # Unrolled loops leave select chains hundreds deep: cast their branches bottom-up, without recursion.
            stack = [(index, False)]
            while stack:
                i, ready = stack.pop()
                if (i, type_) in memo:
                    continue
                n = self.nodes[i]
                branches = [a for a in n['args'][1:] if self.type(a) != type_ and self.nodes[a]['op'] == 'select']
                if ready or not branches:
                    memo[(i, type_)] = self._cast(i, type_, self.type(i))
                else:
                    stack.append((i, True))
                    stack.extend((a, False) for a in branches if (a, type_) not in memo)
            return memo[key]
        memo[key] = self._cast(index, type_, have)
        return memo[key]

    def _cast(self, index, type_, have):
        if have == 'b':
            index, have = self.op('b2i', index), 'i'
            if type_ == 'i':
                return index
        if type_ == 'b':
            return self.op('ine', self.cast(index, 'i'), self.const('i', 0))
        n = self.nodes[index]
        if n['op'] == 'select':
            c, a, b = n['args']
            return self.select(c, self.cast(a, type_), self.cast(b, type_))
        if have == 'f' and type_ == 'i':
            if n['op'] == 'bits2f':
                return n['args'][0]
            return self.op('f2bits', index)
        n = self.nodes[index]
        if n['op'] == 'f2bits':
            return n['args'][0]
        return self.op('bits2f', index)

    def boolean(self, index):
        """The truth of a register value as SM5 tests it (any bit set)."""
        n = self.nodes[index]
        if n['type'] == 'b':
            return index
        if n['op'] == 'b2i':
            return n['args'][0]
        if n['op'] == 'f2bits' and self.nodes[n['args'][0]]['op'] == 'bits2f':
            return self.boolean(self.nodes[n['args'][0]]['args'][0])
        if n['type'] == 'f' and n['op'] == 'bits2f':
            return self.boolean(n['args'][0])
        if n['op'] == 'select':
            c, a, b = n['args']
            av, bv = self.value(a), self.value(b)
            if av is not None and bv is not None:
                at, bt = (av != 0, bv != 0) if n['type'] != 'f' else (float_bits(av) != 0, float_bits(bv) != 0)
                if at and not bt:
                    return c
                if bt and not at:
                    return self.op('b_not', c)
        return self.op('ine', self.cast(index, 'i'), self.const('i', 0))

    # -- operations with folding
    def op(self, name, *args):
        result, arg_types, fold = OPS[name]
        args = tuple(self.cast(a, t) for a, t in zip(args, arg_types))
        values = [self.value(a) for a in args]
        if fold is not None and None not in values:
            try:
                return self.const(result, fold(values))
            except (ValueError, OverflowError, ZeroDivisionError):
                pass
        simplified = self._simplify(name, args, values)
        if simplified is not None:
            return simplified
        return self._add(name, result, args)

    def never_negative(self, index, depth=8):
        """True when the float value is >= 0, -0 or NaN: never less than 0 (comparisons with NaN are false)."""
        n = self.nodes[index]
        op = n['op']
        if op == 'const':
            return n['type'] == 'f' and not n['value'] < 0.0
        if op in ('sat', 'abs', 'sqrt', 'rsq', 'exp2'):
            return True
        if depth == 0:
            return False
        args = n['args']
        if op == 'mul' and any(self.value(x) == 0.0 for x in args):
            return True   # +-0 or NaN
        if op in ('add', 'mul', 'div', 'min'):
            return all(self.never_negative(x, depth - 1) for x in args)
        if op == 'max':   # D3D max returns the other operand for NaN
            return any(self.never_negative(x, depth - 1) for x in args)
        if op == 'select':
            return all(self.never_negative(x, depth - 1) for x in args[1:])
        return False

    def _simplify(self, name, args, values):
        a = args
        v = values
        if name == 'lt' and v[1] is not None and v[1] <= 0.0 and self.never_negative(a[0]):
            return self.const('b', False)
        if name in ('ushr', 'ishr', 'ishl') and v[0] == 0:
            return self.const('i', 0)   # a shifted zero
        if name == 'mul':
            for x, y in ((0, 1), (1, 0)):
                if v[x] == 1.0:
                    return a[y]
        if name == 'add':
            for x, y in ((0, 1), (1, 0)):
                if v[x] is not None and v[x] == 0.0 and float_bits(v[x]) == 0x80000000:
                    return a[y]   # -0 + y == y exactly
        if name in ('b_and', 'b_or'):
            for x, y in ((0, 1), (1, 0)):
                if v[x] is not None:
                    if name == 'b_and':
                        return a[y] if v[x] else self.const('b', False)
                    return self.const('b', True) if v[x] else a[y]
                n = self.nodes[a[x]]
                if n['op'] == 'b_not' and n['args'][0] == a[y]:   # x and not x, x or not x
                    return self.const('b', name == 'b_or')
            if a[0] == a[1]:
                return a[0]
        if name in self.RANGE_TESTS:
            decided = self._range_test(name, a)
            if decided is not None:
                return self.const('b', decided)
            if name in ('ige', 'ilt'):
                floated = self._truncated_test(name, a, v)
                if floated is not None:
                    return floated
        if name == 'b_not' and self.nodes[a[0]]['op'] == 'b_not':
            return self.nodes[a[0]]['args'][0]
        if name == 'and':
            # and(mask, x) with a comparison mask -> select(cond, x, 0)
            for x, y in ((0, 1), (1, 0)):
                n = self.nodes[a[x]]
                if n['op'] == 'b2i':
                    return self.select(n['args'][0], a[y], self.const('i', 0))
                if v[x] == U32:
                    return a[y]
                if v[x] == 0:
                    return self.const('i', 0)
        if name == 'or':
            for x, y in ((0, 1), (1, 0)):
                if v[x] == 0:
                    return a[y]
            na, nb = self.nodes[a[0]], self.nodes[a[1]]
            if na['op'] == 'b2i' and nb['op'] == 'b2i':
                return self.op('b2i', self.op('b_or', na['args'][0], nb['args'][0]))
        if name == 'not' and self.nodes[a[0]]['op'] == 'b2i':
            return self.op('b2i', self.op('b_not', self.nodes[a[0]]['args'][0]))
        if name == 'iadd':
            for x, y in ((0, 1), (1, 0)):
                if v[x] == 0:
                    return a[y]
                n = self.nodes[a[y]]
                if v[x] is not None and n['op'] == 'iadd' and self.value(n['args'][1]) is not None:
                    return self.op('iadd', n['args'][0], self.const('i', self.value(n['args'][1]) + v[x]))   # (p+k1)+k2
        if name == 'ine' and v[1] == 0 and self.nodes[a[0]]['op'] == 'b2i':
            return self.nodes[a[0]]['args'][0]
        if name == 'ieq' and v[1] == 0 and self.nodes[a[0]]['op'] == 'b2i':
            return self.op('b_not', self.nodes[a[0]]['args'][0])
        return None

    # Comparisons decided by value ranges: name -> (signed integer?, test on (low a, high a, low b, high b) -> True/False/None)
    RANGE_TESTS = {
        'ilt': lambda la, ha, lb, hb: True if ha < lb else False if la >= hb else None,
        'ige': lambda la, ha, lb, hb: True if la >= hb else False if ha < lb else None,
        'lt': lambda la, ha, lb, hb: True if ha < lb else False if la >= hb else None,
        'ge': lambda la, ha, lb, hb: True if la >= hb else False if ha < lb else None,
    }

    def _truncated_test(self, name, args, values):
        """A signed test of ftoi(x) against a constant k as a float test of x (x of known range: never NaN, inside int32):
        trunc(x) <= k is x < k + 1 for k >= 0 and x <= k for k < 0."""
        for x, y in ((0, 1), (1, 0)):
            n = self.nodes[args[y]]
            if values[x] is None or n['op'] != 'ftoi':
                continue
            k, inner = _signed(values[x]), n['args'][0]
            r = self.bounds(inner)
            if r is None or not -2 ** 24 < k < 2 ** 24 or r[0] <= -2 ** 31 or r[1] >= 2 ** 31:
                return None
            # name(args[0], args[1]) as 'trunc(x) <= m' (at_most) or its negation
            if (name == 'ige') == (x == 0):   # k >= trunc(x), or trunc(x) < k (= trunc(x) <= k - 1)
                m, at_most = (k if x == 0 else k - 1), True
            else:                             # k < trunc(x), or trunc(x) >= k: not trunc(x) <= m
                m, at_most = (k if x == 0 else k - 1), False
            test = (self.op('lt', inner, self.const('f', m + 1)) if m >= 0
                    else self.op('ge', self.const('f', m), inner))
            return test if at_most else self.op('b_not', test)
        return None

    def _range_test(self, name, args):
        ra, rb = self.bounds(args[0]), self.bounds(args[1])
        if ra is not None and rb is not None:
            decided = self.RANGE_TESTS[name](ra[0], ra[1], rb[0], rb[1])
            if decided is not None:
                return decided
        if name in ('ilt', 'ige'):   # p + k against p, when p + k cannot overflow
            for x, y, flip in ((0, 1, False), (1, 0, True)):
                n = self.nodes[args[x]]
                k = self.value(n['args'][1]) if n['op'] == 'iadd' else None
                if k is None or n['args'][0] != args[y]:
                    continue
                k, rp = _signed(k), self.bounds(args[y])
                if rp is None or not (-2 ** 31 <= rp[0] + k and rp[1] + k < 2 ** 31):
                    return None
                at_least = k >= 0 if not flip else k <= 0   # args[0] >= args[1]
                return at_least if name == 'ige' else not at_least
        return None

    def bounds(self, index, depth=48):
        """(low, high) a node's value always lies in, for integer nodes (signed) and float nodes that are never NaN or
        infinite; None when unknown. Used to decide comparisons, so a loop whose trip count is bounded (parallax steps
        between 8 and 32) ends once its counter passes the bound."""
        memo = self.__dict__.setdefault('_bounds', {})
        if index in memo:
            return memo[index]
        result = self._bounds_of(index, depth)
        if result is not None and not all(math.isfinite(x) and abs(x) < 1e30 for x in result):
            result = None
        if result is not None and self.nodes[index]['type'] == 'f' and self.nodes[index]['op'] not in ('const', 'sat'):
            # float32 rounding can step just past bounds computed in double: widen, so a test at the edge stays open
            result = tuple(x + sign * (abs(x) * 2.0 ** -20 + 1e-30) for x, sign in zip(result, (-1, 1)))
        if result is not None or depth > 0:
            memo[index] = result
        return result

    def _bounds_of(self, index, depth):
        n = self.nodes[index]
        op, t = n['op'], n['type']
        if op == 'const':
            if t == 'f':
                return (n['value'], n['value']) if math.isfinite(n['value']) else None
            return (_signed(n['value']),) * 2 if t == 'i' else None
        if op == 'sat':
            return (0.0, 1.0)
        if depth == 0 or t == 'b' or op not in ('add', 'mul', 'neg', 'min', 'max', 'select', 'ftoi', 'itof', 'iadd',
                                                'round_ne', 'round_ni', 'round_pi', 'round_z', 'imin', 'imax', 'ineg'):
            return None
        args = n['args'][1:] if op == 'select' else n['args']
        r = [self.bounds(x, depth - 1) for x in args]
        if None in r:
            return None
        if op in ('add', 'iadd'):
            low, high = r[0][0] + r[1][0], r[0][1] + r[1][1]
            return (low, high) if op == 'add' or (-2 ** 31 <= low and high < 2 ** 31) else None
        if op == 'mul':
            products = [x * y for x in r[0] for y in r[1]]
            return min(products), max(products)
        if op in ('neg', 'ineg'):
            return (-r[0][1], -r[0][0]) if op == 'neg' or r[0][0] > -2 ** 31 else None
        if op in ('min', 'imin', 'max', 'imax'):
            pick = min if op in ('min', 'imin') else max
            return pick(r[0][0], r[1][0]), pick(r[0][1], r[1][1])
        if op == 'select':
            return min(r[0][0], r[1][0]), max(r[0][1], r[1][1])
        if op == 'ftoi':
            return (max(-2 ** 31, math.trunc(r[0][0])), min(2 ** 31 - 1, math.trunc(r[0][1])))
        if op == 'itof':
            return float(r[0][0]), float(r[0][1])
        fn = {'round_ne': round, 'round_ni': math.floor, 'round_pi': math.ceil, 'round_z': math.trunc}[op]
        return float(fn(r[0][0])), float(fn(r[0][1]))

    def select(self, cond, a, b):
        cond = self.boolean(cond)
        if a == b:
            return a
        c = self.value(cond)
        if c is not None:
            return a if c else b
        va, vb = self.value(a), self.value(b)
        if self.type(a) == 'b' == self.type(b) and va is not None and vb is not None:
            return cond if va else self.op('b_not', cond)   # c ? true : false
        if self.type(a) == 'b' == self.type(b) and (va is not None or vb is not None):
            if va is not None:   # c ? true : b = c or b;  c ? false : b = not c and b
                return self.op('b_or', cond, b) if va else self.op('b_and', self.op('b_not', cond), b)
            return self.op('b_or', self.op('b_not', cond), a) if vb else self.op('b_and', cond, a)
        ta, tb = self.type(a), self.type(b)
        if ta != tb:
            # Mixed register contents: keep the float view when either side is a float constant 0.
            want = 'f' if 'f' in (ta, tb) and 'b' not in (ta, tb) else 'i'
            a, b = self.cast(a, want), self.cast(b, want)
        n = self.nodes[cond]
        if n['op'] == 'b_not':
            return self.select(n['args'][0], b, a)
        return self._add('select', self.type(a), (cond, a, b))

    def channel(self, source, index, type_):
        return self._add('channel', type_, (source,), index=index)

    # -- analysis
    def reachable(self, roots):
        seen, stack = set(), [r for r in roots if r is not None]
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            stack.extend(self.nodes[i]['args'])
        return seen

    def unknowns(self, roots):
        return sorted({self.nodes[i]['reason'] for i in self.reachable(roots) if self.nodes[i]['op'] == 'unknown'})

    def leaves(self, roots, op):
        return sorted({self.nodes[i]['name'] for i in self.reachable(roots) if self.nodes[i]['op'] == op
                       and 'name' in self.nodes[i]})


class Bindings:
    """Values substituted during translation. Every method returns None to leave the input symbolic."""

    def input(self, name):                      # interpolant 'TEXCOORD3.x' or system value 'SV_POSITION.y'
        return None

    def constant(self, name):                   # 'GlobalViewportCBuffer.m_ViewToWorldMat+16'
        return None

    def dynamic_constant(self, buffer, offset, index):   # cb[r.x + n]: buffer name, byte offset of n, index node
        return None

    def resource(self, graph, op, binding, args):        # sample/ld on a named binding -> 4 node ids or None
        return None

    def texture_size(self, binding):   # (width, height, depth or array size, mip count) of the texture bound there
        return None


def _cb_name(reflection, slot, byte):
    buffer = None
    if reflection is not None:
        binding = reflection.binding('cb', slot)
        if binding is not None:
            buffer = reflection.constant_buffer(binding.name)
    if buffer is None:
        return 'cb%d+%d' % (slot, byte)
    for var in buffer.variables:
        if var.offset <= byte < var.offset + max(var.size, 4):
            # A buffer's only struct variable (g_VP, g_World) is left out of the name; any other variable is named.
            path, rest = [buffer.name], byte - var.offset
            if not var.members or len(buffer.variables) > 1:
                path.append(var.name)
            while var.members:
                # A struct: the member with the largest offset at or below the byte (members carry no size).
                name, at, member = max((m for m in var.members if m[1] <= rest), key=lambda m: m[1],
                                       default=(None, 0, None))
                if name is None:
                    break
                path.append(name)
                rest -= at
                var = member
            return '.'.join(path) + ('+%d' % rest if rest else '')
    return '%s+%d' % (buffer.name, byte)


def _binding_name(reflection, kind, register):
    if reflection is not None:
        b = reflection.binding(kind, register)
        if b is not None:
            return b.name
    return '%s%d' % (kind, register)


def _blocks(instructions):
    """Matching structure: for each if/else/loop/switch/case position, its partners."""
    match, stack = {}, []
    for i, ins in enumerate(instructions):
        op = ins.opcode
        if op in ('if', 'loop', 'switch'):
            stack.append([op, i])
        elif op == 'else':
            stack[-1].append(i)
        elif op in ('endif', 'endloop', 'endswitch'):
            entry = stack.pop()
            match[entry[1]] = (entry[2:], i)
    if stack:
        raise dxbc.DxbcError('Unbalanced control flow')
    return match


class _Flow:
    """Predicates of the current point: active lanes, plus per-loop and per-switch break state."""

    def __init__(self, active):
        self.active = active


class Translator:
    FLOAT_OPS = {'add', 'mul', 'mad', 'div', 'min', 'max', 'sqrt', 'rsq', 'rcp', 'log', 'exp', 'frc', 'round_ne',
                 'round_ni', 'round_pi', 'round_z', 'dp2', 'dp3', 'dp4', 'sincos', 'eq', 'ne', 'lt', 'ge',
                 'deriv_rtx', 'deriv_rty', 'deriv_rtx_coarse', 'deriv_rtx_fine', 'deriv_rty_coarse',
                 'deriv_rty_fine', 'ftoi', 'ftou', 'f32tof16', 'mov', 'movc', 'swapc'}

    def __init__(self, shader, bindings=None, graph=None):
        self.shader = shader
        self.reflection = shader.reflection
        self.bindings = bindings or Bindings()
        self.g = graph or Graph()
        self.regs = {}            # ('r', n, c) / ('x', n, i, c) / ('o', n, c) -> node
        self.outputs = {}
        self.discard = self.g.const('b', False)
        self.alive = self.g.const('b', True)    # lanes that have not returned
        self.outer = self.g.const('b', True)    # entry predicate of the region being run (absolute)
        self.stores = []          # UAV stores: dict(binding, address, values {lane: node}, active)
        self.unsupported = []
        self.loops = []           # (position, iterations or None)
        self._loops = []          # the loop/switch states of the block being run
        self._facts_memo = {}     # id(facts) -> {node: specialised node}
        self.instructions = shader.program.instructions
        self.match = _blocks(self.instructions)
        self.icb = shader.program.immediate_constants
        self._inputs, self._patch = {}, {}
        for elements, table in ((shader.inputs, self._inputs), (getattr(shader, 'patch_constants', ()), self._patch)):
            for e in elements:
                for c in range(4):
                    if e.mask >> c & 1:
                        table[(e.register, c)] = e
        self.groupshared = {}     # (g register, address node, byte offset) -> node: this thread's groupshared stores
        self._indexable = {}
        self._dimensions = {}     # t register -> declared resource dimension
        for d in shader.program.declarations:
            if d.opcode == 'dcl_indexabletemp':
                self._indexable[d.data[0]] = (d.data[1], d.data[2])
            elif d.opcode == 'dcl_resource' and d.operands:
                self._dimensions[d.operands[0].indices[0][0]] = d.controls & 0x1F

    # ------------------------------------------------------------------ operands
    def _index(self, imm, rel):
        if rel is None:
            return imm, None
        value = self.read(rel, 'i')[0]
        v = self.g.value(value)
        if v is not None:
            return (imm + v) & U32, None
        return imm, value

    def _read_component(self, operand, comp, want):
        g = self.g
        t = operand.type
        if t == 'l':
            raw = operand.values[comp if len(operand.values) == 4 else 0]
            return g.const('f', bits_float(raw)) if want == 'f' else g.const('i', raw)
        if t == 'r':
            reg = operand.indices[0][0]
            node = self.regs.get(('r', reg, comp))
            return node if node is not None else g.const('i', 0)
        if t == 'x':
            reg = operand.indices[0][0]
            index, dynamic = self._index(*operand.indices[1])
            if dynamic is not None:
                return self._dynamic_indexable(reg, index, dynamic, comp)
            node = self.regs.get(('x', reg, index, comp))
            return node if node is not None else g.const('i', 0)
        if t == 'v':
            if len(operand.indices) != 1 or operand.indices[0][1] is not None:
                raise Unknown('indexed input register')
            reg = operand.indices[0][0]
            element = self._inputs.get((reg, comp))
            name = '%s.%s' % (element.semantic if element else 'v%d' % reg, 'xyzw'[comp])
            bound = self.bindings.input(name)
            if bound is not None:
                return bound
            itype = 'f'
            if element is not None and element.component_type in (1, 2):
                itype = 'i'
            return g.input(name, itype)
        if t == 'cb':
            slot = operand.indices[0][0]
            index, dynamic = self._index(*operand.indices[1])
            if dynamic is not None:
                buffer = _cb_name(self.reflection, slot, index * 16 + 4 * comp).split('.')[0].split('+')[0]
                bound = self.bindings.dynamic_constant(buffer, index * 16 + 4 * comp, dynamic)
                if bound is not None:
                    return bound
                node = g._add('cbload', 'f', (dynamic,), buffer=buffer, offset=index * 16 + 4 * comp)
                return node
            name = _cb_name(self.reflection, slot, index * 16 + 4 * comp)
            bound = self.bindings.constant(name)
            if bound is not None:
                return bound
            return g.input(name)
        if t == 'icb':
            index, dynamic = self._index(*operand.indices[0])
            if dynamic is None:
                at = 4 * index + comp
                if at >= len(self.icb):
                    raise Unknown('immediate constant buffer read past its end')
                raw = self.icb[at]
                return g.const('f', bits_float(raw)) if want == 'f' else g.const('i', raw)
            rows = len(self.icb) // 4
            # Dynamic icb index: a select chain over the rows.
            node = g.const('i', self.icb[comp])
            for row in range(1, rows):
                raw = self.icb[4 * row + comp]
                pick = g.op('ieq', g.op('iadd', dynamic, g.const('i', index)), g.const('i', row))
                node = g.select(pick, g.const('i', raw), node)
            return node
        if t in ('vicp', 'vocp'):   # a control point's input or output register: CP<n>.<semantic>.<c> (OCP for vocp)
            point, dynamic = self._index(*operand.indices[0])
            if dynamic is not None:
                raise Unknown('dynamically indexed control point')
            reg = operand.indices[1][0]
            element = self._inputs.get((reg, comp)) if t == 'vicp' else None
            name = '%s%d.%s.%s' % ('CP' if t == 'vicp' else 'OCP', point, element.semantic if element else 'v%d' % reg,
                                   'xyzw'[comp])
            bound = self.bindings.input(name)
            if bound is not None:
                return bound
            return g.input(name, 'i' if element is not None and element.component_type in (1, 2) else 'f')
        if t == 'vpc':   # a patch constant: PATCH.<semantic>.<c> from the patch-constant signature
            reg = operand.indices[0][0]
            element = self._patch.get((reg, comp))
            name = 'PATCH.%s.%s' % (element.semantic if element else 'vpc%d' % reg, 'xyzw'[comp])
            bound = self.bindings.input(name)
            if bound is not None:
                return bound
            return g.input(name, 'i' if element is not None and element.component_type in (1, 2) else 'f')
        if t == 'vDomain':   # the domain location (barycentric or uv)
            name = 'vDomain.' + 'xyzw'[comp]
            bound = self.bindings.input(name)
            return bound if bound is not None else g.input(name, 'f')
        if t in ('vPrim', 'vCoverage', 'vThreadID', 'vThreadGroupID', 'vThreadIDInGroup', 'vThreadIDInGroupFlattened',
                 'vGSInstanceID', 'vOutputControlPointID', 'vForkInstanceID', 'vJoinInstanceID'):
            return g.input(t + '.' + 'xyzw'[comp], 'i')
        raise Unknown('operand type ' + t)

    def _dynamic_indexable(self, reg, index, dynamic, comp):
        size = self._indexable.get(reg, (0, 4))[0]
        g = self.g
        node = g.const('i', 0)
        for k in range(size):
            value = self.regs.get(('x', reg, k, comp), g.const('i', 0))
            pick = g.op('ieq', g.op('iadd', dynamic, g.const('i', index)), g.const('i', k))
            node = g.select(pick, value, node)
        return node

    def read(self, operand, want):
        """Four component nodes (in destination lane order), converted to ``want`` and with modifiers applied."""
        g = self.g
        lanes = operand.selected()
        cache = {}
        out = []
        for comp in lanes:
            if comp not in cache:
                node = self._read_component(operand, comp, want)
                if want != 'raw':
                    node = g.cast(node, want)
                    if operand.modifier in ('abs', 'absneg'):
                        node = g.op('abs', node) if want == 'f' else g.op('iabs', node)
                    if operand.modifier in ('neg', 'absneg'):
                        node = g.op('neg', node) if want == 'f' else g.op('ineg', node)
                cache[comp] = node
            out.append(cache[comp])
        return out

    def write(self, operand, lane, value, saturate=False, flow=None):
        g = self.g
        if saturate:
            value = g.op('sat', g.cast(value, 'f'))
        t = operand.type
        if t == 'null':
            return
        if operand.components == 4 and operand.mode == 'mask' and not operand.mask >> lane & 1:
            return
        if t == 'r':
            key = ('r', operand.indices[0][0], lane)
        elif t == 'o':
            key = ('o', operand.indices[0][0], lane)
        elif t == 'x':
            reg = operand.indices[0][0]
            index, dynamic = self._index(*operand.indices[1])
            if dynamic is not None:
                size = self._indexable.get(reg, (0, 4))[0]
                for k in range(size):
                    pick = g.op('ieq', g.op('iadd', dynamic, g.const('i', index)), g.const('i', k))
                    old = self.regs.get(('x', reg, k, lane), g.const('i', 0))
                    active = g.op('b_and', flow.active, pick)
                    self.regs[('x', reg, k, lane)] = g.select(active, value, old)
                return
            key = ('x', reg, index, lane)
        elif t == 'oDepth':
            key = ('oDepth', 0, 0)
        elif t == 'oMask':
            key = ('oMask', 0, 0)
        else:
            raise Unknown('destination ' + t)
        old = self.regs.get(key)
        if old is None:
            old = g.const(g.type(value), 0)
        self.regs[key] = g.select(flow.active, value, old) if flow is not None else value

    def lanes(self, operand):
        if operand.type == 'null':
            return []
        if operand.components == 4 and operand.mode == 'mask':
            return [c for c in range(4) if operand.mask >> c & 1]
        if operand.components == 1:
            return [0]
        return [0, 1, 2, 3]

    # ------------------------------------------------------------------ execution
    def run(self):
        flow = _Flow(self.g.const('b', True))
        self._block(0, len(self.instructions), flow, [])
        g = self.g
        outputs = {}
        for key, node in self.regs.items():
            if key[0] in ('o', 'oDepth', 'oMask'):
                outputs[key] = node
        translation = Translation(g, outputs, self.discard, self.unsupported, self.loops, self.shader)
        translation.stores = self.stores
        return translation

    def _block(self, start, end, flow, loops):
        """Execute instructions [start, end) under ``flow``; ``loops`` is the stack of loop/switch states."""
        g = self.g
        self._loops = loops
        i = start
        while i < end:
            ins = self.instructions[i]
            op = ins.opcode
            if op == 'if':
                cond = g.boolean(self.read(ins.operands[0], 'i')[0])
                if not ins.nonzero:
                    cond = g.op('b_not', cond)
                cond = self._given_active(cond, flow, loops)   # both branches are entered only by active lanes
                partners, close = self.match[i]
                else_at = partners[0] if partners else None
                taken = g.value(cond)
                if taken is not None:   # uniform after folding: run only the branch that executes
                    if taken:
                        self._block(i + 1, else_at if else_at is not None else close, flow, loops)
                    elif else_at is not None:
                        self._block(else_at + 1, close, flow, loops)
                elif g.value(flow.active) is not False:
                    active = flow.active
                    then = self._region(i + 1, else_at if else_at is not None else close, flow, loops,
                                        g.op('b_and', active, cond))
                    other = (self._region(else_at + 1, close, flow, loops,
                                          g.op('b_and', active, g.op('b_not', cond))) if else_at is not None
                             else self._identity(loops))
                    self._merge(active, [(cond, then)], other, flow, loops)
                i = close + 1
                continue
            if op == 'loop':
                _partners, close = self.match[i]
                self._loop(i, close, flow, loops)
                i = close + 1
                continue
            if op == 'switch':
                _partners, close = self.match[i]
                self._switch(i, close, flow, loops)
                i = close + 1
                continue
            if op in ('break', 'breakc', 'continue', 'continuec'):
                cond = flow.active
                if op.endswith('c'):
                    c = g.boolean(self.read(ins.operands[0], 'i')[0])
                    if not ins.nonzero:
                        c = g.op('b_not', c)
                    cond = g.op('b_and', cond, self._given_active(c, flow, loops, op == 'breakc'))
                if op.startswith('break'):
                    state = loops[-1]
                    state['broken'] = g.op('b_or', state['broken'], cond)
                else:
                    state = next(s for s in reversed(loops) if s['kind'] == 'loop')
                    state['continued'] = g.op('b_or', state['continued'], cond)
                flow.active = g.op('b_and', flow.active, g.op('b_not', cond))
                i += 1
                continue
            if op in ('ret', 'retc'):
                cond = flow.active
                if op == 'retc':
                    c = g.boolean(self.read(ins.operands[0], 'i')[0])
                    if not ins.nonzero:
                        c = g.op('b_not', c)
                    cond = g.op('b_and', cond, self._given_active(c, flow, loops))
                self.alive = g.op('b_and', self.alive, g.op('b_not', cond))
                flow.active = g.op('b_and', flow.active, g.op('b_not', cond))
                i += 1
                continue
            if op == 'discard':
                c = g.boolean(self.read(ins.operands[0], 'i')[0])
                if not ins.nonzero:
                    c = g.op('b_not', c)
                # Discarded lanes are don't-care for registers and targets: only the predicate records them.
                self.discard = g.op('b_or', self.discard, g.op('b_and', flow.active, self._given_active(c, flow, loops)))
                i += 1
                continue
            if op in ('case', 'default', 'endswitch', 'endif', 'endloop', 'else', 'label'):
                raise dxbc.DxbcError('Unexpected %s at instruction %d' % (op, i))
            if g.value(flow.active) is False:
                i += 1
                continue
            try:
                self._instruction(ins, flow)
            except Unknown as error:
                self.unsupported.append('%s: %s' % (op, error))
                for operand in ins.operands[:self._destinations(ins)]:
                    for lane in self.lanes(operand):
                        try:
                            self.write(operand, lane, g.unknown('%s at %d' % (op, ins.position)), flow=flow)
                        except Unknown:
                            pass
            i += 1

    def _region(self, start, end, flow, loops, entry):
        """Run [start, end) as if every lane entered it; return its effects relative to that entry.

        ``entry`` is the region's entry predicate relative to the enclosing region (kept for UAV stores)."""
        g = self.g
        saved_regs, saved_discard, saved_alive, saved_active = self.regs, self.discard, self.alive, flow.active
        saved_outer = self.outer
        self.outer = self._entered(entry)
        saved_states = [(s['broken'], s['continued']) for s in loops]
        self.regs = dict(saved_regs)
        self.discard, self.alive, flow.active = g.const('b', False), g.const('b', True), g.const('b', True)
        for state in loops:
            state['broken'] = state['continued'] = g.const('b', False)
        self._block(start, end, flow, loops)
        result = dict(regs=self.regs, discard=self.discard, alive=self.alive,
                      states=[(s['broken'], s['continued']) for s in loops])
        self.regs, self.discard, self.alive, flow.active = saved_regs, saved_discard, saved_alive, saved_active
        self.outer = saved_outer
        for state, (broken, continued) in zip(loops, saved_states):
            state['broken'], state['continued'] = broken, continued
        return result

    LEAVES = ('const', 'input', 'unknown', 'channel', 'sample', 'load', 'cbload')

    def _given_active(self, cond, flow, loops=(), exit_test=False, split=True):
        """``cond`` as it reads in an active lane: decided under the facts flow.active and the region entry imply.

        The result is only ANDed with flow.active (a branch, break, return or discard predicate), so it may differ from
        ``cond`` where the lane is inactive. A loop counter written as select(previous active, i + 1, i) is then exactly
        the iteration count, and a test on it folds, so a loop with an early break still ends at its count.

        Facts carry over iterations: a lane active in iteration k passed the top-level breaks of iteration k - 1 that came
        before any continue, so their (decided) conditions were false. When the result still depends on a select
        condition c and one of its cases decides, it becomes c ? that : the other case: a parallax step counter that jumps
        past the step count on a hit reads 'hit or k >= n', which is true once k passes n's bound."""
        g = self.g
        state = next((s for s in reversed(loops) if s['kind'] == 'loop'), None)
        if state is None:   # outside loops nothing needs to end: the plain predicate is exact and cheaper
            return cond
        facts = self._active_facts(flow, state)
        result = self._specialise(cond, facts, self._facts_memo.get(id(facts))) if facts else cond
        if split and g.value(result) is None:
            result = self._case_split(result)
        if (exit_test and state is not None and self.outer == state.get('outer')
                and g.value(state['continued']) is False):   # later iterations: this lane's test was false here
            state['facts'] = dict(facts)
            if not g.is_const(result):
                state['facts'][result] = False
            state['version'] = state.get('version', 0) + 1
            need = self._steps_to_bound(result)
            if need is None and result is not cond:
                need = self._steps_to_bound(cond, shallow=True)
            if need is not None:   # an integer counter against a bounded value: allow just the steps that reach it
                state['until'] = max(state.get('until', 0), min(MAX_LOOP_ITERATIONS, state['iteration'] + need))
        return result

    def _active_facts(self, flow, state):
        """{node: truth} that flow.active, the region entry and the loop's carried facts imply (memoized)."""
        g = self.g
        key = (flow.active, self.outer, id(state), state.get('version', 0) if state is not None else 0)
        cache = self.__dict__.setdefault('_facts_cache', {})
        if key in cache:
            return cache[key]
        facts, pending = {}, []

        def learn(node, value):
            if node in facts or g.is_const(node):
                return
            facts[node] = value
            n = g.nodes[node]
            if n['op'] == 'b_not':
                learn(n['args'][0], not value)
            elif n['op'] == ('b_and' if value else 'b_or'):
                for a in n['args']:
                    learn(a, value)
            elif n['op'] in ('b_and', 'b_or'):
                pending.append(node)   # a false and, a true or: one side follows once the other is known
        if state is not None:
            for node, value in state.get('facts', {}).items():
                learn(node, value)
        learn(flow.active, True)
        learn(self.outer, True)
        changed = True
        while changed:
            changed = False
            for node in pending:
                x, y = g.nodes[node]['args']
                want = g.nodes[node]['op'] == 'b_or'   # true or, one side false: the other true (and dually)
                for p, q in ((x, y), (y, x)):
                    if facts.get(p) is (not want) and q not in facts:
                        learn(q, want)
                        changed = True
        if len(cache) > 4096:
            cache.clear()
            self._facts_memo.clear()
        cache[key] = facts
        self._facts_memo[id(facts)] = {}   # specialised nodes under these facts, shared by every condition read with them
        return facts

    def _specialise(self, root, facts, memo=None):
        """``root`` rebuilt with the nodes in ``facts`` replaced by their truth (selects on them take their branch)."""
        g = self.g
        memo = {} if memo is None else memo
        leaves = self.LEAVES

        def under(i):
            if i in memo:
                return memo[i]
            if i in facts:
                result = g.const('b', facts[i])
            else:
                n = g.nodes[i]
                if n['op'] in leaves or not g.predicated[i]:
                    result = i
                elif n['op'] == 'select':
                    c = under(n['args'][0])
                    taken = g.value(c)
                    if taken is not None:
                        result = under(n['args'][1] if taken else n['args'][2])
                    else:
                        result = g.select(c, under(n['args'][1]), under(n['args'][2]))
                else:
                    args = [under(a) for a in n['args']]
                    result = i if args == n['args'] else g.op(n['op'], *args)
                if result in facts:   # rebuilt into a node the facts decide (hash-consing makes it the same node)
                    result = g.const('b', facts[result])
            memo[i] = result
            return result
        stack, order, seen = [root], [], set()   # children first, so the recursion stays shallow on long counter chains
        while stack:
            i = stack.pop()
            if i in seen or i in facts or i in memo:
                continue
            seen.add(i)
            if g.nodes[i]['op'] not in leaves and g.predicated[i]:
                order.append(i)
                stack.extend(g.nodes[i]['args'])
        for i in reversed(order):
            n = g.nodes[i]
            if n['op'] == 'select' or all(a in memo or a in facts or g.nodes[a]['op'] in leaves or not g.predicated[a]
                                          for a in n['args']):
                under(i)
        return under(root)

    def _case_split(self, root, limit=3, depth=8):
        """c ? root(c true) : root(c false) for the first select condition c near the root (of a few) whose true or
        false case decides root; else root."""
        g = self.g
        conditions, frontier, seen = [], [root], {root}
        for _ in range(depth):
            following = []
            for i in frontier:
                n = g.nodes[i]
                if n['op'] in self.LEAVES:
                    continue
                if n['op'] == 'select' and n['args'][0] not in conditions and not g.is_const(n['args'][0]):
                    conditions.append(n['args'][0])
                for a in n['args']:
                    if a not in seen:
                        seen.add(a)
                        following.append(a)
            frontier = following
            if len(conditions) >= limit or not frontier:
                break
        for c in conditions[:limit]:
            yes, no = self._specialise(root, {c: True}), self._specialise(root, {c: False})
            if g.value(yes) is not None or g.value(no) is not None:
                return g.select(c, yes, no)
        return root

    def _steps_to_bound(self, node, shallow=False):
        """Iterations until an exit test 'k >= y' or 'k < y' (k the counter, y of known range) is decided for every y:
        high(y) - k + 2 (a budget for unrolling, not a correctness bound), or None without such a comparison. A counter
        that is not constant counts from its low bound: in the unspecialised test a parallax counter reads
        select(hit, n + 1, k) + 1, and its test is decided (false) in the specialised one while k stays below n's low
        bound, so only the unspecialised test still shows the budget. ``shallow`` looks only through the boolean
        combinators at the top of the test (the unspecialised one reaches the whole unrolled history)."""
        g = self.g
        need = None
        if shallow:
            found, stack = [], [(node, 0)]
            while stack:
                i, d = stack.pop()
                found.append(i)
                if d < 4 and g.nodes[i]['op'] in ('b_or', 'b_and', 'b_not'):
                    stack.extend((a, d + 1) for a in g.nodes[i]['args'])
        else:
            found = g.reachable([node])
        for i in found:
            n = g.nodes[i]
            if n['op'] in ('ilt', 'ige', 'lt', 'ge'):   # float tests too: ftoi counters fold to x < k + 1
                for x, y in (n['args'], n['args'][::-1]):
                    r = g.bounds(y)
                    if r is None:
                        continue
                    if g.is_const(x):
                        k = _signed(g.value(x)) if n['type'] != 'f' and g.type(x) == 'i' else g.value(x)
                    else:
                        rx = g.bounds(x)
                        if rx is None or not math.isfinite(rx[0]):
                            continue
                        k = rx[0]
                    if not math.isfinite(r[1]):
                        continue
                    step = int(math.ceil(r[1] - k)) + 2
                    need = step if need is None else max(need, step)
        return need if need is not None and need > 0 else None

    def _entered(self, entry):
        """Absolute entry predicate of a region entered by ``entry``: lanes that discarded before it do not enter it."""
        g = self.g
        return g.op('b_and', g.op('b_and', self.outer, g.op('b_not', self.discard)), entry)

    def _identity(self, loops):
        f = self.g.const('b', False)
        return dict(regs=self.regs, discard=f, alive=self.g.const('b', True), states=[(f, f)] * len(loops))

    def _merge(self, active, choices, default, flow, loops):
        """Join region results: ``choices`` [(condition, result)] are mutually exclusive, ``default`` covers the rest,
        and lanes outside ``active`` keep their state."""
        g = self.g

        def pick(get):
            node = get(default)
            for cond, result in reversed(choices):
                node = g.select(cond, get(result), node)
            return node
        before = self.regs
        keys = set(before) | set(default['regs'])
        for _cond, result in choices:
            keys |= set(result['regs'])
        merged = {}
        for key in keys:
            old = before.get(key)
            present = [r['regs'][key] for r in [default] + [c[1] for c in choices] if key in r['regs']]
            zero = g.const(g.type(present[0] if present else old), 0)
            fallback = old if old is not None else zero
            value = pick(lambda r: r['regs'].get(key, fallback))
            merged[key] = g.select(active, value, fallback)
        self.regs = merged
        self.discard = g.op('b_or', self.discard, g.op('b_and', active, pick(lambda r: r['discard'])))
        died = g.op('b_and', active, g.op('b_not', pick(lambda r: r['alive'])))
        self.alive = g.op('b_and', self.alive, g.op('b_not', died))
        for k, state in enumerate(loops):
            state['broken'] = g.op('b_or', state['broken'], g.op('b_and', active, pick(lambda r: r['states'][k][0])))
            state['continued'] = g.op('b_or', state['continued'],
                                      g.op('b_and', active, pick(lambda r: r['states'][k][1])))
        flow.active = self._restore_breaks(active, loops)

    def _restore_breaks(self, active, loops):
        # Lanes that broke, continued or returned inside a branch stay inactive after it joins.
        g = self.g
        for state in loops:
            active = g.op('b_and', active, g.op('b_not', state['broken']))
            if state['kind'] == 'loop':
                active = g.op('b_and', active, g.op('b_not', state['continued']))
        return g.op('b_and', active, self.alive)

    def _written(self, start, end):
        keys = set()
        for ins in self.instructions[start:end]:
            if ins.opcode in ('if', 'else', 'endif', 'loop', 'endloop', 'break', 'breakc', 'continue', 'continuec',
                              'switch', 'case', 'default', 'endswitch', 'ret', 'retc', 'discard'):
                continue
            for operand in ins.operands[:self._destinations(ins)]:
                if operand.type in ('r', 'o'):
                    for lane in self.lanes(operand):
                        keys.add((operand.type, operand.indices[0][0], lane))
                elif operand.type == 'x':
                    reg = operand.indices[0][0]
                    for k in range(self._indexable.get(reg, (0, 4))[0]):
                        for lane in self.lanes(operand):
                            keys.add(('x', reg, k, lane))
        return keys

    def _loop(self, start, close, flow, loops):
        """Run the loop as a region entered by every lane, then merge it with select(entry, after, before).

        Inside the region predicates are relative to the loop entry, so a trip count that folds (a fixed tap count, or
        an empty light list) unrolls even when the loop sits under a data-dependent branch."""
        g = self.g
        entry = flow.active
        before, discard_before, alive_before = dict(self.regs), self.discard, self.alive
        saved_outer = self.outer
        self.outer = self._entered(entry)
        self.discard, self.alive = g.const('b', False), g.const('b', True)
        stores_before = len(self.stores)
        node_mark = len(g.nodes)
        predicated = 0
        body = self.instructions[start + 1:close]
        # Data-dependent trip counts are unrolled only for small loops without nested loops (PCF taps and the like).
        predicated_limit = MAX_PREDICATED_ITERATIONS if len(body) <= 64 and not any(
            x.opcode == 'loop' for x in body) else 0
        state = dict(kind='loop', broken=g.const('b', False), continued=g.const('b', False), outer=self.outer, facts={})
        iterations = 0
        while True:
            state['continued'] = g.const('b', False)
            continuing = g.op('b_and', g.op('b_not', state['broken']), self.alive)
            flow.active = continuing
            if g.value(continuing) is False:
                break
            if not g.is_const(continuing):
                predicated += 1
            state['iteration'] = iterations
            limit = predicated_limit if not predicated_limit else max(predicated_limit, state.get('until', 0) - iterations
                                                                      + predicated)
            if (iterations >= MAX_LOOP_ITERATIONS or predicated > limit
                    or len(g.nodes) - node_mark > 200000):
                iterations = None
                break
            self._block(start + 1, close, flow, [state])
            iterations += 1
        self.loops.append((self.instructions[start].position, iterations))
        self.outer = saved_outer
        if iterations is None:
            del self.stores[stores_before:]
            # Trip count unknown: keep the registers from before and poison what the loop writes.
            reason = 'loop at %d' % self.instructions[start].position
            after = dict(before)
            for key in self._written(start + 1, close):
                after[key] = g.unknown(reason, 'f')
            if not g.is_const(self.discard) or g.value(self.discard):
                self.discard = g.unknown(reason, 'b')
            if not g.is_const(self.alive) or not g.value(self.alive):
                self.alive = g.unknown(reason, 'b')
        else:
            after = self.regs
        merged = {}
        for key in set(before) | set(after):
            old = before.get(key)
            new = after.get(key, old)
            if old is None:
                old = g.const(g.type(new), 0)
            merged[key] = g.select(entry, new, old)
        self.regs = merged
        self.discard = g.op('b_or', discard_before, g.op('b_and', entry, self.discard))
        self.alive = g.op('b_and', alive_before, g.op('b_or', g.op('b_not', entry), self.alive))
        flow.active = self._restore_breaks(g.op('b_and', entry, self.alive), loops)

    def _switch(self, start, close, flow, loops):
        g = self.g
        selector = self.read(self.instructions[start].operands[0], 'i')[0]
        active = flow.active
        # Case labels at this nesting level; consecutive labels share the body that follows them.
        labels, depth = [], 0
        for k in range(start + 1, close):
            op = self.instructions[k].opcode
            if op in ('if', 'loop', 'switch'):
                depth += 1
            elif op in ('endif', 'endloop', 'endswitch'):
                depth -= 1
            elif depth == 0 and op in ('case', 'default'):
                labels.append(k)

        def test(k):
            c = self.instructions[k]
            return g.op('ieq', selector, g.const('i', c.operands[0].values[0])) if c.opcode == 'case' else None
        others = g.const('b', False)
        for k in labels:
            if test(k) is not None:
                others = g.op('b_or', others, test(k))
        state = dict(kind='switch', broken=g.const('b', False), continued=g.const('b', False))
        stack = loops + [state]
        choices, default, hit, has_default = [], None, g.const('b', False), False
        for n, k in enumerate(labels):
            t = test(k)
            has_default = has_default or t is None
            hit = g.op('b_or', hit, t if t is not None else g.op('b_not', others))
            stop = labels[n + 1] if n + 1 < len(labels) else close
            if stop == k + 1:
                continue
            result = self._region(k + 1, stop, flow, stack, g.op('b_and', active, hit))
            if has_default:
                default = result
            else:
                choices.append((hit, result))
            hit, has_default = g.const('b', False), False
        if default is None:
            default = self._identity(stack)
        self._merge(active, choices, default, flow, loops)

    @staticmethod
    def _destinations(ins):
        op = ins.opcode
        if op in ('sincos', 'udiv', 'umul', 'imul', 'uaddc', 'usubb', 'swapc'):
            return 2
        if op in ('store_uav_typed', 'store_raw', 'store_structured') or op.startswith('atomic_'):
            return 0
        if op.startswith('imm_atomic'):
            return 1
        return 1 if ins.operands else 0

    # ------------------------------------------------------------------ instructions
    def _instruction(self, ins, flow):
        g = self.g
        op = ins.opcode
        o = ins.operands
        sat = ins.saturate

        def unary(fn, want='f'):
            dst = o[0]
            a = self.read(o[1], want)
            for lane in self.lanes(dst):
                self.write(dst, lane, fn(a[lane]), sat, flow)

        def binary(fn, want='f'):
            dst = o[0]
            a, b = self.read(o[1], want), self.read(o[2], want)
            for lane in self.lanes(dst):
                self.write(dst, lane, fn(a[lane], b[lane]), sat, flow)

        simple_f = {'add': 'add', 'mul': 'mul', 'div': 'div', 'min': 'min', 'max': 'max'}
        if op in simple_f:
            return binary(lambda a, b: g.op(simple_f[op], a, b))
        if op == 'mad':
            dst = o[0]
            a, b, c = self.read(o[1], 'f'), self.read(o[2], 'f'), self.read(o[3], 'f')
            for lane in self.lanes(dst):
                self.write(dst, lane, g.op('add', g.op('mul', a[lane], b[lane]), c[lane]), sat, flow)
            return
        unary_f = {'sqrt': 'sqrt', 'rsq': 'rsq', 'rcp': 'rcp', 'log': 'log2', 'exp': 'exp2', 'frc': 'frc',
                   'round_ne': 'round_ne', 'round_ni': 'round_ni', 'round_pi': 'round_pi', 'round_z': 'round_z'}
        if op in unary_f:
            return unary(lambda a: g.op(unary_f[op], a))
        if op.startswith('deriv_rt'):
            return unary(lambda a: g.op('ddx' if 'rtx' in op else 'ddy', a))
        if op in ('dp2', 'dp3', 'dp4'):
            n = int(op[2])
            dst = o[0]
            a, b = self.read(o[1], 'f'), self.read(o[2], 'f')
            total = g.op('mul', a[0], b[0])
            for k in range(1, n):
                total = g.op('add', total, g.op('mul', a[k], b[k]))
            for lane in self.lanes(dst):
                self.write(dst, lane, total, sat, flow)
            return
        if op == 'sincos':
            a = self.read(o[2], 'f')
            for k, fn in ((0, 'sin'), (1, 'cos')):
                for lane in self.lanes(o[k]):
                    self.write(o[k], lane, g.op(fn, a[lane]), sat, flow)
            return
        if op in ('eq', 'ne', 'lt', 'ge'):
            return binary(lambda a, b: g.op(op, a, b))
        if op == 'mov':
            dst = o[0]
            a = self.read(o[1], 'f' if (sat or o[1].modifier) else 'raw')
            for lane in self.lanes(dst):
                self.write(dst, lane, a[lane], sat, flow)
            return
        if op == 'movc':
            dst = o[0]
            want = 'f' if sat or o[2].modifier or o[3].modifier else 'raw'
            c, a, b = self.read(o[1], 'raw'), self.read(o[2], want), self.read(o[3], want)
            for lane in self.lanes(dst):
                cond = self._given_active(g.boolean(c[lane]), flow, self._loops, split=False) if flow else g.boolean(c[lane])
                self.write(dst, lane, g.select(cond, a[lane], b[lane]), sat, flow)
            return
        if op == 'swapc':
            c, a, b = self.read(o[2], 'raw'), self.read(o[3], 'raw'), self.read(o[4], 'raw')
            for lane in self.lanes(o[0]):
                self.write(o[0], lane, g.select(g.boolean(c[lane]), b[lane], a[lane]), False, flow)
            for lane in self.lanes(o[1]):
                self.write(o[1], lane, g.select(g.boolean(c[lane]), a[lane], b[lane]), False, flow)
            return
        int_binary = {'iadd': 'iadd', 'and': 'and', 'or': 'or', 'xor': 'xor', 'ishl': 'ishl', 'ishr': 'ishr',
                      'ushr': 'ushr', 'imin': 'imin', 'imax': 'imax', 'umin': 'umin', 'umax': 'umax',
                      'ieq': 'ieq', 'ine': 'ine', 'ilt': 'ilt', 'ige': 'ige', 'ult': 'ult', 'uge': 'uge'}
        if op in int_binary:
            if op in ('and', 'or'):
                return self._logic(ins, flow)
            return binary(lambda a, b: g.op(int_binary[op], a, b), 'i')
        if op in ('ineg', 'not', 'bfrev', 'countbits', 'firstbit_hi', 'firstbit_lo', 'firstbit_shi', 'f16tof32'):
            return unary(lambda a: g.op(op, a), 'i')
        if op in ('itof', 'utof'):
            return unary(lambda a: g.op(op, a), 'i')
        if op in ('ftoi', 'ftou', 'f32tof16'):
            return unary(lambda a: g.op(op, a), 'f')
        if op == 'imad' or op == 'umad':
            dst = o[0]
            a, b, c = self.read(o[1], 'i'), self.read(o[2], 'i'), self.read(o[3], 'i')
            for lane in self.lanes(dst):
                self.write(dst, lane, g.op('iadd', g.op('imul', a[lane], b[lane]), c[lane]), sat, flow)
            return
        if op in ('imul', 'umul'):
            a, b = self.read(o[2], 'i'), self.read(o[3], 'i')
            for lane in self.lanes(o[0]):
                self.write(o[0], lane, g.op('imul_hi' if op == 'imul' else 'umul_hi', a[lane], b[lane]), False, flow)
            for lane in self.lanes(o[1]):
                self.write(o[1], lane, g.op('imul', a[lane], b[lane]), False, flow)
            return
        if op == 'udiv':
            a, b = self.read(o[2], 'i'), self.read(o[3], 'i')
            for lane in self.lanes(o[0]):
                self.write(o[0], lane, g.op('udiv', a[lane], b[lane]), False, flow)
            for lane in self.lanes(o[1]):
                self.write(o[1], lane, g.op('urem', a[lane], b[lane]), False, flow)
            return
        if op in ('ubfe', 'ibfe'):
            dst = o[0]
            w, s, v = self.read(o[1], 'i'), self.read(o[2], 'i'), self.read(o[3], 'i')
            for lane in self.lanes(dst):
                self.write(dst, lane, g.op(op, w[lane], s[lane], v[lane]), False, flow)
            return
        if op == 'bfi':
            dst = o[0]
            w, s, v, b = (self.read(x, 'i') for x in o[1:5])
            for lane in self.lanes(dst):
                self.write(dst, lane, g.op('bfi', w[lane], s[lane], v[lane], b[lane]), False, flow)
            return
        if op in ('sample', 'sample_l', 'sample_b', 'sample_d', 'sample_c', 'sample_c_lz', 'gather4', 'gather4_c',
                  'ld', 'ld_ms', 'ld_structured', 'ld_raw', 'resinfo', 'bufinfo', 'lod', 'sample_info', 'sample_pos',
                  'ld_uav_typed', 'gather4_po', 'gather4_po_c'):
            return self._resource(ins, flow)
        if op == 'store_uav_typed' and o[0].type == 'u' and o[0].indices[0][1] is None:
            values = self.read(o[2], 'raw')
            self.stores.append(dict(binding=_binding_name(self.reflection, 'u', o[0].indices[0][0]),
                                    address=self.read(o[1], 'i'),
                                    values={lane: values[lane] for lane in self.lanes(o[0])},
                                    active=self._entered(flow.active)))
            return
        if op == 'sync':
            return   # a barrier: this thread's own groupshared stores stay what it reads back (_groupshared)
        if op in ('store_raw', 'store_structured') or op.startswith(('atomic_', 'imm_atomic')):
            return self._memory(ins, flow)
        if op == 'store_uav_typed':
            self.unsupported.append(op)
            return
        if op == 'nop':
            return
        raise Unknown('opcode')

    def _logic(self, ins, flow):
        g = self.g
        dst = ins.operands[0]
        a, b = self.read(ins.operands[1], 'raw'), self.read(ins.operands[2], 'raw')
        name = ins.opcode
        for lane in self.lanes(dst):
            x, y = a[lane], b[lane]
            tx, ty = g.type(x), g.type(y)
            if tx == 'b' and ty == 'b':
                value = g.op('b_and' if name == 'and' else 'b_or', x, y)
            elif name == 'and' and 'b' in (tx, ty):
                cond, other = (x, y) if tx == 'b' else (y, x)
                value = g.select(cond, other, g.const(g.type(other), 0))
            else:
                value = g.op(name, x, y)
                # Keep a float view of masked floats (and r0.x, mask, l(1.0)) so later float math stays typed.
                nv = g.nodes[value]
                if nv['op'] == 'select' and 'f' in (tx, ty):
                    value = g.cast(value, 'f')
            self.write(dst, lane, value, False, flow)

    def _memory(self, ins, flow):
        """store_raw and store_structured (UAV: a store; groupshared: this thread's value at the address), and the UAV and
        groupshared atomics. An atomic is recorded as a store with its operation; the value an imm_atomic returns is the
        memory's previous content, which other threads write, so it is unknown, and a groupshared address an atomic
        touched reads unknown afterwards."""
        g = self.g
        op = ins.opcode
        o = ins.operands
        returns = op.startswith('imm_atomic')
        dst = o[1] if returns else o[0]
        if dst.indices and dst.indices[0][1] is not None:
            raise Unknown('dynamically indexed memory')
        register = dst.indices[0][0]
        rest = o[2:] if returns else o[1:]
        if op == 'store_structured':
            address, offset, values = self.read(rest[0], 'i')[0], self.read(rest[1], 'i')[0], self.read(rest[2], 'raw')
        elif op == 'store_raw':
            address, offset, values = self.read(rest[0], 'i')[0], g.const('i', 0), self.read(rest[1], 'raw')
        elif not rest:   # imm_atomic_alloc / consume: the UAV's hidden counter
            address, offset, values = g.const('i', 0), g.const('i', 0), []
        else:   # atomics: address (a structured address takes two lanes), then their operands
            lanes = self.read(rest[0], 'i')
            address, offset = lanes[0], lanes[1] if len(lanes) > 1 else g.const('i', 0)
            values = [x for operand in rest[1:] for x in self.read(operand, 'i')[:1]]
        lanes = self.lanes(dst) if op.startswith('store') else list(range(len(values)))
        if dst.type == 'g':
            base = g.value(offset)
            if op.startswith('store') and base is not None:
                for k, lane in enumerate(lanes):
                    key = (register, address, base + 4 * k)
                    old = self.groupshared.get(key)
                    value = values[lane]
                    self.groupshared[key] = value if old is None or flow is None else g.select(flow.active, value, old)
            else:   # an atomic or a store at a run-time offset: the slot's content is no longer this thread's alone
                for key in [key for key in self.groupshared if key[0] == register and key[1] == address]:
                    del self.groupshared[key]
        else:
            self.stores.append(dict(binding=_binding_name(self.reflection, 'u', register), op=op, address=address,
                                    offset=offset, values={lane: values[lane] for lane in lanes},
                                    active=self._entered(flow.active)))
        if returns:
            for lane in self.lanes(o[0]):
                self.write(o[0], lane, g.unknown('%s result (memory other threads write)' % op), False, flow)

    def _groupshared(self, ins, register, flow):
        """ld_raw / ld_structured of groupshared memory: what this thread stored at the same address, else unknown (a value
        another thread wrote)."""
        g = self.g
        o = ins.operands
        if ins.opcode == 'ld_structured':
            address, offset = self.read(o[1], 'i')[0], self.read(o[2], 'i')[0]
        else:
            address, offset = self.read(o[1], 'i')[0], g.const('i', 0)
        base = g.value(offset)
        resource = o[-1]
        swizzle = resource.selected()
        dst = o[0]
        for lane in self.lanes(dst):
            value = self.groupshared.get((register, address, base + 4 * swizzle[lane])) if base is not None else None
            if value is None:
                value = g.unknown('groupshared memory another thread wrote')
            self.write(dst, lane, value, False, flow)

    def _resource(self, ins, flow):
        g = self.g
        op = ins.opcode
        o = ins.operands
        # Operand layouts (d3d11 tokenized format): dst, address, resource[, sampler][, extra...].
        layout = {'sample': (1, 2, 3, ()), 'sample_l': (1, 2, 3, (4,)), 'sample_b': (1, 2, 3, (4,)),
                  'sample_d': (1, 2, 3, (4, 5)), 'sample_c': (1, 2, 3, (4,)), 'sample_c_lz': (1, 2, 3, (4,)),
                  'gather4': (1, 2, 3, ()), 'gather4_c': (1, 2, 3, (4,)), 'lod': (1, 2, 3, ()),
                  'gather4_po': (1, 3, 4, (2,)), 'gather4_po_c': (1, 3, 4, (2, 5)),
                  'ld': (1, 2, None, ()), 'ld_ms': (1, 2, None, (3,)), 'ld_uav_typed': (1, 2, None, ()),
                  'ld_raw': (1, 2, None, ()), 'ld_structured': (1, 3, None, (2,)),
                  'resinfo': (1, 2, None, ()), 'bufinfo': (None, 1, None, ()), 'sample_info': (None, 1, None, ()),
                  'sample_pos': (2, 1, None, ())}[op]
        address_at, resource_at, sampler_at, extras = layout
        resource = o[resource_at]
        if resource.type == 'g':
            if op not in ('ld_raw', 'ld_structured') or (resource.indices and resource.indices[0][1] is not None):
                raise Unknown('groupshared memory read by ' + op)
            return self._groupshared(ins, resource.indices[0][0], flow)
        kind = 'u' if resource.type == 'u' else 't'
        if resource.indices and resource.indices[0][1] is not None:
            raise Unknown('dynamically indexed resource')
        register = resource.indices[0][0]
        binding = _binding_name(self.reflection, kind, register)
        sampler = None
        if sampler_at is not None:
            sampler = _binding_name(self.reflection, 's', o[sampler_at].indices[0][0])
        args = []
        if address_at is not None:
            want = 'i' if op.startswith('ld') or op in ('resinfo', 'sample_pos') else 'f'
            args.extend(self.read(o[address_at], want))
        for e in extras:
            want = 'i' if op in ('ld_structured', 'ld_ms', 'gather4_po', 'gather4_po_c') and e in (2, 3) else 'f'
            args.extend(self.read(o[e], want))
        if op == 'resinfo':
            bound = self._resinfo(ins, register, binding, args)
        else:
            bound = self.bindings.resource(g, op, binding, args)
        rtype = 'f'
        if ins.return_types and ins.return_types[0] in (3, 4):
            rtype = 'i'
        if op in ('ld_raw', 'ld_structured', 'bufinfo', 'sample_info') or (ins.return_types and ins.return_types[0] == 6):
            rtype = 'i'
        if bound is None:
            source = g._add('sample' if op.startswith(('sample', 'gather', 'lod')) else 'load', 'v', tuple(args),
                            opcode=op, binding=binding, sampler=sampler, offsets=ins.offsets,
                            dimension=ins.resource_dimension, controls=ins.controls)
            channels = [g.channel(source, c, rtype) for c in range(4)]
        else:
            channels = bound
        dst = o[0]
        swizzle = resource.selected()
        for lane in self.lanes(dst):
            self.write(dst, lane, channels[swizzle[lane]], ins.saturate, flow)


    def _resinfo(self, ins, register, binding, args):
        """resinfo of a texture whose size the bindings know, at a constant mip level: four constants, else None.

        SM5: (width, height, depth or array size, mip count) by resource dimension, unused components 0; width, height
        and depth at the level are max(1, size >> level), every size 0 past the last level; returned as float, as
        reciprocals (rcpFloat, the mip count stays a float) or as uint (controls bits 0-1: 0, 1, 2)."""
        g = self.g
        size = self.bindings.texture_size(binding)
        level = g.value(args[0]) if args else None
        dimension = ins.resource_dimension or self._dimensions.get(register)
        if size is None or level is None or dimension not in (2, 3, 4, 5, 6, 7, 8, 9, 10):
            return None
        width, height, depth, mips = size
        inside = level < mips
        at = lambda x: max(1, x >> level) if inside else 0   # noqa: E731
        whole = depth if inside else 0
        values = {2: (at(width), 0, 0), 7: (at(width), whole, 0), 5: (at(width), at(height), at(depth)),
                  8: (at(width), at(height), whole), 9: (at(width), at(height), whole),
                  10: (at(width), at(height), whole)}.get(dimension, (at(width), at(height), 0))
        kind = ins.controls & 3
        if kind == 2:
            return [g.const('i', v) for v in values] + [g.const('i', mips)]
        if kind == 1:
            return [g.const('f', 1.0 / v if v else math.inf) for v in values] + [g.const('f', mips)]
        return [g.const('f', v) for v in values] + [g.const('f', mips)]


class Translation:
    """The translated program: output nodes keyed ('o', register, component), the discard predicate, diagnostics."""

    def __init__(self, graph, outputs, discard, unsupported, loops, shader):
        self.graph, self.outputs, self.discard = graph, outputs, discard
        self.unsupported, self.loops, self.shader = unsupported, loops, shader

    def output(self, register, component):
        return self.outputs.get(('o', register, component))

    def status(self):
        """'ready' when no output depends on an unknown value, else 'partial' (or 'empty' with no outputs)."""
        roots = list(self.outputs.values())
        if not roots:
            return 'empty'
        return 'partial' if self.graph.unknowns(roots + [self.discard]) else 'ready'


def translate(shader, bindings=None):
    return Translator(shader, bindings).run()


# ---------------------------------------------------------------------------------------------- evaluation

def evaluate(graph, roots, inputs=None, resource=None, constants=None):
    """Evaluate nodes numerically. ``inputs``: {name: value}, a float for a float input (pass bits_float(n) for integer
    data the program reads from a constant buffer), an int for an integer input; ``resource(node, args) -> 4 values``;
    ``constants``: {(buffer, offset): value} for dynamic cbloads (index ignored). Returns {root: value}."""
    inputs = inputs if inputs is not None else {}
    constants = constants if constants is not None else {}
    values = {}
    nodes = graph.nodes

    def visit(i):
        if i in values:
            return values[i]
        n = nodes[i]
        op = n['op']
        if op == 'const':
            v = n['value']
        elif op == 'input':
            v = inputs[n['name']]
            v = f32(v) if n['type'] == 'f' else int(v) & U32
        elif op == 'unknown':
            raise Unknown(n['reason'])
        elif op == 'cbload':
            v = f32(constants[(n['buffer'], n['offset'])])
        elif op == 'select':
            v = visit(n['args'][1]) if visit(n['args'][0]) else visit(n['args'][2])
        elif op == 'channel':
            v = visit(n['args'][0])[n['index']]
            v = f32(v) if n['type'] == 'f' else int(v) & U32
        elif op in ('sample', 'load'):
            v = resource(n, [visit(a) for a in n['args']])
        elif op in ('ddx', 'ddy'):
            v = 0.0
        else:
            v = OPS[op][2]([visit(a) for a in n['args']])
            if OPS[op][0] == 'f':
                v = f32(v)
        values[i] = v
        return v
    return {r: visit(r) for r in roots}
