__all__ = [
    'CSI',
    'Color',
    'ColorStr',
    'SGR_RESET',
    'SgrParameter',
    'SgrSequence',
    'ansicolor24Bit',
    'ansicolor4Bit',
    'ansicolor8Bit',
    'color_chain',
    'colorbytes',
    'get_ansi_type',
    'is_vt_enabled',
    'randcolor',
    'rgb2ansi_escape',
]

import collections.abc as abc
import operator as op
import os
import random
import re
import sys
import typing as tp
from collections import Counter
from copy import deepcopy
from ctypes import byref
from enum import IntEnum
from functools import lru_cache
from types import MappingProxyType as mappingproxy, UnionType
from typing import Literal as L

import numpy as np

from .._typing import AnsiColorAlias, ColorDictKeys, Int3Tuple
from .colorconv import (
    ansi_4bit_to_rgb,
    ansi_8bit_to_rgb,
    int2rgb,
    is_u24,
    nearest_ansi_4bit_rgb,
    rgb2int,
    rgb_to_ansi_8bit,
)

CSI: tp.Final[bytes] = b'\x1b['
SGR_RESET: tp.Final[bytes] = b'\x1b[0m'
SGR_RESET_S: tp.Final[str] = '\x1b[0m'


# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR
# int enum {sgr parameter name ==> sgr code (int)}
class SgrParameter(IntEnum):
    RESET = 0
    BOLD = 1
    FAINT = 2
    ITALICS = 3
    SINGLE_UNDERLINE = 4
    SLOW_BLINK = 5
    RAPID_BLINK = 6
    NEGATIVE = 7
    CONCEALED_CHARS = 8
    CROSSED_OUT = 9
    PRIMARY = 10
    FIRST_ALT = 11
    SECOND_ALT = 12
    THIRD_ALT = 13
    FOURTH_ALT = 14
    FIFTH_ALT = 15
    SIXTH_ALT = 16
    SEVENTH_ALT = 17
    EIGHTH_ALT = 18
    NINTH_ALT = 19
    GOTHIC = 20
    DOUBLE_UNDERLINE = 21
    RESET_BOLD_AND_FAINT = 22
    RESET_ITALIC_AND_GOTHIC = 23
    RESET_UNDERLINES = 24
    RESET_BLINKING = 25
    POSITIVE = 26
    REVEALED_CHARS = 28
    RESET_CROSSED_OUT = 29
    BLACK_FG = 30
    RED_FG = 31
    GREEN_FG = 32
    YELLOW_FG = 33
    BLUE_FG = 34
    MAGENTA_FG = 35
    CYAN_FG = 36
    WHITE_FG = 37
    ANSI_256_SET_FG = 38
    DEFAULT_FG_COLOR = 39
    BLACK_BG = 40
    RED_BG = 41
    GREEN_BG = 42
    YELLOW_BG = 43
    BLUE_BG = 44
    MAGENTA_BG = 45
    CYAN_BG = 46
    WHITE_BG = 47
    ANSI_256_SET_BG = 48
    DEFAULT_BG_COLOR = 49
    FRAMED = 50
    ENCIRCLED = 52
    OVERLINED = 53
    NOT_FRAMED_OR_CIRCLED = 54
    IDEOGRAM_UNDER_OR_RIGHT = 55
    IDEOGRAM_2UNDER_OR_2RIGHT = 60
    IDEOGRAM_OVER_OR_LEFT = 61
    IDEOGRAM_2OVER_OR_2LEFT = 62
    CANCEL = 63
    BLACK_BRIGHT_FG = 90
    RED_BRIGHT_FG = 91
    GREEN_BRIGHT_FG = 92
    YELLOW_BRIGHT_FG = 93
    BLUE_BRIGHT_FG = 94
    MAGENTA_BRIGHT_FG = 95
    CYAN_BRIGHT_FG = 96
    WHITE_BRIGHT_FG = 97
    BLACK_BRIGHT_BG = 100
    RED_BRIGHT_BG = 101
    GREEN_BRIGHT_BG = 102
    YELLOW_BRIGHT_BG = 103
    BLUE_BRIGHT_BG = 104
    MAGENTA_BRIGHT_BG = 105
    CYAN_BRIGHT_BG = 106
    WHITE_BRIGHT_BG = 107


# ----------------
# CONSTANT LOOKUPS

_SGR_PARAM_VALUES = frozenset(x.value for x in SgrParameter)

# ansi 4bit {color code (int) ==> (key, RGB)}
_ANSI16C_I2KV: dict[int, tuple[ColorDictKeys, Int3Tuple]] = {
    v: (k, ansi_4bit_to_rgb(v))
    for i, k in enumerate(("fg", "bg"))
    for x in (0, 60)
    for v in (lambda n: range(n, n + 8))((30, 40)[i] + x)
}

# ansi 4bit {(key, RGB) ==> color code (int)}
_ANSI16C_KV2I = {v: k for k, v in _ANSI16C_I2KV.items()}

# ansi 4bit standard color range
_ANSI16C_STD = frozenset(x for i in (30, 40) for x in range(i, i + 8))

# ansi 4bit bright color range
_ANSI16C_BRIGHT = frozenset(_ANSI16C_I2KV.keys() - _ANSI16C_STD)

# ansi 8bit {color code (ascii bytes) ==> color dict key (str)}
_ANSI256_B2KEY: dict[L[b'38', b'48'], ColorDictKeys] = {b'38': 'fg', b'48': 'bg'}

# ansi 8bit {color dict key (str) ==> color code (int)}
_ANSI256_KEY2I = {v: int(k) for k, v in _ANSI256_B2KEY.items()}
# ----------------


@lru_cache
def _issubclass(typ: type, class_or_tuple: type | UnionType | tuple[tp.Any, ...], /):
    return issubclass(typ, class_or_tuple)


class colorbytes(bytes):
    @classmethod
    def from_rgb(cls, rgb, /):
        """Construct a `colorbytes` object from an RGB key-value pair.

        Returns
        -------
        cb
            colorbytes object

        Raises
        ------
        ValueError
            If key-value pair does not match expected structure.

        Examples
        --------
        >>> from chromatic.color.core import ansicolor4Bit, ansicolor8Bit

        >>> rgb_dict = {'fg': (255, 85, 85)}
        >>> old_ansi = ansicolor4Bit.from_rgb(rgb_dict)
        >>> repr(old_ansi)
        "ansicolor4Bit(b'91')"

        >>> new_ansi = ansicolor24Bit.from_rgb(rgb_dict)
        >>> repr(new_ansi)
        "ansicolor24Bit(b'38;2;255;85;85')"

        """

        k: ColorDictKeys
        match rgb:
            case ('fg' | 'bg') as k, v:
                pass
            case {'fg': _} | {'bg': _}:
                k, v = dict(rgb).popitem()
            case _:
                raise ValueError
        r, g, b = (
            (int(x) & 0xFF for x in v)
            if _issubclass(v.__class__, abc.Iterable)
            else int2rgb(v)
        )
        typ = DEFAULT_ANSI if cls is colorbytes else cls
        inst = super().__new__(typ, rgb2ansi_escape(typ, mode=k, rgb=(r, g, b)))
        setattr(inst, '_rgb_dict', {k: (r, g, b)})
        return inst

    def __new__(cls, ansi, /):
        self_issubtype = bool(cls is not colorbytes)
        objtype = ansi.__class__
        if self_issubtype and objtype is cls:
            return ansi
        elif not _issubclass(objtype, (bytes, bytearray)):
            raise TypeError(
                f"Expected bytes-like object, got {objtype.__name__!r} object instead"
            )
        k: ColorDictKeys
        match _unwrap_ansi_escape(ansi):
            case [color]:
                k, rgb = _ANSI16C_I2KV[int(color)]
                typ = ansicolor4Bit
            case [(b'38' | b'48') as sgr1, (b'2' | b'5') as sgr2, *rest]:
                k = _ANSI256_B2KEY[sgr1]
                if sgr2 == b'2':
                    [r, g, b] = map(int, rest)
                    rgb = r, g, b
                    typ = ansicolor24Bit
                else:
                    [color] = rest
                    rgb = ansi_8bit_to_rgb(int(color))
                    typ = ansicolor8Bit
            case _:
                raise ValueError
        if typ is not cls:
            if self_issubtype:
                typ = cls
            ansi = rgb2ansi_escape(typ, mode=k, rgb=rgb)
        inst = super().__new__(typ, ansi)
        setattr(inst, '_rgb_dict', {k: rgb})
        return inst

    def __repr__(self):
        return "{0.__class__.__name__}({0!s})".format(self)

    def to_param_buffer(self):
        obj = object.__new__(SgrParamBuffer)
        obj._value = self
        return obj

    @property
    def rgb_dict(self):
        return self._rgb_dict.items().mapping


class ansicolor4Bit(colorbytes):
    """ANSI 4-bit color format.

    Notes
    -----
    Supports 16 colors.

    +-------+---------+
    | index |  color  |
    +-------+---------+
    |     0 | black   |
    |     1 | red     |
    |     2 | green   |
    |     3 | yellow  |
    |     4 | blue    |
    |     5 | magenta |
    |     6 | cyan    |
    |     7 | white   |
    +-------+---------+

    Each color has a bright variant at ``index + 60``.

    Color codes use escape sequences of the form:
        - `CSI 30–37 m` for foreground colors.
        - `CSI 40–47 m` for background colors.
        - `CSI 90–97 m` for foreground colors (bright).
        - `CSI 100–107 m` for background colors (bright).

    Where `CSI` (Control Sequence Introducer) is `ESC[`.

    Examples
    --------
    bright red fg: `ESC[91m`
    standard green bg: `ESC[42m`
    bright white bg, black fg: `ESC[107;30m`

    """

    alias = '4b'


class ansicolor8Bit(colorbytes):
    """ANSI 8-Bit color format.

    Notes
    -----
    Supports 256 colors, mapped to the following value ranges:
        - ``(0, 15)``: Corresponds to ANSI 4-bit colors.
        - ``(16, 231)``: Represents a 6x6x6 RGB color cube.
        - ``(232, 255)``: Greyscale colors, from black to white.

    Color codes use escape sequences of the form:
        - `CSI 38;5;(n) m` for foreground colors.
        - `CSI 48;5;(n) m` for background colors.

    Where `CSI` (Control Sequence Introducer) is `ESC[` and `n` is an unsigned 8-bit integer.

    Examples
    --------
    white bg: `ESC[48;5;255m`
    bright red fg (ANSI 4-bit): `ESC[38;5;9m`
    bright red fg (color cube): `ESC[38;5;196m`

    """

    alias = '8b'


class ansicolor24Bit(colorbytes):
    """ANSI 24-Bit color format.

    Notes
    -----
    Supports all colors in the RGB color space (16,777,216 total).

    Color codes use escape sequences of the form:
        - `CSI 38;2;(r);(g);(b) m` for foreground colors.
        - `CSI 48;2;(r);(g);(b) m` for background colors.

    Where `CSI` (Control Sequence Introducer) is `ESC[` and `r,g,b` are unsigned 8-bit integers.

    Examples
    --------
    red fg: `ESC[38;2;255;85;85m`
    black bg: `ESC[48;2;0;0;0m`
    white fg, green bg: `ESC[38;2;255;255;255;48;2;0;170;0m`

    """

    alias = '24b'


if os.name == 'nt':
    from ctypes import windll, wintypes

    def _enable_vt_processing(handle: int):
        ENABLE_VT_PROCESSING = 0x0004
        k32 = windll.kernel32
        k32.GetStdHandle.restype = wintypes.HANDLE
        k32.GetConsoleMode.restype = k32.SetConsoleMode.restype = wintypes.BOOL
        h = k32.GetStdHandle(handle)
        if h == -1:
            return False
        mode = wintypes.DWORD()
        if not k32.GetConsoleMode(h, byref(mode)):
            return False
        mode.value |= ENABLE_VT_PROCESSING
        return bool(k32.SetConsoleMode(h, mode))

    def is_vt_enabled() -> bool:
        if os.environ.keys() & {
            'ANSICON',
            'COLORTERM',
            'ConEmuANSI',
            'PYCHARM_HOSTED',
            'TERM',
            'TERMINAL_EMULATOR',
            'TERM_PROGRAM',
            'WT_SESSION',
        }:
            return True
        ok = False
        for fd, handle in [(sys.stdout, -11), (sys.stderr, -12)]:
            if getattr(fd, "isatty", lambda: False)():
                ok |= _enable_vt_processing(handle)
        return ok

else:

    def is_vt_enabled() -> bool:
        return True


DEFAULT_ANSI = ansicolor8Bit if is_vt_enabled() else ansicolor4Bit

AnsiColorFormat: tp.TypeAlias = ansicolor4Bit | ansicolor8Bit | ansicolor24Bit
AnsiColorType: tp.TypeAlias = type[AnsiColorFormat]
AnsiColorParam: tp.TypeAlias = AnsiColorAlias | AnsiColorType
_ANSI_COLOR_TYPES = frozenset({ansicolor4Bit, ansicolor8Bit, ansicolor24Bit})
_ANSI_FORMAT_MAP = {k: x for x in _ANSI_COLOR_TYPES for k in (x, x.alias)}


@lru_cache(maxsize=len(_ANSI_COLOR_TYPES))
def _is_ansi_type(typ: type, /) -> bool:
    try:
        return typ in _ANSI_COLOR_TYPES
    except TypeError:
        return False


@lru_cache(maxsize=len(_ANSI_FORMAT_MAP))
def _get_ansi_type(typ, /):
    try:
        return _ANSI_FORMAT_MAP[typ]
    except (TypeError, KeyError) as e:
        if isinstance(typ, str):
            err = ValueError(f"invalid ANSI color format alias: {typ!r}")
        else:
            err = TypeError(
                str.format(
                    "Expected {}, got {.__class__.__name__!r} object instead",
                    type[AnsiColorFormat] | L[*(t.alias for t in _ANSI_COLOR_TYPES)],
                    typ,
                )
            )
        raise err from e


def get_ansi_type(typ=None, /):
    if typ is None:
        return DEFAULT_ANSI
    return _get_ansi_type(typ)


def set_default_ansi(typ, /):
    """Sets the global `DEFAULT_ANSI` variable to the specified ANSI color format"""
    if valid_typ := get_ansi_type(typ):
        global DEFAULT_ANSI
        DEFAULT_ANSI = valid_typ


@lru_cache(maxsize=1)
def sgr_pattern():
    uint8_re = r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d|\d)"
    ansicolor_re = f"[3-4]8;(?:2(?:;{uint8_re}){{3}}|5;{uint8_re})"
    sgr_param_re = (
        rf"(?:{ansicolor_re}|10[0-7]|9[0-7]|6[0-3]|5[02-5]|2[0-68-9]|[13-4]\d|\d)"
    )

    return re.compile(rf"\x1b\[(?:{sgr_param_re}(?:;{sgr_param_re})*)?m")


def _split_ansi_escape(s: str, /) -> list[tuple['SgrSequence', str]] | None:
    out = []
    i = 0
    for m in sgr_pattern().finditer(s):
        text = s[i : (j := m.start())]
        if i != j:
            out.append(text)
        ansi = _unwrap_ansi_escape(m[0].encode())
        if any(ansi):
            out.append(SgrSequence(map(int, ansi)))
    if i + 1 < len(s):
        out.append(s[i:])
    if not any(isinstance(x, SgrSequence) for x in out):
        return
    n = len(out)
    tmp = []
    for idx, x in enumerate(out):
        if idx + 1 < n and type(x) is type(out[idx + 1]):
            out[idx + 1] = x + out[idx + 1]
        else:
            tmp.append(x)
    out = tmp
    if out and len(out) % 2 != 0:
        out.append({SgrSequence: str, str: SgrSequence}[type(out[-1])]())
    return [
        (a, b) if isinstance(a, SgrSequence) else (b, a)
        for a, b in zip(out[::2], out[1::2])
    ]


def _unwrap_ansi_escape(b: bytes | bytearray, /):
    return bytes(b.removeprefix(CSI).removesuffix(b'm')).split(b';')


def _concat_ansi_escape(iterable: abc.Iterable[bytes | bytearray], /):
    return b'\x1b[%sm' % b';'.join(iterable)


def rgb2ansi_escape(
    fmt: AnsiColorAlias | AnsiColorType, /, mode: ColorDictKeys, rgb: Int3Tuple
):
    fmt = get_ansi_type(fmt)
    if len(rgb) != 3:
        raise ValueError('length of RGB value is not 3')
    try:
        if fmt is ansicolor4Bit:
            return b'%d' % _ANSI16C_KV2I[mode, nearest_ansi_4bit_rgb(rgb)]
        sgr = [_ANSI256_KEY2I[mode]]
        if fmt is ansicolor8Bit:
            sgr += [5, rgb_to_ansi_8bit(rgb)]
        else:
            sgr += [2, *rgb]
        return b';'.join(map(b'%d'.__mod__, sgr))
    except KeyError:
        pass
    if isinstance(mode, str):
        raise ValueError(f"invalid mode: {mode!r}")
    raise TypeError(
        f"expected 'mode' be {str.__name__!r}, "
        f"got {type(mode).__name__!r} object instead"
    )


class Color(int):
    """
    Color([x]) -> color

    Color(x, base=10) -> color

    Convert a number or string into a color, or return ``Color(0)`` if no arguments are given.
    Accepts the same arguments as int, but the value must be in range 0,0xFFFFFF (incl).
    """

    def __new__(cls, *args, **kwargs):
        inst = super().__new__(cls, *args, **kwargs)
        if is_u24(inst, strict=True):
            inst._rgb = int2rgb(inst)
            return inst
        raise RuntimeError("unreachable")

    def __repr__(self):
        return "{0.__class__.__name__}(0x{0:06X})".format(self)

    def __invert__(self):
        return Color(0xFFFFFF ^ self)

    @classmethod
    def from_rgb(cls, rgb, /):
        inst = super().__new__(cls, rgb2int(rgb))
        inst._rgb = int2rgb(inst)
        return inst

    @property
    def rgb(self):
        return getattr(self, '_rgb')


def randcolor():
    """Return a random color as a `Color` object"""
    return Color.from_bytes(random.randbytes(3))


class SgrParamBuffer[_T]:
    __slots__ = ('_value', '_bytes', '_is_color', '_is_reset')

    __match_args__ = ('value',)

    def __buffer__(self, flags, /):
        return self._value.__buffer__(flags)

    def __bytes__(self):
        try:
            return getattr(self, '_bytes')
        except AttributeError:
            setattr(self, '_bytes', bytes(self._value))
            return self._bytes

    def __eq__(self, other, /):
        return self._value == other

    def __hash__(self):
        return hash(self._value)

    def __init__(self, value: tp.Self | bytes = b'', /):
        if value.__class__ is self.__class__:
            self._value = value._value
        elif _issubclass(value.__class__, bytes):
            self._value = value
        else:
            err = TypeError(
                str.format(
                    "expected {0.__class__.__name__!r} or bytes-like object, "
                    "got {1.__class__.__name__!r} instead",
                    self,
                    value,
                )
            )
            raise err

    @property
    def value(self) -> _T:
        return self._value

    def __repr__(self):
        return "{0.__class__.__name__}({0._value!r})".format(self)

    def is_color(self):
        try:
            return getattr(self, '_is_color')
        except AttributeError:
            setattr(self, '_is_color', _issubclass(self._value.__class__, colorbytes))
            return self._is_color

    def is_reset(self):
        try:
            return getattr(self, '_is_reset')
        except AttributeError:
            setattr(self, '_is_reset', self._value == b'0')
            return self._is_reset


@lru_cache
def _get_sgr_nums(x: bytes, /) -> list[int]:
    """Return a list of integers from a bytestring of ANSI SGR parameters.

    Notes
    -----
    Roughly, bitwise equivalent to ``list(map(int, bytes().split(b';')))``

    """
    if x.isdigit():
        return [int(x)]
    x = x.removeprefix(CSI)[: idx if ~(idx := x.find(0x6D)) else None].removesuffix(
        b'm'
    )
    length = len(x)
    mask_indices = enumerate(
        map(
            bool,
            int.to_bytes(
                ~int.from_bytes(b';' * length) & int.from_bytes(x), length=length
            ),
        )
    )
    res = []
    buf = bytearray()
    while True:
        try:
            idx, not_delim = next(mask_indices)
            while not_delim:
                buf.append(x[idx] | 0x30)
                idx, not_delim = next(mask_indices)
            else:
                if buf:
                    res.append(int(buf))
                    buf.clear()
        except StopIteration:
            if buf:
                res.append(int(buf))
            return res


def _iter_normalized_sgr[_T: (abc.Buffer, tp.SupportsInt)](
    iterable: bytes | bytearray | abc.Iterable[_T], /
) -> abc.Iterator[int | AnsiColorFormat]:
    if isinstance(iterable, (bytes, bytearray)):
        iterable = iterable.split(b';')
    elt: object | tp.Any
    for elt in iterable:
        objtype = elt.__class__
        if objtype is SgrParamBuffer:
            elt = elt._value
            if _issubclass(elt.__class__, colorbytes):
                yield elt
            else:
                yield int(elt)
        elif _issubclass(objtype, colorbytes):
            yield elt
        elif _issubclass(objtype, abc.Buffer):
            if objtype is not bytes:
                elt = bytes(elt)
            if elt.isdigit():
                yield int(elt)
            else:
                yield from _get_sgr_nums(elt)
        elif _issubclass(objtype, tp.SupportsInt):
            yield int(elt)
        else:
            raise TypeError(
                str.format(
                    "Expected {.__name__!r} or bytes-like object, "
                    "got {.__class__.__name__!r} instead",
                    int,
                    elt,
                )
            )


def _co_yield_colorbytes(
    iterable: abc.Iterator[int], /
) -> abc.Generator[bytes | AnsiColorFormat, int, None]:
    d: dict[int, ColorDictKeys] = {38: 'fg', 48: 'bg'}
    obj = b''
    while True:
        value = yield obj
        try:
            key = d[value]
        except KeyError:
            if value in _ANSI16C_I2KV:
                obj = ansicolor4Bit.from_rgb(_ANSI16C_I2KV[value])
            else:
                obj = b'%d' % value
        else:
            kind = next(iterable)
            if kind == 5:
                obj = ansicolor8Bit(b'%d;%d;%d' % (value, kind, next(iterable)))
            else:
                r, g, b = (next(iterable) for _ in range(3))
                obj = ansicolor24Bit.from_rgb((key, (r, g, b)))


def _gen_colorbytes(
    iterable: abc.Iterable[int], /
) -> abc.Iterator[bytes | AnsiColorFormat]:
    gen = iter(iterable)
    color_coro = _co_yield_colorbytes(gen)
    next(color_coro)
    for value in gen:
        if _is_ansi_type(value.__class__):
            yield value
        else:
            yield color_coro.send(value)


def _iter_sgr[_T: (abc.Buffer, tp.SupportsInt)](
    x: bytes | bytearray | abc.Iterable[_T], /
):
    return _gen_colorbytes(_iter_normalized_sgr(x))


def _is_ansi_std_16c(value: bytes, /):
    return value.isdigit() and int(value) in _ANSI16C_STD


@lru_cache(maxsize=len(_SGR_PARAM_VALUES))
def _is_sgr_param(value: int, /):
    return value in _SGR_PARAM_VALUES


class SgrSequence(abc.MutableSequence[SgrParamBuffer]):
    _idx_attrs = ("_bg_idx", "_fg_idx")
    _key2idx = mappingproxy({"bg": "_bg_idx", "fg": "_fg_idx"})
    __slots__ = ("_sgr_params", *_idx_attrs)

    class _color_descriptor:
        def __set_name__(self, objtype, name, /):
            self.__objclass__ = objtype
            self.key = name
            self.idx = f"_{name}_idx"
            assert self.idx in objtype._idx_attrs

        def __get__(self, inst, objtype=None):
            if inst is None:
                return self
            try:
                idx = getattr(inst, self.idx)
            except AttributeError:
                params = inst._sgr_params
                for i in reversed(range(len(params))):
                    x = params[i]
                    if not x.is_color():
                        continue
                    rgb = x._value._rgb_dict
                    if self.key not in rgb:
                        continue
                    setattr(inst, self.idx, i)
                    return rgb[self.key]
                else:
                    setattr(inst, self.idx, None)
                    return
            else:
                if idx is None:
                    return
                rgb = inst._sgr_params[idx]._value._rgb_dict
                return rgb[self.key]

        def __set__(self, inst, value, /):
            if inst is None:
                raise TypeError
            if value is None:
                return delattr(inst, self.key)
            params = inst._sgr_params
            idx = hi = None
            for i in reversed(range(len(params))):
                x = params[i]
                if not x.is_color():
                    continue
                rgb = x._value._rgb_dict
                if self.key in rgb:
                    if rgb[self.key] != value:
                        if hi is None:
                            hi = i
                        continue
                    elif hi is None:
                        setattr(inst, self.idx, i)
                        return
                    else:
                        idx = i
                        break
            else:
                raise ValueError
            x = params[idx]
            params[idx] = params[hi]
            params[hi] = x
            setattr(inst, self.idx, hi)

        def __delete__(self, inst, /):
            if inst is None:
                raise TypeError
            idx = getattr(inst, self.idx, None)
            if idx is None:
                return
            params = inst._sgr_params
            new_idx = None
            for i in reversed(range(len(params))):
                if i == idx:
                    continue
                x = params[i]
                if not x.is_color():
                    continue
                if self.key in x._value._rgb_dict:
                    new_idx = i
                    break
            setattr(inst, self.idx, new_idx)

    bg = _color_descriptor()
    fg = _color_descriptor()

    def _invalidate_indices(self):
        for idx_attr in self._idx_attrs:
            try:
                delattr(self, idx_attr)
            except AttributeError:
                pass

    def insert(self, index, value, /):
        if value.__class__ is not SgrParamBuffer:
            value = SgrParamBuffer(value)
        params = self._sgr_params
        n = len(params)
        if index < 0:
            index = max(0, n + index)
        elif index > n:
            index = n
        params.insert(index, value)
        keys = value._value._rgb_dict if value.is_color() else ()
        for k, idx_attr in self._key2idx.items():
            try:
                cur = getattr(self, idx_attr)
            except AttributeError:
                continue
            if cur is not None and cur >= index:
                cur += 1
            if k in keys and (cur is None or cur < index):
                cur = index
            setattr(self, idx_attr, cur)

    def extend(self, iterable, /):
        n = len(self)
        for x in map(SgrParamBuffer, _iter_sgr(iterable)):
            self.insert(n, x)
            n += 1

    def is_color(self):
        return bool(self.bg or self.fg)

    def is_reset(self):
        return any(p.is_reset() for p in self)

    def values(self):
        for p in self._sgr_params:
            yield p._value

    def ansi_type(self):
        if self.is_color():
            typ, _ = max(
                Counter(x._value.__class__ for x in self if x.is_color()).items(),
                key=lambda x: x[1],
            )
            return typ

    def __add__(self, other, /):
        if isinstance(other, self.__class__):
            return self.__class__(x for xs in (self, other) for x in xs)
        return NotImplemented

    def __bool__(self):
        return bool(self._sgr_params)

    def __bytes__(self):
        return _concat_ansi_escape(self.values()) if self else b''

    def __copy__(self):
        inst = object.__new__(self.__class__)
        inst._sgr_params = self._sgr_params.copy()
        for attr in self._idx_attrs:
            try:
                idx = getattr(self, attr)
            except AttributeError:
                continue
            setattr(inst, attr, idx)
        return inst

    copy = __copy__

    def __deepcopy__(self, memo, /):
        inst = memo[id(self)] = object.__new__(self.__class__)
        inst._sgr_params = deepcopy(self._sgr_params, memo)
        for attr in self._idx_attrs:
            try:
                idx = getattr(self, attr)
            except AttributeError:
                continue
            setattr(inst, attr, idx)
        return inst

    def __delitem__(self, index, /):
        del self._sgr_params[index]
        self._invalidate_indices()

    def __getitem__(self, index, /):
        return self._sgr_params[index]

    def __init__(self, iterable=None, /) -> None:
        if iterable is None:
            self._sgr_params = []
        elif isinstance(iterable, SgrSequence):
            self._sgr_params = iterable._sgr_params.copy()
            for attr in self._idx_attrs:
                try:
                    idx = getattr(iterable, attr)
                except AttributeError:
                    continue
                setattr(self, attr, idx)
        else:
            colors: dict = {}
            elts: dict = {}

            def pop_color(key: str):
                if key in colors:
                    elts.pop(colors.pop(key))

            for elt in _iter_sgr(iterable):
                if elt in elts:
                    continue
                elif isinstance(elt, colorbytes):
                    for k in elt.rgb_dict:
                        pop_color(k)
                        colors[k] = elt
                    elts[elt.to_param_buffer()] = None
                    continue
                elif elt == b'0':
                    elts.clear()
                    colors.clear()
                elif elt == b'39':
                    pop_color('fg')
                elif elt == b'49':
                    pop_color('bg')
                elts[SgrParamBuffer(elt)] = None

            self._sgr_params = list(elts)

    def __iter__(self):
        return iter(self._sgr_params)

    def __len__(self):
        return len(self._sgr_params)

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self.values())})"

    def __setitem__(self, index, value, /):
        iterable = map(SgrParamBuffer, _iter_sgr(value))
        if isinstance(index, slice):
            self._sgr_params[index] = iterable
        else:
            [item] = iterable
            self._sgr_params[index] = item
        self._invalidate_indices()

    def __str__(self):
        return bytes(self).decode()

    __hash__ = None

    def clear_colors(self):
        self._sgr_params[:] = [p for p in self._sgr_params if not p.is_color()]
        self._bg_idx = self._fg_idx = None

    def set_colors(self, iterable, /, ansi_type=None):
        new_colors = dict(iterable)
        if not new_colors:
            return
        new_keys = new_colors.keys()
        keys = self._key2idx.keys()
        if not new_keys <= keys:
            raise ValueError
        if len(new_keys) == 2 and all(v is None for v in new_colors.values()):
            return self.clear_colors()
        if ansi_type is None:
            ansi_type = DEFAULT_ANSI
        self._sgr_params[:] = [
            p
            for p in self._sgr_params
            if not p.is_color() or p._value._rgb_dict.keys().isdisjoint(new_colors)
        ]
        for k in keys - new_keys:
            try:
                delattr(self, self._key2idx[k])
            except AttributeError:
                pass
        for k, v in new_colors.items():
            idx_attr = self._key2idx[k]
            if v is None:
                setattr(self, idx_attr, None)
            else:
                new_idx = len(self._sgr_params)
                x = ansi_type.from_rgb((k, v)).to_param_buffer()
                self._sgr_params.append(x)
                setattr(self, idx_attr, new_idx)

    def _rgb_dict_get(self):
        d = {}
        if (bg := self.bg) is not None:
            d["bg"] = bg
        if (fg := self.fg) is not None:
            d["fg"] = fg
        return d

    rgb_dict = property(_rgb_dict_get, set_colors, clear_colors)


_END_RESET_PATTERN = re.compile(r"\x1b\[0?m$")
_unset: tp.Any = object()


def _colorstr[_T](
    supercls: type[_T],
    obj=_unset,
    /,
    fg=None,
    bg=None,
    *,
    encoding=_unset,
    errors=_unset,
    ansi_type=_unset,
    reset=True,
) -> _T:
    buf_kwargs = {}
    if encoding is not _unset:
        buf_kwargs["encoding"] = encoding
    if errors is not _unset:
        buf_kwargs["errors"] = errors
    if buf_kwargs:
        if not _issubclass(obj.__class__, abc.Buffer):
            raise ValueError(f"unexpected keyword arguments: {set(buf_kwargs)}")
        elif not _issubclass(obj.__class__, (bytes, bytearray)):
            obj = bytes(obj)
        obj = obj.decode(**buf_kwargs)
    sgr = SgrSequence()

    if obj is not _unset:
        if _issubclass(obj.__class__, str):
            base_str = getattr(obj, 'base_str', obj)
            sgr_match = sgr_pattern().match
            while m := sgr_match(base_str):
                sgr.extend(m[0].removeprefix("\x1b[").removesuffix('m').encode())
                base_str = base_str[m.end() :]
            base_str = _END_RESET_PATTERN.sub('', base_str)
        else:
            base_str = str(obj)
    else:
        base_str = ''
    reset = bool(reset)
    if ansi_type is not _unset:
        ansi_type = get_ansi_type(ansi_type)
    elif not sgr.is_color():
        ansi_type = DEFAULT_ANSI
    else:
        ansi_type, _ = max(
            Counter(
                x._value.__class__ for x in sgr._sgr_params if x.is_color()
            ).items(),
            key=lambda x: x[1],
        )
    colors = {}
    if fg is not None:
        colors["fg"] = fg
    if bg is not None:
        colors["bg"] = bg
    try:
        for k, v in colors.items():
            match v:
                case Color(rgb=(_ as r, _ as g, _ as b)):
                    pass
                case tp.SupportsInt():
                    r, g, b = int2rgb(v)
                case [tp.SupportsInt(), tp.SupportsInt(), tp.SupportsInt()]:
                    r, g, b = (int(x) & 0xFF for x in v)
                case np.ndarray(shape=(3,)):
                    r, g, b = map(int, np.astype(v, np.uint8))
                case _:
                    raise TypeError(v.__class__)
            sgr.append(ansi_type.from_rgb((k, (r, g, b))).to_param_buffer())
    except TypeError as e:
        [typ] = e.args
        err = TypeError(
            "expected integer or vector of 3 integers, "
            f"got {typ.__name__!r} object instead"
        )
        err.__cause__ = e.__cause__
        raise err
    suffix = SGR_RESET_S if reset else ''
    inst: tp.Any = supercls.__new__(
        supercls.__thisclass__, f"{sgr}{base_str}{suffix}"  # type: ignore
    )
    inst.__dict__ |= {
        '_sgr': sgr,
        '_base_str': base_str,
        '_ansi_type': ansi_type,
        '_reset': suffix,
    }
    return inst


class _IntFloatMixin:
    """Mixin for ``int(ColorStr(...))`` / ``float(ColorStr(...))`` compatibility

    Notes
    -----
    If supplying 'base' to `int`, CPython ignores `nb_int` due to `PyUnicode_Check`.
    Use `ColorStr.base_str` directly in that case.

    """

    def __int__(self):
        try:
            return int(getattr(self, 'base_str'))
        except AttributeError:
            return int(str(self))

    def __float__(self):
        try:
            return float(getattr(self, 'base_str'))
        except AttributeError:
            return float(str(self))


class ColorStr(str, _IntFloatMixin):
    def _weak_var_update(self, **kwargs):
        expected = {"base_str", "sgr", "reset"}
        if not kwargs.keys() <= expected:
            unexpected = kwargs.keys() - expected
            raise ValueError(f'unexpected keys: {unexpected}')
        sgr = kwargs.get('sgr', self._sgr)
        base_str = kwargs.get('base_str', self.base_str)
        suffix = SGR_RESET_S if kwargs.get('reset', self.reset) else ''
        inst = super().__new__(self.__class__, f"{sgr}{base_str}{suffix}")
        inst.__dict__ |= vars(self) | {f'_{k}': v for k, v in kwargs.items()}
        return inst

    def ansi_partition(self):
        r"""Returns a 3-tuple of parts of the string
        (sgr, base string, '\x1B[0m' or '')

        """
        return str(self._sgr), self.base_str, self._reset

    def as_ansi_type(self, ansi_type, /):
        """Convert all ANSI colors in the `ColorStr` to a single ANSI type.

        Parameters
        ----------
        __ansi_type : {'4b', '8b', '24b'} or type[ansicolor4Bit | ansicolor8Bit | ansicolor24Bit]
            ANSI format to which all SGR parameters of type `colorbytes` will be cast.

        Returns
        -------
        ColorStr
            Return `self` if all ANSI formats are already the input type.
            Otherwise, return reformatted `ColorStr`.

        """
        ansi_type = get_ansi_type(ansi_type)
        if self.rgb_dict and ansi_type is not self.ansi_type:
            sgr = self._sgr.copy()
            sgr.set_colors(sgr.rgb_dict, ansi_type)
            inst = super().__new__(self.__class__, f"{sgr}{self.base_str}{self._reset}")
            inst.__dict__ |= vars(self) | {'_sgr': sgr, '_ansi_type': ansi_type}
            return inst
        return self

    def recolor(self, *args, **kwargs):
        """Return a copy of self with a new color spec.

        ``ColorStr.recolor(self, value, /, *, absolute=False) -> ColorStr``
        ``ColorStr.recolor(self, *, fg=None, bg=None, absolute=False) -> ColorStr``

        If no arguments are given, returns self unchanged.
        If 'value' is given and a `ColorStr`, return self with the colors of 'value'.
        Else, use keyword arguments ``{'fg', 'bg'}`` for colors.
        Any other mix of arguments will fail outright,
        since 'value' along with { fg=... | bg=... } is ambiguous which to use for colors.
        The 'absolute' keyword can be used with either signature.

        Keyword Args
        ------------
        fg : SupportsInt, optional
            New foreground color.

        bg : SupportsInt, optional
            New background color.

        absolute : bool, optional
            If True, clear all colors of the copied string before substitution.
            Otherwise, replace colors only where specified (default is False).

        Returns
        -------
        recolored : ColorStr

        Raises
        ------
        ValueError
            If the input arguments do not match any of the expected signatures.

        Examples
        --------
        >>> from chromatic import ColorStr, Color, randcolor
        >>> cs1 = ColorStr('foo', randcolor())
        >>> cs2 = ColorStr('bar', fg=Color(0xFF5555), bg=Color(0xFF00FF))
        >>> new_cs = cs2.recolor(bg=cs1.fg)
        >>> int(new_cs.fg) == 0xFF5555, new_cs.bg == cs1.fg
        (True, True)

        >>> cs = ColorStr("Red text", fg=0xFF0000)
        >>> recolored = cs.recolor(fg=Color(0x00FF00))
        >>> recolored.base_str, f"0x{recolored.fg:06X}"
        ('Red text', '0x00FF00')

        """
        expected = {"absolute", "fg", "bg"}
        if not kwargs.keys() <= expected:
            unexpected = kwargs.keys() - expected
            raise ValueError(f"unexpected keywords: {unexpected}")
        if kwargs.pop('absolute', False):
            if not (args or kwargs):
                return (
                    self
                    if not self._sgr.is_color()
                    else self._weak_var_update(
                        sgr=SgrSequence(p for p in self._sgr if not p.is_color())
                    )
                )
            default_fg = default_bg = None
        else:
            if not (args or kwargs):
                return self
            default_fg = self._sgr.fg
            default_bg = self._sgr.bg
        fg: Int3Tuple | None
        bg: Int3Tuple | None
        match args, kwargs:
            case [ColorStr(fg=fg_color, bg=bg_color)], {}:
                fg = getattr(fg_color, 'rgb', default_fg)
                bg = getattr(bg_color, 'rgb', default_bg)
            case [], _:
                fg = kwargs.pop('fg', default_fg)
                bg = kwargs.pop('bg', default_bg)
            case _:
                raise ValueError(
                    f"expected at most 1 positional arguments, got {len(args)}"
                    if len(args) > 1
                    else f"unexpected keywords: {set(kwargs)}"
                )
        sgr = self._sgr.copy()
        sgr.set_colors({"fg": fg, "bg": bg}, self.ansi_type)
        return self._weak_var_update(sgr=sgr)

    def strip_style(self):
        only_colors = []
        diff = False
        for x in self._sgr:
            if x.is_color():
                only_colors.append(x)
            elif not diff:
                diff = True
        if not diff:
            return self
        sgr = self._sgr.copy()
        sgr[:] = only_colors
        return self._weak_var_update(sgr=sgr)

    def add_reset(self):
        if not self.reset:
            return self._weak_var_update(reset=True)
        return self

    def remove_reset(self):
        if self.reset:
            return self._weak_var_update(reset=False)
        return self

    def swap_reset(self):
        return self.remove_reset() if self.reset else self.add_reset()

    def add_sgr_param(self, x: int, /):
        bx = SgrParamBuffer(b'%d' % SgrParameter(x))
        if bx in self._sgr:
            return self
        sgr = self._sgr.copy()
        sgr.append(bx)
        inst = super().__new__(self.__class__, f"{sgr}{self.base_str}{self._reset}")
        inst.__dict__ |= vars(self) | {
            '_sgr': sgr,
            '_ansi_type': sgr.ansi_type() or self.ansi_type,
        }
        return inst

    def remove_sgr_param(self, x: int, /):
        bx = SgrParamBuffer(b'%d' % SgrParameter(x))
        if bx not in self._sgr:
            return self
        sgr = self._sgr.copy()
        sgr.remove(bx)
        inst = super().__new__(self.__class__, f"{sgr}{self.base_str}{self._reset}")
        inst.__dict__ |= vars(self) | {
            '_sgr': sgr,
            '_ansi_type': sgr.ansi_type() or self.ansi_type,
        }
        return inst

    def blink(self):
        return self.add_sgr_param(SgrParameter.SLOW_BLINK)

    def blink_stop(self):
        return self.add_sgr_param(SgrParameter.RESET_BLINKING)

    def bold(self):
        return self.add_sgr_param(SgrParameter.BOLD)

    def faint(self):
        return self.add_sgr_param(SgrParameter.FAINT)

    def crossed_out(self):
        return self.add_sgr_param(SgrParameter.CROSSED_OUT)

    def encircle(self):
        return self.add_sgr_param(SgrParameter.ENCIRCLED)

    def italicize(self):
        return self.add_sgr_param(SgrParameter.ITALICS)

    def negative(self):
        return self.add_sgr_param(SgrParameter.NEGATIVE)

    def underline(self):
        return self.add_sgr_param(SgrParameter.SINGLE_UNDERLINE)

    def double_underline(self):
        return self.add_sgr_param(SgrParameter.DOUBLE_UNDERLINE)

    def capitalize(self):
        return self._weak_var_update(base_str=self.base_str.capitalize())

    def casefold(self):
        return self._weak_var_update(base_str=self.base_str.casefold())

    def center(self, width, fillchar=' ', /):
        return self._weak_var_update(base_str=self.base_str.center(width, fillchar))

    def count(self, x, /, *args):
        return self.base_str.count(x, *args)

    def endswith(self, suffix, /, *args):
        return self.base_str.endswith(suffix, *args)

    def expandtabs(self, /, tabsize=8):
        return self._weak_var_update(base_str=self.base_str.expandtabs(tabsize))

    def find(self, sub, /, *args):
        return self.base_str.find(sub, *args)

    def format(self, *args, **kwargs):
        return self._weak_var_update(base_str=self.base_str.format(*args, **kwargs))

    def format_map(self, mapping, /):
        return self._weak_var_update(base_str=self.base_str.format_map(mapping))

    def index(self, sub, /, *args):
        return self.base_str.index(sub, *args)

    def isalnum(self):
        return self.base_str.isalnum()

    def isalpha(self):
        return self.base_str.isalpha()

    def isascii(self):
        return self.base_str.isascii()

    def isdecimal(self):
        return self.base_str.isdecimal()

    def isdigit(self):
        return self.base_str.isdigit()

    def isidentifier(self):
        return self.base_str.isidentifier()

    def islower(self):
        return self.base_str.islower()

    def isnumeric(self):
        return self.base_str.isnumeric()

    def isprintable(self):
        return self.base_str.isprintable()

    def isspace(self):
        return self.base_str.isspace()

    def istitle(self):
        return self.base_str.istitle()

    def isupper(self):
        return self.base_str.isupper()

    def join(self, iterable, /):
        return self._weak_var_update(
            base_str=self.base_str.join(
                getattr(elt, 'base_str', elt) for elt in iterable
            )
        )

    def ljust(self, width, fillchar=' ', /):
        return self._weak_var_update(base_str=self.base_str.ljust(width, fillchar))

    def lower(self):
        return self._weak_var_update(base_str=self.base_str.lower())

    def lstrip(self, chars=None, /):
        return self._weak_var_update(base_str=self.base_str.lstrip(chars))

    def partition(self, sep, /):
        lhs, sep, rhs = (
            self._weak_var_update(base_str=s) for s in self.base_str.partition(sep)
        )
        return lhs, sep, rhs

    def removeprefix(self, prefix, /):
        return self._weak_var_update(base_str=self.base_str.removeprefix(prefix))

    def removesuffix(self, prefix, /):
        return self._weak_var_update(base_str=self.base_str.removesuffix(prefix))

    def replace(self, old, new, /, count=-1):
        return self._weak_var_update(base_str=self.base_str.replace(old, new, count))

    def rfind(self, sub, /, *args):
        return self.base_str.rfind(sub, *args)

    def rindex(self, sub, /, *args):
        return self.base_str.rindex(sub, *args)

    def rjust(self, width, fillchar=' ', /):
        return self._weak_var_update(base_str=self.base_str.rjust(width, fillchar))

    def rstrip(self, chars=None, /):
        return self._weak_var_update(base_str=self.base_str.rstrip(chars))

    def rpartition(self, sep, /):
        lhs, sep, rhs = (
            self._weak_var_update(base_str=s) for s in self.base_str.rpartition(sep)
        )
        return lhs, sep, rhs

    def rsplit(self, sep=None, maxsplit=-1):
        return [
            self._weak_var_update(base_str=s)
            for s in self.base_str.rsplit(sep=sep, maxsplit=maxsplit)
        ]

    def split(self, sep=None, maxsplit=-1):
        return [
            self._weak_var_update(base_str=s)
            for s in self.base_str.split(sep=sep, maxsplit=maxsplit)
        ]

    def splitlines(self, keepends=False):
        return [
            self._weak_var_update(base_str=s)
            for s in self.base_str.splitlines(keepends=keepends)
        ]

    def startswith(self, prefix, /, *args):
        return self.base_str.startswith(prefix, *args)

    def strip(self, chars=None, /):
        return self._weak_var_update(base_str=self.base_str.strip(chars))

    def swapcase(self):
        return self._weak_var_update(base_str=self.base_str.swapcase())

    def title(self):
        return self._weak_var_update(base_str=self.base_str.title())

    def translate(self, table, /):
        return self._weak_var_update(base_str=self.base_str.translate(table))

    def upper(self):
        return self._weak_var_update(base_str=self.base_str.upper())

    def zfill(self, width, /):
        return self._weak_var_update(base_str=self.base_str.zfill(width))

    def __add__(self, other, /):
        if isinstance(other, self.__class__):
            return self._weak_var_update(
                sgr=self._sgr + other._sgr, base_str=self.base_str + other.base_str
            )
        elif isinstance(other, str):
            return self._weak_var_update(base_str=self.base_str + other)
        return NotImplemented

    def __contains__(self, key: str, /):
        return self.base_str.__contains__(key)

    def __eq__(self, other, /):
        if _issubclass(other.__class__, self.__class__):
            return hash(self) == hash(other)
        return NotImplemented

    def __format__(self, format_spec='', /):
        """Return a formatted version of the ColorStr as described by format_spec.

        A `colorbytes` subclass alias (ie., '24b', '8b', '4b') can be prepended to
        a `str` format_spec to convert ansi types before applying the format_spec
        to the base string.

        Notes
        -----
        This method returns type `Self` instead of `str`, which can lead to
        surprising behavior when dealing with f-strings.

        Consider the following example:
        >>> from chromatic import ColorStr
        >>> cs = ColorStr("hello", fg=0xFF0000, ansi_type="24b")
        >>> cs._ansi_type
        <class 'chromatic.color.core.ansicolor24Bit'>
        >>> fstring = f"{cs:4b#<20}"
        >>> fstring.__class__
        <class 'chromatic.color.core.ColorStr'>
        >>> fstring._ansi_type
        <class 'chromatic.color.core.ansicolor4Bit'>
        >>> fstring.base_str
        'hello###############'

        In that case, the f-string eval returned a `ColorStr` object,
        because the whole f-string only consists of a single `{...}` span.

        In such cases, the underlying ``format(...) -> ColorStr`` has nothing
        to be concatenated with, so it is returned directly.

        In any case other than the single span f-string, the internals delegate
        to normal `str` concatentation, and we get a `str` result:
        >>> from chromatic import ColorStr
        >>> cs = ColorStr("hello", fg=0xFF0000, ansi_type="24b")
        >>> f"foo {cs} bar".__class__
        <class 'str'>
        >>> cs2 = ColorStr("world", bg=0x00FFFF, ansi_type="8b")
        >>> fstring_concat = f"{cs: >10}{cs2: <10}"
        >>> fstring_concat
        '\\x1b[38;2;255;0;0m     hello\\x1b[0m\\x1b[48;5;51mworld     \\x1b[0m'
        >>> fstring_concat.__class__
        <class 'str'>

        """
        if format_spec.startswith(("24b", "8b", "4b")):
            idx = format_spec.index("b") + 1
            alias = format_spec[:idx]
            format_spec = format_spec[idx:]
            inst = self.as_ansi_type(alias)
        else:
            inst = self
        return inst._weak_var_update(base_str=inst.base_str.__format__(format_spec))

    def __ge__(self, other, /):
        return self.base_str.__ge__(other)

    def __getitem__(self, key, /):
        return self._weak_var_update(base_str=self.base_str[key])

    def __gt__(self, other, /):
        return self.base_str.__gt__(other)

    def __hash__(self):
        return hash((self.__class__, str(self)))

    def __invert__(self):
        """Return a copy of `self` with inverted colors (color ^= 0xFFFFFF)"""
        sgr = self._sgr.copy()
        sgr.set_colors(
            {k: ~Color.from_rgb(v) for k, v in self._sgr.rgb_dict.items()},
            self.ansi_type,
        )
        return self._weak_var_update(sgr=sgr)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __le__(self, other, /):
        return self.base_str.__le__(other)

    def __len__(self):
        return len(self.base_str)

    def __lt__(self, other, /):
        return self.base_str.__lt__(other)

    def __matmul__(self, other, /):
        """Return a new `ColorStr` with the base string of `self` and colors of `other`"""
        if isinstance(other, ColorStr):
            return self._weak_var_update(sgr=other._sgr.copy(), reset=other.reset)
        return NotImplemented

    def __mod__(self, value, /):
        return self._weak_var_update(base_str=self.base_str % value)

    def __mul__(self, value, /):
        return self._weak_var_update(base_str=self.base_str * value)

    __rmul__ = __mul__

    def __new__(cls, obj=_unset, /, *args, **kwargs):
        return _colorstr(super(), obj, *args, **kwargs)  # noqa

    def __radd__(self, other, /):
        if isinstance(other, SgrSequence):
            return self._weak_var_update(sgr=(other + self._sgr))
        return NotImplemented

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def __xor__(self, other, /):
        """Return copy of self with colors ^ other colors"""

        if isinstance(other, self.__class__):
            xor_dict = {
                k: int2rgb(
                    Color.from_rgb(self.rgb_dict[k]) ^ Color.from_rgb(other.rgb_dict[k])
                )
                for k in self.rgb_dict.keys() & other.rgb_dict
            }
        elif isinstance(other, int):
            xor_dict = {
                k: int2rgb(Color.from_rgb(v) ^ other) for k, v in self.rgb_dict.items()
            }
        else:
            return NotImplemented
        if not xor_dict:
            return self
        sgr = self._sgr.copy()
        sgr.set_colors(xor_dict, self.ansi_type)
        return self._weak_var_update(sgr=sgr)

    @property
    def ansi(self):
        return bytes(self._sgr)

    @property
    def ansi_type(self):
        return getattr(self, '_ansi_type')

    @property
    def base_str(self):
        """The non-ANSI part of the string"""
        return getattr(self, '_base_str')

    @property
    def bg(self):
        """Background color"""
        if bg := self._sgr.bg:
            return Color.from_rgb(bg)

    @property
    def fg(self):
        """Foreground color"""
        if fg := self._sgr.fg:
            return Color.from_rgb(fg)

    @property
    def reset(self):
        return bool(self._reset)

    @property
    def rgb_dict(self):
        return self._sgr.rgb_dict


type _ChainMask = tuple[SgrSequence, str]
type _ChainMaskList = list[_ChainMask]
type _ConvertibleToMask = color_chain | ColorStr | str | SgrSequence


def _color_str_to_mask(cs: ColorStr) -> _ChainMask:
    return cs._sgr.copy(), cs.base_str


def _collect_masks(
    *elts: _ConvertibleToMask,
    masks: tp.Optional[_ChainMaskList] = None,
    ansi_type: tp.Optional[AnsiColorParam] = None,
) -> _ChainMaskList:
    if masks is None:
        masks = []
    if ansi_type is not None:
        ansi_type = get_ansi_type(ansi_type)
    for elt in elts:
        if isinstance(elt, (color_chain, ColorStr)):
            other_masks: _ChainMaskList
            try:
                other_masks = [(sgr.copy(), s) for sgr, s in getattr(elt, '_masks')]
            except AttributeError:
                other_masks = [_color_str_to_mask(elt)]
            masks.extend(other_masks)
        elif isinstance(elt, str):
            if other_masks := _split_ansi_escape(elt):
                masks.extend(other_masks)
            else:
                masks.append((SgrSequence(), elt))
        elif isinstance(elt, SgrSequence):
            masks.append((elt.copy(), ''))
        else:
            raise TypeError(elt.__class__.__name__)
    if ansi_type is not None:
        for i in range(len(masks)):
            masks[i][0].set_colors(masks[i][0].rgb_dict, ansi_type)
    return masks


class color_chain(abc.Sequence[tuple[SgrSequence, str]]):
    @staticmethod
    def _is_mask_seq(obj, /):
        if isinstance(obj, abc.Sequence):
            for x in obj:
                match x:
                    case (SgrSequence(), str()):
                        continue
                    case _:
                        break
            else:
                return True
        return False

    @classmethod
    def _from_masks_unchecked(cls, masks, /, ansi_type):
        inst = object.__new__(cls)
        inst._ansi_type = ansi_type
        inst._masks = []
        prev_fg = prev_bg = None
        for sgr, s in masks:
            for k, prev in zip(('fg', 'bg'), (prev_fg, prev_bg)):
                if prev is not None and prev == getattr(sgr, k):
                    if ansi_type is None:
                        sgr.rgb_dict = dict.fromkeys([k])
                    else:
                        sgr.set_colors(dict.fromkeys([k]), ansi_type)
            inst._masks.append((sgr, s))
            prev_fg, prev_bg = sgr.fg, sgr.bg
        return inst

    @classmethod
    def from_masks(cls, masks, /, ansi_type=None):
        if cls._is_mask_seq(masks):
            return cls._from_masks_unchecked(
                masks, ansi_type if ansi_type is None else get_ansi_type(ansi_type)
            )
        raise TypeError

    def shrink(self):
        """Return a copy where SGR sequences are joined for spans of empty string parts"""
        if self:
            maxlen = len(self._masks)
            it = enumerate(self._masks)
            out = []
            while True:
                try:
                    idx, (sgr, s) = next(it)
                    sgr = sgr.copy()
                    while idx + 1 < maxlen and not s:
                        idx, xs = next(it)
                        sgr += xs[0]; s = xs[1]  # fmt: skip
                    else:
                        out.append((sgr, s))
                except StopIteration:
                    break
        else:
            out = self.masks
        return self._from_masks_unchecked(out, ansi_type=self._ansi_type)

    def merge(self, *other):
        if not other:
            return self
        masks = self.masks
        for x in other:
            for sgr, s in x:
                if not masks[-1][-1]:
                    masks[-1] = masks[-1][0] + sgr, s
                else:
                    masks.append((sgr, s))
        return self._from_masks_unchecked(masks, ansi_type=self._ansi_type)

    def __add__(self, other, /):
        try:
            masks = _collect_masks(
                other, masks=deepcopy(self._masks), ansi_type=self._ansi_type
            )
        except TypeError as e:
            tb = e.__traceback__
            if tb and tb.tb_frame.f_code is _collect_masks.__code__:
                return NotImplemented
            raise
        else:
            return self._from_masks_unchecked(masks, ansi_type=self._ansi_type)

    def __bool__(self):
        return bool(self._masks)

    def __call__(self, obj='', /):
        return f"{self}{obj}\x1b[0m"

    def __getitem__(self, index, /):
        return self.masks[index]

    def __init__(self, iterable=None, /, *, ansi_type=None):
        self._ansi_type = None
        if ansi_type is not None:
            self._ansi_type = get_ansi_type(ansi_type)
        iterable = iterable or []
        self._masks = _collect_masks(*iterable, ansi_type=self._ansi_type)

    def __len__(self):
        return len(self._masks)

    def __or__(self, other, /):
        if _issubclass(other.__class__, self.__class__):
            return self.merge(other)
        return NotImplemented

    def __radd__(self, other, /):
        if isinstance(other, ColorStr):
            return self._from_masks_unchecked(
                [_color_str_to_mask(other), *self.masks],
                ansi_type=(
                    self._ansi_type if self._ansi_type is None else other.ansi_type
                ),
            )
        elif isinstance(other, str):
            if (parsed := _split_ansi_escape(other)) is not None:
                return self._from_masks_unchecked(
                    parsed + self.masks, ansi_type=self._ansi_type
                )
            else:
                return self._from_masks_unchecked(
                    [(SgrSequence(), other), *self.masks], ansi_type=self._ansi_type
                )
        return NotImplemented

    def __repr__(self):
        return "{.__class__.__name__}({})".format(
            self,
            ', '.join(
                [
                    repr([f"{sgr}{s}" for sgr, s in self._masks]),
                    *(
                        [f"ansi_type={self._ansi_type.alias!r}"]
                        if self._ansi_type
                        else ()
                    ),
                ]
            ),
        )

    def __str__(self):
        kwargs = {"reset": False}
        if self._ansi_type is not None:
            kwargs["ansi_type"] = self._ansi_type
        return ''.join(ColorStr(f"{sgr}{s}", **kwargs) for sgr, s in self._masks)

    @property
    def masks(self):
        return self._masks[:]
