__all__ = [
    'CSI',
    'Color',
    'ColorChainDType',
    'ColorStr',
    'SGR_RESET',
    'SgrFlag',
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
import enum
import functools as ft
import os
import random
import re
import sys
import typing as tp
from collections import Counter
from copy import deepcopy
from itertools import pairwise
from types import MappingProxyType as mappingproxy
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
class SgrParameter(enum.IntEnum):
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


SgrFlag = enum.IntFlag(
    "SgrFlag",
    [
        x.name
        for x in SgrParameter
        if not any(j <= x <= j + k for i in (30, 40) for j, k in [(i, 8), (i + 60, 7)])
    ],
)
setattr(
    SgrFlag,
    "parameters",
    property(lambda self: [SgrParameter[name] for x in self if (name := x.name)]),
)

_P2F = {SgrParameter[name].value: x.value for x in SgrFlag if (name := x.name)}
_F2P = {v: k for k, v in _P2F.items()}

# ----------------
# CONSTANT LOOKUPS

# ansi 4bit {color code (int) ==> (key, RGB)}
_ANSI16C_I2KV: dict[int, tuple[ColorDictKeys, Int3Tuple]] = {
    v: (k, ansi_4bit_to_rgb(v))
    for i, k in enumerate(("fg", "bg"))
    for x in (0, 60)
    for v in (lambda n: range(n, n + 8))((30, 40)[i] + x)
}

# ansi 4bit {(key, RGB) ==> color code (int)}
_ANSI16C_KV2I = {v: k for k, v in _ANSI16C_I2KV.items()}

# ansi 8bit {color code (ascii bytes) ==> color dict key (str)}
_ANSI256_B2KEY: dict[L[b'38', b'48'], ColorDictKeys] = {b'38': 'fg', b'48': 'bg'}

# ansi 8bit {color dict key (str) ==> color code (int)}
_ANSI256_KEY2I = {v: int(k) for k, v in _ANSI256_B2KEY.items()}
# ----------------

if tp.TYPE_CHECKING:
    _issubclass = issubclass
else:

    @ft.lru_cache
    def _issubclass(typ, class_or_tuple, /):
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
                [(k, v)] = rgb.items()
            case _:
                raise ValueError
        r, g, b = (
            (int(x) & 0xFF for x in v)
            if _issubclass(v.__class__, abc.Iterable)
            else int2rgb(v)
        )
        typ = DEFAULT_ANSI if cls is colorbytes else cls
        inst = super().__new__(typ, rgb2ansi_escape(typ, mode=k, rgb=(r, g, b)))
        inst.rgb_dict = mappingproxy({k: (r, g, b)})
        return inst

    def __new__(cls, ansi, /):
        if (objtype := ansi.__class__) is cls:
            return ansi
        elif not _issubclass(objtype, (bytes, bytearray)):
            raise TypeError(
                f"expected bytes-like object, got {objtype.__name__!r} object instead"
            )
        k: ColorDictKeys
        match _unwrap_ansi_escape(ansi):
            case [color]:
                try:
                    k, rgb = _ANSI16C_I2KV[int(color)]
                except KeyError:
                    raise ValueError(f"invalid 4bit color code: {color}")
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
            if cls is not colorbytes:
                typ = cls
            ansi = rgb2ansi_escape(typ, mode=k, rgb=rgb)
        inst = super().__new__(typ, ansi)
        inst.rgb_dict = mappingproxy({k: rgb})
        return inst

    def __repr__(self):
        return "{0.__class__.__name__}({0!s})".format(self)

    def kind(self):
        [k] = self.rgb_dict
        return k

    def to_param_buffer(self) -> 'SgrParamBuffer[tp.Self]':
        obj = object.__new__(SgrParamBuffer)
        obj._value = self
        obj._is_color = True
        return obj

    rgb_dict: mappingproxy[L["fg"], Int3Tuple] | mappingproxy[L["bg"], Int3Tuple]


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
    typecode = 1


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
    typecode = 2


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
    typecode = 3


if os.name == 'nt':
    from ctypes import byref, windll, wintypes

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
_ANSI_FORMAT_MAP = {k: x for x in _ANSI_COLOR_TYPES for k in (x, x.alias, x.typecode)}


@ft.lru_cache(maxsize=len(_ANSI_COLOR_TYPES))
def _is_ansi_type(typ: type, /) -> bool:
    try:
        return typ in _ANSI_COLOR_TYPES
    except TypeError:
        return False


@ft.lru_cache(maxsize=len(_ANSI_FORMAT_MAP))
def _get_ansi_type(typ, /):
    try:
        return _ANSI_FORMAT_MAP[typ]
    except (TypeError, KeyError) as e:
        if isinstance(typ, str):
            err = ValueError(f"invalid ANSI color format alias: {typ!r}")
        else:
            import operator as op

            expected = ft.reduce(
                op.or_,
                (
                    L[t.alias, t.typecode]
                    for t in sorted(_ANSI_COLOR_TYPES, key=op.attrgetter("typecode"))
                ),
                AnsiColorType,
            )
            err = TypeError(
                str.format(
                    "expected {}, got {.__class__.__name__!r} object instead",
                    expected,
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


@ft.lru_cache(maxsize=1)
def sgr_pattern():
    uint8_re = r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d|\d)"
    truecolor_re = f"(?:2;(?:{uint8_re}?;){{2}}(?:{uint8_re}|;))"
    ansi256_re = f"(?:5;(?:{uint8_re}?|;))"
    color_re = f"[3-4]8;(?:{truecolor_re}|{ansi256_re})"
    sgr_param_re = rf"(?:{color_re}|10[0-7]|9[0-7]|6[0-3]|5[02-5]|2[0-68-9]|[13-4]?\d)?"

    return re.compile(rf"\x1b\[(?:{sgr_param_re}(?:;{sgr_param_re})*)?m")


def _unwrap_ansi_escape(b: bytes | bytearray, /):
    return [
        x or b"0"
        for x in (
            bytes(b).removeprefix(CSI).removesuffix(b"m").removesuffix(b";").split(b";")
        )
    ]


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
            return inst
        raise RuntimeError("unreachable")

    def __repr__(self):
        return "{0.__class__.__name__}(0x{0:06X})".format(self)

    def __invert__(self):
        return self.__class__(0xFFFFFF ^ self)

    @classmethod
    def from_rgb(cls, rgb, /):
        return super().__new__(cls, rgb2int(rgb))

    @property
    def rgb(self):
        return (self >> 16) & 0xFF, (self >> 8) & 0xFF, self & 0xFF


def randcolor():
    """Return a random color as a `Color` object"""
    return Color.from_bytes(random.randbytes(3))


class SgrParamBuffer[_T]:
    """Transparent wrapper type for `SgrSequence` members"""

    __slots__ = ('_value', '_bytes', '_is_color', '_is_reset')
    __match_args__ = ('value',)

    def __buffer__(self, flags, /):
        return self._value.__buffer__(flags)

    def __bytes__(self):
        try:
            return getattr(self, '_bytes')
        except AttributeError:
            res = self._bytes = bytes(self._value)
            return res

    def __eq__(self, other, /):
        if isinstance(other, (self.__class__, bytes)):
            return self._value == getattr(other, "_value", other)
        return NotImplemented

    def __hash__(self):
        return hash(self._value)

    def __new__(cls, value: tp.Self | bytes, /) -> tp.Self:
        if (objtype := value.__class__) is cls:
            return value
        elif _issubclass(objtype, (bytes, bytearray)):
            try:
                return colorbytes(value).to_param_buffer()
            except ValueError:
                if not (value.isdigit() and 0 <= (x := int(value)) <= 107):
                    raise
                elif x in {38, 48}:
                    raise
            inst = object.__new__(cls)
            inst._value = value if objtype is bytes else bytes(value)
            inst._is_reset = x == 0
            return inst
        raise TypeError(
            str.format(
                "expected {0.__name__!r} or bytes-like object, "
                "got {1.__class__.__name__!r} instead",
                cls,
                value,
            )
        )

    @property
    def value(self) -> _T:
        return self._value

    def __repr__(self):
        return "{0.__class__.__name__}({0._value!r})".format(self)

    def is_color(self):
        try:
            return getattr(self, '_is_color')
        except AttributeError:
            res = self._is_color = _issubclass(self._value.__class__, colorbytes)
            return res

    def is_reset(self):
        try:
            return getattr(self, '_is_reset')
        except AttributeError:
            res = self._is_reset = self._value == b'0'
            return res


@ft.lru_cache
def _get_sgr_nums(x: bytes, /) -> list[int]:
    """Return a list of integers from a bytestring of ANSI SGR parameters.

    Notes
    -----
    Roughly, bitwise equivalent to ``list(map(int, bytes().split(b';')))``

    """
    x = x.removeprefix(CSI)[: i if ~(i := x.find(*b"m")) else None].removesuffix(b"m")
    length = len(x)
    mask_indices = enumerate(
        map(
            bool,
            int.to_bytes(
                int.from_bytes(x) ^ int.from_bytes(b';' * length), length=length
            ),
        )
    )
    res = []
    digits = bytearray()
    for i, is_digit in mask_indices:
        try:
            while is_digit:
                digits.append(x[i] | 0x30)
                i, is_digit = next(mask_indices)
        except StopIteration:
            break
        finally:
            res.append(int(digits) if digits else 0)
        digits.clear()
    return res


def _iter_normalized_sgr[_T: (abc.Buffer, tp.SupportsInt)](
    iterable: bytes | bytearray | abc.Iterable[_T], /
) -> abc.Iterator[int | AnsiColorFormat]:
    if isinstance(iterable, (bytes, bytearray)):
        iterable = _unwrap_ansi_escape(iterable)
    for elt in iterable:
        match elt:
            case (colorbytes() as x) | SgrParamBuffer(colorbytes() as x):
                yield x
            case (
                (abc.Buffer() as x) | SgrParamBuffer(x) | (tp.SupportsInt() as x)
            ) if getattr(x, "isdigit", lambda: hasattr(x, "__int__"))():
                yield int(x)
            case abc.Buffer() as x:
                yield from _get_sgr_nums(bytes(x))
            case _:
                raise TypeError(
                    str.format(
                        "expected {.__name__!r} or bytes-like object, "
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
            elif kind != 2:
                raise ValueError(
                    f"invalid param after extended color: '{value};{kind}'"
                )
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


class SgrSequence(abc.MutableSequence[SgrParamBuffer]):
    _idx_attrs = ("_bg_idx", "_fg_idx")
    _key2idx = mappingproxy({"bg": "_bg_idx", "fg": "_fg_idx"})
    __slots__ = ("_sgr_params", *_idx_attrs)
    __match_args__ = ("_sgr_params",)

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
                k = self.key
                params = inst._sgr_params
                for i in reversed(range(len(params))):
                    x = params[i]
                    if (
                        x == b"0"
                        or (x == b"39" and k == "fg")
                        or (x == b"49" and k == "bg")
                    ):
                        break
                    if not x.is_color():
                        continue
                    rgb = x._value.rgb_dict
                    if k not in rgb:
                        continue
                    setattr(inst, self.idx, i)
                    return rgb[k]
                setattr(inst, self.idx, None)
                return
            else:
                if idx is None:
                    return
                rgb = inst._sgr_params[idx]._value.rgb_dict
                return rgb[self.key]

        def __set__(self, inst, value, /):
            if inst is None:
                raise TypeError
            k = self.key
            if value is None:
                return delattr(inst, k)
            params = inst._sgr_params
            idx = hi = None
            for i in reversed(range(len(params))):
                x = params[i]
                if (
                    x == b"0"
                    or (x == b"39" and k == "fg")
                    or (x == b"49" and k == "bg")
                ):
                    return setattr(inst, self.idx, None)
                if not x.is_color():
                    continue
                rgb = x._value.rgb_dict
                if self.key in rgb:
                    if rgb[self.key] != value:
                        if hi is None:
                            hi = i
                        continue
                    elif hi is None:
                        return setattr(inst, self.idx, i)
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
                if self.key in x._value.rgb_dict:
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
        value = SgrParamBuffer(value)
        params = self._sgr_params
        n = len(params)
        if index < 0:
            index = max(0, n + index)
        elif index > n:
            index = n
        params.insert(index, value)
        if value == b"0":
            self._invalidate_indices()
        elif value == b"39":
            try:
                delattr(self, "_fg_idx")
            except AttributeError:
                pass
        elif value == b"49":
            try:
                delattr(self, "_bg_idx")
            except AttributeError:
                pass
        else:
            keys = value._value.rgb_dict if value.is_color() else ()
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
        return super().extend(map(SgrParamBuffer, _iter_sgr(iterable)))

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

    def shrink(self):
        """Mutate self in-place by removing redundant codes from the sequence

        Specifically what is removed:
            - codes that occur before a ``b"0"``
            - fg / bg colors occurring before a respective reset code
                and vice-versa, or a subsequent color of the same kind
            - duplicate codes. the highest-index occurrence is kept
        """

        buf = []
        seen = set()
        seen_fg = seen_bg = False
        for x in reversed(self):
            if x in seen:
                continue
            seen.add(x)
            if x == b"0":
                buf.append(x)
                break
            elif x == b"39":
                if not seen_fg:
                    buf.append(x)
                    seen_fg = True
                continue
            elif x == b"49":
                if not seen_bg:
                    buf.append(x)
                    seen_bg = True
                continue
            elif not x.is_color():
                buf.append(x)
                continue
            elif x._value.kind() == "fg":
                if seen_fg:
                    continue
                seen_fg = True
            elif seen_bg:
                continue
            else:
                seen_bg = True
            buf.append(x)
        self[:] = buf[::-1]

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

    def __eq__(self, other, /):
        if isinstance(other, SgrSequence):
            return bytes(self) == bytes(other)
        return NotImplemented

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
            self._sgr_params = [SgrParamBuffer(x) for x in _iter_sgr(iterable)]

    def __iter__(self) -> abc.Iterator[SgrParamBuffer]:
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
            if not p.is_color() or p._value.rgb_dict.keys().isdisjoint(new_colors)
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
                start, end = m.span(0)
                sgr.extend(base_str[start + 2 : end - 1].encode())
                base_str = base_str[end:]
            if base_str:
                base_str = _END_RESET_PATTERN.sub('', base_str)
            elif sgr and sgr[-1] == b"0":
                del sgr[-1]
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
                raise TypeError(
                    "expected integer or vector of 3 integers, "
                    "got {.__class__.__name__!r} object instead".format(v)
                )
        sgr.append(ansi_type.from_rgb((k, (r, g, b))).to_param_buffer())
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


ColorChainDType = np.dtype(
    [
        ("char", "<U1"),
        ("sgr", "<u8"),
        ("rgb", "u1", (2, 4)),
        # rgb subarray:
        #   [[ansitype, r, g, b],     fg
        #    [ansitype, r, g, b]]     bg
    ]
)


class color_chain(abc.MutableSequence[tuple[SgrSequence, str]]):
    __slots__ = ("_ansi_type", "_items")
    __match_args__ = ("_items",)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("`copy=False` isn't supported. a copy is always created")
        flags = 0
        rgb = np.zeros((2, 4), np.uint8)
        mask_flags: list[int] = []
        mask_rgb: list[np.ndarray] = []
        strs: list[str] = []
        for sgr, s in self:
            for p in sgr:
                v = p._value
                if p.is_color():
                    v: AnsiColorFormat
                    [(k, (r, g, b))] = v.rgb_dict.items()
                    rgb[0 if k == "fg" else 1] = (v.typecode, r, g, b)
                else:
                    iv = int(v)
                    if iv == 0:
                        flags = 0
                        rgb[:] = 0
                    elif iv == 39:
                        rgb[0] = 0
                    elif iv == 49:
                        rgb[1] = 0
                    else:
                        flags |= _P2F[iv]
            mask_flags.append(flags)
            mask_rgb.append(rgb.copy())
            strs.append(s)
        if not strs:
            return np.empty(0, ColorChainDType)
        lengths = np.fromiter(map(len, strs), np.intp, len(strs))
        arr = np.empty(int(lengths.sum()), ColorChainDType)
        if not arr.size:
            return arr
        arr["char"] = np.frombuffer("".join(strs).encode("utf-32-le"), dtype="<U1")
        arr["sgr"] = np.repeat(np.asarray(mask_flags, np.uint64), lengths)
        arr["rgb"] = np.repeat(np.stack(mask_rgb), lengths, axis=0)
        return arr if dtype is None else arr.astype(dtype, copy=False)

    @classmethod
    def fromarray(cls, arr, /, *, ansi_type=None) -> tp.Self:
        arr = np.asarray(arr, ColorChainDType).reshape(-1)
        n = arr.size
        if not n:
            return cls(ansi_type=ansi_type)
        change = np.empty(n, bool)
        change[0] = True
        change[1:] = (
            (arr["sgr"][1:] != arr["sgr"][:-1]) |
            (arr["rgb"][1:] != arr["rgb"][:-1]).any(axis=(1, 2))
        )   # fmt: skip
        prev_flags = 0
        prev_rgb = np.zeros((2, 4), np.uint8)
        masks: list[tuple[SgrSequence, str]] = []
        chars = arr["char"]
        for start, stop in pairwise([*map(int, np.flatnonzero(change)), n]):
            cur_flags = int(arr["sgr"][start])
            cur_rgb = arr["rgb"][start]
            params: list[int] = []
            if prev_flags & ~cur_flags:
                params.append(0)
                emit_flags = cur_flags
                emit_rows = [i for i in (0, 1) if cur_rgb[i, 0]]
            else:
                emit_flags = cur_flags & ~prev_flags
                emit_rows = [
                    i
                    for i in (0, 1)
                    if cur_rgb[i, 0] and (cur_rgb[i] != prev_rgb[i]).any()
                ]
                if prev_rgb[0, 0] and not cur_rgb[0, 0]:
                    params.append(39)
                if prev_rgb[1, 0] and not cur_rgb[1, 0]:
                    params.append(49)
            params.extend(_F2P[m.value] for m in SgrFlag(emit_flags))
            sgr = SgrSequence(sorted(params))
            for i in emit_rows:
                sgr.set_colors(
                    {("fg" if i == 0 else "bg"): tuple(map(int, cur_rgb[i, 1:]))},
                    _ANSI_FORMAT_MAP[int(cur_rgb[i, 0])],
                )
            masks.append((sgr, "".join(chars[start:stop])))
            prev_flags, prev_rgb = cur_flags, cur_rgb
        return cls(masks, ansi_type=ansi_type)

    @staticmethod
    def _coerce(item, /) -> abc.Iterator[tuple[SgrSequence, str]]:
        match item:
            case (SgrSequence() as sgr, _ as s) | ColorStr(
                _sgr=sgr, base_str=_ as s
            ) if s.__class__ is str:
                yield (sgr.copy(), s)
            case str() as s:
                if spans := [m.span(0) for m in sgr_pattern().finditer(s)]:
                    [ix0, *bounds] = [
                        slice(*x)
                        for i, span in enumerate(spans)
                        for x in [
                            (None if i == 0 else spans[i - 1][1], span[0]),
                            (span[0] + 2, span[1] - 1),
                        ]
                    ]
                    if s0 := s[ix0]:
                        yield (SgrSequence(), s0)
                    if not bounds:
                        return
                    bounds.append(slice(spans[-1][1], None))
                    sgr_prev: SgrSequence | None = None
                    for ix_sgr, ix_s in zip(bounds[::2], bounds[1::2], strict=True):
                        params = (
                            int(n or 0) for n in s[ix_sgr].removesuffix(";").split(";")
                        )
                        if sn := s[ix_s]:
                            if sgr_prev is None:
                                yield (SgrSequence(params), sn)
                            else:
                                sgr_prev.extend(params)
                                yield (sgr_prev, sn)
                                sgr_prev = None
                        elif sgr_prev is None:
                            sgr_prev = SgrSequence(params)
                        else:
                            sgr_prev.extend(params)
                    if sgr_prev is not None:
                        yield (sgr_prev, "")
                else:
                    yield (SgrSequence(), s)
            case SgrSequence() as sgr:
                yield (sgr, "")
            case _:
                raise TypeError

    def insert(self, index, value, /):
        [value] = self._coerce(value)
        if (ansi_type := self._ansi_type) and (rgb_d := value[0].rgb_dict):
            value[0].set_colors(rgb_d, ansi_type)
        self._items.insert(index, value)

    def shrink(self):
        """Mutate self in-place by joining SGR sequences for spans of empty string parts
        and vice-versa.

        This operation removes items from the sequence, so prior length assumptions
        should be considered invalidated by calling this method.
        """
        maxlen = len(self)
        if maxlen <= 1:
            return
        buf = []
        it = enumerate(self)
        for idx, (sgr, s) in it:
            while idx + 1 < maxlen and not s:
                idx, (_sgr, s) = next(it)
                sgr += _sgr
            sgr.shrink()
            buf.append((sgr, s))
        idx = len(buf) - 1
        while idx > 0:
            sgr, s = buf[idx]
            while idx - 1 >= 0 and not sgr:
                buf[idx] = None
                idx -= 1
                sgr, _s = buf[idx]
                s = _s + s
            buf[idx] = sgr, s
            idx -= 1
        self[:] = filter(None, buf)

    def splitlines(self):
        if not self:
            return []
        pend_cr = opened = False
        buf, out = [], []
        carry = SgrSequence()
        for sgr, s in self:
            if s:
                if pend_cr:
                    s = s.removeprefix("\n")
                pend_cr = s.endswith("\r")
            if s:
                # cpython/main/Objects/stringlib/split.h#L336
                # splitlines just for '\n'
                lines = []
                str_len = len(s)
                i = j = 0
                while i < str_len:
                    while i < str_len and s[i] != "\n":
                        i += 1
                    eol = i
                    if i < str_len:
                        i += 1
                    lines.append(s[j:eol])
                    j = i

                last = len(lines) - 1
                tail_open = s[-1] not in "\r\n\v\f"
                for i, line in enumerate(lines):
                    if not opened:
                        if carry:
                            buf.append((carry.copy(), ""))
                        opened = True
                    buf.append((sgr, line))
                    if i < last or not tail_open:
                        out.append(buf)
                        buf, opened = [], False
            elif opened:
                buf.append((sgr, ""))
            carry += sgr
            carry.shrink()
        if opened:
            out.append(buf)
        cls = self.__class__
        for i, line in enumerate(out):
            color_chain.shrink(x := cls(line))
            out[i] = x
        return out

    def term_array(self, shape=None, fillchar=""):
        if shape and not (
            isinstance(shape, abc.Sequence)
            and len(shape) == 2
            and all(isinstance(x, int) for x in shape)
        ):
            raise ValueError(f"expected 2d shape: {shape}")
        rows = [*map(np.array, self.splitlines())]
        h, w = (None, None) if shape is None else shape
        if w is not None:
            rows = [r[i : i + w] for r in rows for i in range(0, len(r) or 1, w)]
        if h is not None:
            del rows[h:]
            rows += [np.empty(0, ColorChainDType)] * (h - len(rows))
        if not rows:
            return np.zeros((0, 0), ColorChainDType)
        lengths = np.fromiter(map(len, rows), np.intp, len(rows))
        width = int(lengths.max(initial=0)) if w is None else w
        out = np.zeros((len(rows), width), ColorChainDType)
        mask = np.arange(width) < lengths[:, None]
        out[mask] = np.concatenate(rows)
        if fillchar:
            out["char"][~mask] = fillchar
        return out

    def __add__(self, other, /):
        if isinstance(other, str):
            return color_chain(f"{self}{other}")
        if isinstance(other, abc.Iterable):
            return color_chain(
                (x for xs in (self, other) for x in xs),
                ansi_type=self._ansi_type or getattr(other, "_ansi_type", None),
            )
        return NotImplemented

    def __bool__(self):
        return bool(self._items)

    def __call__(self, obj='', /):
        return f"{self}{obj}\x1b[0m"

    def __delitem__(self, index, /):
        del self._items[index]

    def __eq__(self, other, /):
        if isinstance(other, color_chain):
            return self._items == other._items
        return NotImplemented

    def __getitem__(self, index, /):
        return self._items[index]

    def __init__(self, iterable=None, /, *, ansi_type=None):
        if ansi_type is not None:
            self._ansi_type = ansi_type = get_ansi_type(ansi_type)
        else:
            self._ansi_type = None
        if iterable is None:
            self._items = []
            return
        elif isinstance(iterable, str):
            iterable = [iterable]
        buf = []
        for item in iterable:
            for sgr, s in self._coerce(item):
                if ansi_type and (sgr.bg or sgr.fg):
                    sgr.set_colors(sgr.rgb_dict, ansi_type)
                buf.append((sgr, s))
        self._items = buf

    def __len__(self):
        return len(self._items)

    def __radd__(self, other, /):
        if isinstance(other, str):
            return color_chain(f"{other}{self}")
        if isinstance(other, abc.Iterable):
            return color_chain(
                (x for xs in (other, self) for x in xs),
                ansi_type=getattr(other, "_ansi_type", self._ansi_type),
            )
        return NotImplemented

    def __repr__(self):
        constructor_args = repr([f"{sgr}{s}" for sgr, s in self])
        if self._ansi_type is not None:
            constructor_args += f", ansi_type={self._ansi_type.alias!r}"
        return "{.__class__.__name__}({})".format(self, constructor_args)

    def __setitem__(self, index, value, /):
        ansi_type = self._ansi_type

        def _validate(obj, /):
            if (
                isinstance(obj, tuple)
                and len(obj) == 2
                and isinstance(obj[0], SgrSequence)
                and obj[1].__class__ is str
            ):
                sgr, _ = obj
                if ansi_type and (rgb_d := sgr.rgb_dict):
                    sgr.set_colors(rgb_d, ansi_type)
                return obj
            raise TypeError

        if isinstance(index, slice):
            self._items[index] = list(map(_validate, value))
        else:
            self._items[index] = _validate(value)

    def __str__(self):
        return ''.join(
            ColorStr(f"{sgr}{s}", ansi_type=self._ansi_type, reset=False) if sgr else s
            for sgr, s in self
        )
