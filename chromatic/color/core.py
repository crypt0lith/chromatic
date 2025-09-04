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

import os
import random
import re
import sys
from collections import Counter
from collections.abc import Buffer, Mapping
from copy import deepcopy
from ctypes import byref
from enum import IntEnum
from functools import lru_cache
from typing import (
    Any,
    Final,
    Generator,
    Iterable,
    Iterator,
    Literal as L,
    MutableSequence,
    Sequence,
    SupportsIndex,
    SupportsInt,
    TypeAlias,
    TypeVar,
    cast,
)

import numpy as np

from .colorconv import (
    ansi_4bit_to_rgb,
    ansi_8bit_to_rgb,
    int2rgb,
    is_u24,
    nearest_ansi_4bit_rgb,
    rgb2int,
    rgb_to_ansi_8bit,
)
from .._typing import AnsiColorAlias, ColorDictKeys, Int3Tuple

# os.system('')

CSI: Final[bytes] = b'\x1b['
SGR_RESET: Final[bytes] = CSI + b'0m'
SGR_RESET_S: Final[str] = SGR_RESET.decode()


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
_ANSI16C_I2KV = cast(
    dict[int, tuple[ColorDictKeys, Int3Tuple]],
    {
        v: (k, ansi_4bit_to_rgb(v))
        for x in (
            zip(('fg', 'bg'), (j, j + 10)) for i in (30, 90) for j in range(i, i + 8)
        )
        for (k, v) in x
    },
)

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


class colorbytes(bytes):

    @classmethod
    def from_rgb(cls, __rgb):
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
        >>> rgb_dict = {'fg': (255, 85, 85)}
        >>> old_ansi = ansicolor4Bit.from_rgb(rgb_dict)
        >>> repr(old_ansi)
        "ansicolor4Bit(b'91')"

        >>> new_ansi = ansicolor24Bit.from_rgb(rgb_dict)
        >>> repr(new_ansi)
        "ansicolor24Bit(b'38;2;255;85;85')"
        """

        k: ColorDictKeys
        match __rgb:
            case ('fg' | 'bg') as k, v:
                pass
            case {'fg': _} | {'bg': _}:
                k, v = dict(__rgb).popitem()
            case _:
                raise ValueError
        r, g, b = map(int, v) if isinstance(v, Iterable) else int2rgb(int(v))
        typ: AnsiColorType = cls if cls is not colorbytes else DEFAULT_ANSI
        inst = bytes.__new__(typ, rgb2ansi_escape(typ, mode=k, rgb=(r, g, b)))
        setattr(inst, '_rgb_dict', {k: (r, g, b)})
        return inst

    def __new__(cls, __ansi):
        if not isinstance(__ansi, (bytes, bytearray)):
            raise TypeError(
                f"Expected bytes-like object, got {type(__ansi).__name__!r} object instead"
            )
        issubtype = bool(cls is not colorbytes)
        if issubtype and type(__ansi) is cls:
            return __ansi
        k: ColorDictKeys
        match _unwrap_ansi_escape(__ansi):
            case [color]:
                k, rgb = _ANSI16C_I2KV[int(color)]
                typ = ansicolor4Bit
            case [(b'38' | b'48') as sgr1, (b'2' | b'5') as sgr2, *rest]:
                k = _ANSI256_B2KEY[sgr1]
                if sgr2 == b'2':
                    [r, g, b] = rest
                    rgb = int(r), int(g), int(b)
                    typ = ansicolor24Bit
                else:
                    [color] = rest
                    rgb = ansi_8bit_to_rgb(int(color))
                    typ = ansicolor8Bit
            case _:
                raise ValueError
        if typ is not cls:
            if issubtype:
                __ansi = rgb2ansi_escape(cls, mode=k, rgb=rgb)
                typ = cls
            else:
                __ansi = rgb2ansi_escape(typ, mode=k, rgb=rgb)
        inst = bytes.__new__(typ, __ansi)
        setattr(inst, '_rgb_dict', {k: rgb})
        return inst

    def __repr__(self):
        return f"{type(self).__name__}({self})"

    def to_param_buffer(self):
        obj = object.__new__(SgrParamBuffer)
        obj._value = self
        return obj

    @property
    def rgb_dict(self):
        return self._rgb_dict.items().mapping


class ansicolor4Bit(colorbytes):
    """ANSI 4-bit color format.

    Supports 16 colors:
        * 8 standard colors:
            {0: black, 1: red, 2: green, 3: yellow, 4: blue, 5: magenta, 6: cyan, 7: white}
        * 8 bright colors, each mapping to a standard color (bright = standard + 8).

    Color codes use escape sequences of the form:
        * `CSI 30–37 m` for standard foreground colors.
        * `CSI 40–47 m` for standard background colors.
        * `CSI 90–97 m` for bright foreground colors.
        * `CSI 100–107 m` for bright background colors.
    Where `CSI` (Control Sequence Introducer) is `ESC[`.

    Examples
    --------
    bright red fg:
        `ESC[91m`

    standard green bg:
        `ESC[42m`

    bright white bg, black fg:
        `ESC[107;30m`
    """

    alias = '4b'


class ansicolor8Bit(colorbytes):
    """ANSI 8-Bit color format.

    Supports 256 colors, mapped to the following value ranges:
        * (0, 15): Corresponds to ANSI 4-bit colors.
        * (16, 231): Represents a 6x6x6 RGB color cube.
        * (232, 255): Greyscale colors, from black to white.

    Color codes use escape sequences of the form:
        * `CSI 38;5;(n) m` for foreground colors.
        * `CSI 48;5;(n) m` for background colors.
    Where `CSI` (Control Sequence Introducer) is `ESC[` and `n` is an unsigned 8-bit integer.

    Examples
    --------
    white bg:
        `ESC[48;5;255m`

    bright red fg (ANSI 4-bit):
        `ESC[38;5;9m`

    bright red fg (color cube):
        `ESC[38;5;196m`
    """

    alias = '8b'


class ansicolor24Bit(colorbytes):
    """ANSI 24-Bit color format.

    Supports all colors in the RGB color space (16,777,216 total).

    Color codes use escape sequences of the form:
        * `CSI 38;2;(r);(g);(b) m` for foreground colors.
        * `CSI 48;2;(r);(g);(b) m` for background colors.
    Where `CSI` (Control Sequence Introducer) is `ESC[` and `r,g,b` are unsigned 8-bit integers.

    Examples
    --------
    red fg:
        `ESC[38;2;255;85;85m`

    black bg:
        `ESC[48;2;0;0;0m`

    white fg, green bg:
        `ESC[38;2;255;255;255;48;2;0;170;0m`
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

    def is_vt_enabled():
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

    def is_vt_enabled():
        return True


DEFAULT_ANSI = ansicolor8Bit if is_vt_enabled() else ansicolor4Bit

AnsiColorFormat: TypeAlias = ansicolor4Bit | ansicolor8Bit | ansicolor24Bit
AnsiColorType: TypeAlias = type[AnsiColorFormat]
AnsiColorParam: TypeAlias = AnsiColorAlias | AnsiColorType
_AnsiColor_co = TypeVar('_AnsiColor_co', bound=colorbytes, covariant=True)
_ANSI_COLOR_TYPES = cast(
    frozenset[AnsiColorType], frozenset(colorbytes.__subclasses__())
)
_ANSI_FORMAT_MAP = {k: x for x in _ANSI_COLOR_TYPES for k in [x, x.alias]}


@lru_cache(maxsize=len(_ANSI_COLOR_TYPES))
def _is_ansi_type(typ: type):
    try:
        return typ in _ANSI_COLOR_TYPES
    except TypeError:
        return False


@lru_cache(maxsize=len(_ANSI_FORMAT_MAP))
def get_ansi_type(typ):
    try:
        return _ANSI_FORMAT_MAP[typ]
    except (TypeError, KeyError) as e:
        if isinstance(typ, str):
            err = ValueError(f"invalid ANSI color format alias: {e!r}")
        else:
            from .._typing import unionize

            repr_getter = lambda t: (t if isinstance(t, type) else type(t))
            msg = (
                "Expected {.__name__!r} or {}, got {.__name__!r} object instead".format(
                    str,
                    type[unionize(set(map(repr_getter, _ANSI_FORMAT_MAP.values())))],
                    repr_getter(typ),
                )
            )
            err = TypeError(msg)
        err.__cause__ = e.__cause__
        raise err


def set_default_ansi(typ):
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


def _split_ansi_escape(__s: str) -> list[tuple['SgrSequence', str]] | None:
    out = []
    i = 0
    for m in sgr_pattern().finditer(__s):
        text = __s[i : (j := m.start())]
        if i != j:
            out.append(text)
        ansi = _unwrap_ansi_escape(m[0].encode())
        if any(ansi):
            out.append(SgrSequence(map(int, ansi)))
    if i + 1 < len(__s):
        out.append(__s[i:])
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


def _unwrap_ansi_escape(__b: bytes | bytearray):
    return bytes(__b.removeprefix(CSI).removesuffix(b'm')).split(b';')


def _concat_ansi_escape(__it: Iterable[bytes | bytearray]):
    return b'\x1b[%sm' % b';'.join(__it)


def rgb2ansi_escape(fmt, mode, rgb):
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
    except KeyError as e:
        if isinstance(mode, str):
            err = ValueError(f"invalid mode: {mode!r}")
        else:
            err = TypeError(
                f"expected 'mode' be {str.__name__!r}, "
                f"got {type(mode).__name__!r} object instead"
            )
        err.__cause__ = e.__cause__
        raise err


class Color(int):
    """
    Color([x]) -> color
    Color(x, base=10) -> color

        Convert a number or string into a color, or return Color(0) if no arguments are given.
        Accepts the same arguments as int, but the value must be in range 0,0xFFFFFF (incl).
    """

    def __new__(cls, *args, **kwargs):
        if is_u24(__x := int(*args, **kwargs), strict=True):
            inst = super().__new__(cls, __x)
            inst._rgb = int2rgb(inst)
            return inst

    def __repr__(self):
        return f"{type(self).__name__}(0x{self:06X})"

    def __invert__(self):
        return Color(0xFFFFFF ^ self)

    @classmethod
    def from_rgb(cls, rgb):
        inst = super().__new__(cls, rgb2int(rgb))
        inst._rgb = int2rgb(inst)
        return inst

    @property
    def rgb(self):
        return getattr(self, '_rgb')


def randcolor():
    """Return a random color as a Color object."""
    return Color.from_bytes(random.randbytes(3))


class SgrParamBuffer[_T]:
    __slots__ = ('_value', '_bytes', '_is_color', '_is_reset')

    __match_args__ = ('value',)

    def __buffer__(self, __flags):
        return self.value.__buffer__(__flags)

    def __bytes__(self):
        try:
            return getattr(self, '_bytes')
        except AttributeError:
            setattr(self, '_bytes', bytes(self.value))
            return self._bytes

    def __eq__(self, other):
        return self.value == other

    def __hash__(self):
        return hash(self.value)

    def __init__(self, __value=b''):
        if isinstance(__value, bytes):
            self._value = __value
        elif isinstance(__value, type(self)):
            self._value = __value._value
        else:
            err = TypeError(
                str.format(
                    "expected {0.__class__.__name__!r} or bytes-like object, "
                    "got {1.__class__.__name__!r} instead",
                    self,
                    __value,
                )
            )
            raise err

    @property
    def value(self) -> _T:
        return self._value

    def __repr__(self):
        return f"{type(self).__name__}({self._value!r})"

    def is_color(self):
        try:
            return getattr(self, '_is_color')
        except AttributeError:
            setattr(self, '_is_color', isinstance(self._value, colorbytes))
            return self._is_color

    def is_reset(self):
        try:
            return getattr(self, '_is_reset')
        except AttributeError:
            setattr(self, '_is_reset', self._value == b'0')
            return self._is_reset


@lru_cache
def _get_sgr_nums(__x: bytes) -> list[int]:
    """Return a list of integers from a bytestring of ANSI SGR parameters.

    Roughly, bitwise equivalent to `list(map(int, bytes().split(b';')))`.
    """
    if __x.isdigit():
        return [int(__x)]
    __x = __x.removeprefix(CSI)[
        : idx if ~(idx := __x.find(0x6D)) else None
    ].removesuffix(b'm')
    length = len(__x)
    mask_indices = enumerate(
        map(
            bool,
            int.to_bytes(
                ~int.from_bytes(b';' * length) & int.from_bytes(__x), length=length
            ),
        )
    )
    res = []
    buf = bytearray()
    while True:
        try:
            idx, not_delim = next(mask_indices)
            while not_delim:
                buf.append(__x[idx] | 0x30)
                idx, not_delim = next(mask_indices)
            else:
                if buf:
                    res.append(int(buf))
                    buf.clear()
        except StopIteration:
            if buf:
                res.append(int(buf))
            return res


def _iter_normalized_sgr[_T: (
    Buffer,
    SupportsInt,
)](__iter: bytes | bytearray | Iterable[_T]) -> Iterator[int | AnsiColorFormat]:
    if isinstance(__iter, (bytes, bytearray)):
        __iter = __iter.split(b';')
    for elt in __iter:
        match elt:
            case (colorbytes() as cb) | SgrParamBuffer(colorbytes() as cb):
                yield cb
            case (SupportsInt() as b) | SgrParamBuffer(_ as b):
                yield int(b)
            case Buffer():
                elt = bytes(elt)
                if elt.isdigit():
                    yield int(elt)
                else:
                    yield from _get_sgr_nums(elt)
            case _:
                raise TypeError(
                    "Expected {!r} or bytes-like object, got {!r} instead".format(
                        int.__qualname__, type(elt).__qualname__
                    )
                )


def _co_yield_colorbytes(
    __iter: Iterator[int],
) -> Generator[bytes | AnsiColorFormat, int, None]:
    d: dict[int, ColorDictKeys] = {38: 'fg', 48: 'bg'}
    obj = b''
    while True:
        value = yield obj
        try:
            key = d[value]
            kind = next(__iter)
            if kind == 5:
                obj = ansicolor8Bit(b'%d;%d;%d' % (value, kind, next(__iter)))
            else:
                r, g, b = (next(__iter) for _ in range(3))
                obj = ansicolor24Bit.from_rgb((key, (r, g, b)))
        except KeyError:
            if value in _ANSI16C_I2KV:
                obj = ansicolor4Bit.from_rgb(_ANSI16C_I2KV[value])
            else:
                obj = b'%d' % value


def _gen_colorbytes(__iter: Iterable[int]) -> Iterator[bytes | AnsiColorFormat]:
    gen = iter(__iter)
    color_coro = _co_yield_colorbytes(gen)
    next(color_coro)
    for value in gen:
        if _is_ansi_type(type(value)):
            yield value
        else:
            yield color_coro.send(value)


def _iter_sgr[_T: (Buffer, SupportsInt)](__x: bytes | bytearray | Iterable[_T]):
    return _gen_colorbytes(_iter_normalized_sgr(__x))


def _is_ansi_std_16c(__value: bytes):
    return __value.isdigit() and int(__value) in _ANSI16C_STD


@lru_cache(maxsize=len(_SGR_PARAM_VALUES))
def _is_sgr_param(__value: int):
    return __value in _SGR_PARAM_VALUES


class SgrSequence(MutableSequence[SgrParamBuffer]):
    def _update_colors(self):
        def _iter_values():
            if self._rgb_dict:
                self._rgb_dict.clear()
            rgb_keys = self._rgb_dict.keys()
            for x in reversed(self._sgr_params):
                if x.is_color():
                    if x._value._rgb_dict.keys() <= rgb_keys:
                        continue
                    self._rgb_dict |= x._value._rgb_dict
                yield x

        self._sgr_params[:] = _iter_values()
        self._sgr_params.reverse()

    def copy(self):
        return self.__copy__()

    def insert(self, __index, __value):
        if not isinstance(__value, SgrParamBuffer):
            try:
                __value = SgrParamBuffer(__value)
            except TypeError as e:
                err = TypeError(e)
                err.__cause__ = e.__cause__
                raise err
        self._sgr_params.insert(__index, __value)
        if __value.is_color():
            self._update_colors()

    def extend(self, __iter):
        self._sgr_params += [SgrParamBuffer(x) for x in _iter_sgr(__iter)]
        self._update_colors()

    def is_color(self):
        return any(p.is_color() for p in self)

    def is_reset(self):
        return any(p.is_reset() for p in self)

    def values(self):
        for p in self._sgr_params:
            yield p._value

    def ansi_type(self):
        if self.is_color():
            typ, _ = max(
                Counter(type(x._value) for x in self if x.is_color()).items(),
                key=lambda x: x[1],
            )
            return typ

    def __add__(self, other):
        if isinstance(other, SgrSequence):
            return type(self)(x for xs in (self, other) for x in xs)
        return NotImplemented

    def __bool__(self):
        return bool(self._sgr_params)

    def __bytes__(self):
        return _concat_ansi_escape(self.values()) if self else b''

    def __copy__(self):
        inst = object.__new__(type(self))
        inst._sgr_params = self._sgr_params.copy()
        inst._rgb_dict = self._rgb_dict.copy()
        return inst

    def __deepcopy__(self, memo):
        inst = memo[id(self)] = object.__new__(type(self))
        inst._sgr_params = deepcopy(self._sgr_params, memo)
        inst._rgb_dict = deepcopy(self._rgb_dict, memo)
        return inst

    def __delitem__(self, __index):
        del self._sgr_params[__index]
        self._update_colors()

    def __getitem__(self, __index):
        return self._sgr_params[__index]

    def __init__(self, __iter=None) -> None:
        if __iter is None:
            self._rgb_dict = {}
            self._sgr_params = []
        elif isinstance(__iter, SgrSequence):
            self._sgr_params = __iter._sgr_params.copy()
            self._rgb_dict = __iter._rgb_dict.copy()
        else:
            colors: dict = {}
            elts: dict = {}
            bold_bit = False
            for elt in _iter_sgr(__iter):
                if elt in elts:
                    continue
                match elt:
                    case colorbytes(__class__=ansi_type):
                        elt: AnsiColorFormat
                        if bold_bit and ansi_type is ansicolor4Bit and int(elt) <= 37:
                            elt = ansicolor4Bit(b'%d' % (int(elt) + 60))
                        for k in elt.rgb_dict:
                            x = colors.pop(k, None)
                            if x is not None:
                                elts.pop(x)
                            colors[k] = elt
                        elts[elt.to_param_buffer()] = None
                        continue
                    case b'0' | b'1' | b'22' | b'39' | b'49':
                        if elt == b'0':
                            elts.clear()
                            colors.clear()
                            bold_bit = False
                        elif elt == b'1':
                            bold_bit = True
                        elif elt == b'22':
                            bold_bit = False
                        else:
                            if x := colors.pop({b'39': 'fg', b'49': 'bg'}[elt], None):
                                elts.pop(x)
                elts[SgrParamBuffer(elt)] = None

            self._sgr_params = list(elts)
            self._rgb_dict = {
                k: v for xs in colors.values() for k, v in xs._rgb_dict.items()
            }

    def __iter__(self):
        return iter(self._sgr_params)

    def __len__(self):
        return len(self._sgr_params)

    def __repr__(self):
        return f"{type(self).__name__}({list(self.values())})"

    def __setitem__(self, __index, __value):
        if isinstance(__index, slice):
            self._sgr_params[__index] = map(SgrParamBuffer, _iter_sgr(__value))
        else:
            __index: SupportsIndex
            __value: ...
            xs = list(_iter_sgr(__value))
            if len(xs) != 1:
                err = ValueError(
                    f"parsed {len(xs)} sgr parameters, expected only 1: {xs!r}"
                )
                raise err
            self._sgr_params[__index] = SgrParamBuffer(xs.pop())
        self._update_colors()

    def __str__(self):
        return bytes(self).decode()

    __slots__ = '_rgb_dict', '_sgr_params'

    __hash__ = None

    @property
    def bg(self):
        return self._rgb_dict.get('bg')

    @property
    def fg(self):
        return self._rgb_dict.get('fg')

    @property
    def rgb_dict(self):
        return self._rgb_dict.items().mapping

    @rgb_dict.deleter
    def rgb_dict(self):
        self._rgb_dict.clear()
        self._sgr_params = [p for p in self._sgr_params if not p.is_color()]

    @rgb_dict.setter
    def rgb_dict(self, __value):
        match __value:
            case Mapping() as mapping if __value.keys() <= {'fg', 'bg'}:
                typ = DEFAULT_ANSI
            case [Mapping() as mapping, type(__base__=base) as typ] if (
                mapping.keys() <= {'fg', 'bg'} and base is colorbytes
            ):
                pass
            case _:
                raise ValueError
        mapping: Mapping[ColorDictKeys, Int3Tuple]
        typ: AnsiColorType
        for k, v in mapping.items():
            if v is None:
                if k in self._rgb_dict:
                    self._sgr_params.pop(
                        next(
                            i
                            for i, x in enumerate(self._sgr_params)
                            if x.is_color() and k in x._value._rgb_dict
                        )
                    )
            else:
                self._sgr_params.append(typ.from_rgb((k, v)).to_param_buffer())
        self._update_colors()


_END_RESET_PATTERN = re.compile(r"\x1b\[0?m$")
_unset: Any = object()


def _colorstr[_T: ColorStr](
    cls: type[_T],
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
    if buf_kwargs := {
        k: v
        for k, v in locals().items()
        if k in {"encoding", "errors"} and v is not _unset
    }:
        if not isinstance(obj, Buffer):
            raise ValueError(f"unexpected keyword arguments: {set(buf_kwargs)}")
        elif not isinstance(obj, (bytes, bytearray)):
            obj = bytes(obj)
        obj = obj.decode(**buf_kwargs)
    sgr = SgrSequence()

    if obj is not _unset:
        if isinstance(obj, str):
            base_str = str(obj)
            sgr_match = sgr_pattern().match
            while m := sgr_match(base_str):
                sgr.extend(m[0].removeprefix("\x1b[").removesuffix('m').encode())
                base_str = base_str[m.end() :]
            base_str = _END_RESET_PATTERN.sub('', base_str)
        else:
            base_str = str(obj)
    else:
        base_str = ''
    if type(reset) is not bool:
        reset = bool(reset)
    if ansi_type is not _unset:
        ansi_type = get_ansi_type(ansi_type)
    elif not sgr._rgb_dict:
        ansi_type = DEFAULT_ANSI
    else:
        ansi_type = max(
            Counter(type(x._value) for x in sgr._sgr_params if x.is_color()).items(),
            key=lambda x: x[1],
        )[0]
    try:
        for k, v in locals().items():
            if k not in {"fg", "bg"} or v is None:
                continue
            match v:
                case Color(rgb=(_ as r, _ as g, _ as b)):
                    pass
                case SupportsInt():
                    r, g, b = int2rgb(int(v))
                case [SupportsInt(), SupportsInt(), SupportsInt()]:
                    r, g, b = (int(x) & 0xFF for x in v)
                case np.ndarray(shape=(3,)):
                    r, g, b = map(int, np.astype(v, np.uint8))
                case _:
                    raise TypeError(type(v))
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
    inst: Any = str.__new__(cls, f"{sgr}{base_str}{suffix}")
    inst.__dict__ |= {
        '_sgr': sgr,
        '_base_str': base_str,
        '_ansi_type': ansi_type,
        '_reset': suffix,
    }
    return inst


class ColorStr(str):
    def _weak_var_update(self, **kwargs):
        if not kwargs.keys() <= {'base_str', 'sgr', 'reset'}:
            raise ValueError(
                f'unexpected keys: {(kwargs.keys() - {'base_str', 'sgr', 'reset'})}'
            )
        sgr = kwargs.get('sgr', self._sgr)
        base_str = kwargs.get('base_str', self.base_str)
        suffix = SGR_RESET_S if kwargs.get('reset', self.reset) else ''
        inst = super().__new__(type(self), f"{sgr}{base_str}{suffix}")
        inst.__dict__ |= vars(self) | {f'_{k}': v for k, v in kwargs.items()}
        return inst

    def ansi_partition(self):
        r"""Returns a 3-tuple of parts of the string
        (sgr, base string, '\x1B[0m' or '')
        """
        return str(self._sgr), self.base_str, self._reset

    def as_ansi_type(self, __ansi_type):
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
        ansi_type = get_ansi_type(__ansi_type)
        if self.rgb_dict and ansi_type is not self.ansi_format:
            sgr = SgrSequence(self._sgr)
            sgr.rgb_dict = sgr._rgb_dict, ansi_type
            inst = super().__new__(type(self), f"{sgr}{self.base_str}{self._reset}")
            inst.__dict__ |= vars(self) | {'_sgr': sgr, '_ansi_type': ansi_type}
            return inst
        return self

    def bold(self):
        if b'%d' % SgrParameter.BOLD in self._sgr:
            return self
        return self.update_sgr(SgrParameter.BOLD)

    def italicize(self):
        if b'%d' % SgrParameter.ITALICS in self._sgr:
            return self
        return self.update_sgr(SgrParameter.ITALICS)

    def negative(self):
        return self.update_sgr(SgrParameter.NEGATIVE)

    def recolor(self, *args, **kwargs):
        """ColorStr.recolor(self, __value, *, absolute=False) -> ColorStr
        ColorStr.recolor(self, *, fg=None, bg=None, absolute=False) -> ColorStr

        Return a copy of self with a new color spec.

        If no arguments are given, returns self unchanged.
        If __value is given and a ColorStr, return self with the colors of __value.
        Else, use keyword arguments { 'fg', 'bg' } for colors.
        Any other mix of arguments will fail outright,
        since __value along with { fg=... | bg=... } is ambiguous which to use for colors.
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
        if not kwargs.keys() <= {'absolute', 'fg', 'bg'}:
            raise ValueError(
                f"unexpected keywords: {(kwargs.keys() - {'absolute', 'fg', 'bg'})}"
            )
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
            default_fg = self._sgr._rgb_dict.get('fg')
            default_bg = self._sgr._rgb_dict.get('bg')
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
        sgr.rgb_dict = {'fg': fg, 'bg': bg}, self.ansi_format
        return self._weak_var_update(sgr=sgr)

    def strikethrough(self):
        if b'%d' % SgrParameter.CROSSED_OUT in self._sgr:
            return self
        return self.update_sgr(SgrParameter.CROSSED_OUT)

    def underline(self, double=False):
        if double:
            if b'%d' % SgrParameter.DOUBLE_UNDERLINE in self._sgr:
                return self
            elif b'%d' % SgrParameter.SINGLE_UNDERLINE in self._sgr:
                return self.update_sgr(
                    SgrParameter.SINGLE_UNDERLINE, SgrParameter.DOUBLE_UNDERLINE
                )
            else:
                return self.update_sgr(SgrParameter.DOUBLE_UNDERLINE)
        else:
            if b'%d' % SgrParameter.SINGLE_UNDERLINE in self._sgr:
                return self
            elif b'%d' % SgrParameter.DOUBLE_UNDERLINE in self._sgr:
                return self.update_sgr(
                    SgrParameter.DOUBLE_UNDERLINE, SgrParameter.SINGLE_UNDERLINE
                )
            else:
                return self.update_sgr(SgrParameter.SINGLE_UNDERLINE)

    def update_sgr(self, *args):
        """Return a copy of `self` with updated SGR sequence parameters.

        Parameters
        ----------
        *args: SgrParameter | int
            The SGR parameter value(s) to be added or removed from the `ColorStr`.
            A value already in `self` SGR sequence gets removed, else it gets added.
            If no values are passed, returns `self` unchanged.

        Returns
        -------
        ColorStr
            A new `ColorStr` object with the SGR updates applied.

        Raises
        ------
        ValueError
            If any of the SGR parameters are invalid,
            or if extended color codes { 38, 48 } are passed.

        Examples
        --------
            >>> # creating an empty ColorStr object
            >>> empty_cs = ColorStr(reset=True)
            >>> empty_cs.ansi
            b''

            >>> # adding red foreground color
            >>> red_fg = empty_cs.update_sgr(SgrParameter.RED_FG)
            >>> red_fg.rgb_dict
            {'fg': (170, 0, 0)}

            >>> # removing the same parameter
            >>> empty_cs = red_fg.update_sgr(31)
            >>> empty_cs.ansi, empty_cs.rgb_dict
            (b'', {})

            >>> # adding more parameters
            >>> styles = [SgrParameter.BOLD, SgrParameter.ITALICS, SgrParameter.NEGATIVE]
            >>> stylized_cs = empty_cs.update_sgr(*styles)
            >>> stylized_cs.ansi.replace(CSI, b'ESC[')
            b'ESC[1;3;7m'

            >>> # parameter updates also supported by the `__add__` operator
            >>> stylized_cs += SgrParameter.BLACK_BG    # add background color
            >>> stylized_cs += SgrParameter.BOLD    # remove bold style
            >>> stylized_cs.ansi.replace(CSI, b'ESC['), stylized_cs.rgb_dict
            (b'ESC[3;7;40m', {'bg': (0, 0, 0)})

        See Also
        --------
            ColorStr.recolor : to change colors
            ColorStr.as_ansi_type : to change ANSI color type
        """
        if not args:
            return self
        sgr = self._sgr.copy()
        for x in args:
            if not _is_sgr_param(x):
                raise ValueError
            bx = SgrParamBuffer(b'%d' % x)
            if bx in sgr:
                sgr.remove(bx)
            else:
                sgr.append(bx)
        inst = super().__new__(type(self), f"{sgr}{self.base_str}{self._reset}")
        inst.__dict__ |= vars(self) | {
            '_sgr': sgr,
            '_ansi_type': sgr.ansi_type() or self.ansi_format,
        }
        return inst

    def capitalize(self):
        return self._weak_var_update(base_str=self.base_str.capitalize())

    def casefold(self):
        return self._weak_var_update(base_str=self.base_str.casefold())

    def center(self, __width, __fillchar=' '):
        return self._weak_var_update(base_str=self.base_str.center(__width, __fillchar))

    def count(self, x, *args):
        return self.base_str.count(x, *args)

    def endswith(self, __suffix, *args):
        return self.base_str.endswith(__suffix, *args)

    def expandtabs(self, tabsize=8):
        return self._weak_var_update(base_str=self.base_str.expandtabs(tabsize))

    def find(self, __sub, *args):
        return self.base_str.find(__sub, *args)

    def format(self, *args, **kwargs):
        return self._weak_var_update(base_str=self.base_str.format(*args, **kwargs))

    def format_map(self, __mapping):
        return self._weak_var_update(base_str=self.base_str.format_map(__mapping))

    def index(self, __sub: str, *args):
        return self.base_str.index(__sub, *args)

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

    def join(self, __iterable):
        return self._weak_var_update(
            base_str=self.base_str.join(
                getattr(elt, 'base_str', elt) for elt in __iterable
            )
        )

    def ljust(self, __width, __fillchar=' '):
        return self._weak_var_update(base_str=self.base_str.ljust(__width, __fillchar))

    def lower(self):
        return self._weak_var_update(base_str=self.base_str.lower())

    def lstrip(self, __chars=None):
        return self._weak_var_update(base_str=self.base_str.lstrip(__chars))

    def partition(self, __sep):
        lhs, sep, rhs = (
            self._weak_var_update(base_str=s) for s in self.base_str.partition(__sep)
        )
        return lhs, sep, rhs

    def removeprefix(self, __prefix):
        return self._weak_var_update(base_str=self.base_str.removeprefix(__prefix))

    def removesuffix(self, __prefix):
        return self._weak_var_update(base_str=self.base_str.removesuffix(__prefix))

    def replace(self, __old, __new, __count=-1):
        return self._weak_var_update(
            base_str=self.base_str.replace(__old, __new, __count)
        )

    def rfind(self, __sub, *args):
        return self.base_str.rfind(__sub, *args)

    def rindex(self, __sub, *args):
        return self.base_str.rindex(__sub, *args)

    def rjust(self, __width, __fillchar=' '):
        return self._weak_var_update(base_str=self.base_str.rjust(__width, __fillchar))

    def rstrip(self, __chars=None):
        return self._weak_var_update(base_str=self.base_str.rstrip(__chars))

    def rpartition(self, __sep):
        lhs, sep, rhs = (
            self._weak_var_update(base_str=s) for s in self.base_str.rpartition(__sep)
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

    def startswith(self, __prefix, *args):
        return self.base_str.startswith(__prefix, *args)

    def strip(self, __chars=None):
        return self._weak_var_update(base_str=self.base_str.strip(__chars))

    def swapcase(self):
        return self._weak_var_update(base_str=self.base_str.swapcase())

    def title(self):
        return self._weak_var_update(base_str=self.base_str.title())

    def translate(self, __table):
        return self._weak_var_update(base_str=self.base_str.translate(__table))

    def upper(self):
        return self._weak_var_update(base_str=self.base_str.upper())

    def zfill(self, __width):
        return self._weak_var_update(base_str=self.base_str.zfill(__width))

    def __add__(self, other):
        if isinstance(other, ColorStr):
            return self._weak_var_update(
                sgr=self._sgr + other._sgr, base_str=self.base_str + other.base_str
            )
        elif isinstance(other, str):
            return self._weak_var_update(base_str=self.base_str + other)
        elif isinstance(other, SgrParameter):
            return self.update_sgr(other)
        return NotImplemented

    def __contains__(self, __key: str):
        if type(__key) is not str:
            return False
        if __key == str(self._sgr):
            return True
        if __key == SGR_RESET_S:
            return self.reset
        return bool(__key in self.base_str)

    def __eq__(self, other):
        if type(self) is type(other):
            return hash(self) == hash(other)
        return NotImplemented

    def __format__(self, format_spec=''):
        if ansi_typ := _ANSI_FORMAT_MAP.get(format_spec):  # type: ignore[arg-type]
            return str(self.as_ansi_type(ansi_typ))
        return super().__format__(format_spec)

    def __getitem__(self, __key):
        return self._weak_var_update(base_str=self.base_str[__key])

    def __hash__(self):
        return hash((type(self), str(self)))

    def __invert__(self):
        """Return a copy of `self` with inverted colors (color ^= 0xFFFFFF)"""
        sgr = self._sgr.copy()
        sgr.rgb_dict = (
            {k: ~Color.from_rgb(v) for k, v in self._sgr.rgb_dict.items()},
            self.ansi_format,
        )
        return self._weak_var_update(sgr=sgr)

    def __iter__(self):
        for c in self.base_str:
            yield self._weak_var_update(base_str=c)

    def __len__(self):
        return len(self.base_str)

    def __matmul__(self, other):
        """Return a new `ColorStr` with the base string of `self` and colors of `other`"""
        if isinstance(other, ColorStr):
            return self._weak_var_update(sgr=other._sgr.copy(), reset=other.reset)
        return NotImplemented

    def __mod__(self, __value):
        return self._weak_var_update(base_str=self.base_str % __value)

    def __mul__(self, __value):
        return self._weak_var_update(base_str=self.base_str * __value)

    def __new__(cls, obj=_unset, *args, **kwargs):
        return _colorstr(cls, obj, *args, **kwargs)

    def __radd__(self, other):
        if isinstance(other, SgrSequence):
            return self._weak_var_update(sgr=(other + self._sgr))
        return NotImplemented

    def __repr__(self):
        return f'{type(self).__name__}({str(self)!r})'

    def __xor__(self, other):
        """Return copy of self with colors ^ other colors"""

        k: L['fg', 'bg']
        if isinstance(other, type(self)):
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
        sgr.rgb_dict = xor_dict, self.ansi_format
        return self._weak_var_update(sgr=sgr)

    @property
    def ansi(self):
        return bytes(self._sgr)

    @property
    def ansi_format(self):
        return getattr(self, '_ansi_type')

    @property
    def base_str(self):
        """The non-ANSI part of the string"""
        return getattr(self, '_base_str')

    @property
    def bg(self):
        """Background color"""
        if bg := self._sgr._rgb_dict.get('bg'):
            return Color.from_rgb(bg)

    @property
    def fg(self):
        """Foreground color"""
        if fg := self._sgr._rgb_dict.get('fg'):
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
    masks: _ChainMaskList = None,
    ansi_type: AnsiColorParam = None,
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
            raise TypeError(type(elt).__name__)
    if ansi_type is not None:
        for i in range(len(masks)):
            masks[i][0].rgb_dict = masks[i][0].rgb_dict, ansi_type
    return masks


class color_chain:
    @staticmethod
    def _is_mask_seq(obj):
        if isinstance(obj, Sequence):
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
    def _from_masks_unchecked(cls, masks, ansi_type):
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
                        sgr.rgb_dict = dict.fromkeys([k]), ansi_type
            inst._masks.append((sgr, s))
            prev_fg, prev_bg = sgr.fg, sgr.bg
        return inst

    @classmethod
    def from_masks(cls, masks, ansi_type=None):
        if cls._is_mask_seq(masks):
            return cls._from_masks_unchecked(
                masks, ansi_type if ansi_type is None else get_ansi_type(ansi_type)
            )
        raise TypeError

    def shrink(self):
        if self:
            maxlen = len(self._masks)
            it = enumerate(self._masks)
            out = []
            while True:
                try:
                    idx, (sgr, s) = next(it)
                    while idx + 1 < maxlen and not s:
                        idx, xs = next(it)
                        sgr += xs[0]; s = xs[1]    # fmt: skip
                    else:
                        out.append((sgr, s))
                except StopIteration:
                    break
        else:
            out = self.masks
        return self._from_masks_unchecked(out, ansi_type=self._ansi_type)

    def __add__(self, other):
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

    def __call__(self, __obj=''):
        return f"{self}{__obj}\x1b[0m"

    def __getitem__(self, __index):
        return self.masks[__index]

    def __init__(self, __iterable=None, *, ansi_type=None):
        self._ansi_type = None
        if ansi_type is not None:
            self._ansi_type = get_ansi_type(ansi_type)
        __iterable = __iterable or []
        self._masks = _collect_masks(*__iterable, ansi_type=self._ansi_type)

    def __len__(self):
        return len(self._masks)

    def __radd__(self, other):
        if isinstance(other, ColorStr):
            return self._from_masks_unchecked(
                [_color_str_to_mask(other), *self.masks],
                ansi_type=(
                    self._ansi_type if self._ansi_type is None else other.ansi_format
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
        return "{.__name__}({})".format(
            type(self),
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
