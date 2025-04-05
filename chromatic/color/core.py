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
    'colorbytes',
    'get_ansi_type',
    'randcolor',
    'rgb2ansi_escape',
]

import operator as op
import os
import random
import sys
from collections import Counter
from collections.abc import Buffer
from copy import deepcopy
from ctypes import byref, c_ulong
from enum import IntEnum
from functools import lru_cache
from types import MappingProxyType
from typing import (
    Callable,
    Final,
    Generator,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Self,
    Sequence,
    SupportsIndex,
    SupportsInt,
    TypeAlias,
    TypeVar,
    TypedDict,
    Union,
    cast,
)

import numpy as np

from .colorconv import *
from .._typing import (
    AnsiColorAlias,
    ColorDictKeys,
    Int3Tuple,
    RGBVectorLike,
    is_matching_typed_dict,
    unionize,
)

os.system('')

CSI: Final[bytes] = b'['
SGR_RESET: Final[str] = '[0m'

# ansi color global lookups
# ansi 4bit {color code (int) ==> (key, RGB)}
_ANSI16C_I2KV = cast(
    dict[int, tuple[ColorDictKeys, Int3Tuple]],
    {
        v: (k, ansi_4bit_to_rgb(v))
        for x in (zip(('fg', 'bg'), (j, j + 10)) for i in (30, 90) for j in range(i, i + 8))
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
_ANSI256_B2KEY = {b'38': 'fg', b'48': 'bg'}

# ansi 8bit {color dict key (str) ==> color code (int)}
_ANSI256_KEY2I = {v: int(k) for k, v in _ANSI256_B2KEY.items()}


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


# constant for sgr parameter validation
_SGR_PARAM_VALUES = frozenset(x.value for x in SgrParameter)


class colorbytes(bytes):

    @classmethod
    def from_rgb(cls, __rgb):
        """Construct a `colorbytes` object from a dictionary of RGB values.

        Returns
        -------
        color_bytes : ansicolor4Bit | ansicolor8Bit | ansicolor24Bit
            Constructed from the RGB dictionary, or `__rgb` returned if of same type as `cls`.

        Raises
        ------
        TypeError
            If `__rgb` is not a dictionary.

        ValueError
            If an unexpected key or value type is encountered in the RGB dict.

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
        if isinstance(__rgb, colorbytes):
            return __rgb
        elif not isinstance(__rgb, Mapping):
            raise TypeError
        if __rgb.keys() not in ({'fg'}, {'bg'}):
            raise ValueError
        rgb = {
            k: tuple(map(int, v)) if isinstance(v, Iterable) else hex2rgb(v)
            for k, v in __rgb.items()
        }

        fmt: AnsiColorType = cls if cls is not colorbytes else DEFAULT_ANSI
        try:
            inst = bytes.__new__(fmt, rgb2ansi_escape(fmt, *rgb.copy().popitem()))
        except TypeError:
            print(vars())
            raise
        setattr(inst, '_rgb_dict_', rgb)
        return cast(AnsiColorFormat, inst)

    def __new__(cls, __ansi):
        if not isinstance(__ansi, (bytes, bytearray)):
            raise TypeError(
                f"Expected bytes-like object, got {type(__ansi).__name__} instead"
            ) from None
        if (is_subtype := cls is not colorbytes) and type(__ansi) is cls:
            return cast(AnsiColorFormat, __ansi)
        match __ansi.removeprefix(CSI).removesuffix(b'm').split(b';'):
            case [color]:
                typ = ansicolor4Bit
                k, rgb = _ANSI16C_I2KV[int(color)]
            case [(b'38' | b'48') as k, b'5', color]:
                typ = ansicolor8Bit
                k = _ANSI256_B2KEY[k]
                rgb = ansi_8bit_to_rgb(int(color))
            case [(b'38' | b'48') as k, b'2', r, g, b]:
                typ = ansicolor24Bit
                k = _ANSI256_B2KEY[k]
                rgb = int(r), int(g), int(b)
            case _:
                raise ValueError
        if typ is not cls:
            __ansi = rgb2ansi_escape(
                cls if is_subtype else typ, mode=cast(ColorDictKeys, k), rgb=rgb
            )
        inst = bytes.__new__(typ, __ansi)
        setattr(inst, '_rgb_dict_', {k: rgb})
        return cast(AnsiColorFormat, inst)

    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"

    @property
    def rgb_dict(self):
        return MappingProxyType(self._rgb_dict_)


class ansicolor4Bit(colorbytes):
    """ANSI 4-bit color format.

    alias: '4b'

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

    ...


class ansicolor8Bit(colorbytes):
    """ANSI 8-Bit color format.

    alias: '8b'

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

    ...


class ansicolor24Bit(colorbytes):
    """ANSI 24-Bit color format.

    alias: '24b'

    Supports all colors in the RGB color space (16,777,216 total).

    Color codes use escape sequences of the form:
        * `CSI 38;2;(r);(g);(b) m` for foreground colors.
        * `CSI 48;2;(r);(g);(b) m` for background colors.
    Where `CSI` (Control Sequence Introducer) is `ESC[` and `r`, `g`, `b` are unsigned 8-bit ints.

    Examples
    --------
    red fg:
        `ESC[38;2;255;85;85m`

    black bg:
        `ESC[48;2;0;0;0m`

    white fg, green bg:
        `ESC[38;2;255;255;255;48;2;0;170;0m`
    """

    ...


_SUPPORTS_256 = frozenset(
    [
        'ANSICON',
        'COLORTERM',
        'ConEmuANSI',
        'PYCHARM_HOSTED',
        'TERM',
        'TERMINAL_EMULATOR',
        'TERM_PROGRAM',
        'WT_SESSION',
    ]
)


def is_vt_proc_enabled():
    if os.name != 'nt' or os.environ.keys() & _SUPPORTS_256:
        return True
    from ctypes import windll

    STD_OUTPUT_HANDLE = -11
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    kernel32 = windll.kernel32
    handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    if handle == -1:
        return False
    mode = c_ulong()
    if not kernel32.GetConsoleMode(handle, byref(mode)):
        return False
    mode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
    if not kernel32.SetConsoleMode(handle, mode):
        return False
    return True


DEFAULT_ANSI = ansicolor8Bit if is_vt_proc_enabled() else ansicolor4Bit

_ANSI_COLOR_TYPES = frozenset(colorbytes.__subclasses__())


@lru_cache
def _is_ansi_type(typ: type):
    try:
        return typ in _ANSI_COLOR_TYPES
    except TypeError:
        return False


AnsiColorFormat: TypeAlias = ansicolor4Bit | ansicolor8Bit | ansicolor24Bit
AnsiColorType: TypeAlias = type[AnsiColorFormat]
AnsiColorParam: TypeAlias = AnsiColorAlias | AnsiColorType
_AnsiColor_co = TypeVar('_AnsiColor_co', bound=colorbytes, covariant=True)
_ANSI_FORMAT_MAP = cast(
    dict[AnsiColorParam, AnsiColorType],
    {x: x for x in _ANSI_COLOR_TYPES}
    | {
        k.__args__[0]: t
        for k, t in zip(
            sorted(AnsiColorAlias.__args__, key=lambda x: int(x.__args__[0].removesuffix('b'))),
            sorted(
                _ANSI_COLOR_TYPES,
                key=lambda x: (lambda n: int(n[n.index('r') + 1 : n.rindex('B')]))(x.__name__),
            ),
        )
    },
)


def get_ansi_type(typ):
    try:
        return _ANSI_FORMAT_MAP[typ]
    except (TypeError, KeyError) as e:
        if isinstance(typ, str):
            raise ValueError(f"invalid ANSI color format alias: {str(e)}") from None
        repr_getter = lambda t: (t if isinstance(t, type) else type(t)).__name__
        raise TypeError(
            f": {'Expected {!r} or type[{}], got {!r} instead'.format(
                str.__qualname__,
                ' | '.join(set(map(repr_getter, _ANSI_FORMAT_MAP.values()))),
                repr_getter(typ)
            )}"
        ) from None


def rgb2ansi_escape(ret_format, mode, rgb):
    ret_format = get_ansi_type(ret_format)
    assert len(rgb) == 3, 'length of RGB value is not 3'
    try:
        if ret_format is ansicolor4Bit:
            return b'%d' % _ANSI16C_KV2I[mode, nearest_ansi_4bit_rgb(rgb)]
        return b';'.join(
            b'%d' % b
            for b in [_ANSI256_KEY2I[mode]]
            + ([5, rgb_to_ansi_8bit(rgb)] if ret_format is ansicolor8Bit else [2, *rgb])
        )
    except KeyError:
        if isinstance(mode, str):
            raise ValueError(f"invalid mode: {mode!r}")
        raise TypeError(
            f"'mode' argument must be {str.__qualname__}, " f"not {type(mode).__qualname__}"
        ) from None


class Color(int):

    def __new__(cls, __x):
        """Convert an integer into a `Color` object.

        Parameters
        ----------
        __x : SupportsInt | Color
            If another Color object is given, immediately return it unchanged.
            Otherwise, the value must be an integer within the range (0, 0xFFFFFF).

        Returns
        -------
        Color
            A new Color object.

        Raises
        ------
        TypeError
            If value is of an unexpected type.
        """
        if type(__x) is cls:
            return __x
        if is_hex_rgb(__x, strict=True):
            inst = super().__new__(cls, int(__x))
            inst._rgb_ = hex2rgb(inst)
            return inst

    def __repr__(self):
        return f"{type(self).__qualname__}({self:#08x})"

    def __invert__(self):
        return Color(0xFFFFFF ^ self)

    @classmethod
    def from_rgb(cls, rgb) -> Self:
        inst = super().__new__(cls, rgb2hex(rgb))
        inst._rgb_ = hex2rgb(inst)
        return inst

    @property
    def rgb(self):
        return getattr(self, '_rgb_')


def randcolor():
    """Return a random color as a :class:`Color` object."""
    return Color.from_bytes(random.randbytes(3))


class SgrParamWrapper:
    __slots__ = '_value_'

    def __init__(self, value=b''):
        cls, vt = map(type, (self, value))
        if not issubclass(vt, (cls, bytes)):
            raise TypeError(
                f"expected value to be {cls.__qualname__!r} or bytes-like object,"
                f"got {type(value).__qualname__!r} instead"
            )
        self._value_ = value._value_ if vt is cls else value

    def __hash__(self):
        return hash(self._value_)

    def __eq__(self, other):
        if type(self) is type(other) or isinstance(other, bytes):
            return hash(self) == hash(other)
        return NotImplemented

    def __bytes__(self):
        return self._value_.__bytes__()

    def __repr__(self):
        return f"{type(self).__name__}({self._value_})"

    def is_same_kind(self, other):
        try:
            return self == other or self._value_ == next(_iter_sgr(other))
        except (TypeError, StopIteration, RuntimeError):
            return False

    def is_reset(self):
        return self._value_ == b'0'

    def is_color(self):
        return isinstance(self._value_, colorbytes)


@lru_cache
def _get_sgr_bitmask[_T: (bytes, bytearray, Buffer)](__x: _T) -> list[int]:
    """Return a list of integers from a bytestring of ANSI SGR parameters.

    Bitwise equivalent to `list(map(int, bytes().split(b';')))`.
    """
    __x = __x.removeprefix(CSI)[: idx if ~(idx := __x.find(0x6D)) else None].removesuffix(b'm')
    length = len(__x)
    a, b = map(int.from_bytes, (bytes([0x3B] * length), __x))
    buffer = []
    allocated = []
    alloc = lambda: allocated.append(int(''.join(map(str, buffer))))
    prepass = zip(map(bool, (~b & a).to_bytes(length=length)), (x % 0x30 for x in __x))
    for c, v in prepass:
        if c:
            buffer.append(v)
        else:
            alloc()
            buffer.clear()
    if buffer:
        alloc()
    return allocated


def _iter_normalized_sgr[
    _T: (Buffer, SgrParamWrapper, int)
](__iter: Buffer | Iterable[_T]) -> Iterator[AnsiColorFormat | int]:
    if isinstance(__iter, Buffer):
        yield from _get_sgr_bitmask(__iter)
    else:
        for it in __iter:
            if _is_ansi_type(type(it)):
                yield it
            elif isinstance(it, int):
                yield int(it)
            elif isinstance(it, (Buffer, SgrParamWrapper)):
                if type(it) is SgrParamWrapper:
                    it = it._value_
                yield from _get_sgr_bitmask(it)
            else:
                raise TypeError(
                    f"Expected {int.__qualname__!r} or bytes-like object, "
                    f"got {type(it).__qualname__!r} instead"
                )


def _co_yield_colorbytes(__iter: Iterator[int]) -> Generator[bytes | AnsiColorFormat, int, None]:
    m: dict[int, ColorDictKeys] = {38: 'fg', 48: 'bg'}
    key_pair = m.get
    get_4b = _ANSI16C_I2KV.get
    new_4b = lambda t: ansicolor4Bit.from_rgb({t[0]: t[1]})
    new_8b = lambda *args: ansicolor8Bit(
        b';'.join(map(b'%d'.__mod__, (args[0], args[1], next(__iter))))
    )
    new_24b = lambda x: ansicolor24Bit.from_rgb({x: tuple(next(__iter) for _ in range(3))})
    default = lambda x: ascii(x).encode()
    obj = bytes()
    while True:
        value = yield obj
        if key := key_pair(value):
            kind = next(__iter)
            if kind == 5:
                obj = new_8b(value, kind)
            else:
                obj = new_24b(key)
        elif kv := get_4b(value):
            obj = new_4b(kv)
        else:
            obj = default(value)


def _gen_colorbytes(__iter: Iterable[int]) -> Iterator[bytes | AnsiColorFormat]:
    gen = iter(__iter)
    color_coro = _co_yield_colorbytes(gen)
    next(color_coro)
    while True:
        try:
            value = next(gen)
            if _is_ansi_type(type(value)):
                yield value
                continue
            yield color_coro.send(value)
        except StopIteration:
            break


def _iter_sgr[_T: (Buffer, int)](__x: _T | Iterable[_T]):
    if isinstance(__x, int):
        __x = [__x]
    return _gen_colorbytes(_iter_normalized_sgr(__x))


class SgrSequence(Sequence[SgrParamWrapper]):

    def append(self, __value):
        if __value not in _SGR_PARAM_VALUES:
            raise ValueError(f"{__value!r} is not a valid SGR parameter")
        if kv := _ANSI16C_I2KV.get(__value):
            if __value in _ANSI16C_BRIGHT:
                self._has_bright_colors_ = True
            elif ~(b_idx := self.find(b'1')):
                self._has_bright_colors_ = True
                self.pop(b_idx)
                kv = _ANSI16C_I2KV.get(__value + 60)
            else:
                self._has_bright_colors_ = False
            value = ansicolor4Bit.from_rgb({kv[0]: kv[1]})
        else:
            value = b'%d' % __value
        v = SgrParamWrapper(value)
        if v.is_color():
            for key in (rgb_dict := v._value_._rgb_dict_).keys() & self._rgb_dict_:
                color = self.get_color(key)
                self.pop(self.index(color))
            self._rgb_dict_ |= rgb_dict
        self._sgr_params_.append(v)
        self._bytes_ = None

    def find(self, value):
        try:
            return self.index(value)
        except ValueError:
            return -1

    def get_color(self, __key: ColorDictKeys) -> SgrParamWrapper | None:
        if self.is_color():
            return next((v for v in self if v.is_color() and __key in v._value_.rgb_dict), None)

    def index(self, value, start: SupportsIndex = 0, stop: SupportsIndex = sys.maxsize):
        try:
            return op.index(start) + next(
                i for i, p in enumerate(self[start:stop]) if p.is_same_kind(value)
            )
        except StopIteration:
            raise ValueError(f"{value!r} not in sequence") from None

    def is_color(self):
        return any(p.is_color() for p in self)

    def is_reset(self):
        return any(p.is_reset() for p in self)

    def remove(self, value):
        try:
            self.pop(self.index(value))
        except ValueError as e:
            raise ValueError(e) from None

    def pop(self, __index=-1):
        try:
            obj = self._sgr_params_.pop(__index)
        except IndexError as e:
            raise IndexError(e) from None
        v = obj._value_
        if obj.is_color():
            for k in v._rgb_dict_.keys():
                del self._rgb_dict_[k]
            if self.has_bright_colors and (vx := int(v)) in _ANSI16C_I2KV:
                self._has_bright_colors_ = False
                if vx in _ANSI16C_STD:
                    self.pop(self.index(b'1'))
        elif self.has_bright_colors and v == b'1':
            for p in self._sgr_params_:
                if type(px := p._value_) is not ansicolor4Bit or int(px) not in _ANSI16C_STD:
                    continue
                self._has_bright_colors_ = False
                break
        self._bytes_ = None
        return obj

    def values(self):
        return [p._value_ for p in self._sgr_params_]

    def __add__(self, other):
        if type(self) is type(other):
            return SgrSequence(x for xs in (self, other) for x in xs)
        if isinstance(other, str):
            return str(self) + other

        raise TypeError(
            "can only concatenate {0} (not {1!r}) to {0}".format(
                *map(op.attrgetter('__qualname__'), (SgrSequence, type(other)))
            )
        )

    def __bool__(self):
        return bool(self._sgr_params_)

    def __bytes__(self):
        if self._bytes_ is None:
            if self._sgr_params_:
                self._bytes_ = b'\x1b[%sm' % b';'.join(self.values())
            else:
                self._bytes_ = bytes()
        return self._bytes_

    def __contains__(self, item: ...):
        if self:
            try:
                return set(_iter_sgr(item)).issubset(self.values())
            except (TypeError, RuntimeError):
                pass
        return False

    def __copy__(self):
        cls = type(self)
        inst = object.__new__(cls)
        inst._bytes_ = self._bytes_
        inst._has_bright_colors_ = self._has_bright_colors_
        inst._sgr_params_ = self._sgr_params_.copy()
        inst._rgb_dict_ = self._rgb_dict_.copy()
        return inst

    def __deepcopy__(self, memo):
        cls = type(self)
        inst = object.__new__(cls)
        memo[id(self)] = inst
        inst._bytes_ = self._bytes_
        inst._has_bright_colors_ = self._has_bright_colors_
        inst._sgr_params_ = deepcopy(self._sgr_params_, memo)
        inst._rgb_dict_ = deepcopy(self._rgb_dict_, memo)
        return inst

    def __eq__(self, other: ...):
        if type(self) is type(other):
            other: SgrSequence
            try:
                return all(
                    self_param == other_param
                    for self_param, other_param in zip(self.values(), other.values(), strict=True)
                )
            except ValueError:
                return False
        return NotImplemented

    def __getitem__(self, item):
        return self._sgr_params_[item]

    def __iadd__(self, other: 'SgrSequence'):
        if type(self) is not type(other):
            raise TypeError(
                "can only concatenate {0} (not {1!r}) to {0}".format(
                    *map(op.attrgetter('__qualname__'), (SgrSequence, type(other)))
                )
            )
        return SgrSequence(self._sgr_params_ + other._sgr_params_)

    def __init__(self, __iter=None, *, ansi_type=None) -> None:
        if type(self) is type(__iter):
            __iter: SgrSequence
            other = __iter.__copy__()
            for attr in type(self).__slots__:
                setattr(self, attr, getattr(other, attr))
            return

        self._bytes_ = None
        self._has_bright_colors_ = False
        self._rgb_dict_ = {}
        self._sgr_params_ = []

        if not __iter:
            return

        values = set()
        add_unique = values.add
        append_param = self._sgr_params_.append
        remove_param = self._sgr_params_.remove
        fg_slot: SgrParamWrapper | None
        bg_slot: SgrParamWrapper | None
        color_dict = dict.fromkeys(['fg', 'bg'], None)
        is_bold = has_bold = False

        def update_colors(__param: SgrParamWrapper, __rgb_dict: Mapping[ColorDictKeys, Int3Tuple]):
            k: ColorDictKeys
            for k, slot in color_dict.items():
                if v := __rgb_dict.get(k):
                    if slot:
                        remove_param(slot)
                    color_dict[k] = __param
                    self._rgb_dict_[k] = v

        is_diff_ansi_typ: Callable[[AnsiColorFormat], bool]
        if ansi_type is None:
            is_diff_ansi_typ = lambda _: False
        else:
            assert ansi_type in _ANSI_COLOR_TYPES
            is_diff_ansi_typ = lambda v: type(v) is not ansi_type

        for x in _iter_sgr(__iter):
            if x in values:
                continue
            param = SgrParamWrapper(x)
            if x == b'1':
                if not is_bold:
                    has_bold = is_bold = True
            elif hasattr(x, 'rgb_dict'):
                if is_diff_ansi_typ(x):
                    param = SgrParamWrapper(x := ansi_type.from_rgb(x))
                if type(x) is ansicolor4Bit:
                    if (btoi := int(x)) in _ANSI16C_BRIGHT:
                        self._has_bright_colors_ = True
                    elif is_bold and btoi in _ANSI16C_STD:
                        self._has_bright_colors_ = True
                        param = SgrParamWrapper(x := ansicolor4Bit(b'%d' % (btoi + 60)))
                        if has_bold:
                            self._sgr_params_.pop(
                                next(
                                    i for i, v in enumerate(self._sgr_params_) if v._value_ == b'1'
                                )
                            )
                            has_bold = False
                update_colors(param, x.rgb_dict)
            append_param(param)
            add_unique(x)

        if self._sgr_params_[-1]._value_ == b'0':
            self._has_bright_colors_ = False
            self._sgr_params_ = [self._sgr_params_.pop()]
            self._rgb_dict_ = {}
        self._bytes_ = b'\x1b[%sm' % b';'.join(map(bytes, self._sgr_params_))

    def __iter__(self):
        return iter(self._sgr_params_)

    def __len__(self):
        return len(self._sgr_params_)

    def __radd__(self, other):
        if type(self) is type(other):
            return SgrSequence(x for xs in (other, self) for x in xs)
        if isinstance(other, str):
            return other + str(self)
        raise TypeError(
            "can only concatenate {0} (not {1!r}) to {0}".format(
                *map(op.attrgetter('__qualname__'), (SgrSequence, type(other)))
            )
        )

    def __repr__(self):
        return f"{type(self).__qualname__}({self.values()})"

    def __str__(self):
        return str(bytes(self), 'utf-8')

    __slots__ = '_bytes_', '_has_bright_colors_', '_rgb_dict_', '_sgr_params_'

    @property
    def bg(self):
        return self.rgb_dict.get('bg')

    @property
    def fg(self):
        return self.rgb_dict.get('fg')

    @property
    def has_bright_colors(self):
        return self._has_bright_colors_

    @property
    def rgb_dict(self):
        return MappingProxyType(self._rgb_dict_)

    @rgb_dict.deleter
    def rgb_dict(self) -> None:
        for k in self._rgb_dict_.keys():
            self.pop(self.index(self.get_color(k)))
            self._bytes_ = None

    @rgb_dict.setter
    def rgb_dict(self, __value: tuple[AnsiColorType, dict[ColorDictKeys, Optional[Color]]]) -> None:
        ansi_type, color_dict = __value
        for k, v in color_dict.items():
            if self._rgb_dict_.get(k):
                try:
                    self.pop(self.index(self.get_color(k)))
                except ValueError as e:
                    e.add_note(repr(self))
                    raise e
            if v is not None:
                color_bytes = ansi_type.from_rgb({k: v})
                self._rgb_dict_ |= color_bytes._rgb_dict_
                self._sgr_params_.append(SgrParamWrapper(color_bytes))
            self._bytes_ = None


# type aliases for ColorStr constructor `color_spec` parameter forms
_CSpecScalar: TypeAlias = int | Color | RGBVectorLike
_CSpecDict: TypeAlias = Mapping[ColorDictKeys, _CSpecScalar]
_CSpecKVPair: TypeAlias = tuple[ColorDictKeys, _CSpecScalar]
_CSpecTuplePair: TypeAlias = tuple[_CSpecScalar, _CSpecScalar] | tuple[_CSpecKVPair, _CSpecKVPair]
_CSpecType: TypeAlias = SgrSequence | _CSpecScalar | _CSpecTuplePair | _CSpecKVPair | _CSpecDict
_ColorSpec = TypeVar('_ColorSpec', _CSpecType, str, bytes)


class _ColorDict(TypedDict, total=False):
    fg: Optional[Color | AnsiColorFormat]
    bg: Optional[Color | AnsiColorFormat]


_VALID_KEYS: frozenset[str] = unionize(
    op.attrgetter('__optional_keys__', '__required_keys__')(_ColorDict)
)


def _solve_color_spec[
    _T: (_CSpecType, SgrSequence)
](color_spec: Optional[_T], ansi_type: AnsiColorType):
    keys: list[str] = ['bg', 'fg']

    def resolve(value, *, key=None):
        nonlocal keys
        if key is not None:
            assert key in _VALID_KEYS, 'expected literal keys {}, got {!r}'.format(_VALID_KEYS, key)
            if key in keys:
                keys.remove(key)
        match value:
            case Color() | int() | np.integer() as color:
                yield (key or keys.pop(), Color(color).rgb)
            case [int(), int(), int()] as rgb:
                r, g, b = (x & 0xFF for x in rgb)
                yield (key or keys.pop(), (r, g, b))
            case np.ndarray() as colors:
                assert not colors.shape[-1] % 3, 'array does not contain RGB values'
                it = np.uint8(colors).flat
                for _ in range(colors.ndim):
                    yield (key or keys.pop(), tuple(int(next(it)) for _ in range(3)))
            case {'fg': _, 'bg': _} | {'fg': _} | {'bg': _} as colors:
                for key, color in colors.items():
                    yield from resolve(color, key=key)
            case [str() as key, color] if key in _VALID_KEYS:
                yield from resolve(color, key=key)
            case [_, _] as colors:
                for color in colors:
                    yield from resolve(color)
            case _:
                raise ValueError(repr(value))

    out: dict[str, ...] = {}
    try:
        for k, v in resolve(color_spec):
            if k in out:
                if len(out) > 1 and out[k] != v:
                    raise ValueError(f"multiple possible values for {k!r} {(out[k], v)}")
                out[keys.pop()] = out.pop(k)
                keys.append(k)
            out[k] = v
    except Exception as e:
        if type(e) is IndexError:
            e = ValueError('too many arguments' if len(out) >= 2 else 'args contain non-RGB values')
        context = ('invalid color spec', str(e))
        raise ValueError(': '.join(filter(None, context))) from None
    return SgrSequence(ansi_type.from_rgb({k: v}) for k, v in out.items())


def _get_color_str_vars(
    base_str: Optional[str], color_spec: Optional[_ColorSpec], ansi_type: AnsiColorType = None
) -> tuple[SgrSequence, str]:
    if not color_spec and color_spec != 0:
        return SgrSequence(), base_str or ''
    if ansi_type is None:
        ansi_type = DEFAULT_ANSI
    if isinstance(color_spec, (str, bytes)):
        if hasattr(color_spec, 'encode'):
            color_spec = color_spec.encode()
        if csi_count := color_spec.count(CSI):
            if csi_count > 1:
                color_spec, _, byte_str = (
                    color_spec.removeprefix(CSI).removesuffix(SGR_RESET.encode()).partition(b'm')
                )
                if color_spec.count(CSI) > 1:
                    raise ValueError(
                        f"color spec contains {csi_count} escape sequences, expected only 1"
                    ) from None
                base_str = byte_str.decode()
            sgr_params = SgrSequence(color_spec, ansi_type=ansi_type)
        else:
            is_hex_rgb(color_spec := int.from_bytes(color_spec), strict=True)
            sgr_params = _solve_color_spec(color_spec, ansi_type=ansi_type)
    elif not isinstance(color_spec, SgrSequence):
        sgr_params = _solve_color_spec(color_spec, ansi_type=ansi_type)
    else:
        sgr_params = color_spec
    base_str = base_str or ''
    return sgr_params, base_str


class _ColorStrWeakVars(TypedDict, total=False):
    _base_str_: str
    _sgr_: SgrSequence
    _no_reset_: bool


class _AnsiBytesGetter:

    def __get__(self, instance: Optional['ColorStr'], objtype=None):
        if instance is None:
            return
        return instance._sgr_.__bytes__()


class _SgrParamsGetter:

    def __get__(self, instance: Optional['ColorStr'], objtype=None):
        if instance is None:
            return
        return instance._sgr_._sgr_params_


class _ColorDictGetter:

    def __get__(self, instance: Optional['ColorStr'], objtype=None):
        if instance is None:
            return
        return {k: Color.from_rgb(v) for k, v in instance._sgr_.rgb_dict.items()}


class ColorStr(str):

    def _weak_var_update(self, **kwargs):
        if kwargs.keys().isdisjoint(inst_vars := vars(self)):
            raise ValueError(f"unexpected keys: {kwargs.keys() - inst_vars.keys()}") from None
        sgr = kwargs.get('_sgr_', self._sgr_)
        base_str = kwargs.get('_base_str_', self._base_str_)
        suffix = '' if kwargs.get('_no_reset_', self._no_reset_) else SGR_RESET
        inst = super().__new__(ColorStr, ''.join([str(sgr), base_str, suffix]))
        inst.__dict__ |= {**inst_vars, **kwargs}
        return cast(ColorStr, inst)

    def ansi_partition(self):
        """Returns the 3-tuple: SGR sequence prefix, base string, SGR reset (or empty string)."""
        return str(self._sgr_), self.base_str, '' if self.no_reset else SGR_RESET

    def as_ansi_type(self, __ansi_type):
        """Convert all ANSI color codes in the :class:`ColorStr` to a single ANSI type.

        Parameters
        ----------
        __ansi_type : str or type[ansicolor4Bit | ansicolor8Bit | ansicolor24Bit]
            ANSI format to which all SGR parameters of type :class:`colorbytes` will be cast.

        Returns
        -------
        ColorStr
            Return `self` if all ANSI formats are already the input type.
            Otherwise, return reformatted :class:`ColorStr`.

        """
        ansi_type = get_ansi_type(__ansi_type)
        if self._sgr_.is_color():
            new_params = []
            new_rgb = {}
            for p in self._sgr_params_:
                if p.is_color() and type(p._value_) is not ansi_type:
                    new_ansi = ansi_type(p._value_)
                    new_rgb |= new_ansi.rgb_dict
                    new_params.append(SgrParamWrapper(new_ansi))
                else:
                    new_params.append(p)
            if new_params == self._sgr_params_:
                return self
            new_sgr = SgrSequence()
            for name, value in zip(('_sgr_params_', '_rgb_dict_'), (new_params, new_rgb)):
                setattr(new_sgr, name, value)
            inst = super().__new__(
                type(self),
                ''.join([str(new_sgr), self._base_str_, '' if self._no_reset_ else SGR_RESET]),
            )
            for name, value in {**vars(self), '_sgr_': new_sgr, '_ansi_type_': ansi_type}.items():
                setattr(inst, name, value)
            return cast(ColorStr, inst)
        return self

    def format(self, *args, **kwargs):
        return self._weak_var_update(_base_str_=self.base_str.format(*args, **kwargs))

    def split(self, sep=None, maxsplit=-1):
        return list(
            self._weak_var_update(_base_str_=s)
            for s in self.base_str.split(sep=sep, maxsplit=maxsplit)
        )

    def recolor(self, __value=None, absolute=False, **kwargs):
        """Return a copy of `self` with a new color spec.

        If `__value` is a :class:`ColorStr`, return `self` with the colors of `__value`.

        Parameters
        ----------
        __value : ColorStr, optional
            A :class:`ColorStr` object that the new instance will inherit colors from.

        absolute : bool
            If True, overwrite all colors of the current object with the provided arguments,
            removing any existing colors not explicitly set by the arguments.
            Otherwise, only replace colors where specified (default).

        Keyword Args
        ------------
        fg : Color, optional
            New foreground color.

        bg : Color, optional
            New background color.

        Returns
        -------
        ColorStr
            A new :class:`ColorStr` instance recolored by the input parameters.

        Raises
        ------
        TypeError
            If `__value` is not None but is not an instance of :class:`ColorStr`.

        ValueError
            If any unexpected keys or value types found in `kwargs`.

        Examples
        --------
        >>> cs1 = ColorStr('foo', randcolor())
        >>> cs2 = ColorStr('bar', dict(fg=Color(0xFF5555), bg=Color(0xFF00FF)))
        >>> new_cs = cs2.recolor(bg=cs1.fg)
        >>> int(new_cs.fg) == 0xFF5555, new_cs.bg == cs1.fg
        (True, True)

        >>> cs = ColorStr("Red text", ('fg', 0xFF0000))
        >>> recolored = cs.recolor(fg=Color(0x00FF00))
        >>> recolored.base_str, f"{recolored.fg:06X}"
        ('Red text', '00FF00')
        """
        if __value:
            if isinstance(__value, ColorStr):
                kwargs = __value._color_dict_
            else:
                raise TypeError(
                    f"expected positional argument of type {ColorStr.__qualname__!r}, "
                    f"got {type(__value).__qualname__!r} instead"
                ) from None
        elif not kwargs:
            return self
        valid, context = is_matching_typed_dict(kwargs, _ColorDict)
        if not valid:
            try:
                kwargs = {k: v if v is None else Color(v) for k, v in kwargs.items()}
            except Exception as e:
                raise e from None
        sgr = SgrSequence(self._sgr_)
        if bool(absolute):
            del sgr.rgb_dict
        sgr.rgb_dict = (self.ansi_format, kwargs)
        return self._weak_var_update(_sgr_=sgr)

    def replace(self, __old, __new, __count=-1):
        if isinstance(__new, ColorStr):
            __new = __new.base_str
        return self._weak_var_update(_base_str_=self.base_str.replace(__old, __new, __count))

    def translate(self, __table) -> 'ColorStr':
        return self._weak_var_update(_base_str_=self.base_str.translate(__table))

    def update_sgr(self, *p):
        """Return a copy of `self` with updated SGR sequence parameters.

        Parameters
        ----------
        *p: SgrParameter | int
            The SGR parameter value(s) to be added or removed from the :class:`ColorStr`.
            A value already in `self` SGR sequence gets removed, else it gets added.
            If no values are passed, returns `self` unchanged.

        Returns
        -------
        ColorStr
            A new :class:`ColorStr` object with the SGR updates applied.

        Raises
        ------
        ValueError
            If any of the SGR parameters are invalid, or if extended color codes are passed.

        Notes
        -----
        * The extended color escapes `{38, 48}` require extra parameters and so raise a ValueError.
            :meth:`ColorStr.as_ansi_type` should be used to change ANSI color format instead.

        Examples
        --------
        >>> # creating an empty ColorStr object
        >>> empty_cs = ColorStr(no_reset=True)
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
        """
        if not p:
            return self
        if any(not isinstance(x, int) or x in _ANSI256_KEY2I.values() for x in p):
            raise ValueError
        new_sgr = SgrSequence(self._sgr_)
        for x in p:
            if x in new_sgr:
                new_sgr.pop(new_sgr.index(x))
            elif x == 1 and new_sgr.has_bright_colors:
                for i, param in enumerate(new_sgr):
                    if type(px := param._value_) is not ansicolor4Bit:
                        continue
                    new_sgr.pop(i)
                    new_sgr.append(int(px) - 60)
            else:
                new_sgr.append(x)
        if new_sgr.is_color():
            formats: list[AnsiColorType] = [type(p._value_) for p in new_sgr if p.is_color()]
            ansi_type = max(formats, key=formats.count)
        else:
            ansi_type = self.ansi_format
        inst = super().__new__(
            type(self),
            ''.join([str(new_sgr), self._base_str_, '' if self._no_reset_ else SGR_RESET]),
        )
        inst.__dict__ |= {**vars(self), '_sgr_': new_sgr, '_ansi_type_': ansi_type}
        return cast(ColorStr, inst)

    def __add__(self, other):
        if type(self) is type(other):
            return self._weak_var_update(
                _sgr_=self._sgr_ + other._sgr_,
                _base_str_=''.join([self._base_str_, other._base_str_]),
            )
        if isinstance(other, str):
            return self._weak_var_update(_base_str_=''.join([self._base_str_, other]))
        if isinstance(other, SgrParameter):
            return self.update_sgr(other)
        if hasattr(other, '_sgr_'):
            return NotImplemented
        raise TypeError(
            'can only concatenate {0}, {1}, or {2} (got {3!r}) to {1}'.format(
                *map(op.attrgetter('__name__'), (str, type(self), SgrParameter)),
                type(other).__qualname__,
            )
        )

    def __contains__(self, __key: str):
        if type(__key) is not str:
            return False
        if __key == str(self._sgr_):
            return True
        if __key == SGR_RESET:
            return not self.no_reset
        return self.base_str.__contains__(__key)

    def __eq__(self, other):
        if type(self) is type(other):
            return hash(self) == hash(other)
        return NotImplemented

    def __format__(self, format_spec=''):
        if ansi_typ := {'4b': ansicolor4Bit, '8b': ansicolor8Bit, '24b': ansicolor24Bit}.get(
            format_spec
        ):
            return str(self.as_ansi_type(ansi_typ))
        return str.__format__(self, format_spec)

    def __getitem__(self, __key: Union[SupportsIndex, slice]):
        return self._weak_var_update(_base_str_=self.base_str[__key])

    def __hash__(self):
        return str(self).__hash__()

    # noinspection PyUnusedLocal
    def __init__(self, obj=None, color_spec=None, **kwargs):
        """
        Create a ColorStr object.

        Parameters
        ----------
        obj : object, optional
            The base object to be cast to a ColorStr. If None, uses a null string ('').

        color_spec : type[_ColorSpec | ColorStr], optional
            The color specification for the string.
            The constructor supports various types, such as:

            * An RGB tuple
            * A hex color as an integer
            * A Color object
            * Any tuple pair of the aforementioned types:
                ('fg'=color_spec[0], 'bg'=color_spec[1])
            * A key-value pair or `dict_items`-like tuple:
                ('fg', ...) or (('fg', ...), ('bg', ...))
            * A dictionary mapping:
                dict[Literal['fg', 'bg'], ...]

        Keyword Args
        ------------
        ansi_type : str or type[ansicolor4Bit | ansicolor8Bit | ansicolor24Bit], optional
            An ANSI format to cast all :class:`colorbytes` params to before formatting the string.

            * ANSI format can also be changed on instances using :meth:`ColorStr.as_ansi_type`
            * Reformatting recursively applies to `alt_spec` if `alt_spec` is not None

        no_reset : bool
            If True, create the :class:`ColorStr` without concatenating a 'reset all' SGR sequence.
            Default is False (new instances get concatenated with reset sequences).

        Returns
        -------
        ColorStr
            A new ColorStr object comprised of the base string and provided ANSI sequences.

        Notes
        -----
        * Each of the ANSI color formats can be invoked by their alias in place of the type:
            ``ansicolor4Bit`` == '4b', ``ansicolor8Bit`` == '8b', ``ansicolor24Bit`` == '24b'
        * Use :py:func:`help` with :class:`colorbytes` types for color code ranges and sequences.

        * ``color_spec`` of type :obj:`str` or :obj:`bytes` is parsed as a literal escape sequence.

        Examples
        --------
        >>> cs = ColorStr('Red text', ('fg', 0xFF0000))
        >>> cs.rgb_dict, cs.base_str
        ({'fg': (255, 0, 0)}, 'Red text')

        >>> cs_from_rgb = ColorStr(color_spec={'fg': (255, 85, 85)}, ansi_type='4b')
        >>> cs_from_literal = ColorStr(color_spec=f'{CSI}91m', ansi_type='4b')
        >>> cs_from_rgb == cs_from_literal
        True

        >>> # ANSI 4-bit sequences of the form `ESC[<1 (bold)>;<{30-37} | {40-47}>...`
        >>> # are equivalent to 'bright' counterparts `ESC[<{90-97} | {100-107}>...`
        >>> cs_from_literal_alt = ColorStr(color_spec=f'{CSI}1;31m', ansi_type='4b')
        >>> cs_from_literal_alt == cs_from_literal
        True

        >>> # bold-prefix syntax is autocast to the 'bright' sequence form
        >>> cs_from_literal_alt.ansi.replace(CSI, b'ESC[')
        b'ESC[91m'
        """
        ...

    def __iter__(self):
        for c in self.base_str:
            yield self._weak_var_update(_base_str_=c)

    def __len__(self):
        return len(self.base_str)

    def __matmul__(self, other):
        """Return a new :class:`ColorStr` with the base string of `self` and colors of `other`"""
        if type(self) is type(other):
            return self._weak_var_update(_sgr_=other._sgr_, _no_reset_=other.no_reset)
        raise TypeError(
            'unsupported operand type(s) for @: '
            "{!r} and {!r}".format(*(type(x).__qualname__ for x in (self, other)))
        )

    def __mod__(self, __value):
        return self._weak_var_update(_base_str_=self.base_str.__mod__(__value))

    def __mul__(self, __value):
        return self._weak_var_update(_base_str_=self.base_str.__mul__(__value))

    def __invert__(self):
        """Return a copy of `self` with inverted colors (XORed by '0xFFFFFF')"""
        sgr = SgrSequence(self._sgr_)
        sgr.rgb_dict = (self.ansi_format, {k: ~v for k, v in self._color_dict_.items()})
        return self._weak_var_update(_sgr_=sgr)

    def __new__(cls, obj=None, color_spec=None, **kwargs):
        if ansi_type := kwargs.get('ansi_type'):
            ansi_type = get_ansi_type(ansi_type)
        if type(color_spec) is cls:
            if ansi_type is not None and any(type(a) is not ansi_type for a in color_spec.ansi):
                return color_spec.as_ansi_type(ansi_type)
            inst = super().__new__(cls, str(color_spec))
            for name, value in vars(color_spec).items():
                setattr(inst, name, value)
            return inst
        d = {'_ansi_type_': ansi_type or DEFAULT_ANSI}
        no_reset = d['_no_reset_'] = bool(kwargs.get('no_reset', False))
        suffix = '' if no_reset else SGR_RESET
        if obj is not None:
            if not isinstance(obj, str):
                obj = str(obj, encoding='ansi') if isinstance(obj, Buffer) else str(obj)
            if color_spec is None and obj.startswith(CSI.decode()):
                color_spec = obj.encode()
                obj = None
        elif color_spec is None:
            inst = super().__new__(cls, suffix)
            inst.__dict__ |= {'_sgr_': SgrSequence(), '_base_str_': str(), **d}
            return inst
        sgr, base_str_ = d['_sgr_'], d['_base_str_'] = _get_color_str_vars(
            obj, color_spec, cast(AnsiColorType, ansi_type)
        )
        if ansi_type is None and sgr.is_color():
            d['_ansi_type_'], _ = max(
                Counter(type(p._value_) for p in sgr._sgr_params_ if p.is_color()).items(),
                key=op.itemgetter(1),
            )
        inst = super().__new__(cls, ''.join([str(sgr), base_str_, suffix]))
        inst.__dict__ |= d
        return inst

    def __repr__(self):
        return f"{type(self).__name__}(%r, ansi_type=%s)" % (
            self.ansi.decode() + self.base_str,
            getattr(self.ansi_format, '__name__', type(None).__name__),
        )

    def __sub__(self, other):
        """Return a copy of `self` with colors adjusted by color difference with `other`"""
        if (vt := type(other)) not in {Color, ColorStr}:
            raise TypeError(
                'unsupported operand type(s) for -: '
                f"{ColorStr.__name__!r} and {vt.__qualname__!r}"
            )

        def _rgb_diff_color(a: Int3Tuple, b: Int3Tuple) -> Color:
            return Color.from_rgb(rgb_diff(a, b))

        k: Literal['fg', 'bg']
        if vt is Color:
            diff_dict = {k: _rgb_diff_color(v, other.rgb) for k, v in self.rgb_dict.items()}
        else:
            diff_dict = {
                k: _rgb_diff_color(self.rgb_dict[k], other.rgb_dict[k])
                for k in self.rgb_dict.keys() & other.rgb_dict
            }
        if not diff_dict:
            return self
        sgr = SgrSequence(self._sgr_)
        sgr.rgb_dict = self.ansi_format, diff_dict
        return self._weak_var_update(_sgr_=sgr)

    _ansi_ = _AnsiBytesGetter()
    _color_dict_ = _ColorDictGetter()
    _sgr_params_ = _SgrParamsGetter()

    @property
    def ansi(self):
        return self._ansi_

    @property
    def ansi_format(self):
        return self._ansi_type_

    @property
    def base_str(self):
        """The non-ANSI part of the string"""
        return self._base_str_

    @property
    def bg(self):
        """Background color"""
        return self._color_dict_.get('bg')

    @property
    def fg(self):
        """Foreground color"""
        return self._color_dict_.get('fg')

    @property
    def no_reset(self):
        return self._no_reset_

    @property
    def rgb_dict(self):
        return {k: v.rgb for k, v in self._color_dict_.items()}







