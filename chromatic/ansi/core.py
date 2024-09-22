__all__ = [
    'CSI',
    'Color',
    'ColorStr',
    'SgrParameter',
    'SgrSequence',
    'ansicolor24Bit',
    'ansicolor4Bit',
    'ansicolor8Bit',
    'colorbytes',
    'get_ansi_type',
    'get_term_ansi_default',
    'hexstr2rgb',
    'hsl_gradient',
    'is_hex_rgb',
    'randcolor',
    'rgb2ansi_color_esc',
    'rgb2color',
    'rgb2hexstr',
    'rgb_diff',
    'rgb_luma_transform',
    'SGR_RESET'
]

import ctypes
import math
import os
import random
from collections.abc import Buffer
from copy import deepcopy
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
    Sequence,
    SupportsIndex,
    SupportsInt,
    TypeVar,
    TypedDict,
    Union,
    cast
)

import numpy as np
from numpy import ndarray

from chromatic._typing import (
    AnsiColorAlias,
    ColorDictKeys,
    Float3Tuple,
    Int3Tuple,
    RGBVector,
    is_matching_typed_dict
)
from chromatic.ansi._colorconv import *

os.system('')

CSI: Final[bytes] = b'['
SGR_RESET: Final[str] = '[0m'


def is_hex_rgb(value: int):
    if not isinstance(value, (int, np.uint8)):
        return False
    if isinstance(value, Color):
        return True
    return 0x0 <= value <= 0xFFFFFF


def rgb2hexstr(rgb: RGBVector) -> str:
    r, g, b = rgb
    return f'{r:02x}{g:02x}{b:02x}'


def hexstr2rgb(__str: str) -> Int3Tuple:
    if not is_hex_rgb(hex_val := int(__str, 16)):
        raise TypeError(
            f"{repr(__str)} is not a valid RGB color value") from None
    return hex2rgb(hex_val)


# ansi color global lookups
# ansi 4bit {color code (int) ==> (key, RGB)}
_ANSI16C_I2KV = cast(
    dict[int, tuple[ColorDictKeys, Int3Tuple]], dict(
        xs for x in
        (zip((n, n + 10), (('fg', ansi_4bit_to_rgb(n)), ('bg', ansi_4bit_to_rgb(n + 10)))) for i in
         (30, 90) for n in range(i, i + 8)) for xs in x))

# ansi 4bit {(key, RGB) ==> color code (int)}
_ANSI16C_KV2I = {v: k for k, v in _ANSI16C_I2KV.items()}

# ansi 4bit standard color range
_ANSI16C_STD = frozenset(set(range(30, 38)).union(range(40, 48)))

# ansi 4bit bright color range
_ANSI16C_BRIGHT = frozenset(_ANSI16C_I2KV.keys() - _ANSI16C_STD)

# ansi 8bit {color code (bytes) ==> color dict key (str)}
_ANSI256_B2KEY = {b'38': 'fg', b'48': 'bg'}

# ansi 8bit {color dict key (str) ==> color code (int)}
_ANSI256_KEY2I = {v: int(k) for k, v in _ANSI256_B2KEY.items()}


# see also: https://en.wikipedia.org/wiki/ANSI_escape_code#SGR
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
_SGR_PARAM_VALUES = frozenset(param.value for param in SgrParameter)


class colorbytes(bytes):

    @classmethod
    def from_rgb(cls, __rgb):
        """
        Construct an `ansi_color_bytes` object from a dictionary of RGB values.

        Returns
        -------
            color_bytes : ansicolor4Bit | ansicolor8Bit | ansicolor24Bit
                Constructed from the RGB dictionary, or `__rgb` returned if `type(__rgb)` is `cls`.

        Raises
        ------
            TypeError
                If `__rgb` is not a dictionary

            ValueError
                If an unexpected key or value type is encountered in the RGB dict.

        Examples
        --------
            >>> rgb_dict = {'fg': (255, 85, 85)}
            >>> old_ansi = repr(ansicolor4Bit.from_rgb(rgb_dict))
            >>> old_ansi
            "ansi_color_4Bit(b'91')"

            >>> new_ansi = repr(ansicolor24Bit.from_rgb(rgb_dict))
            >>> new_ansi
            "ansi_color_24Bit(b'38;2;255;85;85')"

        """

        def ParseError(__k: int | tuple[int, int]):
            d = {
                0: (TypeError,
                    "Expected {dict.__qualname__!r}, got {type(__rgb).__qualname__!r} instead"),
                1: (ValueError, "Expected dictionary with length of 1, got {len(__rgb)} instead"),
                2: (ValueError, "Unexpected keys: {unexpected}"),
                (3, 0): (ValueError,
                         "Invalid key-value pair for RGB dict: {}: {k!r} not in {"
                         "expected_keys=!r}"),
                (3, 1): (ValueError,
                         "Invalid key-value pair for RGB dict: {}: {v!r} is not a valid RGB tuple")}
            typ, context = d[__k]
            return lambda *args, **kwargs: typ(context.format(*args, **kwargs))

        if isinstance(__rgb, colorbytes):
            if type(__rgb) is cls:
                return __rgb
            rgb = __rgb.rgb_dict.copy()
        else:
            if not isinstance(__rgb, (dict, MappingProxyType)):
                raise ParseError(
                    0)(dict=dict, **{'type(__rgb)': type(__rgb)}) from None

            if len(__rgb) != 1:
                raise ParseError(
                    1)(**{'len(__rgb)': len(__rgb)}) from None

            if (expected := {'fg', 'bg'}).isdisjoint(__rgb.keys()):
                raise ParseError(
                    2)(unexpected=(__rgb.keys() - expected)) from None

            rgb = __rgb.copy()
            for k, v in __rgb.items():
                try:
                    rgb[k] = Color(v).rgb
                except (KeyError, ValueError, TypeError) as e:
                    raise ParseError((3, 0) if (err_k := type(e) is KeyError) else (3, 1))(
                        (k, v), **{
                                      'k': k,
                                      'v': v} | ({
                                                     'excepted_keys': expected} if err_k else {
                        })) from None

        obj_format: AnsiColorType = cls if cls is not colorbytes else DEFAULT_ANSI
        obj = bytes.__new__(obj_format, rgb2ansi_color_esc(obj_format, *next(iter(rgb.items()))))
        setattr(obj, '_rgb_dict_', rgb)
        return cast(AnsiColorFormat, obj)

    def __new__(cls, __ansi):
        if not isinstance(__ansi, bytes):
            raise TypeError(
                f"Expected bytes-like object, got {type(__ansi).__qualname__} instead") from None
        if (is_subtype := cls is not colorbytes) and type(__ansi) is cls:
            return cast(AnsiColorFormat, __ansi)
        parts = __ansi.removeprefix(CSI).removesuffix(b'm').split(b';')
        if (n := len(parts)) not in {1, 3, 5}:
            raise ValueError
        if n == 1:
            typ = ansicolor4Bit
            k, rgb = _ANSI16C_I2KV[int(*parts)]
        else:
            if parts[1] not in {b'2', b'5'}:
                raise ValueError
            if (typ := ansicolor24Bit if parts[1] == b'2' else ansicolor8Bit) is ansicolor8Bit:
                rgb = ansi_8bit_to_rgb(int(parts[2]))
            else:
                rgb = tuple(map(int, parts[2:]))
            k = _ANSI256_B2KEY[parts[0]]
        if typ is not cls:
            __ansi = rgb2ansi_color_esc(
                cls if is_subtype else typ, mode=cast(ColorDictKeys, k), rgb=rgb)
        obj = cast(AnsiColorFormat, bytes.__new__(cls, __ansi))
        setattr(obj, '_rgb_dict_', {k: rgb})
        return obj

    def __repr__(self):
        return f"{type(self).__qualname__}({super().__repr__()})"

    @property
    def rgb_dict(self):
        return MappingProxyType(self._rgb_dict_)


class ansicolor4Bit(colorbytes):
    """
    ANSI 4-bit color format

    alias: '4b'

    Supports 16 colors:
        * 8 standard colors:
            {0: black, 1: red, 2: green, 3: yellow, 4: blue, 5: magenta, 6: cyan, 7: white}
        * 8 bright colors, each mapping to a standard color (bright = standard + 8)

    Color codes use escape sequences of the form:
        * `CSI 30–37 m` for standard foreground colors
        * `CSI 40–47 m` for standard background colors
        * `CSI 90–97 m` for bright foreground colors
        * `CSI 100–107 m` for bright background colors

        Where `CSI` (Control Sequence Introducer) is `ESC[`

    Examples
    --------
        bright red fg:
            `ESC[91m`

        standard green bg:
            `ESC[42m`

        bright white bg, black fg:
            `ESC[107;30m`
    """
    pass


class ansicolor8Bit(colorbytes):
    """
    ANSI 8-Bit color format

    alias: '8b'

    Supports 256 colors, mapped to the following value ranges:
        * (0, 15): Corresponds to ANSI 4-bit colors
        * (16, 231): Represents a 6x6x6 RGB color cube
        * (232, 255): Greyscale colors, from black to white

    Color codes use escape sequences of the form:
        * `CSI 38;5;(n) m` for foreground colors
        * `CSI 48;5;(n) m` for background colors

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
    pass


class ansicolor24Bit(colorbytes):
    """
    ANSI 24-Bit color format

    alias: '24b'

    Supports all colors in the RGB color space (16,777,216 total).

    Color codes use escape sequences of the form:
        * `CSI 38;2;(r);(g);(b) m` for foreground colors
        * `CSI 48;2;(r);(g);(b) m` for background colors

        Where `CSI` (Control Sequence Introducer) is `ESC[`
        and `(r, g, b)` are unsigned 8-bit integers.

    Examples
    --------
        red fg:
            `ESC[38;2;255;85;85m`

        black bg:
            `ESC[48;2;0;0;0m`

        white fg, green bg:
            `ESC[38;2;255;255;255;48;2;0;170;0m`
    """
    pass


_SUPPORTS_256 = frozenset(
    {'ANSICON', 'COLORTERM', 'ConEmuANSI', 'PYCHARM_HOSTED', 'TERM', 'TERMINAL_EMULATOR',
     'TERM_PROGRAM', 'WT_SESSION'})


def is_vt_proc_enabled():
    if not _SUPPORTS_256 & set(os.environ):
        if os.name == 'nt':
            STD_OUTPUT_HANDLE = -11
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
            if handle == -1:
                return False
            mode = ctypes.c_ulong()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            mode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
            if not kernel32.SetConsoleMode(handle, mode):
                return False
    return True


def get_term_ansi_default():
    return ansicolor8Bit if is_vt_proc_enabled() else ansicolor4Bit


@lru_cache
def _is_ansi_type(typ: type):
    try:
        return typ in _ANSI_COLOR_TYPES
    except TypeError:
        return False


DEFAULT_ANSI = get_term_ansi_default()

_ANSI_COLOR_TYPES = frozenset(colorbytes.__subclasses__())
type AnsiColorFormat = ansicolor4Bit | ansicolor8Bit | ansicolor24Bit
type AnsiColorType = type[AnsiColorFormat]
type AnsiColorParam = Union[AnsiColorAlias, AnsiColorType]
_AnsiColor_co = TypeVar('_AnsiColor_co', bound=colorbytes, covariant=True)

_ANSI_FORMAT_MAP = cast(
    dict[AnsiColorParam, AnsiColorType], {t: t for t in _ANSI_COLOR_TYPES} | dict(
        (k.__args__[0], t) for k, t in zip(
            sorted(
                AnsiColorAlias.__args__, key=lambda x: int(x.__args__[0].rstrip('b'))), sorted(
                _ANSI_COLOR_TYPES, key=lambda x: int(
                    x.__name__.lstrip('ansicolor').rstrip('Bit'))))))


def get_ansi_type(typ):
    try:
        return _ANSI_FORMAT_MAP[typ]
    except (TypeError, KeyError) as e:
        if isinstance(typ, str):
            raise ValueError(
                f"invalid ANSI color format alias: {str(e)}") from None
        raise TypeError(
            f"Expected {str.__qualname__!r} or type[%s | %s | %s], got %r instead" % tuple(
                (t if isinstance(t, type) else type(t)).__qualname__ for t in (*sorted(
                    _ANSI_COLOR_TYPES,
                    key=lambda x: int(x.__name__.lstrip('ansicolor').rstrip('Bit'))),
                                                                               typ))) from None


def rgb2ansi_color_esc(ret_format, mode, rgb):
    ret_format = get_ansi_type(ret_format)
    try:
        if ret_format is ansicolor4Bit:
            return b'%d' % _ANSI16C_KV2I[mode, nearest_ansi_4bit_rgb(rgb)]
        return b';'.join(
            map(
                b'%d'.__mod__, (_ANSI256_KEY2I[mode], *(
                    (5, rgb_to_ansi_8bit(rgb)) if ret_format is ansicolor8Bit else (2, *rgb)))))
    except KeyError:
        if isinstance(mode, str):
            raise ValueError(
                f"invalid mode: {mode!r}")
        raise TypeError(
            f"'mode' argument must be {str.__qualname__}, not {type(mode).__qualname__}") from None


is_uint8_equiv = np.frompyfunc(lambda n: 0 <= n <= 255, 1, 1)


class Color(int):

    def __new__(cls, __x):
        """
        Color(tuple[int, int, int]) -> Color
        Color(int) -> Color

        Convert an integer or RGB tuple into a Color object.

        Parameters
        ----------
            __x : tuple[int, int, int] | Sequence[int] | ndarray[Any, dtype[uint8]] | SupportsInt
            | Color
                If another Color object is given, immediately return the object unchanged.
                Otherwise, the parameter should meet the following criteria:
                * Sequences or arrays must contain 3 unsigned integers less than 256.
                * Integers must be within range(0, 0xFFFFFF).
                * Any other types must support casting to int else raise a TypeError.

        Returns
        -------
            obj : Color
                A new Color object

        Raises
        ------
            TypeError
                If value is of an unexpected type
        """
        if (vt := type(__x)) is cls:
            return __x
        elif vt is not int:
            if vt in {tuple, np.ndarray} or isinstance(__x, Sequence):
                return cls.from_rgb(__x)
            if isinstance(__x, SupportsInt):
                __x = int(__x)
        if is_hex_rgb(__x):
            obj = int.__new__(cls, __x)
            obj._hex_ = __x
            obj._rgb_ = hex2rgb(__x)
            return cast(Color, obj)
        raise TypeError(
            f"'{__x}' is not a valid hex color value: {type(__x)=}") from None

    def __bool__(self):
        return 0 <= self < 0x1000000

    def __repr__(self):
        return f"{type(self).__qualname__}(0x{self.hex:06X})"

    def __neg__(self):
        return Color(0xFFFFFF ^ self)

    @classmethod
    def from_rgb(cls, rgb):
        if len(rgb) != 3:
            raise ValueError(
                f"Expected 3 items, got {len(rgb)}")
        if issubclass(vt := type(rgb), np.ndarray) and rgb.dtype != np.uint8:
            if not np.all(is_uint8_equiv(rgb)):
                raise ValueError(
                    f"{rgb!r}: {rgb.dtype=}, expected {np.dtype(np.uint8)!r}")
        if vt is not tuple:
            rgb = tuple(map(int, rgb))
        obj = int.__new__(cls, x := rgb2hex(rgb))
        obj._rgb_ = rgb
        obj._hex_ = x
        return cast(Color, obj)

    @property
    def rgb(self):
        return self._rgb_

    @property
    def hex(self):
        return self._hex_


def randcolor():
    """Return a random integer between 0 and 0xFFFFFF as a color object"""
    return Color(random.randint(0x0, 0xFFFFFF))


def rgb2color(rgb: Int3Tuple):
    return Color(rgb2hex(rgb))


def rgb_diff(rgb1: Int3Tuple,
             rgb2: Int3Tuple) -> Int3Tuple:
    lab1 = rgb2lab(rgb1)
    lab2 = rgb2lab(rgb2)
    mid_lab = [(lab1[i] + lab2[i]) / 2 for i in range(3)]
    return lab2rgb(mid_lab)


class SgrParamWrapper:
    __slots__ = '_value_'

    def __init__(self, value=b''):
        self._value_ = value._value_ if type(value) is type(self) else value

    def __hash__(self):
        return hash(self._value_)

    def __eq__(self, other):
        if type(self) is type(other):
            return hash(self) == hash(other)
        return False

    def __bytes__(self):
        return self._value_.__bytes__()

    def __repr__(self):
        return f"{type(self).__name__}({self._value_})"

    def is_same_kind(self, other):
        try:
            return next(_iter_sgr(other)) == self._value_
        except (TypeError, StopIteration, RuntimeError):
            return False

    def is_reset(self):
        return self._value_ == b'0'

    def is_color(self):
        return isinstance(self._value_, colorbytes)


SgrParamWrapper.__name__ = SgrParameter.__name__.lower()


@lru_cache
def _get_sgr_bitmask[_T: (bytes, bytearray, Buffer)](__x: _T) -> list[int]:
    """
    Return a list of integers from a bytestring of ANSI SGR parameters.

        Bitwise equivalent to `list(map(int, bytes().split(b';')))`.
    """
    __x = __x.lstrip(CSI)[:idx if (idx := __x.find(0x6d)) != -1 else None].rstrip(b'm')
    length = len(__x)
    cmp_mask = bytes([0x3b] * length)
    a = int.from_bytes(cmp_mask)
    b = int.from_bytes(__x)
    i = a & b
    j = i ^ a
    buffer = []
    buffer_append = buffer.append
    clear_buffer = buffer.clear
    allocated = []
    allocate_params = lambda: allocated.append(
        int(''.join(map(str, buffer))))
    prepass = ((cmp, val) for cmp, val in
               zip(
                   map(bool, j.to_bytes(length=length)),
                   (x if x < 0x30 else x - 0x30 for x in __x)))
    for c, v in prepass:
        if c is False:
            allocate_params()
            clear_buffer()
        else:
            buffer_append(v)
    if buffer:
        allocate_params()
    return allocated


def _iter_normalized_sgr(__iter) -> Iterator[AnsiColorFormat | int]:
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
                    f"got {type(it).__qualname__!r} instead")


def _co_yield_colorbytes(__iter: Iterator[int]) -> Generator[bytes | AnsiColorFormat, int, None]:
    m: dict[int, ColorDictKeys] = {38: 'fg', 48: 'bg'}
    key_pair = m.get
    get_4b = _ANSI16C_I2KV.get
    new_4b = lambda t: ansicolor4Bit.from_rgb({t[0]: t[1]})
    new_8b = lambda *args: ansicolor8Bit(f"{args[0]};{args[1]};{next(__iter)}".encode())
    new_24b = lambda x: ansicolor24Bit.from_rgb({x: tuple(next(__iter) for _ in range(3))})
    default = lambda x: bytes(ascii(x), 'ansi')
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


def _iter_sgr(__x):
    if isinstance(__x, int):
        __x = [__x]
    return _gen_colorbytes(_iter_normalized_sgr(__x))


class SgrSequence:

    def append(self, __value):
        if __value not in _SGR_PARAM_VALUES:
            raise ValueError(
                f"{__value!r} is not a valid SGR parameter")
        if kv := _ANSI16C_I2KV.get(__value):
            if __value in _ANSI16C_BRIGHT:
                self._has_bright_colors_ = True
            elif (b_idx := self.find(b'1')) > -1:
                self._has_bright_colors_ = True
                self.pop(b_idx)
                kv = _ANSI16C_I2KV.get(__value + 60)
            else:
                self._has_bright_colors_ = False
            value = ansicolor4Bit.from_rgb(dict({kv}))
        else:
            value = b'%d' % __value
        v = SgrParamWrapper(value)
        if v.is_color():
            for key in (rgb_dict := v._value_._rgb_dict_).keys() & self._rgb_dict_:
                color = self.get_color(key)
                self.pop(self.index(color))
            self._rgb_dict_ |= rgb_dict
        self._sgr_params_.append(v)
        self._bytes_ = False

    def find(self, value):
        try:
            return self.index(value)
        except ValueError:
            return -1

    def get_color(self, __key: ColorDictKeys):
        if not self.is_color():
            return None
        return next((v for v in self if v.is_color() and __key in v._value_.rgb_dict), None)

    def index(self, value):
        try:
            return next(i for i, p in enumerate(self) if p.is_same_kind(value))
        except StopIteration:
            raise ValueError(
                f"{value!r} not in sequence") from None

    def is_color(self):
        return any(p.is_color() for p in self)

    def is_reset(self):
        return any(p.is_reset() for p in self)

    def remove(self, value):
        try:
            self.pop(self.index(value))
        except ValueError as e:
            raise ValueError(
                e) from None

    def pop(self, __index=-1):
        try:
            obj = self._sgr_params_.pop(__index)
        except IndexError as e:
            raise IndexError(
                e) from None
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
        self._bytes_ = False
        return obj

    def values(self):
        return list(p._value_ for p in self._sgr_params_)

    def __add__(self, other):
        if type(self) is type(other):
            return SgrSequence([*self, *other])
        if isinstance(other, str):
            return str(self) + other
        raise TypeError(
            f"can only concatenate {SgrSequence.__qualname__} "
            f"(not {type(other).__qualname__!r}) to {SgrSequence.__qualname__}")

    def __bool__(self):
        return bool(self._sgr_params_)

    def __bytes__(self):
        to_param_bytes = lambda b: f"\x1b[{';'.join(map(bytes.decode, b))}m".encode()
        if not self._bytes_:
            self._bytes_ = to_param_bytes(self.values()) if self._sgr_params_ else bytes()
        return self._bytes_

    def __contains__(self, item: ...):
        if not self:
            return False
        try:
            return set(_iter_sgr(item)).issubset(self.values())
        except (TypeError, RuntimeError):
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
                    self_param == other_param for self_param, other_param in
                    zip(self.values(), other.values(), strict=True))
            except ValueError:
                return False
        return False

    def __getitem__(self, item):
        return self._sgr_params_[item]

    def __iadd__(self, other: 'SgrSequence'):
        if type(self) is not type(other):
            raise TypeError(
                f"can only concatenate {SgrSequence.__qualname__} "
                f"(not {type(other).__qualname__!r}) to {SgrSequence.__qualname__}")
        return SgrSequence(self._sgr_params_ + other._sgr_params_)

    def __init__(self, __iter=None, *, ansi_type=None) -> None:
        cls = type(self)
        if type(__iter) is cls:
            other = __iter.__copy__()
            for attr in cls.__slots__:
                setattr(self, attr, getattr(other, attr))
            return

        self._bytes_ = bytes()
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
        fg_slot = bg_slot = None
        is_bold = has_bold = False

        def update_colors(__param: SgrParamWrapper, __rgb_dict: Mapping[ColorDictKeys, Int3Tuple]):
            nonlocal fg_slot, bg_slot
            k: ColorDictKeys
            for k, slot in {'fg': fg_slot, 'bg': bg_slot}.items():
                if v := __rgb_dict.get(k):
                    if slot:
                        remove_param(slot)
                    if k == 'fg':
                        fg_slot = __param
                    else:
                        bg_slot = __param
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
                                    i for i, v in enumerate(self._sgr_params_) if
                                    v._value_ == b'1'))
                            has_bold = False
                update_colors(param, x.rgb_dict)
            append_param(param)
            add_unique(x)

        if self._sgr_params_[-1]._value_ == b'0':
            self._has_bright_colors_ = False
            self._sgr_params_ = [self._sgr_params_.pop()]
            self._rgb_dict_ = {}
        conv = lambda p: bytes(p).decode()
        self._bytes_ = f"\x1b[{';'.join(map(conv, self._sgr_params_))}m".encode()

    def __iter__(self):
        return iter(self._sgr_params_)

    def __radd__(self, other):
        if type(self) is type(other):
            return SgrSequence([*other, *self])
        if isinstance(other, str):
            return other + str(self)
        raise TypeError(
            f"can only concatenate {SgrSequence.__qualname__} "
            f"(not {type(other).__qualname__!r}) to {SgrSequence.__qualname__}")

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
        for k in set(self._rgb_dict_.keys()):
            self.pop(self.index(self.get_color(k)))
            self._bytes_ = False

    @rgb_dict.setter
    def rgb_dict[_AnsiColorType: type[AnsiColorFormat]](
        self,
        __value: tuple[_AnsiColorType, dict[ColorDictKeys, Union[Color, None]]]
    ) -> None:
        ansi_type, color_dict = __value
        for k, v in color_dict.items():
            if k in self._rgb_dict_.keys():
                self.pop(self.index(self.get_color(k)))
            if v is not None:
                color_bytes = ansi_type.from_rgb({k: v})
                self._rgb_dict_ |= color_bytes._rgb_dict_
                self._sgr_params_.append(SgrParamWrapper(color_bytes))
            self._bytes_ = False


# type alias types for ColorStr constructor `color_spec` parameter forms
type _CSpecScalar = Union[int, Color, RGBVector]
type _CSpecKVPair = tuple[ColorDictKeys, _CSpecScalar]
type _CSpecTuplePair = tuple[_CSpecScalar, _CSpecScalar] | tuple[_CSpecKVPair, _CSpecKVPair]
type _CSpecDict = Mapping[ColorDictKeys, _CSpecScalar]
type _CSpecType = (Union[SgrSequence, str, bytes]
                   | _CSpecScalar
                   | _CSpecTuplePair
                   | _CSpecKVPair
                   | _CSpecDict)

_ColorSpec = TypeVar('_ColorSpec', bound=_CSpecType)


class _ColorStrKwargs(TypedDict, total=False):
    alt_spec: _ColorSpec
    ansi_type: Optional[AnsiColorParam]
    no_reset: bool


class _ColorDict(TypedDict, total=False):
    fg: Union[Color, AnsiColorFormat, None]
    bg: Union[Color, AnsiColorFormat, None]


_NumberLike = frozenset({int, Color})
_CSpecPairTypeSet = frozenset({tuple, list, ndarray} | _NumberLike)


def _solve_color_spec[_T: (_CSpecType, SgrSequence)](
    color_spec: _T | None,
    ansi_type: AnsiColorType
):
    if isinstance(color_spec, SgrSequence):
        return color_spec
    if color_spec is None:
        return SgrSequence()
    typ: type[_T] = type(color_spec)
    if typ in _NumberLike:
        return SgrSequence(
            [ansi_type.from_rgb({'fg': hex2rgb(color_spec) if typ is int else color_spec.rgb})])
    if typ is tuple:
        if (n_args := len(color_spec)) == 3:  # rgb value
            return SgrSequence([ansi_type.from_rgb({'fg': Color.from_rgb(color_spec).rgb})])
        elif n_args == 2:
            t1, t2 = tuple(map(type, color_spec))
            if {t1, t2} <= _CSpecPairTypeSet:
                v1, v2 = color_spec
                if t1 in _NumberLike:
                    v1 = hex2rgb(v1) if t1 is int else v1.rgb
                if t2 in _NumberLike:
                    v2 = hex2rgb(v2) if t2 is int else v2.rgb
                if len(v1) == len(v2) == 3:  # rgb value pair
                    return SgrSequence(
                        [ansi_type.from_rgb(
                            {'fg': Color.from_rgb(v1).rgb, 'bg': Color.from_rgb(v2).rgb})])
                else:  # assume dict-items
                    d = dict(color_spec)
                    for k, v in d.copy().items():
                        if k not in {'fg', 'bg'}:
                            raise ValueError(
                                f"{k!r} is not a valid color dict key") from None
                        try:
                            f = {int: hex2rgb,
                                 Color: lambda i: cast(Color, i).rgb,
                                 tuple: lambda i: i}[type(v)]
                        except KeyError:
                            raise TypeError(
                                f"invalid key-value pair {(k, v)}: {v!r} is not an RGB value"
                            ) from None
                        d[k] = f(v)
                    return SgrSequence([ansi_type.from_rgb({k: v}) for k, v in d.items()])
            elif t1 is str:  # kv pair
                v = color_spec[1]
                if t2 is not tuple:
                    v = hex2rgb(v) if t2 is not Color else v.rgb
                return SgrSequence([ansi_type.from_rgb({color_spec[0]: v})])
    elif issubclass(typ, Mapping):
        if unexpected_keys := color_spec.keys() - {'fg', 'bg'}:
            raise ValueError(
                f"unexpected keys: {set(unexpected_keys)}")
        d = {}
        for k, v in color_spec.items():
            d[k] = v if (vt := type(v)) is tuple else hex2rgb(v) if vt is not Color else v.rgb
        return SgrSequence([ansi_type.from_rgb({k: v}) for k, v in d.items()])
    raise TypeError(
        f"unexpected type: {typ.__qualname__!r} is not a valid color spec")


def _get_color_str_vars(base_str: Optional[str],
                        color_spec: Optional[_ColorSpec],
                        ansi_type: type[AnsiColorFormat] = None) -> tuple[SgrSequence, str]:
    if color_spec is None:
        return SgrSequence([]), base_str or ''
    if ansi_type is None:
        ansi_type = DEFAULT_ANSI
    if isinstance(color_spec, (str, Buffer)):
        if hasattr(color_spec, 'encode'):
            color_spec = color_spec.encode()
        if (csi_count := color_spec.count(CSI)) > 1:
            color_spec = color_spec.lstrip(CSI).rstrip(SGR_RESET.encode())
            color_spec, _, byte_str = color_spec.partition(b'm')
            base_str = byte_str.decode()
            if color_spec.count(CSI) > 1:
                raise ValueError(
                    f"color spec contains {csi_count} escape sequences, "
                    f"expected only 1") from None
        kw = dict() if csi_count != 0 else dict(ansi_type=ansi_type)
        sgr_params = SgrSequence(color_spec, **kw)
    else:
        sgr_params = _solve_color_spec(color_spec, ansi_type=ansi_type)
    base_str = base_str or ''
    return sgr_params, base_str


class _ColorStrWeakVars(TypedDict, total=False):
    _base_str_: str
    _sgr_: SgrSequence
    _no_reset_: bool


class _AnsiBytesGetter:

    def __get__(self, instance: Union['ColorStr', None], objtype=None):
        if instance is None:
            return
        return instance._sgr_.__bytes__()


class _SgrParamsGetter:

    def __get__(self, instance: Union['ColorStr', None], objtype=None):
        if instance is None:
            return
        return instance._sgr_._sgr_params_


class _ColorDictGetter:

    def __get__(self, instance: Union['ColorStr', None], objtype=None):
        if instance is None:
            return
        return {k: Color.from_rgb(v) for k, v in instance._sgr_.rgb_dict.items()}


class ColorStr(str):

    def _weak_var_update(self, **kwargs):
        obj_vars = vars(self).copy()
        if not kwargs.keys() <= obj_vars.keys():
            raise ValueError(
                f"unexpected keys: {set(kwargs.keys() - obj_vars.keys())}") from None
        obj_vars |= kwargs
        sgr = kwargs.get('_sgr_', self._sgr_)
        base_str = kwargs.get('_base_str_', self._base_str_)
        suffix = '' if kwargs.get('_no_reset_', self._no_reset_) else SGR_RESET
        obj = super().__new__(ColorStr, f"{sgr}{base_str}{suffix}")
        obj.__dict__ |= obj_vars
        return cast(ColorStr, obj)

    def as_ansi_type(self, __ansi_type):
        """
        Convert all ANSI color codes in the `ColorStr` to a single ANSI type.

        Parameters
        ----------
            __ansi_type : str or type[ansicolor4Bit | ansicolor8Bit | ansicolor24Bit]
                ANSI format to which all SGR parameters of type `colorbytes` will be cast.

        Returns
        -------
            ColorStr
                Return `self` if all ANSI formats are already the input type.
                Otherwise, return reformatted ColorStr.

        """
        ansi_type = get_ansi_type(__ansi_type)
        if self._sgr_.is_color():
            new_params = []
            new_rgb = {}
            for p in self._sgr_params_:
                if not p.is_color() or type(p._value_) is ansi_type:
                    new_params.append(p)
                else:
                    new_ansi = ansi_type.from_rgb(p._value_)
                    new_rgb |= new_ansi._rgb_dict_
                    new_params.append(SgrParamWrapper(new_ansi))
            if new_params == self._sgr_params_:
                return self
            new_sgr = SgrSequence()
            for name, value in zip(('_sgr_params_', '_rgb_dict_'), (new_params, new_rgb)):
                setattr(new_sgr, name, value)
            s = f"{new_sgr}{self.base_str}{'' if self.no_reset else SGR_RESET}"
            obj = super().__new__(type(self), s)
            obj_vars = vars(self).copy() | {'_sgr_': new_sgr, '_ansi_type_': ansi_type}
            for name, value in obj_vars.items():
                setattr(obj, name, value)
            if getattr(self, '_alt_', None):
                setattr(obj, '_alt_', self.alt.as_ansi_type(ansi_type))
            return cast(ColorStr, obj)
        return self

    def format(self, *args, **kwargs):
        return self._weak_var_update(_base_str_=self.base_str.format(*args, **kwargs))

    def split(self, sep=None, maxsplit=-1):
        return list(
            self._weak_var_update(_base_str_=s) for s in
            self.base_str.split(sep=sep, maxsplit=maxsplit))

    def recolor(self, __value=None, absolute=False, **kwargs):
        """
        Return a copy of `self` with a new color spec.

        If `__value` is a ColorStr, return `self` with the colors of `__value`.

        Parameters
        ----------
            __value : ColorStr, optional
                A ColorStr object that the new instance will inherit colors from.

            absolute : bool
                If True, overwrite all colors of the current object with the provided arguments,
                removing any existing colors not explicitly set by the arguments.
                Otherwise, only replace colors where specified (default).

        Keyword Args
        ------------
            fg : Color, optional
                New foreground color

            bg : Color, optional
                New background color

        Returns
        -------
            recolored_cs : ColorStr
                A new ColorStr instance recolored by the input parameters

        Raises
        ------
            TypeError
                If `__value` is not None but is not an instance of ColorStr

            ValueError
                If any unexpected keys or value types found in `kwargs`

        Examples
        --------
            # Passing a value from `ColorStr` as keyword argument
            >>> cs1 = ColorStr('foo', randcolor())
            >>> cs2 = ColorStr('bar', dict(fg=Color(0xFF0000), bg=Color(0xFF00FF)))
            >>> new_cs = cs2.recolor(bg=cs1.fg)
            >>> new_cs.fg.hex == 0xFF0000, new_cs.bg == cs1.fg
            (True, True)

            >>> cs = ColorStr("Red text", ('fg', 0xFF0000))
            >>> recolored = cs.recolor(fg=Color(0x00FF00))
            >>> recolored.base_str, f"{recolored.hex_dict['fg']:06X}"
            ('Red text', '00FF00')

        See Also
        --------
            ColorStr.__matmul__ : Transfer colors between ColorStr objects using the '@' operator
        """
        if __value:
            if isinstance(__value, ColorStr):
                kwargs = __value._color_dict_
            else:
                raise TypeError(
                    f"expected positional argument of type {ColorStr.__qualname__!r}, "
                    f"got {type(__value).__qualname__!r} instead") from None
        elif not kwargs:
            return self
        valid, context = is_matching_typed_dict(kwargs, _ColorDict)
        if not valid:
            raise ValueError(
                context)
        sgr = SgrSequence(self._sgr_)
        if bool(absolute):
            del sgr.rgb_dict
        sgr.rgb_dict = (self.ansi_format, kwargs)
        return self._weak_var_update(_sgr_=sgr)

    def replace(self, __old, __new, __count=-1):
        if isinstance(__new, ColorStr):
            __new = __new.base_str
        return self._weak_var_update(_base_str_=self.base_str.replace(__old, __new, __count))

    def update_sgr(self, *p):
        """
        Return a copy of `self` with updated SGR sequence parameters.

        Parameters
        ----------
            *p: SgrParameter | int
                The SGR parameter value(s) to be added or removed from the ColorStr.
                A value already in `self` SGR sequence gets removed, else it gets added.
                If no values are passed, returns `self` unchanged.

        Returns
        -------
            ColorStr
                A new ColorStr object with the SGR updates applied.

        Raises
        ------
            ValueError
                If any of the SGR parameters are invalid, or if extended color codes are passed.

        Notes
        -----
        * The extended color escapes `{38, 48}` require extra parameters and so raise a ValueError.
            `ColorStr.as_ansi_type` should be used to change ANSI color format instead.

        Examples
        --------
            # creating an empty ColorStr object
            >>> empty_cs = ColorStr(no_reset=True)
            >>> empty_cs.ansi
            b''

            # adding red foreground color
            >>> red_fg = empty_cs.update_sgr(SgrParameter.RED_FG)
            >>> red_fg.rgb_dict
            {'fg': (170, 0, 0)}

            # removing the same parameter
            >>> empty_cs = red_fg.update_sgr(31)
            >>> empty_cs.rgb_dict
            {}
            >>> empty_cs.ansi
            b''

            # adding more parameters
            >>> styles = [SgrParameter.BOLD, SgrParameter.ITALICS, SgrParameter.NEGATIVE]
            >>> stylized_cs = empty_cs.update_sgr(*styles)
            >>> stylized_cs.ansi.replace(CSI, b'ESC[')
            b'ESC[1;3;7m'

            # parameter updates also supported by the `__add__` operator
            >>> stylized_cs += SgrParameter.BLACK_BG    # add background color
            >>> stylized_cs += SgrParameter.BOLD    # remove bold style
            >>> stylized_cs.ansi.replace(CSI, b'ESC[')
            b'ESC[3;7;40m'
            >>> stylized_cs.rgb_dict
            {'bg': (0, 0, 0)}
        """
        if not p:
            return self
        not_allowed = _ANSI256_KEY2I.values()
        new_sgr = SgrSequence(self._sgr_)
        for x in p:
            assert isinstance(x, int)
            assert x not in not_allowed
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
            formats: list[AnsiColorType] = list(type(p._value_) for p in new_sgr if p.is_color())
            ansi_type = max(formats, key=formats.count)
        else:
            ansi_type = self.ansi_format
        s = f"{new_sgr}{self.base_str}{'' if self.no_reset else SGR_RESET}"
        obj = super().__new__(type(self), s)
        obj.__dict__ |= vars(self).copy() | {'_sgr_': new_sgr, '_ansi_type_': ansi_type}
        return cast(ColorStr, obj)

    def __add__(self, other):
        if isinstance(other, SgrParameter):
            return self.update_sgr(other)
        if isinstance(other, ColorStr):
            sgr = SgrSequence(self._sgr_) + SgrSequence(other._sgr_)
            base_str = self._base_str_ + other._base_str_
            return self._weak_var_update(_base_str_=base_str, _sgr_=sgr)
        if isinstance(other, str):
            base_str = self._base_str_ + other
            return self._weak_var_update(_base_str_=base_str)
        raise TypeError(
            f"can only concatenate "
            f"{str.__qualname__}, {ColorStr.__qualname__}, or {SgrParameter.__qualname__} "
            f"(got {type(other).__qualname__!r}) "
            f"to {type(self).__qualname__}")

    def __and__(self, other):
        """
        Return `self.alt` or `other.alt` if colors intersect between them, otherwise return `self`
        """
        if not isinstance(other, ColorStr):
            return self
        if (self.alt or other.alt) and set(self.hex_dict.values()).intersection(
                other.hex_dict.values()):
            return self.alt or other.alt
        return self

    def __contains__(self, __key: str):
        if type(__key) is not str:
            return False
        if __key == str(self._sgr_):
            return True
        if self.no_reset is False and __key == SGR_RESET:
            return True
        return self.base_str.__contains__(__key)

    def __eq__(self, other):
        if type(self) is type(other):
            return hash(self) == hash(other)
        return False

    def __format__(self, format_spec=''):
        if format_spec in (d := {'4b': ansicolor4Bit, '8b': ansicolor8Bit, '24b': ansicolor24Bit}):
            return str(self.as_ansi_type(d[format_spec]))
        return str.__format__(self, format_spec)

    def __getitem__(self, __key: Union[SupportsIndex, slice]):
        try:
            it = list(self)[__key]
            return it if type(it) is str else ''.join(it)
        except IndexError as e:
            raise IndexError(
                str(e).replace('list', 'string')) from None

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
                * Any tuple pair of the aforementioned types
                    ('fg'=color_spec[0], 'bg'=color_spec[1])
                * A key-value pair or `dict_items`-like tuple
                    ('fg', ...) or (('fg', ...), ('bg', ...))
                * A dictionary mapping
                    dict[Literal['fg', 'bg'], ...]

        Keyword Args
        ------------
            alt_spec : type[_ColorSpec | ColorStr], optional
                An alternate color spec returned by `ColorStr.__and__`

            ansi_type : str or type[ansi_color_4Bit | ansi_color_8Bit | ansi_color_24Bit], optional
                An ANSI format to cast all `colorbytes` params to before formatting the string.
                * ANSI format can also be changed on instances using `ColorStr.as_ansi_type()`
                * Reformatting recursively applies to `alt_spec` if `alt_spec` is not None

            no_reset : bool
                If True, create the ColorStr without concatenating a 'reset all' SGR sequence.
                Default is False (new instances get concatenated with reset sequences).

        Returns
        -------
            ColorStr
                A new ColorStr object comprised of the base string and provided ANSI sequences.

        Notes
        -----
        * Each of the ANSI color formats can be invoked by their alias in place of the type object:
            (ansicolor4Bit, '4b'), (ansicolor8Bit, '8b'), (ansicolor24Bit, '24b')
        * See `help(ansicolor4Bit | ansicolor8Bit | ansicolor24Bit)`
            for color code ranges and formatting examples.

        * If `obj` is undefined and `color_spec` is an instance of `str` or `bytes`,
            the value will be parsed as a literal escape sequence (see examples).

        Examples
        --------
            >>> cs = ColorStr('Red text', ('fg', 0xFF0000))
            >>> cs.rgb_dict, cs.base_str
            ({'fg': (255, 0, 0)}, 'Red text')

            >>> cs_from_rgb = ColorStr(color_spec={'fg': (255, 85, 85)}, ansi_type='4b')
            >>> cs_from_literal = ColorStr(color_spec='\x1b[91m')
            >>> cs_from_rgb == cs_from_literal
            True

            # ANSI 4-bit sequences of the form `ESC[<1 (bold)>;<{30-37} | {40-47}>...`
            # are equivalent to their 'bright' counterparts `ESC[<{90-97} | {100-107}>...`
            >>> cs_from_literal_alt = ColorStr(color_spec='\x1b[1;31m')
            >>> cs_from_literal_alt == cs_from_literal
            True

            # the alt version will be autocast to the 'bright' sequence form
            >>> cs_from_literal_alt.ansi.replace(CSI, b'ESC[')
            b'ESC[91m'

        See Also
        --------
            ColorStr.__and__ : Return the alt spec of two ColorStr objects using the '&' operator.
            Color: Subclasses `int`, used for intermediate conversions.
            colorbytes: Base class of the ANSI formats; implements shared methods.
            ansicolor4Bit: colorbytes subclass for 4-Bit '16color' escape sequences.
            ansicolor8Bit: colorbytes subclass for 8-Bit escape sequences, with 256 color values.
            ansicolor24Bit: colorbytes subclass for 24-Bit escape sequences, which uses RGB values.
        """
        self._alt_ = None
        if (alt_spec := kwargs.get('alt_spec')) is not None:
            if alt_spec != self:
                self._alt_ = ColorStr(self._base_str_, color_spec=alt_spec)

    def __iter__(self):
        while True:
            yield str(self._sgr_)
            yield from self.base_str
            if self.no_reset is False:
                yield SGR_RESET
            return

    def __len__(self):
        return self.base_str.__len__()

    def __matmul__(self, other):
        """Return a new `ColorStr` with the base string of `self` and colors of `other`"""
        if type(self) is type(other):
            return self._weak_var_update(_sgr_=other._sgr_, _no_reset_=other.no_reset)
        raise TypeError(
            'unsupported operand type(s) for @: '
            f"{type(self).__qualname__!r} and {type(other).__qualname__!r}")

    def __mod__(self, __value):
        return self._weak_var_update(_base_str_=self.base_str % __value)

    def __mul__(self, __value):
        base_str = self.base_str
        base_str *= __value
        return self._weak_var_update(_base_str_=base_str)

    def __neg__(self):
        """Return a copy of `self` with negative colors (XORed by '0xFFFFFF')"""
        sgr = SgrSequence(self._sgr_)
        sgr.rgb_dict = (
            self.ansi_format, {k: Color(0xFFFFFF ^ v) for k, v in self.hex_dict.items()})
        obj = self._weak_var_update(_sgr_=sgr)
        if getattr(self, '_alt_', None):
            setattr(obj, '_alt_', -self.alt)
        return obj

    def __new__(cls, obj=None, color_spec=None, **kwargs):
        if ansi_type := kwargs.get('ansi_type'):
            ansi_type = get_ansi_type(ansi_type)
        if type(color_spec) is cls:
            if ansi_type is not None and any(type(a) is not ansi_type for a in color_spec.ansi):
                return color_spec.as_ansi_type(ansi_type)
            inst = super().__new__(cls, str(color_spec))
            for name, value in vars(color_spec).items():
                setattr(inst, name, value)
            return cast(ColorStr, inst)
        if obj is not None:
            if not isinstance(obj, str):
                kw = {'encoding': 'ansi'} if isinstance(obj, Buffer) else {}
                obj = str(obj, **kw)
            if color_spec is None and obj.startswith(CSI.decode()):
                color_spec = obj.encode()
                obj = None
        sgr, base_str_ = _get_color_str_vars(obj, color_spec, cast(AnsiColorType, ansi_type))
        if ansi_type is None:
            if sgr.is_color():
                formats = list(type(p._value_) for p in sgr._sgr_params_ if p.is_color())
                ansi_type = max(formats, key=formats.count)
            else:
                ansi_type = DEFAULT_ANSI
        no_reset = bool(kwargs.get('no_reset', False))
        inst = super().__new__(cls, f"{sgr}{base_str_}{'' if no_reset else SGR_RESET}")
        inst.__dict__ |= {
            '_ansi_type_': ansi_type,
            '_base_str_': base_str_,
            '_sgr_': sgr,
            '_no_reset_': no_reset
        }
        return cast(ColorStr, inst)

    def __repr__(self):
        s = (self.base_str, self.ansi.decode(),
             getattr(self.ansi_format, '__qualname__', type(None).__qualname__))
        return f"{type(self).__qualname__}(%r, color_spec=%r, ansi_type=%s)" % s

    def __reversed__(self):
        while True:
            if self.no_reset is False:
                yield SGR_RESET
            yield from reversed(self.base_str)
            yield str(self._sgr_)
            return

    def __sub__(self, other):
        """Return a copy of `self` with colors adjusted by color difference with `other`"""
        if (vt := type(other)) not in {Color, ColorStr}:
            raise TypeError(
                'unsupported operand type(s) for -: '
                f"{ColorStr.__qualname__!r} and {vt.__qualname__!r}")
        k: Literal['fg', 'bg']
        if vt is Color:
            diff_dict = {k: rgb2color(rgb_diff(v, other.rgb)) for k, v in self.rgb_dict.items()}
        else:
            if not (shared_keys := self.rgb_dict.keys() & other.rgb_dict):
                return self
            f = lambda s_rgb, o_rgb: Color.from_rgb(rgb_diff(s_rgb, o_rgb))
            diff_dict = {k: f(self.rgb_dict[k], other.rgb_dict[k]) for k in shared_keys}
        sgr = SgrSequence(self._sgr_)
        sgr.rgb_dict = (self.ansi_format, diff_dict)
        return self._weak_var_update(_sgr_=sgr)

    _ansi_ = _AnsiBytesGetter()
    _color_dict_ = _ColorDictGetter()
    _sgr_params_ = _SgrParamsGetter()

    @property
    def alt(self) -> Union['ColorStr', None]:
        return self._alt_

    @alt.deleter
    def alt(self):
        self._alt_ = None

    @alt.setter
    def alt(self, __value: Union['ColorStr', None]) -> None:
        cls = type(self)
        if isinstance(__value, cls) or __value is None:
            self._alt_ = __value
            return
        raise TypeError(
            f"Expected {cls.__qualname__!r} or {type(None).__qualname__}, "
            f"got {type(__value).__qualname__!r} instead")

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
    def hex_dict(self):
        return {k: v.hex for k, v in self._color_dict_.items()}

    @property
    def no_reset(self):
        return self._no_reset_

    @property
    def rgb_dict(self):
        return {k: v.rgb for k, v in self._color_dict_.items()}


def hsl_gradient(start: Int3Tuple | Float3Tuple,
                 stop: Int3Tuple | Float3Tuple,
                 step: SupportsIndex,
                 num: SupportsIndex = None,
                 ncycles: int | float = float('inf'),
                 replace_idx: tuple[
                     SupportsIndex | Iterable[SupportsIndex], Iterator[Color]] = None,
                 dtype: Callable[[Int3Tuple], int] | type[Color] = Color):
    replace_idx, rgb_iter = _resolve_replacement_indices(replace_idx)
    while abs(float(step)) < 1:
        step *= 10
    color_vec = _init_gradient_color_vec(num, start, step, stop)
    color_iter = iter(color_vec)
    type_map: dict[type[Color | int], ...] = {Color: lambda x: x.rgb, int: lambda x: hex2rgb(x)}
    get_rgb_iter_idx: Callable[[Color | int, SupportsIndex], int] = lambda x, ix: \
        rgb2hsl(type_map[type(x)](x))[ix]
    next_rgb_iter = None
    prev_output = None
    while ncycles > 0:
        try:
            cur_iter = next(color_iter)
            if cur_iter != prev_output:
                for idx in replace_idx:
                    try:
                        next_rgb_iter = next(rgb_iter)
                        cur_iter = list(cur_iter)
                        cur_iter[idx] = get_rgb_iter_idx(next_rgb_iter, idx)
                    except StopIteration:
                        raise GeneratorExit
                    except KeyError:
                        raise TypeError(
                            f"Expected iterator to return "
                            f"{repr(Color.__qualname__)} or {repr(int.__qualname__)}, "
                            f"got {repr(type(next_rgb_iter).__qualname__)} instead") from None
                output = hsl2rgb(cast(Float3Tuple, cur_iter))
                if callable(dtype):
                    output = dtype(output)
                yield output
            prev_output = cur_iter
        except StopIteration:
            ncycles -= 1
            color_vec.reverse()
            color_iter = iter(color_vec)
        except GeneratorExit:
            break


def _resolve_replacement_indices(
    replace_idx: tuple[SupportsIndex | Sequence[SupportsIndex], Iterator[Color]] = None
):
    if replace_idx is not None:
        replace_idx, rgb_iter = replace_idx
        if not isinstance(rgb_iter, Iterator):
            raise TypeError(
                f"Expected 'replace_idx[1]' to be an iterator, got {type(rgb_iter).__name__} "
                f"instead")
        if not isinstance(replace_idx, Sequence):
            replace_idx = {replace_idx}
        else:
            replace_idx = set(replace_idx)
        valid_idx_range = range(3)
        if any(idx_diff := replace_idx.difference(valid_idx_range)):
            raise ValueError(
                f"Invalid replacement indices: {idx_diff}")
        if replace_idx == set(valid_idx_range):
            raise ValueError(
                f"All 3 indexes selected for replacement: {replace_idx=}")
    else:
        rgb_iter = None
        replace_idx = []
    return replace_idx, rgb_iter


def _init_gradient_color_vec(num: SupportsIndex,
                             start: Int3Tuple | Float3Tuple,
                             step: SupportsIndex,
                             stop: Int3Tuple | Float3Tuple):
    def convert_bounds(rgb: Int3Tuple):
        if all(0 <= n <= 255 for n in rgb):
            return rgb2hsl(rgb)
        raise ValueError

    start, stop = tuple(map(convert_bounds, (start, stop)))
    start_h, start_s, start_l = start
    stop_h, stop_s, stop_l = stop
    if num:
        num_samples = num
    else:
        abs_h = abs(stop_h - start_h)
        h_diff = min(abs_h, 360 - abs_h)
        dist = math.sqrt(h_diff ** 2 + (stop_s - start_s) ** 2 + (stop_l - start_l) ** 2)
        num_samples = max(int(dist / float(step)), 1)
    color_vec = [np.linspace(*bounds, num=num_samples, dtype=float) for bounds in zip(start, stop)]
    color_vec = list(zip(*color_vec))
    return color_vec


def rgb_luma_transform(rgb: Int3Tuple,
                       start: SupportsIndex = None,
                       num: SupportsIndex = 50,
                       step: SupportsIndex = 1,
                       cycle: bool | Literal['wave'] = False,
                       ncycles: int | float = float('inf'),
                       gradient: Int3Tuple = None,
                       dtype: type[Color] = None) -> Iterator[Int3Tuple | int | Color]:
    if dtype is None:
        ret_type = tuple
    elif issubclass(dtype, int):
        ret_type = lambda x: dtype(rgb2hex(x))
    is_cycle = bool(cycle is not False)
    is_oscillator = cycle == 'wave'
    if is_oscillator:
        ncycles *= 2
    h, s, luma = rgb2hsl(rgb)
    luma_linspace = [*np.linspace(start=0, stop=1, num=num)][::step]
    if start:
        start = min(max(float(start), 0), 1)
        luma = min(luma_linspace, key=lambda x: abs(x - start))
        start_idx = luma_linspace.index(luma)
        remaining_indices = luma_linspace[start_idx:]
        luma_iter = iter(remaining_indices)
    else:
        luma_iter = iter(luma_linspace)

    def _generator():
        nonlocal luma_iter, ncycles
        if step == 0:
            yield rgb
            return
        prev_output = None
        while ncycles > 0:
            try:
                output = hsl2rgb((h, s, next(luma_iter)))
                if output != prev_output:
                    yield ret_type(output)
                prev_output = output
            except StopIteration as STOP_IT:
                if not is_cycle:
                    raise STOP_IT
                ncycles -= 1
                if is_oscillator:
                    luma_linspace.reverse()
                luma_iter = iter(luma_linspace)

    if gradient is not None:
        _gradient = hsl_gradient(
            start=rgb, stop=gradient, step=step, num=num, replace_idx=(2, _generator()))
        return iter(_gradient)
    return iter(_generator())
