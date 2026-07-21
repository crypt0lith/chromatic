__all__ = [
    "CSI",
    "Color",
    "ColorChainDType",
    "ColorStr",
    "SGR_RESET",
    "SgrFlag",
    "SgrParameter",
    "SgrSequence",
    "ansicolor24Bit",
    "ansicolor4Bit",
    "ansicolor8Bit",
    "color_chain",
    "colorbytes",
    "get_ansi_type",
    "is_vt_enabled",
    "randcolor",
    "rgb2ansi_escape",
]

import collections.abc as abc
import enum
import re
import typing as tp
from types import MappingProxyType as mappingproxy
from typing import Literal as L

import numpy as np
from _typeshed import ConvertibleToInt, SupportsKeysAndGetItem

from .._typing import (
    Ansi4BitAlias,
    Ansi8BitAlias,
    Ansi24BitAlias,
    AnsiColorAlias,
    ColorDictKeys,
    Int3Tuple,
    RGBVectorLike,
    ShapedNDArray,
    TupleOf3,
)

CSI: tp.Final[bytes]
SGR_RESET: tp.Final[bytes]
SGR_RESET_S: tp.Final[str]

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

class SgrFlag(enum.IntFlag):
    RESET = 0x1
    BOLD = 0x2
    FAINT = 0x4
    ITALICS = 0x8
    SINGLE_UNDERLINE = 0x10
    SLOW_BLINK = 0x20
    RAPID_BLINK = 0x40
    NEGATIVE = 0x80
    CONCEALED_CHARS = 0x100
    CROSSED_OUT = 0x200
    PRIMARY = 0x400
    FIRST_ALT = 0x800
    SECOND_ALT = 0x1000
    THIRD_ALT = 0x2000
    FOURTH_ALT = 0x4000
    FIFTH_ALT = 0x8000
    SIXTH_ALT = 0x10000
    SEVENTH_ALT = 0x20000
    EIGHTH_ALT = 0x40000
    NINTH_ALT = 0x80000
    GOTHIC = 0x100000
    DOUBLE_UNDERLINE = 0x200000
    RESET_BOLD_AND_FAINT = 0x400000
    RESET_ITALIC_AND_GOTHIC = 0x800000
    RESET_UNDERLINES = 0x1000000
    RESET_BLINKING = 0x2000000
    POSITIVE = 0x4000000
    REVEALED_CHARS = 0x8000000
    RESET_CROSSED_OUT = 0x10000000
    DEFAULT_FG_COLOR = 0x20000000
    DEFAULT_BG_COLOR = 0x40000000
    FRAMED = 0x80000000
    ENCIRCLED = 0x100000000
    OVERLINED = 0x200000000
    NOT_FRAMED_OR_CIRCLED = 0x400000000
    IDEOGRAM_UNDER_OR_RIGHT = 0x800000000
    IDEOGRAM_2UNDER_OR_2RIGHT = 0x1000000000
    IDEOGRAM_OVER_OR_LEFT = 0x2000000000
    IDEOGRAM_2OVER_OR_2LEFT = 0x4000000000
    CANCEL = 0x8000000000

    @property
    def parameters(self) -> list[SgrParameter]: ...

_P2F: tp.Final[dict[int, int]]
_F2P: tp.Final[dict[int, int]]
_ANSI16C_I2KV: tp.Final[dict[int, tuple[ColorDictKeys, Int3Tuple]]]
_ANSI16C_KV2I: tp.Final[dict[tuple[ColorDictKeys, Int3Tuple], int]]
_ANSI256_B2KEY: tp.Final[dict[L[b"38", b"48"], ColorDictKeys]]
_ANSI256_KEY2I: tp.Final[dict[ColorDictKeys, int]]

class colorbytes(bytes):
    @classmethod
    @tp.overload
    def from_rgb[_T, _KT: (L["bg"], L["fg"]), _VT: (tp.SupportsInt, RGBVectorLike)](
        cls: type[_T], rgb: tuple[ColorDictKeys, _VT] | abc.Mapping[_KT, _VT], /
    ) -> _T: ...
    @classmethod
    @tp.overload
    def from_rgb[_T, _VT: (tp.SupportsInt, RGBVectorLike)](
        cls: type[_T], rgb: tuple[str, _VT] | abc.Mapping[str, _VT], /
    ) -> _T: ...

    @tp.overload
    def __new__[_T: (ansicolor4Bit, ansicolor8Bit, ansicolor24Bit)](
        cls: type[_T], ansi: bytes | AnsiColorFormat, /
    ) -> _T: ...
    @tp.overload
    def __new__(cls, ansi: bytes, /) -> AnsiColorFormat: ...

    def kind(self) -> ColorDictKeys: ...
    def to_param_buffer(self) -> SgrParamBuffer[tp.Self]: ...

    rgb_dict: mappingproxy[L["fg"], Int3Tuple] | mappingproxy[L["bg"], Int3Tuple]

class ansicolor4Bit(colorbytes):
    alias: tp.ClassVar[L["4b"]]
    typecode: tp.ClassVar[L[1]]

class ansicolor8Bit(colorbytes):
    alias: tp.ClassVar[L["8b"]]
    typecode: tp.ClassVar[L[2]]

class ansicolor24Bit(colorbytes):
    alias: tp.ClassVar[L["24b"]]
    typecode: tp.ClassVar[L[3]]

def is_vt_enabled() -> bool: ...

DEFAULT_ANSI: tp.Final[type[ansicolor8Bit | ansicolor4Bit]]

AnsiColorFormat: tp.TypeAlias = ansicolor4Bit | ansicolor8Bit | ansicolor24Bit
AnsiColorType: tp.TypeAlias = type[AnsiColorFormat]
AnsiColorParam: tp.TypeAlias = AnsiColorAlias | AnsiColorType
_ANSI_COLOR_TYPES: tp.Final[frozenset[AnsiColorType]]
_ANSI_FORMAT_MAP: tp.Final[dict[AnsiColorAlias | AnsiColorType, AnsiColorType]]

@tp.overload
def get_ansi_type(typ: None = None, /) -> type[ansicolor8Bit | ansicolor4Bit]: ...
@tp.overload
def get_ansi_type[_T: AnsiColorType](typ: _T, /) -> _T: ...
@tp.overload
def get_ansi_type(typ: Ansi4BitAlias, /) -> type[ansicolor4Bit]: ...
@tp.overload
def get_ansi_type(typ: Ansi8BitAlias, /) -> type[ansicolor8Bit]: ...
@tp.overload
def get_ansi_type(typ: Ansi24BitAlias, /) -> type[ansicolor24Bit]: ...

def set_default_ansi(typ: AnsiColorAlias | AnsiColorType, /) -> None: ...
def sgr_pattern() -> re.Pattern[str]: ...
def rgb2ansi_escape(
    fmt: AnsiColorAlias | AnsiColorType, /, mode: ColorDictKeys, rgb: Int3Tuple
) -> bytes: ...

class Color(int):
    @tp.overload
    def __new__(cls, x: ConvertibleToInt = ..., /) -> tp.Self: ...
    @tp.overload
    def __new__(
        cls, x: str | bytes | bytearray, /, base: tp.SupportsIndex = 10
    ) -> tp.Self: ...

    def __invert__(self) -> Color: ...
    @classmethod
    def from_rgb(cls, rgb: RGBVectorLike, /) -> tp.Self: ...
    @property
    def rgb(self) -> Int3Tuple: ...

def randcolor() -> Color: ...

class SgrParamBuffer[_VT: (bytes, ansicolor4Bit, ansicolor8Bit, ansicolor24Bit)]:
    __slots__ = ("_value", "_bytes", "_is_color", "_is_reset")
    __match_args__ = ("value",)

    def __buffer__(self, flags: int, /) -> memoryview: ...
    def __bytes__(self) -> bytes: ...

    @tp.overload
    def __eq__(self, other: SgrParamBuffer, /) -> tp.TypeIs[tp.Self]: ...
    @tp.overload
    def __eq__(self, other: bytes, /) -> tp.TypeIs[_VT]: ...

    def __eq__(self, other, /) -> bool: ...
    def __hash__(self) -> int: ...

    @tp.overload
    def __new__[_T: SgrParamBuffer](cls, value: _T, /) -> _T: ...
    @tp.overload
    def __new__[_T: (bytes, ansicolor4Bit, ansicolor8Bit, ansicolor24Bit)](
        cls, value: _T, /
    ) -> SgrParamBuffer[_T]: ...
    @tp.overload
    def __new__(cls, value: bytearray, /) -> SgrParamBuffer[bytes]: ...

    @property
    def value(self) -> _VT: ...
    def is_color(self) -> bool: ...
    def is_reset(self) -> bool: ...

    _value: _VT
    _bytes: bytes
    _is_color: bool
    _is_reset: bool

class SgrSequence(abc.MutableSequence[SgrParamBuffer]):
    __slots__ = ("_sgr_params", "_bg_idx", "_fg_idx")
    __match_args__ = ("_sgr_params",)

    class _color_descriptor:
        def __set_name__(self, objtype: type, name: str, /) -> None: ...
        def __get__(self, inst, objtype: type | None = None) -> Int3Tuple | None: ...
        def __set__(self, inst, value: Int3Tuple | None, /) -> None: ...
        def __delete__(self, inst, /) -> None: ...
        __objclass__: type
        key: str
        idx: str

    bg: _color_descriptor
    fg: _color_descriptor
    def insert(
        self, index: tp.SupportsIndex, value: bytes | SgrParamBuffer, /
    ) -> None: ...
    def extend(
        self, iterable: bytes | bytearray | abc.Iterable[abc.Buffer], /
    ) -> None: ...
    def is_color(self) -> bool: ...
    def is_reset(self) -> bool: ...
    def values(self) -> abc.Iterator[bytes | AnsiColorFormat]: ...
    def ansi_type(self) -> AnsiColorType | None: ...
    def shrink(self) -> None: ...
    def __add__(self, other: tp.Self, /) -> tp.Self: ...
    def __bool__(self) -> bool: ...
    def __bytes__(self) -> bytes: ...
    def __copy__(self) -> tp.Self: ...
    copy = __copy__
    def __deepcopy__(self, memo, /) -> tp.Self: ...

    @tp.overload
    def __delitem__(self, index: tp.SupportsIndex, /) -> None: ...
    @tp.overload
    def __delitem__(self, index: slice, /) -> None: ...

    def __eq__(self, other: tp.Any, /) -> bool: ...

    @tp.overload
    def __getitem__(self, index: tp.SupportsIndex, /) -> SgrParamBuffer: ...
    @tp.overload
    def __getitem__(self, index: slice, /) -> list[SgrParamBuffer]: ...

    def __init__[_T: (int, abc.Buffer, SgrParamBuffer)](
        self, iterable: abc.Iterable[_T] | None = None, /
    ) -> None: ...
    def __iter__(self) -> abc.Iterator[SgrParamBuffer]: ...
    def __len__(self) -> int: ...

    @tp.overload
    def __setitem__(
        self, index: tp.SupportsIndex, value: bytes | SgrParamBuffer, /
    ) -> None: ...
    @tp.overload
    def __setitem__(
        self, index: slice, value: abc.Iterable[bytes | SgrParamBuffer], /
    ) -> None: ...

    __hash__: tp.ClassVar[None]  # type: ignore[assignment]

    def clear_colors(self) -> None: ...

    @tp.overload
    def set_colors(
        self,
        mapping: SupportsKeysAndGetItem[ColorDictKeys, Int3Tuple | None],
        /,
        ansi_type: AnsiColorParam | None = None,
    ) -> None: ...
    @tp.overload
    def set_colors(
        self,
        iterable: abc.Iterable[tuple[ColorDictKeys, Int3Tuple | None]],
        /,
        ansi_type: AnsiColorParam | None = None,
    ) -> None: ...

    @property
    def rgb_dict(self) -> dict[ColorDictKeys, Int3Tuple]: ...
    @rgb_dict.setter
    def rgb_dict(
        self,
        value: (
            SupportsKeysAndGetItem[ColorDictKeys, Int3Tuple | None]
            | abc.Iterable[tuple[ColorDictKeys, Int3Tuple | None]]
        ),
    ) -> None: ...
    @rgb_dict.deleter
    def rgb_dict(self) -> None: ...

    _sgr_params: list[SgrParamBuffer]
    _bg_idx: int | None
    _fg_idx: int | None

class _IntFloatMixin:
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...

class _RecolorKwargs(tp.TypedDict, total=False):
    absolute: bool
    fg: tp.SupportsInt | None
    bg: tp.SupportsInt | None

class _ColorStrWeakVars(tp.TypedDict, total=False):
    sgr: SgrSequence
    base_str: str
    reset: bool

class ColorStr(str, _IntFloatMixin):
    def _weak_var_update(self, **kwargs: tp.Unpack[_ColorStrWeakVars]) -> tp.Self: ...
    def ansi_partition(self) -> TupleOf3[str]: ...
    def as_ansi_type(self, ansi_type: AnsiColorParam, /) -> tp.Self: ...

    @tp.overload
    def recolor(self, value: ColorStr, /, *, absolute: bool = ...) -> tp.Self: ...
    @tp.overload
    def recolor(self, **kwargs: tp.Unpack[_RecolorKwargs]) -> tp.Self: ...

    def strip_style(self) -> tp.Self: ...
    def add_reset(self) -> tp.Self: ...
    def remove_reset(self) -> tp.Self: ...
    def swap_reset(self) -> tp.Self: ...
    def add_sgr_param(self, x: int, /) -> tp.Self: ...
    def remove_sgr_param(self, x: int, /) -> tp.Self: ...
    def blink(self) -> tp.Self: ...
    def blink_stop(self) -> tp.Self: ...
    def bold(self) -> tp.Self: ...
    def faint(self) -> tp.Self: ...
    def crossed_out(self) -> tp.Self: ...
    def encircle(self) -> tp.Self: ...
    def italicize(self) -> tp.Self: ...
    def negative(self) -> tp.Self: ...
    def underline(self) -> tp.Self: ...
    def double_underline(self) -> tp.Self: ...
    def capitalize(self) -> tp.Self: ...
    def casefold(self) -> tp.Self: ...
    def center(self, width: tp.SupportsIndex, fillchar: str = " ", /) -> tp.Self: ...
    def count(
        self,
        sub: str,
        start: tp.SupportsIndex | None = ...,
        end: tp.SupportsIndex | None = ...,
        /,
    ) -> int: ...
    def endswith(
        self,
        suffix: str | tuple[str, ...],
        start: tp.SupportsIndex | None = ...,
        end: tp.SupportsIndex | None = ...,
        /,
    ) -> bool: ...
    def expandtabs(self, /, tabsize: tp.SupportsIndex = 8) -> tp.Self: ...
    def find(
        self,
        sub: str,
        start: tp.SupportsIndex | None = ...,
        end: tp.SupportsIndex | None = ...,
        /,
    ) -> int: ...
    def format(self, *args, **kwargs) -> tp.Self: ...
    def format_map(
        self, mapping: SupportsKeysAndGetItem[str, object], /
    ) -> tp.Self: ...
    def index(
        self,
        sub: str,
        start: tp.SupportsIndex | None = ...,
        end: tp.SupportsIndex | None = ...,
        /,
    ) -> int: ...
    def isalnum(self) -> bool: ...
    def isalpha(self) -> bool: ...
    def isascii(self) -> bool: ...
    def isdecimal(self) -> bool: ...
    def isdigit(self) -> bool: ...
    def isidentifier(self) -> bool: ...
    def islower(self) -> bool: ...
    def isnumeric(self) -> bool: ...
    def isprintable(self) -> bool: ...
    def isspace(self) -> bool: ...
    def istitle(self) -> bool: ...
    def isupper(self) -> bool: ...
    def join(self, iterable: abc.Iterable[str], /) -> tp.Self: ...
    def ljust(self, width: tp.SupportsIndex, fillchar: str = " ", /) -> tp.Self: ...
    def lower(self) -> tp.Self: ...
    def lstrip(self, chars: str | None = None, /) -> tp.Self: ...
    def partition(self, sep: str, /) -> TupleOf3[tp.Self]: ...
    def removeprefix(self, prefix: str, /) -> tp.Self: ...
    def removesuffix(self, suffix: str, /) -> tp.Self: ...
    def replace(
        self, old: str, new: str, /, count: tp.SupportsIndex = -1
    ) -> tp.Self: ...
    def rfind(
        self,
        sub: str,
        start: tp.SupportsIndex | None = ...,
        end: tp.SupportsIndex | None = ...,
        /,
    ) -> int: ...
    def rindex(
        self,
        sub: str,
        start: tp.SupportsIndex | None = ...,
        end: tp.SupportsIndex | None = ...,
        /,
    ) -> int: ...
    def rjust(self, width: tp.SupportsIndex, fillchar: str = " ", /) -> tp.Self: ...
    def rstrip(self, chars: str | None = None, /) -> tp.Self: ...
    def rpartition(self, sep: str, /) -> TupleOf3[tp.Self]: ...
    def rsplit(
        self, /, sep: str | None = None, maxsplit: tp.SupportsIndex = -1
    ) -> list[tp.Self]: ...
    def split(
        self, /, sep: str | None = None, maxsplit: tp.SupportsIndex = -1
    ) -> list[tp.Self]: ...
    def splitlines(self, /, keepends: bool = False) -> list[tp.Self]: ...
    def startswith(
        self,
        prefix: str | tuple[str, ...],
        start: tp.SupportsIndex | None = ...,
        end: tp.SupportsIndex | None = ...,
        /,
    ) -> bool: ...
    def strip(self, chars: str | None = None, /) -> tp.Self: ...
    def swapcase(self) -> tp.Self: ...
    def title(self) -> tp.Self: ...
    def translate(self, table, /) -> tp.Self: ...
    def upper(self) -> tp.Self: ...
    def zfill(self, width: tp.SupportsIndex, /) -> tp.Self: ...
    def __add__[_T: (ColorStr, str)](self, other: _T, /) -> tp.Self: ...
    def __contains__(self, key: str, /) -> bool: ...
    def __eq__(self, other, /) -> bool: ...
    def __format__(self, format_spec: str = "", /) -> tp.Self: ...
    def __ge__(self, other: str, /) -> bool: ...

    @tp.overload
    def __getitem__(self, key: tp.SupportsIndex, /) -> tp.Self: ...
    @tp.overload
    def __getitem__(self, key: slice, /) -> tp.Self: ...

    def __gt__(self, other: str, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __invert__(self) -> tp.Self: ...
    def __iter__(self) -> abc.Iterator[tp.Self]: ...
    def __le__(self, other: str, /) -> bool: ...
    def __len__(self) -> int: ...
    def __lt__(self, other: str, /) -> bool: ...
    def __matmul__(self, other: ColorStr, /) -> tp.Self: ...
    def __mod__(self, value, /) -> ColorStr: ...
    def __mul__(self, value: tp.SupportsIndex, /) -> tp.Self: ...
    __rmul__ = __mul__

    @tp.overload
    def __new__[_RGBVectorLike: RGBVectorLike](
        cls,
        obj: object = ...,
        /,
        fg: tp.SupportsInt | _RGBVectorLike | None = None,
        bg: tp.SupportsInt | _RGBVectorLike | None = None,
        *,
        ansi_type: AnsiColorParam = ...,
        reset: bool = ...,
    ) -> tp.Self: ...
    @tp.overload
    def __new__[_RGBVectorLike: RGBVectorLike](
        cls,
        obj: abc.Buffer,
        /,
        fg: tp.SupportsInt | _RGBVectorLike | None = None,
        bg: tp.SupportsInt | _RGBVectorLike | None = None,
        *,
        encoding: str = ...,
        errors: str = ...,
        ansi_type: AnsiColorParam = ...,
        reset: bool = ...,
    ) -> tp.Self: ...

    def __radd__(self, other: SgrSequence, /) -> tp.Self: ...
    def __xor__(self, other: ColorStr | int, /) -> tp.Self: ...
    @property
    def ansi(self) -> bytes: ...
    @property
    def ansi_type(self) -> AnsiColorType: ...
    @property
    def base_str(self) -> str: ...
    @property
    def bg(self) -> Color | None: ...
    @property
    def fg(self) -> Color | None: ...
    @property
    def reset(self) -> bool: ...
    @property
    def rgb_dict(self) -> dict[ColorDictKeys, Int3Tuple]: ...

    _sgr: SgrSequence
    _base_str: str
    _ansi_type: AnsiColorType
    _reset: L["\x1b[0m", ""]

ColorChainDType: tp.Final[np.dtype[np.void]]

class color_chain(abc.MutableSequence[tuple[SgrSequence, str]]):
    __slots__ = ("_ansi_type", "_items")
    __match_args__ = ("_items",)

    dtype: tp.ClassVar[np.dtype[np.void]]

    @classmethod
    def fromarray(
        cls, arr: np.ndarray, /, *, ansi_type: AnsiColorParam | None = None
    ) -> tp.Self: ...
    def insert(
        self,
        index: tp.SupportsIndex,
        value: tuple[SgrSequence, str | SgrSequence] | SgrSequence | str,
        /,
    ) -> None: ...
    def shrink(self) -> None: ...
    def splitlines(self) -> list[tp.Self]: ...

    @tp.overload
    def term_array[_Shape: tuple[int, int]](
        self, shape: _Shape, fillchar=""
    ) -> ShapedNDArray[_Shape, np.void]: ...
    @tp.overload
    def term_array(
        self, shape: tp.Any | None = None, fillchar=""
    ) -> ShapedNDArray[tuple[int, int], np.void]: ...

    def __add__(
        self, other: abc.Iterable[tuple[SgrSequence, str] | SgrSequence | str], /
    ) -> color_chain: ...
    def __array__(
        self, dtype: tp.Any | None = None, copy: bool | None = None
    ) -> ShapedNDArray[tuple[int], np.void]: ...
    def __bool__(self) -> bool: ...
    def __call__(self, obj: object = "", /) -> str: ...

    @tp.overload
    def __delitem__(self, index: tp.SupportsIndex, /) -> None: ...
    @tp.overload
    def __delitem__(self, index: slice, /) -> None: ...

    def __eq__(self, other, /) -> bool: ...

    @tp.overload
    def __getitem__(self, index: tp.SupportsIndex, /) -> tuple[SgrSequence, str]: ...
    @tp.overload
    def __getitem__(self, index: slice, /) -> list[tuple[SgrSequence, str]]: ...

    @tp.overload
    def __init__(
        self, iterable: None = None, /, *, ansi_type: AnsiColorParam | None = None
    ) -> None: ...
    @tp.overload
    def __init__(
        self, iterable: str, /, *, ansi_type: AnsiColorParam | None = None
    ) -> None: ...
    @tp.overload
    def __init__(
        self,
        iterable: abc.Iterable[tuple[SgrSequence, str] | SgrSequence | str],
        /,
        *,
        ansi_type: AnsiColorParam | None = None,
    ) -> None: ...

    def __len__(self) -> int: ...
    def __radd__(
        self, other: abc.Iterable[tuple[SgrSequence, str] | SgrSequence | str], /
    ) -> color_chain: ...

    @tp.overload
    def __setitem__(
        self, index: tp.SupportsIndex, value: tuple[SgrSequence, str], /
    ) -> None: ...
    @tp.overload
    def __setitem__(
        self, index: slice, value: abc.Iterable[tuple[SgrSequence, str]], /
    ) -> None: ...

    _ansi_type: AnsiColorType | None
    _items: list[tuple[SgrSequence, str]]
