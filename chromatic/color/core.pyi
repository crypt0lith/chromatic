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
    'randcolor',
    'rgb2ansi_escape',
]

import re
from _typeshed import ConvertibleToInt
from collections.abc import Buffer
from enum import IntEnum
from types import MappingProxyType
from typing import (
    ClassVar,
    Final,
    Iterable,
    Iterator,
    Literal as L,
    Mapping,
    MutableSequence,
    Optional,
    Self,
    Sequence,
    SupportsIndex,
    SupportsInt,
    TypeAlias,
    TypeGuard,
    TypeIs,
    TypeVar,
    TypedDict,
    Union,
    Unpack,
    overload,
)

from chromatic._typing import (
    Ansi24BitAlias,
    Ansi4BitAlias,
    Ansi8BitAlias,
    AnsiColorAlias,
    ColorDictKeys,
    Int3Tuple,
    RGBVectorLike,
    TupleOf3,
)

def is_vt_enabled() -> bool: ...
@overload
def get_ansi_type[_T: AnsiColorType](typ: _T) -> _T: ...
@overload
def get_ansi_type(typ: Ansi4BitAlias) -> type[ansicolor4Bit]: ...
@overload
def get_ansi_type(typ: Ansi8BitAlias) -> type[ansicolor8Bit]: ...
@overload
def get_ansi_type(typ: Ansi24BitAlias) -> type[ansicolor24Bit]: ...
def set_default_ansi(typ: AnsiColorAlias | AnsiColorType): ...
def randcolor() -> Color: ...
def rgb2ansi_escape(
    fmt: AnsiColorAlias | AnsiColorType, mode: ColorDictKeys, rgb: Int3Tuple
) -> bytes: ...
def sgr_pattern() -> re.Pattern[str]: ...

class ansicolor4Bit(colorbytes):
    alias: ClassVar[L['4b']]

class ansicolor8Bit(colorbytes):
    alias: ClassVar[L['8b']]

class ansicolor24Bit(colorbytes):
    alias: ClassVar[L['24b']]

class Color(int):
    @classmethod
    def from_rgb[_T: RGBVectorLike](cls, rgb: _T) -> Self: ...
    def __invert__(self) -> Color: ...
    @overload
    def __new__(cls, __x: ConvertibleToInt = ...) -> Self: ...
    @overload
    def __new__(cls, __x: str | bytes | bytearray, base: SupportsIndex) -> Self: ...
    @property
    def rgb(self) -> Int3Tuple: ...

class colorbytes(bytes):
    @classmethod
    @overload
    def from_rgb[_T, _VT: (
        SupportsInt,
        RGBVectorLike,
    )](
        cls: type[_T],
        __rgb: (
            tuple[ColorDictKeys, _VT] | Mapping[L['fg'], _VT] | Mapping[L['bg'], _VT]
        ),
    ) -> _T: ...
    @classmethod
    @overload
    def from_rgb[_T, _VT: (
        SupportsInt,
        RGBVectorLike,
    )](cls: type[_T], __rgb: tuple[str, _VT] | Mapping[str, _VT]) -> _T: ...
    @overload
    def __new__[_T: (
        ansicolor4Bit,
        ansicolor8Bit,
        ansicolor24Bit,
    )](cls: AnsiColorType, __ansi: bytes | AnsiColorFormat) -> Self: ...
    @overload
    def __new__(cls, __ansi: bytes | _AnsiColor_co) -> AnsiColorFormat: ...
    def to_param_buffer(self) -> SgrParamBuffer[Self]: ...

    _rgb_dict: dict[L['fg'], Int3Tuple] | dict[L['bg'], Int3Tuple]
    @property
    def rgb_dict(
        self,
    ) -> (
        MappingProxyType[L['fg'], Int3Tuple] | MappingProxyType[L['bg'], Int3Tuple]
    ): ...

class ColorStr(str):
    def _weak_var_update(self, **kwargs: Unpack[_ColorStrWeakVars]) -> Self: ...
    def ansi_partition(self) -> TupleOf3[str]: ...
    def as_ansi_type(self, __ansi_type: AnsiColorParam, /) -> Self: ...
    def bold(self) -> Self: ...
    def italicize(self) -> Self: ...
    def negative(self) -> Self: ...
    @overload
    def recolor(self, __value: ColorStr, /, *, absolute: bool = ...) -> Self: ...
    @overload
    def recolor(self, **kwargs: Unpack[_RecolorKwargs]) -> Self: ...
    def strikethrough(self) -> Self: ...
    def underline(self, double=False) -> Self: ...
    def update_sgr[_T: (int, SgrParameter)](self, *args: _T) -> Self: ...
    def capitalize(self) -> Self: ...
    def casefold(self) -> Self: ...
    def center(self, width, fillchar=' ') -> Self: ...
    def count(
        self,
        x: str,
        __start: SupportsIndex | None = ...,
        __end: SupportsIndex | None = ...,
        /,
    ): ...
    def endswith(
        self,
        __prefix: str | tuple[str, ...],
        __start: SupportsIndex | None = ...,
        __end: SupportsIndex | None = ...,
        /,
    ) -> bool: ...
    def expandtabs(self, /, tabsize=8) -> Self: ...
    def find(
        self,
        __sub: str,
        __start: SupportsIndex | None = ...,
        __end: SupportsIndex | None = ...,
        /,
    ) -> int: ...
    def format(self, *args, **kwargs) -> Self: ...
    def format_map(self, __mapping: Mapping[str, object], /) -> Self: ...
    def index(
        self,
        __sub: str,
        __start: SupportsIndex | None = ...,
        __end: SupportsIndex | None = ...,
        /,
    ) -> int: ...
    def join(self, __iterable: Iterable[str]) -> Self: ...
    def ljust(self, __width: SupportsIndex, __fillchar=' ') -> Self: ...
    def lower(self) -> Self: ...
    def lstrip(self, __chars: str | None = None, /) -> Self: ...
    def partition(self, __sep: str, /) -> TupleOf3[Self]: ...
    def removeprefix(self, __prefix: str, /) -> Self: ...
    def removesuffix(self, __suffix: str, /) -> Self: ...
    def replace(self, __old: str, __new: str, __count: SupportsIndex = -1) -> Self: ...
    def rfind(
        self,
        __sub: str,
        __start: SupportsIndex | None = ...,
        __end: SupportsIndex | None = ...,
        /,
    ) -> int: ...
    def rindex(
        self,
        __sub: str,
        __start: SupportsIndex | None = ...,
        __end: SupportsIndex | None = ...,
        /,
    ) -> int: ...
    def rjust(self, __width: SupportsIndex, __fillchar=' ') -> Self: ...
    def rpartition(self, __sep: str) -> TupleOf3[Self]: ...
    def rsplit(self, sep: str = None, maxsplit=-1) -> list[Self]: ...
    def rstrip(self, __chars: str | None = None) -> Self: ...
    def split(self, sep: str = None, maxsplit=-1) -> list[Self]: ...
    def splitlines(self, /, keepends=False) -> list[Self]: ...
    def startswith(
        self,
        __prefix: str | tuple[str, ...],
        __start: SupportsIndex | None = ...,
        __end: SupportsIndex | None = ...,
        /,
    ) -> bool: ...
    def strip(self, __chars: str | None = None) -> Self: ...
    def swapcase(self) -> Self: ...
    def title(self) -> Self: ...
    def upper(self) -> Self: ...
    def zfill(self, __width: SupportsIndex) -> Self: ...
    def __add__[_T: (Self, str, SgrParameter)](self, other: _T) -> Self: ...
    def __eq__(self, other) -> bool: ...
    def __format__(self, format_spec: str = '') -> str: ...
    @overload
    def __getitem__(self, __key: SupportsIndex) -> Self: ...
    @overload
    def __getitem__(self, __key: slice) -> Self: ...
    def __hash__(self) -> int: ...
    def __invert__(self) -> Self: ...
    def __iter__(self) -> Iterator[Self]: ...
    def __len__(self) -> int: ...
    def __matmul__(self, other: ColorStr) -> Self: ...
    def __mod__(self, __value) -> ColorStr: ...
    def __mul__(self, __value: SupportsIndex) -> Self: ...
    @overload
    def __new__[_RGBVectorLike: RGBVectorLike](
        cls,
        obj: object = ...,
        fg: SupportsInt | _RGBVectorLike = None,
        bg: SupportsInt | _RGBVectorLike = None,
        *,
        ansi_type: AnsiColorParam = ...,
        reset: bool = ...,
    ) -> Self: ...
    @overload
    def __new__[_RGBVectorLike: RGBVectorLike](
        cls,
        obj: Buffer,
        fg: SupportsInt | _RGBVectorLike = None,
        bg: SupportsInt | _RGBVectorLike = None,
        *,
        encoding: str = ...,
        errors: str = ...,
        ansi_type: AnsiColorParam = ...,
        reset: bool = ...,
    ) -> Self: ...
    def __xor__(self, other: ColorStr | Color) -> Self: ...
    @property
    def ansi(self) -> bytes: ...
    @property
    def ansi_format(self) -> AnsiColorType: ...
    @property
    def base_str(self) -> str: ...
    @property
    def bg(self) -> Color | None: ...
    @property
    def fg(self) -> Color | None: ...
    @property
    def reset(self) -> bool: ...
    @property
    def rgb_dict(self) -> MappingProxyType[ColorDictKeys, Int3Tuple]: ...
    _sgr: SgrSequence
    _reset: L["\x1b[0m", ""]

class _ColorChainKwargs(TypedDict, total=False):
    ansi_type: AnsiColorAlias | type[AnsiColorFormat]
    fg: Color | int | Int3Tuple
    bg: Color | int | Int3Tuple

class color_chain:
    @staticmethod
    def _is_mask_seq(obj: object) -> TypeGuard[Sequence[tuple[SgrSequence, str]]]: ...
    @classmethod
    def _from_masks_unchecked(
        cls, masks: Iterable[tuple[SgrSequence, str]], ansi_type: type[AnsiColorFormat]
    ) -> Self: ...
    @classmethod
    def from_masks(
        cls,
        masks: Sequence[tuple[SgrSequence, str]],
        ansi_type: type[AnsiColorFormat] = None,
    ) -> Self: ...
    def shrink(self) -> Self: ...
    def __add__[_T: (
        color_chain,
        SgrSequence,
        ColorStr,
        str,
    )](self, other: _T) -> color_chain: ...
    def __bool__(self) -> bool: ...
    def __call__(self, __obj=None) -> str: ...
    @overload
    def __getitem__(self, __index: SupportsIndex) -> tuple[SgrSequence, str]: ...
    @overload
    def __getitem__(self, __index: slice) -> list[tuple[SgrSequence, str]]: ...
    def __init__[_T: color_chain | ColorStr | str | SgrSequence](
        self, __sgr: Iterable[_T] = ..., *, ansi_type: AnsiColorParam = ...
    ) -> None: ...
    def __len__(self) -> int: ...
    def __radd__[_T: (
        color_chain,
        ColorStr,
        str,
        SgrSequence,
    )](self, other: _T) -> color_chain: ...

    _ansi_type: type[AnsiColorFormat] | None
    _masks: list[tuple[SgrSequence, str]]

    @property
    def masks(self) -> list[tuple[SgrSequence, str]]: ...

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

class SgrParamBuffer[
    _T: (bytes, ansicolor4Bit, ansicolor8Bit, ansicolor24Bit) = bytes | AnsiColorFormat
]:
    def is_color(self) -> bool: ...
    def is_reset(self) -> bool: ...
    def __buffer__(self, __flags: int) -> memoryview: ...
    def __bytes__(self) -> bytes: ...
    def __eq__[_T](
        self: SgrParamBuffer[_T], other
    ) -> TypeIs[SgrParamBuffer[_T] | _T]: ...
    def __hash__(self) -> int: ...
    @overload
    def __new__(cls, __value: _T = ...) -> Self: ...
    @overload
    def __new__(cls, __value: SgrParamBuffer[_T]) -> Self: ...
    @property
    def value(self) -> _T: ...
    __slots__ = ('_bytes', '_is_color', '_is_reset', '_value')
    _value: _T

class SgrSequence(MutableSequence[SgrParamBuffer]):
    def _update_colors(self): ...
    def is_color(self) -> bool: ...
    def is_reset(self) -> bool: ...
    def has_bright_colors(self) -> bool: ...
    def values(self) -> Iterator[bytes | AnsiColorFormat]: ...
    def copy(self) -> Self: ...
    def insert(self, __index: SupportsIndex, __value: bytes | SgrParamBuffer): ...
    def extend(self, __iter: bytes | bytearray | Iterable[Buffer]): ...
    def ansi_type(self) -> AnsiColorType | None: ...
    @overload
    def __add__(self, other: Self) -> Self: ...
    @overload
    def __add__(self, other: str) -> ColorStr: ...
    @overload
    def __add__(self, other: ColorStr) -> color_chain: ...
    @overload
    def __radd__(self, other: ColorStr) -> color_chain: ...
    def __bool__(self) -> bool: ...
    def __bytes__(self) -> bytes: ...
    def __len__(self) -> int: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self) -> Self: ...
    @overload
    def __getitem__(self, __index: slice) -> list[SgrParamBuffer]: ...
    @overload
    def __getitem__(self, __index: SupportsIndex) -> SgrParamBuffer: ...
    @overload
    def __setitem__(
        self, __index: slice, __value: Iterable[bytes | SgrParamBuffer]
    ): ...
    @overload
    def __setitem__(self, __index: SupportsIndex, __value: bytes | SgrParamBuffer): ...
    @overload
    def __delitem__(self, __index: slice): ...
    @overload
    def __delitem__(self, __index: SupportsIndex): ...
    def __init__[_T: (
        int,
        Buffer,
        SgrParamBuffer,
    )](self, __iter: Iterable[_T] = ...): ...
    def __iter__(self) -> Iterator[SgrParamBuffer]: ...

    __slots__ = '_rgb_dict', '_sgr_params'
    _rgb_dict: dict[ColorDictKeys, Int3Tuple]
    _sgr_params: list[SgrParamBuffer]

    @property
    def bg(self) -> Optional[Int3Tuple]: ...
    @property
    def fg(self) -> Optional[Int3Tuple]: ...
    @property
    def rgb_dict(self) -> MappingProxyType[ColorDictKeys, Int3Tuple]: ...
    @rgb_dict.deleter  # type: ignore[no-redef]
    def rgb_dict(self): ...
    @overload
    @rgb_dict.setter  # type: ignore[no-redef]
    def rgb_dict(self, __value: dict[ColorDictKeys, Int3Tuple | None]): ...
    @overload
    @rgb_dict.setter  # type: ignore[no-redef]
    def rgb_dict(
        self, __value: tuple[dict[ColorDictKeys, Int3Tuple | None], AnsiColorType]
    ): ...

class _RecolorKwargs(TypedDict, total=False):
    absolute: bool
    fg: SupportsInt | None
    bg: SupportsInt | None

class _ColorStrWeakVars(TypedDict, total=False):
    sgr: SgrSequence
    base_str: str
    reset: bool

CSI: Final[bytes]
SGR_RESET: Final[bytes]
SGR_RESET_S: Final[str]
DEFAULT_ANSI: Final[type[ansicolor8Bit | ansicolor4Bit]]

_ANSI16C_BRIGHT: Final[frozenset[int]]
_ANSI16C_I2KV: Final[dict[int, tuple[ColorDictKeys, Int3Tuple]]]
_ANSI16C_KV2I: Final[dict[tuple[ColorDictKeys, Int3Tuple], int]]
_ANSI16C_STD: Final[frozenset[int]]
_ANSI256_B2KEY: Final[dict[bytes, str]]
_ANSI256_KEY2I: Final[dict[str, int]]
_ANSI_COLOR_TYPES: Final[frozenset[AnsiColorType]]
_ANSI_FORMAT_MAP: Final[dict[AnsiColorAlias | AnsiColorType, AnsiColorType]]
_SGR_PARAM_VALUES: Final[frozenset[int]]

AnsiColorFormat: TypeAlias = ansicolor4Bit | ansicolor8Bit | ansicolor24Bit
AnsiColorType: TypeAlias = type[AnsiColorFormat]
AnsiColorParam: TypeAlias = Union[AnsiColorAlias, AnsiColorType]

_AnsiColor_co = TypeVar('_AnsiColor_co', colorbytes, covariant=True)
