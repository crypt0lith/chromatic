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
    'hsl_gradient',
    'randcolor',
    'rgb_luma_transform',
    'rgb2ansi_escape',
]

from collections.abc import Buffer, Callable, Iterable, Iterator, Mapping, Sequence
import sys
from enum import IntEnum
from types import MappingProxyType
from typing import (
    Final,
    Literal,
    Optional,
    Self,
    SupportsIndex,
    SupportsInt,
    TypeAlias,
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
    Float3Tuple,
    Int3Tuple,
    RGBVectorLike,
    TupleOf3,
)

@overload
def get_ansi_type[_T: AnsiColorType](typ: _T) -> _T: ...
@overload
def get_ansi_type(typ: Ansi4BitAlias) -> type[ansicolor4Bit]: ...
@overload
def get_ansi_type(typ: Ansi8BitAlias) -> type[ansicolor8Bit]: ...
@overload
def get_ansi_type(typ: Ansi24BitAlias) -> type[ansicolor24Bit]: ...
def hsl_gradient(
    start: Int3Tuple | Float3Tuple,
    stop: Int3Tuple | Float3Tuple,
    step: SupportsIndex,
    num: SupportsIndex = None,
    ncycles: int | float = float('inf'),
    replace_idx: tuple[SupportsIndex | Iterable[SupportsIndex], Iterator[Color]] = None,
    dtype: type[Color] | Callable[[Int3Tuple], int] = Color,
): ...
def randcolor() -> Color: ...
def rgb2ansi_escape(
    ret_format: AnsiColorAlias | AnsiColorType, mode: ColorDictKeys, rgb: Int3Tuple
) -> bytes: ...
def rgb_luma_transform(
    rgb: Int3Tuple,
    start: SupportsIndex = None,
    num: SupportsIndex = 50,
    step: SupportsIndex = 1,
    cycle: bool | Literal['wave'] = False,
    ncycles: int | float = float('inf'),
    gradient: Int3Tuple = None,
    dtype: type[Color] = None,
) -> Iterator[Int3Tuple | int | Color]: ...

class ansicolor4Bit(colorbytes): ...
class ansicolor8Bit(colorbytes): ...
class ansicolor24Bit(colorbytes): ...

class Color(int):
    @classmethod
    def from_rgb(cls, rgb: _RGBVectorLike) -> Self: ...
    def __invert__(self) -> Color: ...
    def __new__(cls, __x: _RgbCoercible) -> Color: ...
    @property
    def rgb(self) -> Int3Tuple: ...

class colorbytes(bytes):
    @classmethod
    def from_rgb(cls, __rgb: Union[_RgbMapping, _AnsiColor_co]) -> Self: ...
    def __new__(cls, __ansi: Union[bytes, _AnsiColor_co]) -> AnsiColorFormat: ...

    _rgb_dict_: dict[ColorDictKeys, Int3Tuple]

    @property
    def rgb_dict(self) -> MappingProxyType[ColorDictKeys, Int3Tuple]: ...

class ColorStr(str):
    def _weak_var_update(self, **kwargs: Unpack[_ColorStrWeakVars]) -> ColorStr: ...
    def ansi_partition(self) -> TupleOf3[str]: ...
    def as_ansi_type(self, __ansi_type: AnsiColorParam) -> ColorStr: ...
    def format(self, *args, **kwargs) -> ColorStr: ...
    def recolor(
        self, __value: ColorStr = None, absolute: bool = False, **kwargs: Unpack[_ColorDict]
    ) -> ColorStr: ...
    def replace(self, __old: str, __new: str, __count: SupportsIndex = -1) -> ColorStr: ...
    def split(self, sep=None, maxsplit=-1) -> list[ColorStr]: ...
    def update_sgr(self, *p: *tuple[Union[int, SgrParameter], ...]) -> ColorStr: ...
    def __add__[_T: (str, ColorStr, SgrParameter)](self, other: _T) -> ColorStr: ...
    def __and__(self, other: ColorStr) -> Union[ColorStr, Self]: ...
    def __eq__(self, other) -> bool: ...
    def __format__(self, format_spec: str = '') -> str: ...
    def __getitem__(self, __key: Union[SupportsIndex, slice]) -> str: ...
    def __hash__(self) -> int: ...
    def __init__(
        self,
        obj: object = None,
        color_spec: Union[_ColorSpec, ColorStr] = None,
        **kwargs: Unpack[_ColorStrKwargs],
    ) -> None: ...
    def __iter__(self) -> Iterator[ColorStr]: ...
    def __len__(self) -> int: ...
    def __matmul__(self, other: ColorStr) -> ColorStr: ...
    def __mod__(self, __value) -> ColorStr: ...
    def __mul__(self, __value: SupportsIndex) -> ColorStr: ...
    def __invert__(self) -> ColorStr: ...
    def __new__(
        cls,
        obj: object = None,
        color_spec: Union[_ColorSpec, ColorStr] = None,
        **kwargs: Unpack[_ColorStrKwargs],
    ) -> ColorStr: ...
    def __sub__(self, other: Union[Color, ColorStr]) -> ColorStr: ...

    _ansi_: bytes
    _ansi_type_: AnsiColorType
    _base_str_: str
    _color_dict_: MappingProxyType[ColorDictKeys, Color]
    _no_reset_: bool
    _sgr_: SgrSequence
    _sgr_params_: list[SgrParamWrapper]

    @property
    def ansi(self) -> bytes: ...
    @property
    def ansi_format(self) -> AnsiColorType: ...
    @property
    def base_str(self) -> str: ...
    @property
    def bg(self) -> Optional[Color]: ...
    @property
    def fg(self) -> Optional[Color]: ...
    @property
    def no_reset(self) -> bool: ...
    @property
    def rgb_dict(self) -> dict[ColorDictKeys, Int3Tuple]: ...

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

class SgrParamWrapper:
    def is_color(self) -> bool: ...
    def is_reset(self) -> bool: ...
    def is_same_kind(self, other: ...) -> bool: ...
    def __bytes__(self) -> bytes: ...
    def __eq__(self, other: ...) -> bool: ...
    def __hash__(self) -> int: ...
    def __init__[_T: (bytes, AnsiColorFormat, SgrParamWrapper)](self, value: _T = b'') -> None: ...

    _value_: Union[bytes, AnsiColorFormat]

class SgrSequence(Sequence[SgrParamWrapper]):
    def append(self, __value: int) -> None: ...
    def find(self, value: ...) -> int: ...
    def get_color(self, __key: ColorDictKeys): ...
    def index(self, value, start: SupportsIndex = 0, stop: SupportsIndex = sys.maxsize) -> int: ...
    def is_color(self) -> bool: ...
    def is_reset(self) -> bool: ...
    def pop(self, __index: SupportsIndex = -1) -> SgrParamWrapper: ...
    def values(self) -> list[Union[bytes, AnsiColorFormat]]: ...
    def __add__[_T: (SgrSequence, str)](self, other: _T) -> _T: ...
    def __bool__(self) -> bool: ...
    def __bytes__(self) -> bytes: ...
    def __contains__(self, item: ...) -> bool: ...
    def __copy__(self) -> SgrSequence: ...
    def __deepcopy__(self) -> SgrSequence: ...
    def __eq__(self, other: ...) -> bool: ...
    @overload
    def __getitem__(self, item: SupportsIndex) -> SgrParamWrapper: ...
    @overload
    def __getitem__(self, item: slice) -> list[SgrParamWrapper]: ...
    def __init__[
        _T: (SgrParamWrapper, Buffer, int), _AnsiType: type[AnsiColorFormat]
    ](self, __iter: Iterable[_T] = None, *, ansi_type: _AnsiType = None) -> None: ...
    def __iter__(self) -> Iterator[SgrParamWrapper]: ...
    def __len__(self) -> int: ...
    def __radd__[_T: (SgrSequence, str)](self, other: _T) -> _T: ...
    def __str__(self) -> str: ...

    __slots__ = '_bytes_', '_has_bright_colors_', '_rgb_dict_', '_sgr_params_'

    _bytes_: Optional[bytes]
    _has_bright_colors_: bool
    _rgb_dict_: dict[ColorDictKeys, Int3Tuple]
    _sgr_params_: list[SgrParamWrapper]

    @property
    def bg(self) -> Optional[Int3Tuple]: ...
    @property
    def fg(self) -> Optional[Int3Tuple]: ...
    @property
    def has_bright_colors(self) -> bool: ...
    @property
    def rgb_dict(self) -> MappingProxyType[ColorDictKeys, Int3Tuple]: ...

    # noinspection PyUnresolvedReferences
    @rgb_dict.deleter
    def rgb_dict(self) -> None: ...

    # noinspection PyUnresolvedReferences
    @rgb_dict.setter
    def rgb_dict[
        _AnsiColorType: type[AnsiColorFormat]
    ](self, __value: tuple[_AnsiColorType, dict[ColorDictKeys, Optional[Color]]]) -> None: ...

class _ColorDict(TypedDict, total=False):
    bg: Optional[Color | AnsiColorFormat]
    fg: Optional[Color | AnsiColorFormat]

class _ColorStrKwargs(TypedDict, total=False):
    ansi_type: Optional[AnsiColorAlias | type[AnsiColorFormat]]
    no_reset: bool

class _ColorStrWeakVars(TypedDict, total=False):
    _base_str_: str
    _no_reset_: bool
    _sgr_: SgrSequence

CSI: Final[bytes] = b'['
SGR_RESET: Final[str] = '[0m'
DEFAULT_ANSI: Final[type[ansicolor8Bit | ansicolor4Bit]]

_ANSI16C_BRIGHT: Final[frozenset[int]]
_ANSI16C_I2KV: Final[dict[int, tuple[ColorDictKeys, Int3Tuple]]]
_ANSI16C_KV2I: Final[dict[tuple[Literal['fg', 'bg'], tuple[int, int, int]], int]]
_ANSI16C_STD: Final[frozenset[int]]
_ANSI256_B2KEY: Final[dict[bytes, str]]
_ANSI256_KEY2I: Final[dict[str, int]]
_ANSI_COLOR_TYPES: Final[frozenset[AnsiColorType]]
_ANSI_FORMAT_MAP: Final[dict[AnsiColorAlias | AnsiColorType, AnsiColorType]]
_SGR_PARAM_VALUES: Final[frozenset[int]]

AnsiColorFormat: TypeAlias = ansicolor4Bit | ansicolor8Bit | ansicolor24Bit
AnsiColorType: TypeAlias = type[AnsiColorFormat]
AnsiColorParam: TypeAlias = Union[AnsiColorAlias, AnsiColorType]

_CSpecDict: TypeAlias = Mapping[ColorDictKeys, _CSpecScalar]
_CSpecKVPair: TypeAlias = tuple[ColorDictKeys, _CSpecScalar]
_CSpecScalar: TypeAlias = int | Color | RGBVectorLike
_CSpecTuplePair: TypeAlias = tuple[_CSpecScalar, _CSpecScalar] | tuple[_CSpecKVPair, _CSpecKVPair]
_CSpecType: TypeAlias = SgrSequence | _CSpecScalar | _CSpecTuplePair | _CSpecKVPair | _CSpecDict
_ColorSpec = TypeVar('_ColorSpec', _CSpecType, str, bytes)
_AnsiColor_co = TypeVar('_AnsiColor_co', bound=colorbytes, covariant=True)
_RGBVectorLike = TypeVar('_RGBVectorLike', bound=RGBVectorLike)
_RgbCoercible: TypeAlias = Color | SupportsInt
type _RgbMapping[_KT: Union[ColorDictKeys, str], _VT: _RgbCoercible] = Mapping[_KT, _VT]
