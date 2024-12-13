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
    'hsl_gradient',
    'randcolor',
    'rgb2ansi_color_esc',
    'rgb_luma_transform',
    'SGR_RESET',
    'DEFAULT_ANSI',
]

from collections.abc import Buffer, Callable, Iterable, Iterator, Mapping, Sequence
from enum import IntEnum
from types import MappingProxyType
from typing import (
    Final,
    Literal,
    overload,
    Self,
    SupportsIndex,
    SupportsInt,
    TypedDict,
    TypeVar,
    Union,
    Unpack,
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

AnsiColorFormat = ansicolor4Bit | ansicolor8Bit | ansicolor24Bit
AnsiColorParam = AnsiColorAlias | AnsiColorType
AnsiColorType = type[AnsiColorFormat]

type _CSpecDict = Mapping[ColorDictKeys, _CSpecScalar]
type _CSpecKVPair = tuple[ColorDictKeys, _CSpecScalar]
type _CSpecScalar = Union[int, Color, RGBVectorLike]
type _CSpecTuplePair = Union[
    tuple[_CSpecScalar, _CSpecScalar], tuple[_CSpecKVPair, _CSpecKVPair]
]
type _CSpecType = Union[
    SgrSequence, str, bytes, _CSpecScalar, _CSpecTuplePair, _CSpecKVPair, _CSpecDict
]
type _RgbCoercible = Union[Color, SupportsInt]
type _RgbMapping[_KT: Union[ColorDictKeys, str], _VT: _RgbCoercible] = Mapping[_KT, _VT]

@overload
def get_ansi_type[_T: AnsiColorType](typ: _T) -> _T: ...
@overload
def get_ansi_type(typ: Ansi4BitAlias) -> type[ansicolor4Bit]: ...
@overload
def get_ansi_type(typ: Ansi8BitAlias) -> type[ansicolor8Bit]: ...
@overload
def get_ansi_type(typ: Ansi24BitAlias) -> type[ansicolor24Bit]: ...
def get_term_ansi_default() -> type[ansicolor8Bit | ansicolor4Bit]: ...
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
def rgb2ansi_color_esc(
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

class ansicolor24Bit(colorbytes):
    pass

class ansicolor4Bit(colorbytes):
    pass

class ansicolor8Bit(colorbytes):
    pass

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
        self,
        __value: ColorStr = None,
        absolute: bool = False,
        **kwargs: Unpack[_ColorDict],
    ) -> ColorStr: ...
    def replace(
        self, __old: str, __new: str, __count: SupportsIndex = -1
    ) -> ColorStr: ...
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
    _ansi_type_: type[AnsiColorFormat]
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
    def bg(self) -> Color | None: ...
    @property
    def fg(self) -> Color | None: ...
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
    def __init__[
        _T: (bytes, AnsiColorFormat, SgrParamWrapper)
    ](self, value: _T = b'') -> None: ...

    _value_: Union[bytes, AnsiColorFormat]

class SgrSequence:
    def append(self, __value: int) -> None: ...
    def find(self, value: ...) -> int: ...
    def get_color(self, __key: ColorDictKeys): ...
    def index(self, value: ...) -> int: ...
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
    def __getitem__(self, item): ...
    def __init__[
        _T: (Buffer, int), _AnsiType: type[AnsiColorFormat]
    ](
        self,
        __iter: Union[Sequence[_T], Sequence[SgrParamWrapper], SgrSequence] = None,
        *,
        ansi_type: _AnsiType = None,
    ) -> None: ...
    def __iter__(self) -> Iterator[SgrParamWrapper]: ...
    def __radd__[_T: (SgrSequence, str)](self, other: _T) -> _T: ...
    def __str__(self) -> str: ...

    __slots__ = '_bytes_', '_has_bright_colors_', '_rgb_dict_', '_sgr_params_'
    _bytes_: Union[bytes, None]
    _has_bright_colors_: bool
    _rgb_dict_: dict[ColorDictKeys, Int3Tuple]
    _sgr_params_: list[SgrParamWrapper]

    @property
    def bg(self) -> Int3Tuple | None: ...
    @property
    def fg(self) -> Int3Tuple | None: ...
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
    ](
        self, __value: tuple[_AnsiColorType, dict[ColorDictKeys, Union[Color, None]]]
    ) -> None: ...

class _ColorDict(TypedDict, total=False):
    bg: Union[Color, AnsiColorFormat, None]
    fg: Union[Color, AnsiColorFormat, None]

class _ColorStrKwargs(TypedDict, total=False):
    ansi_type: Union[AnsiColorAlias, type[AnsiColorFormat], None]
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
_ANSI_FORMAT_MAP: Final[dict[AnsiColorAlias | AnsiColorType, type[AnsiColorFormat]]]
_AnsiColor_co = TypeVar('_AnsiColor_co', bound=colorbytes, covariant=True)
_ColorSpec = TypeVar('_ColorSpec', bound=_CSpecType)
_RGBVectorLike = TypeVar('_RGBVectorLike', bound=RGBVectorLike)
_SGR_PARAM_VALUES: Final[frozenset[int]]
