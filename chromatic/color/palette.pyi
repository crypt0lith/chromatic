__all__ = [
    'Back',
    'ColorNamespace',
    'Fore',
    'Style',
    'color_str_wrapper',
    'rgb_dispatch',
    'named_color',
]

import inspect
from collections.abc import Sequence
from types import MappingProxyType
from typing import Callable, Iterator, Literal, TypeVar, TypedDict, Union, Unpack, overload

from chromatic._typing import AnsiColorAlias, Int3Tuple
from chromatic.color.core import AnsiColorFormat, Color, ColorStr, SgrParameter, SgrSequence

type _ColorLike = Union[Color, Int3Tuple, int]

def display_ansi256_color_range() -> list[list[ColorStr]]: ...
def named_color_idents() -> list[ColorStr]: ...

class AnsiBack(ColorNamespace[color_str_wrapper]):
    RESET: color_str_wrapper

    def __call__(self, bg: _ColorLike) -> color_str_wrapper: ...

class AnsiFore(ColorNamespace[color_str_wrapper]):
    RESET: color_str_wrapper

    def __call__(self, fg: _ColorLike) -> color_str_wrapper: ...

class AnsiStyle[StyleStr: color_str_wrapper](DynamicNamespace[StyleStr]):
    RESET: StyleStr
    BOLD: StyleStr
    FAINT: StyleStr
    ITALICS: StyleStr
    SINGLE_UNDERLINE: StyleStr
    SLOW_BLINK: StyleStr
    RAPID_BLINK: StyleStr
    NEGATIVE: StyleStr
    CONCEALED_CHARS: StyleStr
    CROSSED_OUT: StyleStr
    PRIMARY: StyleStr
    FIRST_ALT: StyleStr
    SECOND_ALT: StyleStr
    THIRD_ALT: StyleStr
    FOURTH_ALT: StyleStr
    FIFTH_ALT: StyleStr
    SIXTH_ALT: StyleStr
    SEVENTH_ALT: StyleStr
    EIGHTH_ALT: StyleStr
    NINTH_ALT: StyleStr
    GOTHIC: StyleStr
    DOUBLE_UNDERLINE: StyleStr
    RESET_BOLD_AND_FAINT: StyleStr
    RESET_ITALIC_AND_GOTHIC: StyleStr
    RESET_UNDERLINES: StyleStr
    RESET_BLINKING: StyleStr
    POSITIVE: StyleStr
    REVEALED_CHARS: StyleStr
    RESET_CROSSED_OUT: StyleStr
    BLACK_FG: StyleStr
    RED_FG: StyleStr
    GREEN_FG: StyleStr
    YELLOW_FG: StyleStr
    BLUE_FG: StyleStr
    MAGENTA_FG: StyleStr
    CYAN_FG: StyleStr
    WHITE_FG: StyleStr
    ANSI_256_SET_FG: StyleStr
    DEFAULT_FG_COLOR: StyleStr
    BLACK_BG: StyleStr
    RED_BG: StyleStr
    GREEN_BG: StyleStr
    YELLOW_BG: StyleStr
    BLUE_BG: StyleStr
    MAGENTA_BG: StyleStr
    CYAN_BG: StyleStr
    WHITE_BG: StyleStr
    ANSI_256_SET_BG: StyleStr
    DEFAULT_BG_COLOR: StyleStr
    FRAMED: StyleStr
    ENCIRCLED: StyleStr
    OVERLINED: StyleStr
    NOT_FRAMED_OR_CIRCLED: StyleStr
    IDEOGRAM_UNDER_OR_RIGHT: StyleStr
    IDEOGRAM_2UNDER_OR_2RIGHT: StyleStr
    IDEOGRAM_OVER_OR_LEFT: StyleStr
    IDEOGRAM_2OVER_OR_2LEFT: StyleStr
    CANCEL: StyleStr
    BLACK_BRIGHT_FG: StyleStr
    RED_BRIGHT_FG: StyleStr
    GREEN_BRIGHT_FG: StyleStr
    YELLOW_BRIGHT_FG: StyleStr
    BLUE_BRIGHT_FG: StyleStr
    MAGENTA_BRIGHT_FG: StyleStr
    CYAN_BRIGHT_FG: StyleStr
    WHITE_BRIGHT_FG: StyleStr
    BLACK_BRIGHT_BG: StyleStr
    RED_BRIGHT_BG: StyleStr
    GREEN_BRIGHT_BG: StyleStr
    YELLOW_BRIGHT_BG: StyleStr
    BLUE_BRIGHT_BG: StyleStr
    MAGENTA_BRIGHT_BG: StyleStr
    CYAN_BRIGHT_BG: StyleStr
    WHITE_BRIGHT_BG: StyleStr

class color_str_wrapper:
    def __add__(self, other) -> ColorStr: ...
    def __call__(self, __obj=None) -> color_str_wrapper | ColorStr: ...
    def __init__(self, **kwargs: Unpack[_ColorStrWrapperKwargs]) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

    _ansi_type_: type[AnsiColorFormat]
    _sgr_: SgrSequence

class DynamicNamespace[_VT](metaclass=DynamicNSMeta[_VT]):
    def as_dict(self) -> dict[str, _VT]: ...
    def __init__[_KT](self, **kwargs: dict[_KT, _VT]) -> None: ...
    def __init_subclass__(cls, **kwargs) -> DynamicNamespace[_VT]: ...
    def __iter__(self) -> Iterator[_VT]: ...
    def __new__(cls, *args, **kwargs) -> DynamicNamespace[_VT]: ...
    def __setattr__(self, name, value) -> None: ...

    __members__: list[_VT]

class DynamicNSMeta[_VT](type):
    def __new__(
        mcls, clsname: str, bases: tuple[type, ...], mapping: dict[str, ...], **kwargs
    ) -> DynamicNSMeta[_VT]: ...

class ColorNamespace[NamedColor: Color](DynamicNamespace[NamedColor]):
    BLACK: NamedColor
    DIM_GREY: NamedColor
    GREY: NamedColor
    DARK_GREY: NamedColor
    SILVER: NamedColor
    LIGHT_GREY: NamedColor
    WHITE_SMOKE: NamedColor
    WHITE: NamedColor
    ROSY_BROWN: NamedColor
    INDIAN_RED: NamedColor
    BROWN: NamedColor
    FIREBRICK: NamedColor
    LIGHT_CORAL: NamedColor
    MAROON: NamedColor
    DARK_RED: NamedColor
    RED: NamedColor
    SNOW: NamedColor
    MISTY_ROSE: NamedColor
    SALMON: NamedColor
    TOMATO: NamedColor
    BURNT_SIENNA: NamedColor
    DARK_SALMON: NamedColor
    CORAL: NamedColor
    ORANGE_RED: NamedColor
    LIGHT_SALMON: NamedColor
    SIENNA: NamedColor
    SEASHELL: NamedColor
    CHOCOLATE: NamedColor
    SADDLE_BROWN: NamedColor
    SANDY_BROWN: NamedColor
    PEACH_PUFF: NamedColor
    PERU: NamedColor
    LINEN: NamedColor
    BISQUE: NamedColor
    DARK_ORANGE: NamedColor
    BURLY_WOOD: NamedColor
    ANTIQUE_WHITE: NamedColor
    TAN: NamedColor
    NAVAJO_WHITE: NamedColor
    BLANCHED_ALMOND: NamedColor
    PAPAYA_WHIP: NamedColor
    MOCCASIN: NamedColor
    ORANGE: NamedColor
    WHEAT: NamedColor
    OLD_LACE: NamedColor
    FLORAL_WHITE: NamedColor
    DARK_GOLDENROD: NamedColor
    GOLDENROD: NamedColor
    CORNSILK: NamedColor
    GOLD: NamedColor
    LEMON_CHIFFON: NamedColor
    KHAKI: NamedColor
    PALE_GOLDENROD: NamedColor
    DARK_KHAKI: NamedColor
    BEIGE: NamedColor
    LIGHT_GOLDENROD_YELLOW: NamedColor
    OLIVE: NamedColor
    YELLOW: NamedColor
    LIGHT_YELLOW: NamedColor
    IVORY: NamedColor
    OLIVE_DRAB: NamedColor
    YELLOW_GREEN: NamedColor
    DARK_OLIVE_GREEN: NamedColor
    GREEN_YELLOW: NamedColor
    CHARTREUSE: NamedColor
    LAWN_GREEN: NamedColor
    DARK_SEA_GREEN: NamedColor
    FOREST_GREEN: NamedColor
    LIME_GREEN: NamedColor
    LIGHT_GREEN: NamedColor
    PALE_GREEN: NamedColor
    DARK_GREEN: NamedColor
    GREEN: NamedColor
    LIME: NamedColor
    HONEYDEW: NamedColor
    SEA_GREEN: NamedColor
    MEDIUM_SEA_GREEN: NamedColor
    SPRING_GREEN: NamedColor
    MINT_CREAM: NamedColor
    MEDIUM_SPRING_GREEN: NamedColor
    MEDIUM_AQUAMARINE: NamedColor
    AQUAMARINE: NamedColor
    TURQUOISE: NamedColor
    LIGHT_SEA_GREEN: NamedColor
    MEDIUM_TURQUOISE: NamedColor
    DARK_SLATE_GREY: NamedColor
    PALE_TURQUOISE: NamedColor
    TEAL: NamedColor
    DARK_CYAN: NamedColor
    CYAN: NamedColor
    LIGHT_CYAN: NamedColor
    AZURE: NamedColor
    DARK_TURQUOISE: NamedColor
    CADET_BLUE: NamedColor
    POWDER_BLUE: NamedColor
    LIGHT_BLUE: NamedColor
    DEEP_SKY_BLUE: NamedColor
    SKY_BLUE: NamedColor
    LIGHT_SKY_BLUE: NamedColor
    STEEL_BLUE: NamedColor
    ALICE_BLUE: NamedColor
    DODGER_BLUE: NamedColor
    SLATE_GREY: NamedColor
    LIGHT_SLATE_GREY: NamedColor
    LIGHT_STEEL_BLUE: NamedColor
    CORNFLOWER_BLUE: NamedColor
    ROYAL_BLUE: NamedColor
    MIDNIGHT_BLUE: NamedColor
    LAVENDER: NamedColor
    NAVY: NamedColor
    DARK_BLUE: NamedColor
    MEDIUM_BLUE: NamedColor
    BLUE: NamedColor
    GHOST_WHITE: NamedColor
    SLATE_BLUE: NamedColor
    DARK_SLATE_BLUE: NamedColor
    MEDIUM_SLATE_BLUE: NamedColor
    MEDIUM_PURPLE: NamedColor
    REBECCA_PURPLE: NamedColor
    BLUE_VIOLET: NamedColor
    INDIGO: NamedColor
    DARK_ORCHID: NamedColor
    DARK_VIOLET: NamedColor
    MEDIUM_ORCHID: NamedColor
    THISTLE: NamedColor
    PLUM: NamedColor
    VIOLET: NamedColor
    PURPLE: NamedColor
    DARK_MAGENTA: NamedColor
    FUCHSIA: NamedColor
    ORCHID: NamedColor
    MEDIUM_VIOLET_RED: NamedColor
    DEEP_PINK: NamedColor
    HOT_PINK: NamedColor
    LAVENDER_BLUSH: NamedColor
    PALE_VIOLET_RED: NamedColor
    CRIMSON: NamedColor
    PINK: NamedColor
    LIGHT_PINK: NamedColor

# fmt: off
_ColorName = TypeVar(
    '_ColorName',
    bound=Literal[
        'ALICE_BLUE', 'ANTIQUE_WHITE', 'AQUAMARINE', 'AZURE', 'BEIGE', 'BISQUE', 'BLACK',
        'BLANCHED_ALMOND', 'BLUE', 'BLUE_VIOLET', 'BROWN', 'BURLY_WOOD', 'BURNT_SIENNA',
        'CADET_BLUE', 'CHARTREUSE', 'CHOCOLATE', 'CORAL', 'CORNFLOWER_BLUE', 'CORNSILK',
        'CRIMSON', 'CYAN', 'DARK_BLUE', 'DARK_CYAN', 'DARK_GOLDENROD', 'DARK_GREEN', 'DARK_GREY',
        'DARK_KHAKI', 'DARK_MAGENTA', 'DARK_OLIVE_GREEN', 'DARK_ORANGE', 'DARK_ORCHID',
        'DARK_RED', 'DARK_SALMON', 'DARK_SEA_GREEN', 'DARK_SLATE_BLUE', 'DARK_SLATE_GREY',
        'DARK_TURQUOISE', 'DARK_VIOLET', 'DEEP_PINK', 'DEEP_SKY_BLUE', 'DIM_GREY', 'DODGER_BLUE',
        'FIREBRICK', 'FLORAL_WHITE', 'FOREST_GREEN', 'FUCHSIA', 'GHOST_WHITE', 'GOLD',
        'GOLDENROD', 'GREEN', 'GREEN_YELLOW', 'GREY', 'HONEYDEW', 'HOT_PINK', 'INDIAN_RED',
        'INDIGO', 'IVORY', 'KHAKI', 'LAVENDER', 'LAVENDER_BLUSH', 'LAWN_GREEN', 'LEMON_CHIFFON',
        'LIGHT_BLUE', 'LIGHT_CORAL', 'LIGHT_CYAN', 'LIGHT_GOLDENROD_YELLOW', 'LIGHT_GREEN',
        'LIGHT_GREY', 'LIGHT_PINK', 'LIGHT_SALMON', 'LIGHT_SEA_GREEN', 'LIGHT_SKY_BLUE',
        'LIGHT_SLATE_GREY', 'LIGHT_STEEL_BLUE', 'LIGHT_YELLOW', 'LIME', 'LIME_GREEN', 'LINEN',
        'MAROON', 'MEDIUM_AQUAMARINE', 'MEDIUM_BLUE', 'MEDIUM_ORCHID', 'MEDIUM_PURPLE',
        'MEDIUM_SEA_GREEN', 'MEDIUM_SLATE_BLUE', 'MEDIUM_SPRING_GREEN', 'MEDIUM_TURQUOISE',
        'MEDIUM_VIOLET_RED', 'MIDNIGHT_BLUE', 'MINT_CREAM', 'MISTY_ROSE', 'MOCCASIN',
        'NAVAJO_WHITE', 'NAVY', 'OLD_LACE', 'OLIVE', 'OLIVE_DRAB', 'ORANGE', 'ORANGE_RED',
        'ORCHID', 'PALE_GOLDENROD', 'PALE_GREEN', 'PALE_TURQUOISE', 'PALE_VIOLET_RED',
        'PAPAYA_WHIP', 'PEACH_PUFF', 'PERU', 'PINK', 'PLUM', 'POWDER_BLUE', 'PURPLE',
        'REBECCA_PURPLE', 'RED', 'ROSY_BROWN', 'ROYAL_BLUE', 'SADDLE_BROWN', 'SALMON',
        'SANDY_BROWN', 'SEASHELL', 'SEA_GREEN', 'SIENNA', 'SILVER', 'SKY_BLUE', 'SLATE_BLUE',
        'SLATE_GREY', 'SNOW', 'SPRING_GREEN', 'STEEL_BLUE', 'TAN', 'TEAL', 'THISTLE', 'TOMATO',
        'TURQUOISE', 'VIOLET', 'WHEAT', 'WHITE', 'WHITE_SMOKE', 'YELLOW', 'YELLOW_GREEN'
    ]
)
_ColorName4Bit = TypeVar(
    '_ColorName4Bit',
    bound=Literal[
        'BLACK', 'RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN', 'GREY', 'DARK_GREY',
        'BRIGHT_RED', 'BRIGHT_GREEN', 'BRIGHT_YELLOW', 'BRIGHT_BLUE', 'BRIGHT_MAGENTA',
        'BRIGHT_CYAN', 'WHITE'
    ]
)
# fmt: on
named_color: Union[
    dict[Literal['4b'], Callable[[_ColorName4Bit], Color]],
    dict[Literal['24b'], Callable[[_ColorName], Color]],
]

class _ColorStrWrapperKwargs(TypedDict, total=False):
    ansi_type: Union[AnsiColorAlias, type[AnsiColorFormat]]
    bg: _ColorLike
    fg: _ColorLike
    sgr_params: Sequence[Union[int, SgrParameter]]

class rgb_dispatch[**P, R]:
    color_ns: MappingProxyType[str, Int3Tuple]

    __signature__: inspect.Signature

    @overload
    def __new__(cls, __f, /, *, args: Sequence[str] = ()):
        return __f

    @property
    def __wrapped__(self) -> Callable[P, R]: ...

Back = AnsiBack()
Fore = AnsiFore()
Style = AnsiStyle()
