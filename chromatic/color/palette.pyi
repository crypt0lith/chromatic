__all__ = ['Back', 'ColorNamespace', 'Fore', 'Style', 'rgb_dispatch', 'named_color']

from types import MappingProxyType
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    Literal,
    Mapping,
    Self,
    TypeAlias,
    overload,
)

from .core import Color, ColorStr, color_chain
from .._typing import Int3Tuple

_ColorLike: TypeAlias = int | Color | Int3Tuple

def named_color_idents() -> list[ColorStr]: ...

class AnsiBack(ColorNamespace[color_chain]):
    RESET: ClassVar[color_chain]

    def __call__(self, bg: _ColorLike) -> color_chain: ...

class AnsiFore(ColorNamespace[color_chain]):
    RESET: ClassVar[color_chain]

    def __call__(self, fg: _ColorLike) -> color_chain: ...

class AnsiStyle(DynamicNamespace[color_chain]):
    RESET: ClassVar[color_chain]
    BOLD: ClassVar[color_chain]
    FAINT: ClassVar[color_chain]
    ITALICS: ClassVar[color_chain]
    SINGLE_UNDERLINE: ClassVar[color_chain]
    SLOW_BLINK: ClassVar[color_chain]
    RAPID_BLINK: ClassVar[color_chain]
    NEGATIVE: ClassVar[color_chain]
    CONCEALED_CHARS: ClassVar[color_chain]
    CROSSED_OUT: ClassVar[color_chain]
    PRIMARY: ClassVar[color_chain]
    FIRST_ALT: ClassVar[color_chain]
    SECOND_ALT: ClassVar[color_chain]
    THIRD_ALT: ClassVar[color_chain]
    FOURTH_ALT: ClassVar[color_chain]
    FIFTH_ALT: ClassVar[color_chain]
    SIXTH_ALT: ClassVar[color_chain]
    SEVENTH_ALT: ClassVar[color_chain]
    EIGHTH_ALT: ClassVar[color_chain]
    NINTH_ALT: ClassVar[color_chain]
    GOTHIC: ClassVar[color_chain]
    DOUBLE_UNDERLINE: ClassVar[color_chain]
    RESET_BOLD_AND_FAINT: ClassVar[color_chain]
    RESET_ITALIC_AND_GOTHIC: ClassVar[color_chain]
    RESET_UNDERLINES: ClassVar[color_chain]
    RESET_BLINKING: ClassVar[color_chain]
    POSITIVE: ClassVar[color_chain]
    REVEALED_CHARS: ClassVar[color_chain]
    RESET_CROSSED_OUT: ClassVar[color_chain]
    BLACK_FG: ClassVar[color_chain]
    RED_FG: ClassVar[color_chain]
    GREEN_FG: ClassVar[color_chain]
    YELLOW_FG: ClassVar[color_chain]
    BLUE_FG: ClassVar[color_chain]
    MAGENTA_FG: ClassVar[color_chain]
    CYAN_FG: ClassVar[color_chain]
    WHITE_FG: ClassVar[color_chain]
    DEFAULT_FG_COLOR: ClassVar[color_chain]
    BLACK_BG: ClassVar[color_chain]
    RED_BG: ClassVar[color_chain]
    GREEN_BG: ClassVar[color_chain]
    YELLOW_BG: ClassVar[color_chain]
    BLUE_BG: ClassVar[color_chain]
    MAGENTA_BG: ClassVar[color_chain]
    CYAN_BG: ClassVar[color_chain]
    WHITE_BG: ClassVar[color_chain]
    DEFAULT_BG_COLOR: ClassVar[color_chain]
    FRAMED: ClassVar[color_chain]
    ENCIRCLED: ClassVar[color_chain]
    OVERLINED: ClassVar[color_chain]
    NOT_FRAMED_OR_CIRCLED: ClassVar[color_chain]
    IDEOGRAM_UNDER_OR_RIGHT: ClassVar[color_chain]
    IDEOGRAM_2UNDER_OR_2RIGHT: ClassVar[color_chain]
    IDEOGRAM_OVER_OR_LEFT: ClassVar[color_chain]
    IDEOGRAM_2OVER_OR_2LEFT: ClassVar[color_chain]
    CANCEL: ClassVar[color_chain]
    BLACK_BRIGHT_FG: ClassVar[color_chain]
    RED_BRIGHT_FG: ClassVar[color_chain]
    GREEN_BRIGHT_FG: ClassVar[color_chain]
    YELLOW_BRIGHT_FG: ClassVar[color_chain]
    BLUE_BRIGHT_FG: ClassVar[color_chain]
    MAGENTA_BRIGHT_FG: ClassVar[color_chain]
    CYAN_BRIGHT_FG: ClassVar[color_chain]
    WHITE_BRIGHT_FG: ClassVar[color_chain]
    BLACK_BRIGHT_BG: ClassVar[color_chain]
    RED_BRIGHT_BG: ClassVar[color_chain]
    GREEN_BRIGHT_BG: ClassVar[color_chain]
    YELLOW_BRIGHT_BG: ClassVar[color_chain]
    BLUE_BRIGHT_BG: ClassVar[color_chain]
    MAGENTA_BRIGHT_BG: ClassVar[color_chain]
    CYAN_BRIGHT_BG: ClassVar[color_chain]
    WHITE_BRIGHT_BG: ClassVar[color_chain]

class DynamicNamespace[_VT](metaclass=DynamicNSMeta[_VT]):
    @classmethod
    def asdict(cls) -> dict[str, _VT]: ...

class DynamicNSMeta[_VT](type):
    __members__: tuple[str, ...]

    @classmethod
    def __class_getitem__[_T](mcls: _T, _) -> _T: ...
    @classmethod
    def __prepare__(
        mcls, clsname: str, bases: tuple[type, ...], /, **kwds
    ) -> Mapping[str, object]: ...
    @overload
    def __new__(
        mcls: type[Self],
        clsname: str,
        bases: tuple[type, ...],
        namespace: dict[str, ...],
        /,
    ) -> Self: ...
    @overload
    def __new__(
        mcls: type[Self],
        clsname: str,
        bases: tuple[type, ...],
        namespace: dict[str, ...],
        /,
        *,
        iterable: Iterable[_VT],
    ) -> Self: ...
    @overload
    def __new__[_T](
        mcls: type[Self],
        clsname: str,
        bases: tuple[type, ...],
        namespace: dict[str, ...],
        /,
        *,
        member_type: Callable[[_T], _VT],
    ) -> Self: ...
    def asdict(cls) -> dict[str, _VT]: ...

class ColorNamespace[NamedColor = Color](DynamicNamespace[NamedColor]):
    BLACK: NamedColor
    DIM_GREY: NamedColor
    GREY: NamedColor
    DARK_GREY: NamedColor
    SILVER: NamedColor
    LIGHT_GREY: NamedColor
    WHITE_SMOKE: NamedColor
    WHITE: NamedColor
    MAROON: NamedColor
    DARK_RED: NamedColor
    RED: NamedColor
    FIREBRICK: NamedColor
    BROWN: NamedColor
    INDIAN_RED: NamedColor
    LIGHT_CORAL: NamedColor
    ROSY_BROWN: NamedColor
    MISTY_ROSE: NamedColor
    SNOW: NamedColor
    SIENNA: NamedColor
    ORANGE_RED: NamedColor
    TOMATO: NamedColor
    BURNT_SIENNA: NamedColor
    CORAL: NamedColor
    SALMON: NamedColor
    DARK_SALMON: NamedColor
    LIGHT_SALMON: NamedColor
    SEASHELL: NamedColor
    SADDLE_BROWN: NamedColor
    CHOCOLATE: NamedColor
    PERU: NamedColor
    SANDY_BROWN: NamedColor
    PEACH_PUFF: NamedColor
    LINEN: NamedColor
    DARK_ORANGE: NamedColor
    BURLY_WOOD: NamedColor
    BISQUE: NamedColor
    ANTIQUE_WHITE: NamedColor
    ORANGE: NamedColor
    TAN: NamedColor
    WHEAT: NamedColor
    NAVAJO_WHITE: NamedColor
    MOCCASIN: NamedColor
    BLANCHED_ALMOND: NamedColor
    PAPAYA_WHIP: NamedColor
    OLD_LACE: NamedColor
    FLORAL_WHITE: NamedColor
    DARK_GOLDENROD: NamedColor
    GOLDENROD: NamedColor
    CORNSILK: NamedColor
    DARK_KHAKI: NamedColor
    GOLD: NamedColor
    KHAKI: NamedColor
    PALE_GOLDENROD: NamedColor
    BEIGE: NamedColor
    LIGHT_GOLDENROD_YELLOW: NamedColor
    LEMON_CHIFFON: NamedColor
    OLIVE: NamedColor
    YELLOW: NamedColor
    LIGHT_YELLOW: NamedColor
    IVORY: NamedColor
    DARK_GREEN: NamedColor
    GREEN: NamedColor
    DARK_OLIVE_GREEN: NamedColor
    FOREST_GREEN: NamedColor
    OLIVE_DRAB: NamedColor
    LIME_GREEN: NamedColor
    DARK_SEA_GREEN: NamedColor
    LIME: NamedColor
    YELLOW_GREEN: NamedColor
    LAWN_GREEN: NamedColor
    CHARTREUSE: NamedColor
    LIGHT_GREEN: NamedColor
    GREEN_YELLOW: NamedColor
    PALE_GREEN: NamedColor
    HONEYDEW: NamedColor
    SEA_GREEN: NamedColor
    MEDIUM_SEA_GREEN: NamedColor
    SPRING_GREEN: NamedColor
    MINT_CREAM: NamedColor
    DARK_SLATE_GREY: NamedColor
    TEAL: NamedColor
    DARK_CYAN: NamedColor
    LIGHT_SEA_GREEN: NamedColor
    MEDIUM_TURQUOISE: NamedColor
    MEDIUM_AQUAMARINE: NamedColor
    TURQUOISE: NamedColor
    MEDIUM_SPRING_GREEN: NamedColor
    CYAN: NamedColor
    PALE_TURQUOISE: NamedColor
    AQUAMARINE: NamedColor
    LIGHT_CYAN: NamedColor
    AZURE: NamedColor
    STEEL_BLUE: NamedColor
    CADET_BLUE: NamedColor
    DEEP_SKY_BLUE: NamedColor
    DARK_TURQUOISE: NamedColor
    SKY_BLUE: NamedColor
    LIGHT_SKY_BLUE: NamedColor
    LIGHT_BLUE: NamedColor
    POWDER_BLUE: NamedColor
    ALICE_BLUE: NamedColor
    MIDNIGHT_BLUE: NamedColor
    ROYAL_BLUE: NamedColor
    SLATE_GREY: NamedColor
    DODGER_BLUE: NamedColor
    LIGHT_SLATE_GREY: NamedColor
    CORNFLOWER_BLUE: NamedColor
    LIGHT_STEEL_BLUE: NamedColor
    LAVENDER: NamedColor
    NAVY: NamedColor
    DARK_BLUE: NamedColor
    MEDIUM_BLUE: NamedColor
    BLUE: NamedColor
    GHOST_WHITE: NamedColor
    INDIGO: NamedColor
    DARK_VIOLET: NamedColor
    DARK_SLATE_BLUE: NamedColor
    REBECCA_PURPLE: NamedColor
    BLUE_VIOLET: NamedColor
    DARK_ORCHID: NamedColor
    SLATE_BLUE: NamedColor
    MEDIUM_ORCHID: NamedColor
    MEDIUM_SLATE_BLUE: NamedColor
    MEDIUM_PURPLE: NamedColor
    THISTLE: NamedColor
    PURPLE: NamedColor
    DARK_MAGENTA: NamedColor
    MEDIUM_VIOLET_RED: NamedColor
    FUCHSIA: NamedColor
    DEEP_PINK: NamedColor
    ORCHID: NamedColor
    HOT_PINK: NamedColor
    VIOLET: NamedColor
    PLUM: NamedColor
    LAVENDER_BLUSH: NamedColor
    CRIMSON: NamedColor
    PALE_VIOLET_RED: NamedColor
    LIGHT_PINK: NamedColor
    PINK: NamedColor

named_color: MappingProxyType[tuple[Literal['4b', '24b'], str], Color]

@overload
def rgb_dispatch[_F: Callable[..., Any]](__f: _F, /) -> _F: ...
@overload
def rgb_dispatch[_F: Callable[..., Any]](
    names: tuple[str, ...] = (),
) -> Callable[[_F], _F]: ...

Back: AnsiBack
Fore: AnsiFore
Style: AnsiStyle
