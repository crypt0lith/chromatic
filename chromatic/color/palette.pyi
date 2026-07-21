__all__ = ["Back", "ColorNamespace", "Fore", "Style", "rgb_dispatch", "named_color"]

import collections.abc as abc
import typing as tp
from types import MappingProxyType

from .._typing import Int3Tuple
from .core import Color, ColorStr, color_chain

if tp.TYPE_CHECKING:
    class _SupportsClassGetItem(tp.Protocol):
        @classmethod
        def __class_getitem__(cls: type[tp.Self], args: tp.Any, /) -> type[tp.Self]: ...

class _DynamicNSMeta(type):
    __ignore__: tuple[str, ...]
    __members__: MappingProxyType[str, tp.Any]

    @tp.overload
    def __getitem__(cls, key: str, /) -> tp.Any: ...
    @tp.overload
    def __getitem__[_T: _SupportsClassGetItem](
        cls: type[_T], key: tp.Any, /
    ) -> type[_T]: ...

    @classmethod
    def __prepare__(
        mcls, clsname: str, bases: tuple[type, ...], /, **kwds
    ) -> abc.MutableMapping[str, object]: ...
    def asdict(cls) -> MappingProxyType[str, tp.Any]: ...

class DynamicNamespace(metaclass=_DynamicNSMeta): ...

class ColorNamespace[NamedColor = Color](DynamicNamespace):
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

class _frozen_color_chain(color_chain):
    def __hash__(self) -> int: ...
    def __delitem__(self, index, /) -> tp.Never: ...
    def __setitem__(self, index, value, /) -> tp.Never: ...
    def insert(self, index, value, /) -> tp.Never: ...

class AnsiStyle(DynamicNamespace):
    RESET: tp.ClassVar[_frozen_color_chain]
    BOLD: tp.ClassVar[_frozen_color_chain]
    FAINT: tp.ClassVar[_frozen_color_chain]
    ITALICS: tp.ClassVar[_frozen_color_chain]
    SINGLE_UNDERLINE: tp.ClassVar[_frozen_color_chain]
    SLOW_BLINK: tp.ClassVar[_frozen_color_chain]
    RAPID_BLINK: tp.ClassVar[_frozen_color_chain]
    NEGATIVE: tp.ClassVar[_frozen_color_chain]
    CONCEALED_CHARS: tp.ClassVar[_frozen_color_chain]
    CROSSED_OUT: tp.ClassVar[_frozen_color_chain]
    PRIMARY: tp.ClassVar[_frozen_color_chain]
    FIRST_ALT: tp.ClassVar[_frozen_color_chain]
    SECOND_ALT: tp.ClassVar[_frozen_color_chain]
    THIRD_ALT: tp.ClassVar[_frozen_color_chain]
    FOURTH_ALT: tp.ClassVar[_frozen_color_chain]
    FIFTH_ALT: tp.ClassVar[_frozen_color_chain]
    SIXTH_ALT: tp.ClassVar[_frozen_color_chain]
    SEVENTH_ALT: tp.ClassVar[_frozen_color_chain]
    EIGHTH_ALT: tp.ClassVar[_frozen_color_chain]
    NINTH_ALT: tp.ClassVar[_frozen_color_chain]
    GOTHIC: tp.ClassVar[_frozen_color_chain]
    DOUBLE_UNDERLINE: tp.ClassVar[_frozen_color_chain]
    RESET_BOLD_AND_FAINT: tp.ClassVar[_frozen_color_chain]
    RESET_ITALIC_AND_GOTHIC: tp.ClassVar[_frozen_color_chain]
    RESET_UNDERLINES: tp.ClassVar[_frozen_color_chain]
    RESET_BLINKING: tp.ClassVar[_frozen_color_chain]
    POSITIVE: tp.ClassVar[_frozen_color_chain]
    REVEALED_CHARS: tp.ClassVar[_frozen_color_chain]
    RESET_CROSSED_OUT: tp.ClassVar[_frozen_color_chain]
    BLACK_FG: tp.ClassVar[_frozen_color_chain]
    RED_FG: tp.ClassVar[_frozen_color_chain]
    GREEN_FG: tp.ClassVar[_frozen_color_chain]
    YELLOW_FG: tp.ClassVar[_frozen_color_chain]
    BLUE_FG: tp.ClassVar[_frozen_color_chain]
    MAGENTA_FG: tp.ClassVar[_frozen_color_chain]
    CYAN_FG: tp.ClassVar[_frozen_color_chain]
    WHITE_FG: tp.ClassVar[_frozen_color_chain]
    DEFAULT_FG_COLOR: tp.ClassVar[_frozen_color_chain]
    BLACK_BG: tp.ClassVar[_frozen_color_chain]
    RED_BG: tp.ClassVar[_frozen_color_chain]
    GREEN_BG: tp.ClassVar[_frozen_color_chain]
    YELLOW_BG: tp.ClassVar[_frozen_color_chain]
    BLUE_BG: tp.ClassVar[_frozen_color_chain]
    MAGENTA_BG: tp.ClassVar[_frozen_color_chain]
    CYAN_BG: tp.ClassVar[_frozen_color_chain]
    WHITE_BG: tp.ClassVar[_frozen_color_chain]
    DEFAULT_BG_COLOR: tp.ClassVar[_frozen_color_chain]
    FRAMED: tp.ClassVar[_frozen_color_chain]
    ENCIRCLED: tp.ClassVar[_frozen_color_chain]
    OVERLINED: tp.ClassVar[_frozen_color_chain]
    NOT_FRAMED_OR_CIRCLED: tp.ClassVar[_frozen_color_chain]
    IDEOGRAM_UNDER_OR_RIGHT: tp.ClassVar[_frozen_color_chain]
    IDEOGRAM_2UNDER_OR_2RIGHT: tp.ClassVar[_frozen_color_chain]
    IDEOGRAM_OVER_OR_LEFT: tp.ClassVar[_frozen_color_chain]
    IDEOGRAM_2OVER_OR_2LEFT: tp.ClassVar[_frozen_color_chain]
    CANCEL: tp.ClassVar[_frozen_color_chain]
    BLACK_BRIGHT_FG: tp.ClassVar[_frozen_color_chain]
    RED_BRIGHT_FG: tp.ClassVar[_frozen_color_chain]
    GREEN_BRIGHT_FG: tp.ClassVar[_frozen_color_chain]
    YELLOW_BRIGHT_FG: tp.ClassVar[_frozen_color_chain]
    BLUE_BRIGHT_FG: tp.ClassVar[_frozen_color_chain]
    MAGENTA_BRIGHT_FG: tp.ClassVar[_frozen_color_chain]
    CYAN_BRIGHT_FG: tp.ClassVar[_frozen_color_chain]
    WHITE_BRIGHT_FG: tp.ClassVar[_frozen_color_chain]
    BLACK_BRIGHT_BG: tp.ClassVar[_frozen_color_chain]
    RED_BRIGHT_BG: tp.ClassVar[_frozen_color_chain]
    GREEN_BRIGHT_BG: tp.ClassVar[_frozen_color_chain]
    YELLOW_BRIGHT_BG: tp.ClassVar[_frozen_color_chain]
    BLUE_BRIGHT_BG: tp.ClassVar[_frozen_color_chain]
    MAGENTA_BRIGHT_BG: tp.ClassVar[_frozen_color_chain]
    CYAN_BRIGHT_BG: tp.ClassVar[_frozen_color_chain]
    WHITE_BRIGHT_BG: tp.ClassVar[_frozen_color_chain]

_ColorLike: tp.TypeAlias = int | Int3Tuple

class AnsiBack(ColorNamespace[_frozen_color_chain]):
    RESET: tp.ClassVar[_frozen_color_chain]

    def __call__(self, bg: _ColorLike) -> color_chain: ...

class AnsiFore(ColorNamespace[_frozen_color_chain]):
    RESET: tp.ClassVar[_frozen_color_chain]

    def __call__(self, fg: _ColorLike) -> color_chain: ...

@tp.overload
def rgb_dispatch[_F: abc.Callable[..., tp.Any]](f: _F, /, *names: str) -> _F: ...
@tp.overload
def rgb_dispatch[_F: abc.Callable[..., tp.Any]](
    *names: str,
) -> abc.Callable[[_F], _F]: ...

named_color: MappingProxyType[str | tuple[str, tp.Literal["4b", "24b"]], Color]

def named_color_idents() -> list[ColorStr]: ...

Back: AnsiBack
Fore: AnsiFore
Style: AnsiStyle
