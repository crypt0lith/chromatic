__all__ = ['ColorNamespace', 'Fore', 'Back', 'Style', 'color_str_wrapper', 'display_named_colors', 'display_ansi256_color_range']
from typing import Iterator, TypedDict, Unpack
from chromatic._typing import AnsiColorAlias
from chromatic.ansi.core import AnsiColorFormat, Color, ColorStr, SgrParameter
class DynamicNSMeta[_VT](type):
    def __new__(mcls, clsname: str, bases: tuple[type, ...], mapping: dict[str, ...], **kwargs) -> DynamicNSMeta[_VT]: ...
class DynamicNamespace[_VT](metaclass=DynamicNSMeta[_VT]):
    def __new__(cls, *args, **kwargs) -> DynamicNamespace[_VT]: ...
    def __init__[_KT](self, **kwargs: dict[_KT, _VT]) -> None: ...
    def __init_subclass__(cls, **kwargs) -> DynamicNamespace[_VT]: ...
    def __setattr__(self, name, value) -> None: ...
    def __iter__(self) -> Iterator[_VT]: ...
    def as_dict(self) -> dict[str, _VT]: ...
    __members__: list[_VT]
class _ColorNamespace[NamedColor: Color](DynamicNamespace[NamedColor]):
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
# noinspection PyTypedDict
class ColorStrWrapperKwargs(TypedDict, total=False):
    fg: int | Color | tuple[int, int, int]
    bg: int | Color | tuple[int, int, int]
    ansi_type: AnsiColorAlias | type[AnsiColorFormat]
    sgr_params: tuple[SgrParameter, ...]
class color_str_wrapper:
    def __init__(self, **kwargs: Unpack[ColorStrWrapperKwargs]) -> None: ...
    def __call__(self, __obj: object = None) -> ColorStr: ...
    def __add__(self, other) -> ColorStr: ...
    def __str__(self) -> ColorStr: ...
    def __repr__(self) -> str: ...
    ansi: ColorStr
class AnsiFore(_ColorNamespace[color_str_wrapper]): ...
class AnsiBack(_ColorNamespace[color_str_wrapper]): ...
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
ColorNamespace = _ColorNamespace()
Fore = AnsiFore()
Back = AnsiBack()
Style = AnsiStyle()
def display_named_colors() -> list[ColorStr]: ...
def display_ansi256_color_range() -> list[list[ColorStr]]: ...
