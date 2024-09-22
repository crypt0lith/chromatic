from types import FunctionType
from typing import Callable, Iterator, TYPE_CHECKING, TypeGuard, TypedDict, Unpack, cast, dataclass_transform

from chromatic._typing import AnsiColorAlias
from chromatic.ansi.core import AnsiColorFormat, Color, ColorStr, SgrParameter, get_default_ansi

DEFAULT_ANSI = get_default_ansi()
null = object()


class Member[_T]:

    def __init__(self, name, clsname, offset):
        self.name = name
        self.clsname = clsname
        self.offset = offset

    def __get__(self, instance, owner) -> _T:
        if instance is None:
            return self
        value = instance.__members__[self.offset]
        if value is null:
            raise AttributeError(self.name)
        try:
            value.name = self.name
        except AttributeError:
            pass
        return value

    def __set__(self, instance, value: _T):
        instance.__members__[self.offset] = value

    def __repr__(self):
        return f"<{type(self).__qualname__} {self.name!r} of {self.clsname!r}>"


@dataclass_transform()
class DynamicNSMeta[_VT](type):

    def __new__(mcls, clsname: str, bases: tuple[type, ...], mapping: dict[str, ...], **kwargs):
        slot_names: dict[str, ...] = mapping.get('__annotations__', {})
        member: Member[_VT]
        for offset, name in enumerate(slot_names):
            member = Member(name, clsname, offset)
            mapping[name] = member
        return type.__new__(mcls, clsname, bases, mapping, **kwargs)


class DynamicNamespace[_VT](metaclass=DynamicNSMeta[_VT]):
    if TYPE_CHECKING:
        __members__: list[_VT]

    def __new__(cls, *args, **kwargs):
        inst = super().__new__(cls)
        if hasattr(cls, '__annotations__'):
            slots = kwargs.pop('slots', list(filter(_type_param_callback_filter(cls), cls.__annotations__)))
            empty_slots = [null] * len(slots)
            object.__setattr__(inst, '__members__', empty_slots)
        return inst

    def __init__[_KT](self, **kwargs: dict[_KT, _VT]):
        for name, value in kwargs.items():
            self.__setattr__(name, value)

    def __init_subclass__(cls, **kwargs):
        if DynamicNamespace in cls.__bases__:
            return super().__new__(cls)
        factory: Callable[[...], _VT] | FunctionType = kwargs.get('factory')
        if not callable(factory):
            raise ValueError(
                f"{cls.__qualname__!r} does not inherit {DynamicNamespace.__qualname__!r} as a base class "
                f"and does not provide callable 'factory' keyword argument")
        try:
            base: type[DynamicNamespace] = cast(
                type[...], next(typ for typ in cls.mro() if DynamicNamespace in typ.__bases__))
        except StopIteration:
            raise TypeError(
                f"{cls.__qualname__!r} does not have any base classes of "
                f"type {DynamicNamespace.__qualname__!r}") from None
        d = dict(zip(base.__annotations__, map(factory, base().__members__)))
        init = DynamicNamespace.__init__
        new = cls.__new__
        cls.__annotations__: dict[str, ...] = dict(map(lambda x: (x[0], type(x[-1])), d.items()))
        cls.__new__ = lambda typ: new(typ, **dict(slots=d))
        cls.__init__ = lambda typ: init(typ, **d)
        return super().__new__(cls)

    def __setattr__(self, name, value):
        cls = type(self)
        if hasattr(cls, '__annotations__') and name not in cls.__annotations__:
            raise AttributeError(
                f'{cls.__name__!r} object has no attribute {name!r}') from None
        super().__setattr__(name, value)

    def as_dict(self):
        cls = type(self)
        return dict(zip(cls.__annotations__, self.__members__))

    def __iter__(self):
        return iter(self.__members__)


def _type_param_callback_filter(cls: type):
    type_param = cls.__type_params__
    if type_param and len(type_param) == 1:
        anno = cls.__annotations__

        def f(x: str) -> TypeGuard[Member]:
            return type_param[0] == anno.get(x)

        return f


def _ns_from_iter[_KT, _VT](
    __iter: Iterator[_KT] | Callable[[], Iterator[_KT]], member_type: _VT = null
) -> Callable[[type[DynamicNamespace[_VT]]], type[DynamicNamespace[_VT]]]:
    def decorator(cls: type[DynamicNamespace[_VT]]):
        anno = cls.__annotations__
        type_params = cls.__type_params__
        member_iter = iter(__iter() if callable(__iter) else __iter)
        members = member_iter if member_type == null else map(member_type, member_iter)
        d = dict(zip((k for k, v in anno.items() if v in type_params), members))
        init = DynamicNamespace.__init__
        cls.__init__ = lambda t: init(t, **d)
        return cls

    return decorator


def _gen_named_color_values() -> Iterator[int]:
    it = iter(
        x for x in (
            0x000000, 0x696969, 0x808080, 0xA9A9A9, 0xC0C0C0, 0xD3D3D3, 0xF5F5F5, 0xFFFFFF, 0xBC8F8F, 0xCD5C5C,
            0xA52A2A, 0xB22222, 0xF08080, 0x800000, 0x8B0000, 0xFF0000, 0xFFFAFA, 0xFFE4E1, 0xFA8072, 0xFF6347,
            0xEA7E5D, 0xE9967A, 0xFF7F50, 0xFF4500, 0xFFA07A, 0xA0522D, 0xFFF5EE, 0xD2691E, 0x8B4513, 0xF4A460,
            0xFFDAB9, 0xCD853F, 0xFAF0E6, 0xFFE4C4, 0xFF8C00, 0xDEB887, 0xFAEBD7, 0xD2B48C, 0xFFDEAD, 0xFFEBCD,
            0xFFEFD5, 0xFFE4B5, 0xFFA500, 0xF5DEB3, 0xFDF5E6, 0xFFFAF0, 0xB8860B, 0xDAA520, 0xFFF8DC, 0xFFD700,
            0xFFFACD, 0xF0E68C, 0xEEE8AA, 0xBDB76B, 0xF5F5DC, 0xFAFAD2, 0x808000, 0xFFFF00, 0xFFFFE0, 0xFFFFF0,
            0x6B8E23, 0x9ACD32, 0x556B2F, 0xADFF2F, 0x7FFF00, 0x7CFC00, 0x8FBC8F, 0x228B22, 0x32CD32, 0x90EE90,
            0x98FB98, 0x006400, 0x008000, 0x00FF00, 0xF0FFF0, 0x2E8B57, 0x3CB371, 0x00FF7F, 0xF5FFFA, 0x00FA9A,
            0x66CDAA, 0x7FFFD4, 0x40E0D0, 0x20B2AA, 0x48D1CC, 0x2F4F4F, 0xAFEEEE, 0x008080, 0x008B8B, 0x00FFFF,
            0xE0FFFF, 0xF0FFFF, 0x00CED1, 0x5F9EA0, 0xB0E0E6, 0xADD8E6, 0x00BFFF, 0x87CEEB, 0x87CEFA, 0x4682B4,
            0xF0F8FF, 0x1E90FF, 0x708090, 0x778899, 0xB0C4DE, 0x6495ED, 0x4169E1, 0x191970, 0xE6E6FA, 0x000080,
            0x00008B, 0x0000CD, 0x0000FF, 0xF8F8FF, 0x6A5ACD, 0x483D8B, 0x7B68EE, 0x9370DB, 0x663399, 0x8A2BE2,
            0x4B0082, 0x9932CC, 0x9400D3, 0xBA55D3, 0xD8BFD8, 0xDDA0DD, 0xEE82EE, 0x800080, 0x8B008B, 0xFF00FF,
            0xDA70D6, 0xC71585, 0xFF1493, 0xFF69B4, 0xFFF0F5, 0xDB7093, 0xDC143C, 0xFFC0CB, 0xFFB6C1))
    while True:
        yield next(it)


@_ns_from_iter(_gen_named_color_values, Color)
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
    sgr_params: tuple[SgrParameter, ...]
    ansi_type: AnsiColorAlias | type[AnsiColorFormat]


# noinspection PyUnresolvedReferences
class color_str_wrapper:

    def __init__(self, **kwargs: Unpack[ColorStrWrapperKwargs]):
        self.__ansi_type = kwargs.pop('ansi_type', DEFAULT_ANSI)
        self.__params: tuple[SgrParameter, ...] = kwargs.pop('sgr_params', tuple())
        if self.__params:
            self.__kw = ColorStr('', no_reset=True).update_sgr(*self.__params).ansi
            self.ansi = ColorStr(self.__kw, ansi_type=self.__ansi_type, no_reset=True)
        else:
            self.__kw = kwargs
            self.ansi = ColorStr(
                color_spec=self.__kw,
                ansi_type=self.__ansi_type,
                no_reset=True)

    def __call__(self, __obj: object = None):
        cls = type(self)
        if type(__obj) is cls:
            other = cast(color_str_wrapper, __obj)
            if isinstance(self.__kw, dict) and isinstance(other.__kw, dict):
                return cls(**self.__kw | other.__kw).ansi
            return self.ansi + other.ansi
        if isinstance(__obj, ColorStr):
            no_reset = not __obj.endswith(('\x1b[m', '\x1b[0m'))
            return ColorStr(
                color_spec=self.__kw,
                ansi_type=self.__ansi_type,
                no_reset=no_reset) + __obj
        return ColorStr(
            __obj,
            color_spec=self.__kw,
            ansi_type=self.__ansi_type)

    def __add__(self, other):
        return self.__call__(other)

    def __str__(self):
        return self.ansi

    def __repr__(self):
        if isinstance(self.__kw, dict):
            return f"{type(self).__qualname__}({', '.join(f'{k}={v!r}' for k, v in self.__kw.items())})"
        return f"{type(self).__qualname__}(color_spec={self.__kw!r})"


def _fg_wrapper_factory(__x: Color):
    return color_str_wrapper(fg=__x)


def _bg_wrapper_factory(__x: Color):
    return color_str_wrapper(bg=__x)


class AnsiFore(_ColorNamespace[color_str_wrapper], factory=_fg_wrapper_factory):
    pass


class AnsiBack(_ColorNamespace[color_str_wrapper], factory=_bg_wrapper_factory):
    pass


# noinspection PyUnresolvedReferences
def _ansi_style_wrapper(__x: SgrParameter):
    base = color_str_wrapper
    if __x in {SgrParameter.ANSI_256_SET_FG, SgrParameter.ANSI_256_SET_BG}:
        return base()
    inst = base(sgr_params=tuple([__x]))
    fg = inst.ansi.fg
    bg = inst.ansi.bg
    if fg or bg:
        return base(fg=fg) if fg else base(bg=bg)
    return inst


@_ns_from_iter(map(_ansi_style_wrapper, SgrParameter))
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


if TYPE_CHECKING:
    ColorNamespace: _ColorNamespace
    Fore: AnsiFore
    Back: AnsiBack
    Style: AnsiStyle


def display_named_colors():
    return list(
        ColorStr(name.replace('_', ' ').lower(), color_spec=color, ansi_type='24b') for name, color in
        _ColorNamespace().as_dict().items())


def display_ansi256_color_range():
    from numpy import asarray
    from chromatic.ansi._colorconv import ansi_8bit_to_rgb

    ansi256_range = asarray(range(256)).reshape([16] * 2).tolist()
    return list(
        list(ColorStr(obj='###', color_spec=Color(ansi_8bit_to_rgb(v)), ansi_type='8b') for v in arr) for arr in
        ansi256_range)


def __getattr__(name):
    globals_dict = dict(globals())
    if name in globals_dict:
        return globals_dict[name]
    if name == 'ColorNamespace':
        return _ColorNamespace()
    if name == 'Back':
        return AnsiBack()
    if name == 'Fore':
        return AnsiFore()
    if name == 'Style':
        return AnsiStyle()
    raise AttributeError(
        f"Module '{__name__}' has no attribute '{name}'")


__all__ = [
    'ColorNamespace',
    'Back',
    'Fore',
    'Style',
    'color_str_wrapper',
    'display_ansi256_color_range',
    'display_named_colors'
]
