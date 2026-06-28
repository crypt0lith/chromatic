__all__ = ['Back', 'ColorNamespace', 'Fore', 'Style', 'rgb_dispatch', 'named_color']

import collections.abc as abc
import functools as ft
import typing as tp
from types import FunctionType, MappingProxyType as mappingproxy

from .._typing import Int3Tuple
from .core import Color, ColorStr, SgrSequence, color_chain


class _ns_member_descriptor:
    def __init__(self, value):
        self.value = value

    def __set_name__(self, owner, name):
        self.__objclass__ = owner
        self.name = name
        owner.__members__[self.name] = self.value

    def __get__(self, inst, owner=None):
        typ = inst.__class__ if owner is None else owner
        return typ.__members__[self.name]


class _DynamicNSMeta(type):
    __members__: abc.Mapping[str, tp.Any]

    @classmethod
    def __prepare__(mcls, name, bases, /, **_) -> abc.MutableMapping[str, object]:
        return {
            "__members__": {
                k: v
                for base in bases
                if isinstance(base, mcls)
                for k, v in base.__members__.items()
            }
        }

    def __new__(mcls, name, bases, ns, /, **kwargs):
        ignored = ns.get("__ignore__", ())
        ns.update(
            {
                k: _ns_member_descriptor(v)
                for k, v in ns.items()
                if not (
                    k in ignored
                    or k.startswith("__")
                    or callable(getattr(v, "__get__", None))
                )
            }
        )
        res = type.__new__(mcls, name, bases, ns)
        if (wrapper_f := kwargs.get("wrapper")) is not None:
            if not callable(wrapper_f):
                raise ValueError(f"expected callable object: {wrapper_f!r}")
            assert isinstance(res.__members__, abc.MutableMapping)
            res.__members__.update(
                {k: wrapper_f(v) for k, v in res.__members__.items()}
            )
        return res

    def asdict(cls):
        return mappingproxy(cls.__members__)


class DynamicNamespace(metaclass=_DynamicNSMeta):
    pass


class ColorNamespace(DynamicNamespace, wrapper=Color):
    BLACK = 0x000000
    DIM_GREY = 0x696969
    GREY = 0x808080
    DARK_GREY = 0xA9A9A9
    SILVER = 0xC0C0C0
    LIGHT_GREY = 0xD3D3D3
    WHITE_SMOKE = 0xF5F5F5
    WHITE = 0xFFFFFF
    MAROON = 0x800000
    DARK_RED = 0x8B0000
    RED = 0xFF0000
    FIREBRICK = 0xB22222
    BROWN = 0xA52A2A
    INDIAN_RED = 0xCD5C5C
    LIGHT_CORAL = 0xF08080
    ROSY_BROWN = 0xBC8F8F
    MISTY_ROSE = 0xFFE4E1
    SNOW = 0xFFFAFA
    SIENNA = 0xA0522D
    ORANGE_RED = 0xFF4500
    TOMATO = 0xFF6347
    BURNT_SIENNA = 0xEA7E5D
    CORAL = 0xFF7F50
    SALMON = 0xFA8072
    DARK_SALMON = 0xE9967A
    LIGHT_SALMON = 0xFFA07A
    SEASHELL = 0xFFF5EE
    SADDLE_BROWN = 0x8B4513
    CHOCOLATE = 0xD2691E
    PERU = 0xCD853F
    SANDY_BROWN = 0xF4A460
    PEACH_PUFF = 0xFFDAB9
    LINEN = 0xFAF0E6
    DARK_ORANGE = 0xFF8C00
    BURLY_WOOD = 0xDEB887
    BISQUE = 0xFFE4C4
    ANTIQUE_WHITE = 0xFAEBD7
    ORANGE = 0xFFA500
    TAN = 0xD2B48C
    WHEAT = 0xF5DEB3
    NAVAJO_WHITE = 0xFFDEAD
    MOCCASIN = 0xFFE4B5
    BLANCHED_ALMOND = 0xFFEBCD
    PAPAYA_WHIP = 0xFFEFD5
    OLD_LACE = 0xFDF5E6
    FLORAL_WHITE = 0xFFFAF0
    DARK_GOLDENROD = 0xB8860B
    GOLDENROD = 0xDAA520
    CORNSILK = 0xFFF8DC
    DARK_KHAKI = 0xBDB76B
    GOLD = 0xFFD700
    KHAKI = 0xF0E68C
    PALE_GOLDENROD = 0xEEE8AA
    BEIGE = 0xF5F5DC
    LIGHT_GOLDENROD_YELLOW = 0xFAFAD2
    LEMON_CHIFFON = 0xFFFACD
    OLIVE = 0x808000
    YELLOW = 0xFFFF00
    LIGHT_YELLOW = 0xFFFFE0
    IVORY = 0xFFFFF0
    DARK_GREEN = 0x006400
    GREEN = 0x008000
    DARK_OLIVE_GREEN = 0x556B2F
    FOREST_GREEN = 0x228B22
    OLIVE_DRAB = 0x6B8E23
    LIME_GREEN = 0x32CD32
    DARK_SEA_GREEN = 0x8FBC8F
    LIME = 0x00FF00
    YELLOW_GREEN = 0x9ACD32
    LAWN_GREEN = 0x7CFC00
    CHARTREUSE = 0x7FFF00
    LIGHT_GREEN = 0x90EE90
    GREEN_YELLOW = 0xADFF2F
    PALE_GREEN = 0x98FB98
    HONEYDEW = 0xF0FFF0
    SEA_GREEN = 0x2E8B57
    MEDIUM_SEA_GREEN = 0x3CB371
    SPRING_GREEN = 0x00FF7F
    MINT_CREAM = 0xF5FFFA
    DARK_SLATE_GREY = 0x2F4F4F
    TEAL = 0x008080
    DARK_CYAN = 0x008B8B
    LIGHT_SEA_GREEN = 0x20B2AA
    MEDIUM_TURQUOISE = 0x48D1CC
    MEDIUM_AQUAMARINE = 0x66CDAA
    TURQUOISE = 0x40E0D0
    MEDIUM_SPRING_GREEN = 0x00FA9A
    CYAN = 0x00FFFF
    PALE_TURQUOISE = 0xAFEEEE
    AQUAMARINE = 0x7FFFD4
    LIGHT_CYAN = 0xE0FFFF
    AZURE = 0xF0FFFF
    STEEL_BLUE = 0x4682B4
    CADET_BLUE = 0x5F9EA0
    DEEP_SKY_BLUE = 0x00BFFF
    DARK_TURQUOISE = 0x00CED1
    SKY_BLUE = 0x87CEEB
    LIGHT_SKY_BLUE = 0x87CEFA
    LIGHT_BLUE = 0xADD8E6
    POWDER_BLUE = 0xB0E0E6
    ALICE_BLUE = 0xF0F8FF
    MIDNIGHT_BLUE = 0x191970
    ROYAL_BLUE = 0x4169E1
    SLATE_GREY = 0x708090
    DODGER_BLUE = 0x1E90FF
    LIGHT_SLATE_GREY = 0x778899
    CORNFLOWER_BLUE = 0x6495ED
    LIGHT_STEEL_BLUE = 0xB0C4DE
    LAVENDER = 0xE6E6FA
    NAVY = 0x000080
    DARK_BLUE = 0x00008B
    MEDIUM_BLUE = 0x0000CD
    BLUE = 0x0000FF
    GHOST_WHITE = 0xF8F8FF
    INDIGO = 0x4B0082
    DARK_VIOLET = 0x9400D3
    DARK_SLATE_BLUE = 0x483D8B
    REBECCA_PURPLE = 0x663399
    BLUE_VIOLET = 0x8A2BE2
    DARK_ORCHID = 0x9932CC
    SLATE_BLUE = 0x6A5ACD
    MEDIUM_ORCHID = 0xBA55D3
    MEDIUM_SLATE_BLUE = 0x7B68EE
    MEDIUM_PURPLE = 0x9370DB
    THISTLE = 0xD8BFD8
    PURPLE = 0x800080
    DARK_MAGENTA = 0x8B008B
    MEDIUM_VIOLET_RED = 0xC71585
    FUCHSIA = 0xFF00FF
    DEEP_PINK = 0xFF1493
    ORCHID = 0xDA70D6
    HOT_PINK = 0xFF69B4
    VIOLET = 0xEE82EE
    PLUM = 0xDDA0DD
    LAVENDER_BLUSH = 0xFFF0F5
    CRIMSON = 0xDC143C
    PALE_VIOLET_RED = 0xDB7093
    LIGHT_PINK = 0xFFB6C1
    PINK = 0xFFC0CB


class AnsiStyle(DynamicNamespace, wrapper=lambda x: color_chain([SgrSequence([x])])):
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
    DEFAULT_FG_COLOR = 39
    BLACK_BG = 40
    RED_BG = 41
    GREEN_BG = 42
    YELLOW_BG = 43
    BLUE_BG = 44
    MAGENTA_BG = 45
    CYAN_BG = 46
    WHITE_BG = 47
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


class AnsiBack(
    ColorNamespace,
    wrapper=lambda x: color_chain([ColorStr(bg=x)._sgr], ansi_type='24b'),
):
    __ignore__ = ("RESET",)
    RESET = AnsiStyle.DEFAULT_BG_COLOR

    def __call__(self, bg: Color | int | tuple[int, int, int]):
        return color_chain([ColorStr(bg=bg)._sgr])


class AnsiFore(
    ColorNamespace,
    wrapper=lambda x: color_chain([ColorStr(fg=x)._sgr], ansi_type='24b'),
):
    __ignore__ = ("RESET",)
    RESET = AnsiStyle.DEFAULT_FG_COLOR

    def __call__(self, fg: Color | int | tuple[int, int, int]):
        return color_chain([ColorStr(fg=fg)._sgr])


_ASCII_UPCASE = mappingproxy({x: x ^ 0x20 for x in range(0x61, 0x7B)} | {0x20: 0x5F})


def rgb_dispatch(*names):
    def decorator(f: FunctionType, /):
        def _prepare():
            assert isinstance(names, set)
            code = f.__code__
            n_args = code.co_argcount
            n_posonly = code.co_posonlyargcount
            n_pos_or_kw = n_args - n_posonly
            n_kwonly = code.co_kwonlyargcount
            flags = code.co_flags
            has_varargs = bool(flags & 0x4)
            has_varkwds = bool(flags & 0x8)
            total = sum([n_args, n_kwonly, has_varargs, has_varkwds])
            params = list(code.co_varnames[:total])
            if not names:
                names.update(
                    name
                    for name in params
                    if (
                        name in {"bg", "fg"}
                        or name.startswith(("bg_", "fg_"))
                        or name.endswith(("_bg", "_fg"))
                    )
                )
            paramd = {param: param in names for param in params}
            positions, keywords = [], {}
            if names:
                if total == 0 or not (has_varkwds or names <= paramd.keys()):
                    unexpected = ", ".join(
                        f"{name!r}" for name in names.difference(paramd)
                    )
                    raise ValueError(f"unexpected parameter names: {unexpected}")
            elif total == 0 or not (n_pos_or_kw or n_kwonly or has_varkwds):
                if total > 0:
                    positions.append(slice(None))
                return tuple(positions), mappingproxy(keywords)
            else:
                raise ValueError("no parameters specified and none could be inferred")
            i = 0
            if n_posonly > 0:
                posonly = params[:n_posonly]
                for name in posonly:
                    if paramd[name]:
                        positions.append(i)
                    i += 1
                del params[:n_posonly]
            if n_pos_or_kw > 0:
                pos_or_kw = params[:n_pos_or_kw]
                for name in pos_or_kw:
                    if paramd[name]:
                        positions.append(i)
                        keywords[name] = positions[-1]
                    i += 1
                del params[:n_pos_or_kw]
            if n_kwonly > 0:
                kwonly = params[:n_kwonly]
                for name in kwonly:
                    if paramd[name]:
                        keywords[name] = None
                del params[:n_kwonly]
            if has_varargs:
                varargs = params.pop(0)
                if paramd[varargs]:
                    positions.append(slice(i, None))
            if has_varkwds:
                varkwds = params.pop(0)
                if paramd[varkwds]:
                    keywords[None] = None
            return tuple(positions), mappingproxy(keywords)

        POSITIONS, KEYWORDS = _prepare()
        HAS_VARKW = None in KEYWORDS

        @ft.cache
        def _lookup(s: str, /) -> Int3Tuple:
            return ColorNamespace.__members__[s.translate(_ASCII_UPCASE)].rgb

        @ft.wraps(f)
        def wrapper(*args, **kwargs):
            _kwargs = kwargs.copy()
            n_args = len(args)
            mask = [False for _ in range(n_args)]
            for idx in POSITIONS:
                if isinstance(idx, slice):
                    for i in range(*idx.indices(n_args)):
                        mask[i] = True
                elif idx < n_args:
                    mask[idx] = True
            for k, v in kwargs.items():
                if not isinstance(v, str):
                    continue
                if not (k in KEYWORDS or HAS_VARKW):
                    continue
                try:
                    v = _lookup(v)
                except KeyError:
                    continue
                _kwargs[k] = v
                if (i := KEYWORDS.get(k)) is None or i >= n_args:
                    continue
                mask[i] = False
            _args = []
            for g, v in zip(mask, args):
                if not (g and isinstance(v, str)):
                    _args.append(v)
                    continue
                try:
                    res = _lookup(v)
                except KeyError:
                    _args.append(v)
                    continue
                _args.append(res)
            return f(*_args, **_kwargs)

        return wrapper

    f = None
    if names and callable(names[0]):
        f, *names = names
    names = set(names)
    return decorator if f is None else decorator(f)


def _named_color():
    class NamedColorDict(dict):
        def __missing__(self, key, /):
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and isinstance(key[0], str)
                and key[1] in ("4b", "24b")
            ):
                name, subkey = key
                name = name.translate(_ASCII_UPCASE)
                key_norm = name, subkey
            elif isinstance(key, str):
                key_norm = key.translate(_ASCII_UPCASE)
            else:
                raise KeyError(key)
            if key_norm in self:
                return self[key_norm]
            raise KeyError(key)

    from .colorconv import ANSI_4BIT_RGB

    ansi_4bit_names = (
        'BLACK',
        'RED',
        'GREEN',
        'YELLOW',
        'BLUE',
        'MAGENTA',
        'CYAN',
        'GREY',
        'DARK_GREY',
        'BRIGHT_RED',
        'BRIGHT_GREEN',
        'BRIGHT_YELLOW',
        'BRIGHT_BLUE',
        'BRIGHT_MAGENTA',
        'BRIGHT_CYAN',
        'WHITE',
    )
    ansi_4bit_dict = dict(zip(ansi_4bit_names, map(Color.from_rgb, ANSI_4BIT_RGB)))

    out = NamedColorDict()
    for k, mp in [("4b", ansi_4bit_dict), ("24b", ColorNamespace.asdict())]:
        for name, v in mp.items():
            out[name] = out[name, k] = v

    return mappingproxy(out)


def named_color_idents():
    return [
        ColorStr(name.translate({0x5F: 0x20}).lower(), color, ansi_type='24b')
        for name, color in ColorNamespace.asdict().items()
    ]


def __getattr__(name: str, /):
    if name in {"Back", "Fore", "Style"}:
        inst = globals()[f"Ansi{name}"]()
        return globals().setdefault(name, inst)
    elif name == "named_color":
        return globals().setdefault(name, _named_color())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if tp.TYPE_CHECKING:
    Back: AnsiBack
    Fore: AnsiFore
    Style: AnsiStyle
    named_color: mappingproxy[str | tuple[str, tp.Literal["4b", "24b"]], Color]
