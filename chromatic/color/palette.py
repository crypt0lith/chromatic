__all__ = ['Back', 'ColorNamespace', 'Fore', 'Style', 'rgb_dispatch', 'named_color']

from functools import lru_cache, update_wrapper
from inspect import getfullargspec, isbuiltin, signature
from typing import Iterable, Iterator, Mapping, Never, TYPE_CHECKING, Union, final

from .colorconv import ANSI_4BIT_RGB
from .core import Color, ColorStr, SgrParameter, color_chain
from .._typing import Int3Tuple

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem


class DynamicNSMeta[_VT](type):
    @classmethod
    def __class_getitem__(mcls, _):
        return mcls

    @classmethod
    def __prepare__(mcls, clsname, bases, /, **kwds):
        return {'__members__': ()}

    def __new__(mcls, clsname, bases, namespace, /, **kwds):
        namespace['__members__'] = tuple(
            dict.fromkeys(
                member
                for base in bases
                if isinstance(base, mcls)
                for member in base.__members__
            )
        )
        if kwds:
            keys = kwds.keys()
            if not keys <= {'iterable', 'member_type'}:
                raise ValueError(
                    f"unexpected keywords: {(keys - {'iterable', 'member_type'})}"
                )
            if keys == {'iterable', 'member_type'}:
                raise ValueError("cannot use keywords 'iterable' with 'member_type'")
            key, value = kwds.popitem()
            if key == 'iterable':
                ann = namespace.get('__annotations__', {})
                member_names = set(namespace['__members__']).union(
                    k for k, v in ann.items() if (k not in namespace and v is _Member)
                )
                namespace['__members__'] = tuple(
                    sorted(member_names, key=(*ann, *namespace['__members__']).index)
                )
                iterable = value
                if not isinstance(iterable, (Iterable, Mapping)):
                    raise TypeError(
                        "expected 'iterable' to be iterable object, "
                        f"got {type(iterable).__name__!r} instead"
                    )
                elif isinstance(iterable, Mapping):
                    if iterable.keys() != member_names:
                        if member_names.isdisjoint(iterable):
                            raise ValueError(
                                "mapping contains unexpected keys: "
                                + str(iterable.keys() - member_names)
                            )
                        raise ValueError
                    namespace |= iterable
                else:
                    namespace.update(
                        zip(namespace['__members__'], iterable, strict=True)
                    )
            else:
                assert key == 'member_type', f"expected key='member_type', got {key=}"
                member_func = value
                if not callable(member_func):
                    raise ValueError(
                        "expected 'member_type' to be type or callable object, "
                        f"got {type(member_func).__name__!r} object instead"
                    )
                namespace.update(
                    (
                        member,
                        member_func(
                            namespace[member]
                            if member in namespace
                            else next(
                                getattr(base, member)
                                for base in bases
                                if isinstance(base, mcls) and hasattr(base, member)
                            )
                        ),
                    )
                    for member in namespace['__members__']
                )

        return super().__new__(mcls, clsname, bases, namespace)

    def __init__(cls, clsname, bases, namespace, /, **__):
        super().__init__(clsname, bases, namespace)

    def asdict(cls):
        return {member: getattr(cls, member) for member in cls.__members__}


class DynamicNamespace[_T](metaclass=DynamicNSMeta[_T]): ...


@final
class _Member[_T]:
    def __new__(cls: type[_T]) -> _T: ...


def _gen_named_color_values() -> Iterator[int]:
    yield from [
        0x000000, 0x696969, 0x808080, 0xA9A9A9, 0xC0C0C0, 0xD3D3D3, 0xF5F5F5, 0xFFFFFF, 0x800000,
        0x8B0000, 0xFF0000, 0xB22222, 0xA52A2A, 0xCD5C5C, 0xF08080, 0xBC8F8F, 0xFFE4E1, 0xFFFAFA,
        0xA0522D, 0xFF4500, 0xFF6347, 0xEA7E5D, 0xFF7F50, 0xFA8072, 0xE9967A, 0xFFA07A, 0xFFF5EE,
        0x8B4513, 0xD2691E, 0xCD853F, 0xF4A460, 0xFFDAB9, 0xFAF0E6, 0xFF8C00, 0xDEB887, 0xFFE4C4,
        0xFAEBD7, 0xFFA500, 0xD2B48C, 0xF5DEB3, 0xFFDEAD, 0xFFE4B5, 0xFFEBCD, 0xFFEFD5, 0xFDF5E6,
        0xFFFAF0, 0xB8860B, 0xDAA520, 0xFFF8DC, 0xBDB76B, 0xFFD700, 0xF0E68C, 0xEEE8AA, 0xF5F5DC,
        0xFAFAD2, 0xFFFACD, 0x808000, 0xFFFF00, 0xFFFFE0, 0xFFFFF0, 0x006400, 0x008000, 0x556B2F,
        0x228B22, 0x6B8E23, 0x32CD32, 0x8FBC8F, 0x00FF00, 0x9ACD32, 0x7CFC00, 0x7FFF00, 0x90EE90,
        0xADFF2F, 0x98FB98, 0xF0FFF0, 0x2E8B57, 0x3CB371, 0x00FF7F, 0xF5FFFA, 0x2F4F4F, 0x008080,
        0x008B8B, 0x20B2AA, 0x48D1CC, 0x66CDAA, 0x40E0D0, 0x00FA9A, 0x00FFFF, 0xAFEEEE, 0x7FFFD4,
        0xE0FFFF, 0xF0FFFF, 0x4682B4, 0x5F9EA0, 0x00BFFF, 0x00CED1, 0x87CEEB, 0x87CEFA, 0xADD8E6,
        0xB0E0E6, 0xF0F8FF, 0x191970, 0x4169E1, 0x708090, 0x1E90FF, 0x778899, 0x6495ED, 0xB0C4DE,
        0xE6E6FA, 0x000080, 0x00008B, 0x0000CD, 0x0000FF, 0xF8F8FF, 0x4B0082, 0x9400D3, 0x483D8B,
        0x663399, 0x8A2BE2, 0x9932CC, 0x6A5ACD, 0xBA55D3, 0x7B68EE, 0x9370DB, 0xD8BFD8, 0x800080,
        0x8B008B, 0xC71585, 0xFF00FF, 0xFF1493, 0xDA70D6, 0xFF69B4, 0xEE82EE, 0xDDA0DD, 0xFFF0F5,
        0xDC143C, 0xDB7093, 0xFFB6C1, 0xFFC0CB  # fmt: skip
    ]


class ColorNamespace(
    DynamicNamespace[Color], iterable=map(Color, _gen_named_color_values())
):
    BLACK: _Member
    DIM_GREY: _Member
    GREY: _Member
    DARK_GREY: _Member
    SILVER: _Member
    LIGHT_GREY: _Member
    WHITE_SMOKE: _Member
    WHITE: _Member
    MAROON: _Member
    DARK_RED: _Member
    RED: _Member
    FIREBRICK: _Member
    BROWN: _Member
    INDIAN_RED: _Member
    LIGHT_CORAL: _Member
    ROSY_BROWN: _Member
    MISTY_ROSE: _Member
    SNOW: _Member
    SIENNA: _Member
    ORANGE_RED: _Member
    TOMATO: _Member
    BURNT_SIENNA: _Member
    CORAL: _Member
    SALMON: _Member
    DARK_SALMON: _Member
    LIGHT_SALMON: _Member
    SEASHELL: _Member
    SADDLE_BROWN: _Member
    CHOCOLATE: _Member
    PERU: _Member
    SANDY_BROWN: _Member
    PEACH_PUFF: _Member
    LINEN: _Member
    DARK_ORANGE: _Member
    BURLY_WOOD: _Member
    BISQUE: _Member
    ANTIQUE_WHITE: _Member
    ORANGE: _Member
    TAN: _Member
    WHEAT: _Member
    NAVAJO_WHITE: _Member
    MOCCASIN: _Member
    BLANCHED_ALMOND: _Member
    PAPAYA_WHIP: _Member
    OLD_LACE: _Member
    FLORAL_WHITE: _Member
    DARK_GOLDENROD: _Member
    GOLDENROD: _Member
    CORNSILK: _Member
    DARK_KHAKI: _Member
    GOLD: _Member
    KHAKI: _Member
    PALE_GOLDENROD: _Member
    BEIGE: _Member
    LIGHT_GOLDENROD_YELLOW: _Member
    LEMON_CHIFFON: _Member
    OLIVE: _Member
    YELLOW: _Member
    LIGHT_YELLOW: _Member
    IVORY: _Member
    DARK_GREEN: _Member
    GREEN: _Member
    DARK_OLIVE_GREEN: _Member
    FOREST_GREEN: _Member
    OLIVE_DRAB: _Member
    LIME_GREEN: _Member
    DARK_SEA_GREEN: _Member
    LIME: _Member
    YELLOW_GREEN: _Member
    LAWN_GREEN: _Member
    CHARTREUSE: _Member
    LIGHT_GREEN: _Member
    GREEN_YELLOW: _Member
    PALE_GREEN: _Member
    HONEYDEW: _Member
    SEA_GREEN: _Member
    MEDIUM_SEA_GREEN: _Member
    SPRING_GREEN: _Member
    MINT_CREAM: _Member
    DARK_SLATE_GREY: _Member
    TEAL: _Member
    DARK_CYAN: _Member
    LIGHT_SEA_GREEN: _Member
    MEDIUM_TURQUOISE: _Member
    MEDIUM_AQUAMARINE: _Member
    TURQUOISE: _Member
    MEDIUM_SPRING_GREEN: _Member
    CYAN: _Member
    PALE_TURQUOISE: _Member
    AQUAMARINE: _Member
    LIGHT_CYAN: _Member
    AZURE: _Member
    STEEL_BLUE: _Member
    CADET_BLUE: _Member
    DEEP_SKY_BLUE: _Member
    DARK_TURQUOISE: _Member
    SKY_BLUE: _Member
    LIGHT_SKY_BLUE: _Member
    LIGHT_BLUE: _Member
    POWDER_BLUE: _Member
    ALICE_BLUE: _Member
    MIDNIGHT_BLUE: _Member
    ROYAL_BLUE: _Member
    SLATE_GREY: _Member
    DODGER_BLUE: _Member
    LIGHT_SLATE_GREY: _Member
    CORNFLOWER_BLUE: _Member
    LIGHT_STEEL_BLUE: _Member
    LAVENDER: _Member
    NAVY: _Member
    DARK_BLUE: _Member
    MEDIUM_BLUE: _Member
    BLUE: _Member
    GHOST_WHITE: _Member
    INDIGO: _Member
    DARK_VIOLET: _Member
    DARK_SLATE_BLUE: _Member
    REBECCA_PURPLE: _Member
    BLUE_VIOLET: _Member
    DARK_ORCHID: _Member
    SLATE_BLUE: _Member
    MEDIUM_ORCHID: _Member
    MEDIUM_SLATE_BLUE: _Member
    MEDIUM_PURPLE: _Member
    THISTLE: _Member
    PURPLE: _Member
    DARK_MAGENTA: _Member
    MEDIUM_VIOLET_RED: _Member
    FUCHSIA: _Member
    DEEP_PINK: _Member
    ORCHID: _Member
    HOT_PINK: _Member
    VIOLET: _Member
    PLUM: _Member
    LAVENDER_BLUSH: _Member
    CRIMSON: _Member
    PALE_VIOLET_RED: _Member
    LIGHT_PINK: _Member
    PINK: _Member


def style():
    for x in SgrParameter:
        if x not in {38, 48}:
            yield color_chain([x])


class AnsiStyle(DynamicNamespace[color_chain], iterable=style()):
    RESET: _Member
    BOLD: _Member
    FAINT: _Member
    ITALICS: _Member
    SINGLE_UNDERLINE: _Member
    SLOW_BLINK: _Member
    RAPID_BLINK: _Member
    NEGATIVE: _Member
    CONCEALED_CHARS: _Member
    CROSSED_OUT: _Member
    PRIMARY: _Member
    FIRST_ALT: _Member
    SECOND_ALT: _Member
    THIRD_ALT: _Member
    FOURTH_ALT: _Member
    FIFTH_ALT: _Member
    SIXTH_ALT: _Member
    SEVENTH_ALT: _Member
    EIGHTH_ALT: _Member
    NINTH_ALT: _Member
    GOTHIC: _Member
    DOUBLE_UNDERLINE: _Member
    RESET_BOLD_AND_FAINT: _Member
    RESET_ITALIC_AND_GOTHIC: _Member
    RESET_UNDERLINES: _Member
    RESET_BLINKING: _Member
    POSITIVE: _Member
    REVEALED_CHARS: _Member
    RESET_CROSSED_OUT: _Member
    BLACK_FG: _Member
    RED_FG: _Member
    GREEN_FG: _Member
    YELLOW_FG: _Member
    BLUE_FG: _Member
    MAGENTA_FG: _Member
    CYAN_FG: _Member
    WHITE_FG: _Member
    DEFAULT_FG_COLOR: _Member
    BLACK_BG: _Member
    RED_BG: _Member
    GREEN_BG: _Member
    YELLOW_BG: _Member
    BLUE_BG: _Member
    MAGENTA_BG: _Member
    CYAN_BG: _Member
    WHITE_BG: _Member
    DEFAULT_BG_COLOR: _Member
    FRAMED: _Member
    ENCIRCLED: _Member
    OVERLINED: _Member
    NOT_FRAMED_OR_CIRCLED: _Member
    IDEOGRAM_UNDER_OR_RIGHT: _Member
    IDEOGRAM_2UNDER_OR_2RIGHT: _Member
    IDEOGRAM_OVER_OR_LEFT: _Member
    IDEOGRAM_2OVER_OR_2LEFT: _Member
    CANCEL: _Member
    BLACK_BRIGHT_FG: _Member
    RED_BRIGHT_FG: _Member
    GREEN_BRIGHT_FG: _Member
    YELLOW_BRIGHT_FG: _Member
    BLUE_BRIGHT_FG: _Member
    MAGENTA_BRIGHT_FG: _Member
    CYAN_BRIGHT_FG: _Member
    WHITE_BRIGHT_FG: _Member
    BLACK_BRIGHT_BG: _Member
    RED_BRIGHT_BG: _Member
    GREEN_BRIGHT_BG: _Member
    YELLOW_BRIGHT_BG: _Member
    BLUE_BRIGHT_BG: _Member
    MAGENTA_BRIGHT_BG: _Member
    CYAN_BRIGHT_BG: _Member
    WHITE_BRIGHT_BG: _Member


def background(__x: Color):
    return color_chain(bg=__x, ansi_type='24b')


def foreground(__x: Color):
    return color_chain(fg=__x, ansi_type='24b')


class AnsiBack(ColorNamespace, member_type=background):
    RESET = AnsiStyle.DEFAULT_BG_COLOR

    def __call__(self, bg: Union[Color, int, tuple[int, int, int]]):
        return color_chain(bg=bg)


class AnsiFore(ColorNamespace, member_type=foreground):
    RESET = AnsiStyle.DEFAULT_FG_COLOR

    def __call__(self, fg: Union[Color, int, tuple[int, int, int]]):
        return color_chain(fg=fg)


class _color_ns_getter:
    mapping = (
        {name.casefold(): color.rgb for name, color in ColorNamespace.asdict().items()}
        .items()
        .mapping
    )

    def __str__(self):
        return str(
            {
                str(ColorStr(k, fg=v, ansi_type='24b')): v
                for k, v in type(self).mapping.items()
            }
        )

    @staticmethod
    @lru_cache
    def _normalize_key(__key: str):
        return __key.translate({0x20: 0x5F}).casefold()

    def __contains__(self, __key):
        if type(__key) is str:
            return self._normalize_key(__key) in self.mapping
        return False

    def keys(self):
        return self.mapping.keys()

    def __getitem__(self, __key: str):
        return self.mapping[self._normalize_key(__key)]

    def __getattr__(self, __name):
        try:
            return getattr(self.mapping, __name)
        except AttributeError as e:
            msg = str(e)
            msg = msg.replace(type(self.mapping).__name__, type(self).__name__)
            err = AttributeError(msg)
            err.__cause__ = e.__cause__
            raise err


def rgb_dispatch(*names: str):
    def decorator(__f):
        def fix_signature(__f):
            from .._typing import eval_annotation

            nonlocal names, rgb_args, variadic
            try:
                argspec = getfullargspec(__f)
                sig = signature(__f)
            except TypeError:
                if not (isbuiltin(__f) or getattr(__f, '__module__', '') == 'builtins'):
                    raise
                return signature(lambda *args, **kwargs: ...)
            variadic = {
                name for name in [argspec.varargs, argspec.varkw] if name is not None
            }
            all_arg_names = variadic.union(argspec.args + argspec.kwonlyargs)
            rgb_args = all_arg_names.intersection(
                dict.get({'*': argspec.varargs, '**': argspec.varkw}, arg, arg)
                for arg in names
            )
            eitherwith = lambda s, *args: str.startswith(s, *args) or str.endswith(
                s, *args
            )
            if not rgb_args:
                for name in all_arg_names:
                    if eitherwith(name, ('fg', 'bg')):
                        rgb_args.add(name)
            variadic &= rgb_args
            parameters = []
            for name, p in sig.parameters.items():
                if name not in rgb_args or p.annotation is p.empty:
                    parameters.append(p)
                else:
                    annotation = p.annotation
                    try:
                        annotation |= str
                    except TypeError:
                        union_repr = f"{annotation} | str"
                        try:
                            annotation = eval_annotation(
                                union_repr, globals=__f.__globals__
                            )
                        except NameError:
                            annotation = union_repr
                    parameters.append(p.replace(annotation=annotation))
            return sig.replace(parameters=parameters)

        def wrapper(*args, **kwargs):
            bound = wrapper_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            for name, value in bound.arguments.items():
                if name not in rgb_args:
                    continue
                if name in variadic:
                    bound.arguments[name] = (
                        tuple(color_ns[v] if v in color_ns else v for v in value)
                        if isinstance(value, tuple)
                        else {
                            k: color_ns[v] if v in color_ns else v
                            for k, v in value.items()
                        }
                    )
                elif value in color_ns:
                    bound.arguments[name] = color_ns[value]
            return __f(*bound.args, **bound.kwargs)

        color_ns: SupportsKeysAndGetItem[str, Int3Tuple] = _color_ns_getter()
        rgb_args, variadic = set[str](), set[str]()
        wrapper_sig = fix_signature(__f)
        if not hasattr(__f, '__text_signature__'):
            setattr(__f, '__signature__', wrapper_sig)
        return update_wrapper(wrapper, __f)

    if names and callable(names[0]):
        func, *names = names
    else:
        func = None
    for x in names:
        if type(x) is not str:
            raise TypeError(
                f"found {type(x).__name__!r} object in names, expected only str"
            )
    return decorator if func is None else decorator(func)


def _make_named_color_map() -> ...:
    class NamedColorMapping(dict):
        def __setitem__(self, *args: Never):
            raise NotImplementedError

        def __getitem__(self, __key: tuple[str, str]):
            try:
                k1, k2 = __key
                if type(k2) is not str:
                    raise TypeError
                return super().__getitem__((k1, k2.upper()))
            except (TypeError, KeyError, ValueError):
                pass
            raise KeyError(__key)

    return NamedColorMapping(
        ((k1, k2), rgb)
        for k1, items in dict.items(
            {
                '4b': dict.items(
                    dict(
                        zip(
                            [
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
                            ],
                            map(Color.from_rgb, ANSI_4BIT_RGB),
                        )
                    )
                ),
                '24b': ColorNamespace.asdict().items(),
            }
        )
        for k2, rgb in items
    )


named_color = _make_named_color_map()


def named_color_idents():
    return [
        ColorStr(name.translate({0x5F: 0x20}).lower(), color, ansi_type='24b')
        for name, color in ColorNamespace.asdict().items()
    ]


def __getattr__(name: str) -> ...:
    try:
        return globals().setdefault(
            name, {'Back': AnsiBack, 'Fore': AnsiFore, 'Style': AnsiStyle}[name]()
        )
    except KeyError:
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    Back: AnsiBack
    Fore: AnsiFore
    Style: AnsiStyle
