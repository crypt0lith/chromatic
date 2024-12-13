from functools import lru_cache, update_wrapper, wraps
from inspect import getfullargspec, signature
from types import FunctionType, MappingProxyType
from typing import (
    Callable, cast, dataclass_transform, Iterable, Iterator, Sequence, TYPE_CHECKING, TypedDict,
    Union,
    Unpack
)

from .colorconv import hex2rgb
from .core import (
    AnsiColorFormat, Color, ColorStr, DEFAULT_ANSI, get_ansi_type, SgrParameter, SgrSequence
)
from .._typing import AnsiColorAlias, Int3Tuple

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
        return f"<{type(self).__name__} {self.name!r} of {self.clsname!r}>"


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
            is_member = _check_if_ns_member(cls)
            slots = kwargs.pop('slots', list(filter(is_member, cls.__annotations__)))
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
                f"{cls.__name__!r} does not inherit {DynamicNamespace.__name__!r} as a "
                f"base class "
                f"and does not provide callable 'factory' keyword argument")
        base: type[DynamicNamespace] = cast(
            type[...], next(
                (typ for typ in cls.mro()
                 if DynamicNamespace in typ.__bases__), null))
        if base is null:
            raise TypeError(
                f"{cls.__qualname__!r} does not have any base classes of "
                f"type {DynamicNamespace.__qualname__!r}") from None
        d = dict(zip(base.__annotations__, map(factory, base().__members__)))
        cls.__annotations__: dict[str, ...] = {k: type(v) for k, v in d.items()}
        new = cls.__new__

        @wraps(cls.__new__)
        def wrapped_new(*args, **kw):
            return new(*args, **(kw | dict(slots=d)))

        @wraps(cls.__init__)
        def wrapped_init(*args, **kw):
            return DynamicNamespace.__init__(*args, **(kw | d))

        cls.__new__ = wrapped_new
        cls.__init__ = wrapped_init
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


def _check_if_ns_member(cls: type) -> Callable[[str], bool]:
    type_params = cls.__type_params__
    if type_params and len(type_params) == 1:
        anno_dict = cls.__annotations__
        member_type = type_params[0]

        def f(x: str):
            return member_type == anno_dict.get(x)

        return f
    else:
        return lambda _: False


def _ns_from_iter[_KT, _VT](
    __iter: Iterator[_KT] | Callable[[], Iterator[_KT]], member_type: _VT = null
) -> Callable[[type[DynamicNamespace[_VT]]], type[DynamicNamespace[_VT]]]:
    def decorator(cls: type[DynamicNamespace[_VT]]):
        anno = cls.__annotations__
        type_params = cls.__type_params__
        m_iter = __iter() if callable(__iter) else iter(__iter)
        members: Iterator[_KT] = m_iter if member_type == null else map(member_type, m_iter)
        d = dict(zip((k for k, v in anno.items() if v in type_params), members))

        @wraps(cls.__init__)
        def wrapped(*args, **kwargs):
            return cls.__base__.__init__(*args, **(kwargs | d))

        cls.__init__ = wrapped
        return cls

    return decorator


def _gen_named_color_values() -> Iterator[int]:
    yield from [
        0x000000, 0x696969, 0x808080, 0xa9a9a9, 0xc0c0c0, 0xd3d3d3, 0xf5f5f5, 0xffffff, 0x800000,
        0x8b0000, 0xff0000, 0xb22222, 0xa52a2a, 0xcd5c5c, 0xf08080, 0xbc8f8f, 0xffe4e1, 0xfffafa,
        0xa0522d, 0xff4500, 0xff6347, 0xea7e5d, 0xff7f50, 0xfa8072, 0xe9967a, 0xffa07a, 0xfff5ee,
        0x8b4513, 0xd2691e, 0xcd853f, 0xf4a460, 0xffdab9, 0xfaf0e6, 0xff8c00, 0xdeb887, 0xffe4c4,
        0xfaebd7, 0xffa500, 0xd2b48c, 0xf5deb3, 0xffdead, 0xffe4b5, 0xffebcd, 0xffefd5, 0xfdf5e6,
        0xfffaf0, 0xb8860b, 0xdaa520, 0xfff8dc, 0xbdb76b, 0xffd700, 0xf0e68c, 0xeee8aa, 0xf5f5dc,
        0xfafad2, 0xfffacd, 0x808000, 0xffff00, 0xffffe0, 0xfffff0, 0x006400, 0x008000, 0x556b2f,
        0x228b22, 0x6b8e23, 0x32cd32, 0x8fbc8f, 0x00ff00, 0x9acd32, 0x7cfc00, 0x7fff00, 0x90ee90,
        0xadff2f, 0x98fb98, 0xf0fff0, 0x2e8b57, 0x3cb371, 0x00ff7f, 0xf5fffa, 0x2f4f4f, 0x008080,
        0x008b8b, 0x20b2aa, 0x48d1cc, 0x66cdaa, 0x40e0d0, 0x00fa9a, 0x00ffff, 0xafeeee, 0x7fffd4,
        0xe0ffff, 0xf0ffff, 0x4682b4, 0x5f9ea0, 0x00bfff, 0x00ced1, 0x87ceeb, 0x87cefa, 0xadd8e6,
        0xb0e0e6, 0xf0f8ff, 0x191970, 0x4169e1, 0x708090, 0x1e90ff, 0x778899, 0x6495ed, 0xb0c4de,
        0xe6e6fa, 0x000080, 0x00008b, 0x0000cd, 0x0000ff, 0xf8f8ff, 0x4b0082, 0x9400d3, 0x483d8b,
        0x663399, 0x8a2be2, 0x9932cc, 0x6a5acd, 0xba55d3, 0x7b68ee, 0x9370db, 0xd8bfd8, 0x800080,
        0x8b008b, 0xc71585, 0xff00ff, 0xff1493, 0xda70d6, 0xff69b4, 0xee82ee, 0xdda0dd, 0xfff0f5,
        0xdc143c, 0xdb7093, 0xffb6c1, 0xffc0cb]


@_ns_from_iter(_gen_named_color_values, Color)
class ColorNamespace[NamedColor: Color](DynamicNamespace[NamedColor]):
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


# noinspection PyTypedDict
class _ColorStrWrapperKwargs(TypedDict, total=False):
    ansi_type: Union[AnsiColorAlias, type[AnsiColorFormat]]
    bg: Union[Color, tuple[int, int, int], int]
    fg: Union[Color, tuple[int, int, int], int]
    sgr_params: Sequence[Union[int, SgrParameter]]


# noinspection PyUnresolvedReferences
class color_str_wrapper:

    def __init__(self, **kwargs: Unpack[_ColorStrWrapperKwargs]):
        self._rhs_ = kwargs.get('_rhs_', False)
        self._concat_ = kwargs.get('_concat_', '')
        if typ := kwargs.get('ansi_type'):
            self._ansi_type_ = get_ansi_type(typ)
        else:
            self._ansi_type_ = DEFAULT_ANSI
        if params := kwargs.get('sgr_params'):
            self._sgr_ = SgrSequence(params)
        else:
            self._sgr_ = SgrSequence()
        for k in kwargs.keys() & {'fg', 'bg'}:
            if v := kwargs[k]:
                if not isinstance(v, Iterable):
                    v = hex2rgb(v)
                self._sgr_ += SgrSequence(self._ansi_type_.from_rgb({k: v}))

    def __call__(self, __obj=None):
        if type(self) is type(__obj):
            new_sgr = self._sgr_ + __obj._sgr_
            new_kwargs = {
                'ansi_type': self._ansi_type_,
                '_concat_': self.__dict__.get('_concat_', '').removesuffix(
                    str(self._sgr_)) + str(new_sgr),
                '_rhs_': self.__dict__['_rhs_'],
                **new_sgr.rgb_dict}
            new_kwargs['sgr_params'] = [
                int(v._value_) for v in new_sgr if not v.is_color()
            ]
            return color_str_wrapper(**new_kwargs)
        if isinstance(__obj, ColorStr):
            if getattr(self, '_rhs_', False):
                new_kwargs = {
                    'ansi_type': self._ansi_type_,
                    'sgr_params': list(self._sgr_),
                    '_concat_': (
                            getattr(self, '_concat_', '') + __obj
                    ).removesuffix('[0m'),
                    '_rhs_': True,
                    **self._sgr_.rgb_dict
                }
                return color_str_wrapper(**new_kwargs)
            new_params = [v for v in __obj._sgr_.values() if v not in self._sgr_.values()]
            return ColorStr(
                __obj.base_str,
                color_spec=SgrSequence(new_params),
                no_reset=__obj.no_reset,
                ansi_type=self._ansi_type_)
        if getattr(self, '_rhs_', False):
            new_kwargs = {
                'ansi_type': self._ansi_type_,
                'sgr_params': list(self._sgr_),
                '_concat_': (
                        getattr(self, '_concat_', '') + f"{__obj}"
                ).removesuffix('[0m'),
                '_rhs_': True,
                **self._sgr_.rgb_dict
            }
            return color_str_wrapper(**new_kwargs)
        return ColorStr(__obj, color_spec=self._sgr_, ansi_type=self._ansi_type_)

    def __add__(self, other):
        return self.__call__(other)

    def __radd__(self, other):
        if getattr(self, '_rhs_') is False and type(other) is ColorStr:
            setattr(self, '_rhs_', True)
        return self.__call__(other)

    def __str__(self):
        return self.__dict__['_concat_'] + str(self._sgr_)

    def __repr__(self):
        return (f"{type(self).__name__}"
                f"(sgr_params={self._sgr_.values()}, ansi_type={self._ansi_type_.__name__})")

    def __getattr__(self, name):
        if hasattr(str, name):
            return getattr(self.__str__(), name)
        raise AttributeError


def _style_wrappers():
    yield from (
        color_str_wrapper()
        if x in {38, 48}
        else color_str_wrapper(sgr_params=[x])
        for x in SgrParameter
    )


@_ns_from_iter(_style_wrappers)
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


def _bg_wrapper_factory(__x: Color):
    return color_str_wrapper(bg=__x, ansi_type='24b')


def _fg_wrapper_factory(__x: Color):
    return color_str_wrapper(fg=__x, ansi_type='24b')


class AnsiBack(ColorNamespace[color_str_wrapper], factory=_bg_wrapper_factory):
    RESET = getattr(AnsiStyle(), 'DEFAULT_BG_COLOR')

    def __call__(self, bg: Union[Color, int, tuple[int, int, int]]):
        return color_str_wrapper(bg=bg)


class AnsiFore(ColorNamespace[color_str_wrapper], factory=_fg_wrapper_factory):
    RESET = getattr(AnsiStyle(), 'DEFAULT_FG_COLOR')

    def __call__(self, fg: Union[Color, int, tuple[int, int, int]]):
        return color_str_wrapper(fg=fg)


class _color_ns_getter:
    mapping = MappingProxyType(
        {k.casefold(): v.rgb
         for (k, v) in
         ColorNamespace().as_dict().items()})

    def __get__(self, instance, owner: type = None):
        if instance:
            return self
        dummy = type.__new__(type, (cls_name := type(self).__name__), (), {})
        dummy_str = f"<attr {cls_name!r} of {owner.__name__!r} objects>"
        dummy.__str__ = lambda _: dummy_str
        return dummy()

    @staticmethod
    @lru_cache
    def _normalize_key(__key: str):
        return __key.translate({0x20: 0x5f}).casefold()

    def __contains__(self, __key):
        if type(__key) is str:
            return self._normalize_key(__key) in self.mapping
        return False

    def __getitem__(self, __key: str):
        return self.mapping[self._normalize_key(__key)]

    def __getattr__(self, __name):
        try:
            return getattr(self.mapping, __name)
        except AttributeError as e:
            raise AttributeError(
                str(e).replace(
                    *map(
                        lambda x: type(x).__name__,
                        (self.mapping, self)))
            ) from None


def _handle_singleton(__obj: ...):
    return (__obj,) if isinstance(__obj, (str, int)) else __obj


def _scalar_union(s: set, value: object):
    return s.union(_handle_singleton(value))


# noinspection PyUnresolvedReferences
class rgb_dispatch[** P, R]:
    color_ns = cast(
        MappingProxyType[str, Int3Tuple],
        _color_ns_getter())

    def __new__(cls, func: Callable[P, R] = None, /, *, args: Sequence[str | int] = ()):
        args = _handle_singleton(args)
        if func is None:
            return lambda f, **kwargs: (
                cls(f, args=tuple(_scalar_union(set(args), kwargs.get('args', ()))))
            )
        inst = super().__new__(cls)
        setattr(inst, 'func', func)
        getattr(inst, '_init_wrapper')(*args)
        return inst

    def _init_wrapper(
        self,
        *params: *tuple[str | int, ...]
    ):
        if not callable(self.func):
            raise ValueError
        try:
            argspec = getfullargspec(self.func)
            sig = signature(self.func)
        except TypeError:
            if not (getattr(self.func, '__module__', '') == 'builtins'
                    or inspect.isbuiltin(self.func)):
                raise
            generic_spec = lambda *args, **kwargs: ...
            argspec = getfullargspec(generic_spec)
            sig = signature(generic_spec)
        self.variadic = {argspec.varargs, argspec.varkw}
        self.variadic.discard(None)
        all_args = self.variadic.union(argspec.args + argspec.kwonlyargs)
        self.rgb_args = all_args & {
            *params,
            *(v for (s, v) in zip(
                ('*' * x for x in range(1, 3)),
                (argspec.varargs, argspec.varkw))
              if s in params)
        }
        if not self.rgb_args:
            keys = frozenset({'fg', 'bg'})
            for arg in all_args:
                if (arg[:2] in keys) or (arg[-2:] in keys):
                    self.rgb_args.add(arg)
        self.variadic &= self.rgb_args
        self.signature = sig.replace(
            parameters=[
                param.replace(
                    annotation=' | '.join(
                        {'str'}.union(filter(None, f"{param.annotation}".split(' | ')))))
                if name in self.rgb_args and param.annotation is not param.empty
                else param
                for (name, param) in sig.parameters.items()
            ])
        update_wrapper(self, self.func)

    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs
    ) -> R:
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        for arg, value in bound.arguments.items():
            if arg not in self.rgb_args:
                continue
            if arg in self.variadic:
                bound.arguments[arg] = (
                    tuple(
                        self.color_ns[v] if v in self.color_ns else v
                        for v in value)
                    if isinstance(value, tuple) else
                    {k: self.color_ns[v] if v in self.color_ns else v
                     for k, v in value.items()}
                )
            elif value in self.color_ns:
                bound.arguments[arg] = self.color_ns[value]
        return self.func(*bound.args, **bound.kwargs)


def display_named_colors():
    return [ColorStr(name.replace('_', ' ').lower(), color_spec=color, ansi_type='24b')
            for name, color in ColorNamespace().as_dict().items()]


def display_ansi256_color_range():
    from numpy import asarray
    from chromatic.color.colorconv import ansi_8bit_to_rgb

    ansi256_range = asarray(range(256)).reshape([16] * 2).tolist()
    return [[ColorStr(
        obj='###',
        color_spec=ansi_8bit_to_rgb(v),
        ansi_type='8b')
        for v in arr]
        for arr in ansi256_range]


def __getattr__(name: ...) -> ...:
    if name == 'Back':
        return AnsiBack()
    if name == 'Fore':
        return AnsiFore()
    if name == 'Style':
        return AnsiStyle()
    raise AttributeError(
        f"Module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    Back: AnsiBack
    Fore: AnsiFore
    Style: AnsiStyle

__all__ = [
    'Back',
    'ColorNamespace',
    'Fore',
    'Style',
    'color_str_wrapper',
    'rgb_dispatch',
]
