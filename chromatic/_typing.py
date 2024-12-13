from __future__ import annotations

from functools import reduce
from numbers import Number
from types import UnionType
from typing import (
    Any,
    Callable,
    Concatenate,
    get_args,
    get_origin,
    get_type_hints,
    Iterable,
    Literal,
    ParamSpec,
    Protocol,
    Sequence,
    SupportsRound,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

from numpy import dtype, float64, generic, ndarray, number, uint8
from numpy._typing import _ArrayLike, NDArray
from PIL.Image import Image
from PIL.ImageFont import FreeTypeFont

from chromatic.data import UserFont

_P = ParamSpec('_P')
_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)
_T_contra = TypeVar('_T_contra', contravariant=True)
_AnyNumber_co = TypeVar('_AnyNumber_co', number, Number, covariant=True)

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison, SupportsDivMod

    class SupportsRoundAndDivMod(
        SupportsRound[_T_co], SupportsDivMod[Any, _T_co], Protocol
    ): ...


type ArrayReducerFunc[_SCT: generic] = Callable[
    Concatenate[_ArrayLike[_SCT], _P], NDArray[_SCT]
]
type KeyFunction[_T] = Callable[[_T], SupportsRichComparison]
type ShapedNDArray[_Shape: tuple[int, ...], _SCT: generic] = ndarray[
    _Shape, dtype[_SCT]
]
type MatrixLike[_SCT: generic] = ShapedNDArray[TupleOf2[int], _SCT]
type SquareMatrix[_I: int, _SCT: generic] = ShapedNDArray[TupleOf2[_I], _SCT]
type GlyphArray[_SCT: generic] = SquareMatrix[Literal[24], _SCT]
type TupleOf2[_T] = tuple[_T, _T]
type TupleOf3[_T] = tuple[_T, _T, _T]
Float3Tuple = TupleOf3[float]
Int3Tuple = TupleOf3[int]
FloatSequence = Sequence[float]
IntSequence = Sequence[int]
GlyphBitmask = GlyphArray[bool]
Bitmask = MatrixLike[bool]
GreyscaleGlyphArray = GlyphArray[float64]
GreyscaleArray = MatrixLike[float64]
RGBArray = ShapedNDArray[tuple[int, int, Literal[3]], uint8]
RGBPixel = ShapedNDArray[tuple[Literal[3]], uint8]

RGBImageLike = Union[Image, RGBArray]
RGBVectorLike = Union[Int3Tuple, IntSequence, RGBPixel]
ColorDictKeys = Literal['fg', 'bg']
Ansi4BitAlias = Literal['4b']
Ansi8BitAlias = Literal['8b']
Ansi24BitAlias = Literal['24b']
AnsiColorAlias = Ansi4BitAlias | Ansi8BitAlias | Ansi24BitAlias
FontArgType = Union[FreeTypeFont | UserFont, str, int]


def is_matching_type(value, typ):
    if typ is Any:
        return True
    origin, args = deconstruct_type(typ)
    if origin is Union:
        return any(is_matching_type(value, arg) for arg in args)
    elif origin is Literal:
        return value in args
    elif isinstance(typ, TypeVar):
        if typ.__constraints__:
            return any(
                is_matching_type(value, constraint)
                for constraint in typ.__constraints__
            )
        else:
            return True
    elif origin is type:
        if not isinstance(value, type):
            return False
        target_type = args[0]
        target_origin = get_origin(target_type)
        target_args = get_args(target_type)
        if target_origin is Union:
            return any(issubclass(value, t) for t in target_args)
        else:
            return issubclass(value, target_type)
    elif origin is Callable:
        return is_matching_callable(value, typ)
    elif origin is list:
        if not isinstance(value, list):
            return False
        if not args:
            return True
        return all(is_matching_type(item, args[0]) for item in value)
    elif origin is dict:
        if not isinstance(value, dict):
            return False
        if not args:
            return True
        key_type, val_type = args
        return all(
            is_matching_type(k, key_type) and is_matching_type(v, val_type)
            for k, v in value.items()
        )
    elif origin is tuple:
        if not isinstance(value, tuple):
            return False
        if len(args) == 2 and args[1] is ...:
            return all(is_matching_type(item, args[0]) for item in value)
        if len(value) != len(args):
            return False
        return all(is_matching_type(v, t) for v, t in zip(value, args))
    else:
        try:
            return isinstance(value, typ)
        except TypeError:
            return False


def deconstruct_type(tp):
    origin = get_origin(tp) or tp
    args = get_args(tp)
    return origin, args


def is_matching_callable(value, expected_type):
    if not callable(value):
        return False
    return id(value) == id(expected_type)


def pseudo_union(ts: Iterable[type]) -> Union[type, UnionType]:
    return reduce(lambda x, y: x | y, ts)


def type_error_msg(err_obj, *expected, context: str = '', obj_repr=False):
    n_expected = len(expected)
    name_slots = [f"{{{n}.__qualname__!r}}" for n in range(n_expected)]
    if name_slots and n_expected > 1:
        name_slots[-1] = f"or {name_slots[-1]}"
    names = (
        (', ' if n_expected > 2 else ' ')
        .join([context.strip()] + name_slots)
        .format(*expected)
    )
    if not obj_repr:
        if not isinstance(err_obj, type):
            err_obj = type(err_obj)
        oops = repr(err_obj.__qualname__)
    elif not isinstance(err_obj, str):
        oops = repr(err_obj)
    else:
        oops = err_obj
    return f"expected {names}, got {oops} instead"


def is_matching_typed_dict(__d: dict, typed_dict: type[dict]) -> tuple[bool, str]:
    if not isinstance(__d, dict):
        return False, type_error_msg(__d, dict)
    expected = get_type_hints(typed_dict)
    if unexpected := __d.keys() - expected:
        return False, f"unexpected keyword arguments: {unexpected}"
    if missing := set(getattr(typed_dict, '__required_keys__', expected)) - __d.keys():
        return False, f"missing required keys: {missing}"
    for name, typ in expected.items():
        if ((field := __d.get(name)) is not None) and not is_matching_type(field, typ):
            return False, type_error_msg(
                field, typ, context=f'keyword argument {name!r} of type'
            )
    return True, ''
