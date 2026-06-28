import collections.abc as abc
import sys
import types
import typing as tp
from functools import reduce
from typing import Literal as L

import numpy as np
from numpy._typing import NDArray, _ArrayLike
from PIL.Image import Image
from PIL.ImageFont import FreeTypeFont

if tp.TYPE_CHECKING:
    from .data import UserFont

type ArrayReducerFunc[_SCT: np.generic, **_P] = abc.Callable[
    tp.Concatenate[_ArrayLike[_SCT], _P], NDArray[_SCT]
]
if sys.version_info >= (3, 13):
    type ShapedNDArray[_Shape: tuple[int, ...], _SCT = np.generic] = np.ndarray[
        _Shape, np.dtype[_SCT]
    ]
else:
    type ShapedNDArray[_Shape: tuple[int, ...], _SCT: np.generic] = np.ndarray[
        _Shape, np.dtype[_SCT]
    ]
type MatrixLike[_SCT: np.generic] = ShapedNDArray[TupleOf2[int], _SCT]
type SquareMatrix[_I: int, _SCT: np.generic] = ShapedNDArray[TupleOf2[_I], _SCT]
type GlyphArray[_SCT: np.generic] = SquareMatrix[L[24], _SCT]
type TupleOf2[_T] = tuple[_T, _T]
type TupleOf3[_T] = tuple[_T, _T, _T]
type TupleOf4[_T] = tuple[_T, _T, _T, _T]

Float3Tuple: tp.TypeAlias = TupleOf3[float]
Int3Tuple: tp.TypeAlias = TupleOf3[int]
FloatSequence: tp.TypeAlias = abc.Sequence[float]
IntSequence: tp.TypeAlias = abc.Sequence[int]
GlyphBitmask: tp.TypeAlias = GlyphArray[np.bool_]
Bitmask: tp.TypeAlias = MatrixLike[np.bool_]
GreyscaleGlyphArray: tp.TypeAlias = GlyphArray[np.float64]
GreyscaleArray: tp.TypeAlias = MatrixLike[np.float64]
RGBArray: tp.TypeAlias = ShapedNDArray[tuple[int, int, L[3]], np.uint8]
RGBPixel: tp.TypeAlias = ShapedNDArray[tuple[L[3]], np.uint8]

RGBImageLike: tp.TypeAlias = Image | RGBArray
RGBVectorLike: tp.TypeAlias = IntSequence | RGBPixel
ColorDictKeys = L['fg', 'bg']
Ansi4BitAlias = L['4b']
Ansi8BitAlias = L['8b']
Ansi24BitAlias = L['24b']
AnsiColorAlias = Ansi4BitAlias | Ansi8BitAlias | Ansi24BitAlias
FontArgType: tp.TypeAlias = 'FreeTypeFont | UserFont | str'


def type_error_msg(err_obj, *expected, context: str = '', obj_repr=False):
    n_expected = len(expected)
    name_slots = ["{%d.__name__!r}" % n for n in range(n_expected)]
    if n_expected > 1:
        name_slots[-1] = f"or {name_slots[-1]}"
    names = (
        (', ' if n_expected > 2 else ' ')
        .join([context.strip(), *name_slots])
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


def unionize[_T: type](iterable: abc.Iterable[_T], /) -> types.UnionType | _T:
    return reduce(type.__or__, iterable)


class TypedDictMatcher[_T: tp.TypedDict]:
    def __init__(self, typeddict: type[_T], /):
        if not tp.is_typeddict(typeddict):
            raise ValueError(f"not a TypedDict: {typeddict}")
        self.required = typeddict.__required_keys__
        self.optional = typeddict.__optional_keys__
        self.annotations = types.MappingProxyType(tp.get_type_hints(typeddict))

    def keys(self):
        return frozenset(self.required | self.optional)

    def match(self, mapping: abc.Mapping[str, tp.Any], /) -> tp.TypeGuard[_T]:
        return (
            mapping.keys() <= self.keys()
            and all(
                k in mapping and isinstance(mapping[k], self.annotations[k])
                for k in self.required
            )
            and all(
                isinstance(mapping[k], self.annotations[k])
                for k in self.optional
                if k in mapping
            )
        )
