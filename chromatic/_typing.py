from __future__ import annotations

import inspect
import operator as op
import re
import types
from collections import OrderedDict, namedtuple
from collections.abc import Callable as ABC_Callable
from functools import reduce, wraps
from numbers import Number
from typing import (
    Any,
    Callable,
    Concatenate,
    Hashable,
    Iterable,
    Literal as L,
    NamedTuple,
    Optional,
    ParamSpec,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    Type,
    TypeAlias,
    TypeAliasType,
    TypeGuard,
    TypeVar,
    TypedDict,
    Union,
    Unpack,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

from PIL.Image import Image
from PIL.ImageFont import FreeTypeFont
from numpy import dtype, float64, generic, ndarray, number, uint8
from numpy._typing import NDArray, _ArrayLike

if TYPE_CHECKING:
    from .data import UserFont

_P = ParamSpec('_P')
_T = TypeVar('_T')
_T_co = TypeVar('_T_co', covariant=True)
_T_contra = TypeVar('_T_contra', contravariant=True)
_AnyNumber_co = TypeVar('_AnyNumber_co', number, Number, covariant=True)

type ArrayReducerFunc[_SCT: generic] = Callable[Concatenate[_ArrayLike[_SCT], _P], NDArray[_SCT]]
type ShapedNDArray[_Shape: tuple[int, ...], _SCT = generic] = ndarray[_Shape, dtype[_SCT]]
type MatrixLike[_SCT: generic] = ShapedNDArray[TupleOf2[int], _SCT]
type SquareMatrix[_I: int, _SCT: generic] = ShapedNDArray[TupleOf2[_I], _SCT]
type GlyphArray[_SCT: generic] = SquareMatrix[L[24], _SCT]
type TupleOf2[_T] = tuple[_T, _T]
type TupleOf3[_T] = tuple[_T, _T, _T]

Float3Tuple: TypeAlias = TupleOf3[float]
Int3Tuple: TypeAlias = TupleOf3[int]
FloatSequence: TypeAlias = Sequence[float]
IntSequence: TypeAlias = Sequence[int]
GlyphBitmask: TypeAlias = GlyphArray[bool]
Bitmask: TypeAlias = MatrixLike[bool]
GreyscaleGlyphArray: TypeAlias = GlyphArray[float64]
GreyscaleArray: TypeAlias = MatrixLike[float64]
RGBArray: TypeAlias = ShapedNDArray[tuple[int, int, L[3]], uint8]
RGBPixel: TypeAlias = ShapedNDArray[tuple[L[3]], uint8]

RGBImageLike: TypeAlias = Image | RGBArray
RGBVectorLike: TypeAlias = IntSequence | RGBPixel
ColorDictKeys = L['fg', 'bg']
Ansi4BitAlias = L['4b']
Ansi8BitAlias = L['8b']
Ansi24BitAlias = L['24b']
AnsiColorAlias = Ansi4BitAlias | Ansi8BitAlias | Ansi24BitAlias
FontArgType: TypeAlias = 'FreeTypeFont | UserFont | str'


def eval_annotation(annotation: str, **kwargs) -> Any:
    globals_ = kwargs.get('globals', {}) | globals()
    locals_ = kwargs.get('locals', {})
    try:
        return subtype(eval(annotation, globals_.copy(), locals_.copy()))
    except NameError as e:
        try:
            import typing

            globals_[e.name] = getattr(typing, e.name)
            return eval_annotation(annotation, globals=globals_, locals=locals_)
        except AttributeError:
            pass
        raise


def type_error_msg(err_obj, *expected, context: str = '', obj_repr=False):
    n_expected = len(expected)
    name_slots = ["{%d.__name__!r}" % n for n in range(n_expected)]
    if n_expected > 1:
        name_slots[-1] = f"or {name_slots[-1]}"
    names = (', ' if n_expected > 2 else ' ').join([context.strip(), *name_slots]).format(*expected)
    if not obj_repr:
        if not isinstance(err_obj, type):
            err_obj = type(err_obj)
        oops = repr(err_obj.__qualname__)
    elif not isinstance(err_obj, str):
        oops = repr(err_obj)
    else:
        oops = err_obj
    return f"expected {names}, got {oops} instead"


def is_matching_type(value, typ):
    if typ is Any:
        return True
    origin, args = deconstruct_type(typ)
    if origin is Union:
        return any(is_matching_type(value, arg) for arg in args)
    elif origin is L:
        return value in args
    elif isinstance(typ, TypeVar):
        if typ.__constraints__:
            return any(is_matching_type(value, constraint) for constraint in typ.__constraints__)
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


def is_matching_typed_dict(__d: dict, typed_dict: type[dict]) -> tuple[bool, str]:
    if TypedDict not in getattr(__d, '__orig_bases__', ()):
        return False, type_error_msg(__d, dict)
    expected = get_type_hints(typed_dict)
    if unexpected := __d.keys() - expected:
        return False, f"unexpected keyword arguments: {unexpected}"
    if missing := set(getattr(typed_dict, '__required_keys__', expected)) - __d.keys():
        return False, f"missing required keys: {missing}"
    for name, typ in expected.items():
        field = __d.get(name)
        if field is None or is_matching_type(field, typ):
            continue
        return False, type_error_msg(field, typ, context=f'keyword argument {name!r} of type')
    return True, ''


def is_matching_callable(value, expected_type):
    return callable(value) and value is expected_type


def deconstruct_type(tp):
    origin = get_origin(tp) or tp
    args = get_args(tp)
    return origin, args


@runtime_checkable
class SupportsUnion(Protocol[_T_contra, _T_co]):

    def __or__(self, x: _T_contra, /) -> _T_co: ...


def unionize(__iterable: Iterable[SupportsUnion[_T_contra, _T_co]]) -> _T_co:
    return reduce(op.or_, __iterable)


_GenericAlias = type(Type[...]) | types.GenericAlias
_UnionGenericType = type(Union[..., None])
_LiteralGenericType = type(L[''])
_CallableGenericType = type(Callable[[], ...]) | type(ABC_Callable[[], ...])
_CallableType = type(Callable) | ABC_Callable


class _BoundedDict[_KT, _VT](OrderedDict[_KT, _VT]):
    """Bounded OrderedDict, mimics FIFO behavior of `functools.lru_cache`"""

    def __init__(self, *, maxsize: Optional[int] = 128):
        super().__init__()
        if maxsize is not None:
            maxsize = max(maxsize, 0)

            @wraps(_BoundedDict.__setitem__)
            def _fifo(_, key, value):
                if maxsize <= len(self):
                    self.popitem(last=True)
                super(_BoundedDict, self).__setitem__(key, value)

            self.__setitem__ = _fifo

    __repr__ = dict.__repr__


_SUBTYPE_CACHE: _BoundedDict[int, ...] = _BoundedDict()
_ATTR_GETTERS: _BoundedDict[..., tuple[Callable[[Iterable], NamedTuple], op.attrgetter]] = (
    _BoundedDict()
)


def _unique_attrs(obj) -> Optional['NamedTuple']:
    tp = type(obj)
    tp_name: str = tp.__name__
    if tp is type:
        return
    elif tp in _ATTR_GETTERS:
        constructor, getter = _ATTR_GETTERS[tp]
        return constructor(getter(obj))
    ignored_attrs = frozenset({'__module__', '__slots__'})
    attr_names = sorted(
        s
        for t in tp.mro()[:-1]
        for s in set(dir(t)).difference(ignored_attrs, *map(dir, t.mro()[1:]))
        if not callable(getattr(tp, s, None))
    )
    if '__dict__' in attr_names:
        attr_names = sorted(cast(dict[str, ...], obj.__dict__).keys() - ignored_attrs)
    if not attr_names:
        return
    attr_names, field_names = _sort_attrs(obj, tp_name, attr_names)
    tup_name = (
        re.sub(
            r'\b[a-z0-9_]+\b',
            lambda m: ''.join(s.capitalize() for s in m.group(0).split('_')),
            tp_name.strip(),
        )
        + 'Attrs'
    ).strip('_')
    UniqueAttrs = cast(NamedTuple, namedtuple(tup_name, field_names))
    _ATTR_GETTERS[tp] = [constructor, getter] = [
        UniqueAttrs._make,
        op.attrgetter(*attr_names),  # noqa
    ]
    return constructor(getter(obj))


def _sort_attrs(obj, tp_name, attr_names):
    field_names = [name.strip('_') for name in attr_names]
    inf = float('inf')
    try:
        sig = inspect.signature(type(obj))
        indices = (
            dict.fromkeys(field_names, inf) | {p: i for i, p in enumerate(sig.parameters)}
        ).values()
        for names in (attr_names, field_names):
            names.sort(key=dict(zip(names, indices)).__getitem__)
        return attr_names, field_names
    except ValueError:
        maybe_sigs: set[str | None] = set()
        text_signature = '__text_signature__'
        if text_signature in attr_names:
            attr_names.remove(text_signature)
            field_names.remove(text_signature.strip('_'))
            maybe_sigs.add(getattr(obj, text_signature, None))
        elif doc := getattr(obj, '__doc__', None):
            lines = iter(inspect.cleandoc(doc).splitlines(keepends=True))
            no_square_parens = {ord(c): None for c in '[]'}
            sig_start = tp_name + '('
            while True:
                try:
                    line = next(lines)
                    while sig_start not in line:
                        line = next(lines)
                    _, _, params = line.partition(sig_start)
                    params, _, _ = (s.translate(no_square_parens) for s in params.partition(')'))
                    maybe_sigs.add(params)
                except StopIteration:
                    break
    maybe_sigs.discard(None)
    if maybe_sigs:
        if len(maybe_sigs) > 1:
            sig = max(
                maybe_sigs, key=lambda s: sum(1 for sub in s.split(', ') if sub in field_names)
            )
        else:
            sig = maybe_sigs.pop()
        positions = {x: i for i, x in enumerate(sig.split(', ')) if x}
        sorted_field_names = sorted(field_names, key=lambda k: positions.get(k, inf))
        transitions = {idx: sorted_field_names.index(x) for idx, x in enumerate(field_names)}
        field_names = sorted_field_names
        attr_names = [attr_names[k] for k in map(transitions.__getitem__, range(len(attr_names)))]
    for Names in (attr_names, field_names):
        name_attr = next((s for s in Names if s.strip('_') == 'name'), None)
        if name_attr is not None:
            Names.sort(key=name_attr.__eq__, reverse=True)
    return attr_names, field_names


def subtype[_T](typ: _T) -> _T:
    type TypeVarDict = dict[TypeVar, ...]

    def _serialize(__item: tuple[Any, Hashable]):
        key, value = __item
        return _unique_attrs(key), hash(value)

    def _cache_key(tp, tvars: TypeVarDict):
        if tvars:
            value = tp, *sorted(tvars.items(), key=_serialize)
        else:
            value = tp
        return value

    def _inner(tp: ..., tvars: TypeVarDict):
        key = _cache_key(tp, tvars)
        recursive = lambda x: _inner(x, tvars)
        try:
            if key in _SUBTYPE_CACHE:
                return _SUBTYPE_CACHE[key]
            elif tp in tvars:
                return tvars[tp]
        except TypeError:
            pass
        if isinstance(tp, (_UnionGenericType, types.UnionType)):
            args = tp.__args__
            args_list = list(args)
            if (
                literals := [
                    idx for idx, elem in enumerate(args) if isinstance(elem, _LiteralGenericType)
                ]
            ) and len(literals) > 1:

                def _next_args(__index: int):
                    idx = args_list.index(args[__index])
                    value = args_list.pop(idx)
                    return getattr(value, '__args__', ())

                start = args[literals.pop(0)]
                args_list[args_list.index(start)] = L[
                    *start.__args__,
                    *dict.fromkeys(arg for idx in literals for arg in _next_args(idx)),
                ]
            try:
                return unionize(map(recursive, args_list))
            except TypeError:
                return Union[*map(recursive, args_list)]
        elif isinstance(tp, TypeAliasType):
            result = recursive(tp.__value__)
        elif isinstance(tp, _CallableGenericType):
            ts, rtype = get_args(tp)
            if isinstance(ts, list):
                ts = list(map(recursive, ts))
            result = ABC_Callable[ts, recursive(rtype)]
        elif isinstance(tp, _GenericAlias):
            origin, args = cast(tuple[types.GenericAlias, tuple], deconstruct_type(tp))
            if origin_params := dict(zip(getattr(origin, '__parameters__', ()), args)):
                if arg_match := origin_params.keys() & tvars:
                    _union = unionize(
                        origin[*map(f, arg_match)]
                        for f in (tvars.__getitem__, origin_params.__getitem__)
                    )
                    key = _cache_key(_union, {})
                    result = _SUBTYPE_CACHE[key] = _inner(_union, {})
                    return result
                for param, arg in origin_params.items():
                    tvars[param] = recursive(arg)
            if isinstance(origin, TypeAliasType):
                result = recursive(origin)
            elif (
                isinstance(origin, type)
                and issubclass(origin, tuple)
                and len(args) == 2
                and args[-1] is Ellipsis
            ):
                result = origin[recursive(args[0]), ...]
            else:
                try:
                    result = origin[*map(recursive, args)]
                except TypeError:
                    if origin is Unpack or origin is TypeGuard:
                        result = origin[recursive(args[0])]
                    else:
                        raise
        elif tp is Ellipsis:
            return Any
        elif tp is None or tp is types.NoneType:
            return None
        else:
            return tp
        _SUBTYPE_CACHE[key] = result
        return result

    return _inner(typ, {})
