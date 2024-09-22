from typing import Any, Callable, Literal, Sequence, TypeVar, Union, get_args, get_origin, get_type_hints

from numpy import dtype, ndarray, uint8

type TupleOf3[T] = tuple[T, T, T]
type Float3Tuple = TupleOf3[float]
type Int3Tuple = TupleOf3[int]
type FloatSequence = Sequence[float]
type IntSequence = Sequence[int]
type RGBArrayLike = ndarray[Any, dtype[uint8]]
type RGBVector = Union[Int3Tuple, IntSequence, RGBArrayLike]
ColorDictKeys = Literal['fg', 'bg']
Ansi4BitAlias = Literal['4b']
Ansi8BitAlias = Literal['8b']
Ansi24BitAlias = Literal['24b']
AnsiColorAlias = Ansi4BitAlias | Ansi8BitAlias | Ansi24BitAlias


def is_matching_type(value, expected_type):
    if expected_type is Any:
        return True
    origin, args = deconstruct_type(expected_type)
    if origin is Union:
        return any(is_matching_type(value, arg) for arg in args)
    elif origin is Literal:
        return value in args
    elif isinstance(expected_type, TypeVar):
        if expected_type.__constraints__:
            return any(is_matching_type(value, constraint) for constraint in expected_type.__constraints__)
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
        return is_matching_callable(value, expected_type)
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
        return all(is_matching_type(k, key_type) and is_matching_type(v, val_type) for k, v in value.items())
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
            return isinstance(value, expected_type)
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


def is_matching_typed_dict(__d: dict, typed_dict: type[dict]) -> tuple[bool, str]:
    if not isinstance(__d, dict):
        return False, f"expected {dict.__qualname__}, got {type(__d).__qualname__!r} instead"
    expected = get_type_hints(typed_dict)
    if unexpected := set(__d).difference(expected):
        return False, f"unexpected keyword arguments: {unexpected}"
    if missing := set(k for k in getattr(typed_dict, '__required_keys__', expected.keys()) if k not in __d):
        return False, f"missing required keys: {missing}"
    for name, typ in expected.items():
        if name in __d:
            field = __d[name]
            if not is_matching_type(field, typ):
                return False, (f"expected keyword argument {name!r} to be {typ.__qualname__}, "
                               f"got {type(field).__qualname__!r} instead")
    return True, ''
