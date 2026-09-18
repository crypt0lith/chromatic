import cProfile
import functools
import io
import pstats
import time
from inspect import isbuiltin, signature
from types import FunctionType
from typing import Any, Callable


def coerce_argspec[**P, R](
    f: Callable[P, R] | FunctionType | type,
    args: P.args = None,
    kwargs: P.kwargs = None,
    *,
    retfunc: bool = False,
) -> Callable[[], R] | tuple[P.args, P.kwargs]:
    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()
    if isbuiltin(f) or getattr(f, "__module__", "") == "builtins":
        if not isinstance(args, tuple):
            args = tuple([args])
    else:
        try:
            sig = signature(f)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            args, kwargs = bound_args.args, bound_args.kwargs
        except (TypeError, ValueError):
            if not isinstance(args, tuple):
                args = tuple([args])
    if retfunc is True:
        return lambda: f(*args, **kwargs)
    return args, kwargs


class cprofile_wrapper:

    def __init__(self, *, number=10000, use_perf_counter=False):
        self.number = number
        self.use_perf_counter = use_perf_counter

    def __call__[**P](self, f: Callable[P, Any], /) -> Callable[P, None]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
            profiler_kwargs = {}
            if self.use_perf_counter:
                profiler_kwargs["timer"] = time.perf_counter
            profiler = cProfile.Profile(**profiler_kwargs)
            profiler.enable()
            for _ in range(max(self.number, 0)):
                f(*args, **kwargs)
            profiler.disable()
            out_stream = io.StringIO()
            p = pstats.Stats(profiler, stream=out_stream).sort_stats("cumulative")
            p.print_stats()
            print(out_stream.getvalue())
            out_stream.close()

        return wrapper
