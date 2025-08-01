import math
from typing import Callable, Iterable, Iterator, Literal, Sequence, SupportsIndex, cast

import numpy as np

from .colorconv import int2rgb, hsl2rgb, rgb2int, rgb2hsl
from .core import Color
from .._typing import Float3Tuple, Int3Tuple


def _resolve_replacement_indices(
    replace_idx: tuple[SupportsIndex | Sequence[SupportsIndex], Iterator[Color]] = None,
):
    if replace_idx is not None:
        replace_idx, rgb_iter = replace_idx
        if not isinstance(rgb_iter, Iterator):
            raise TypeError(
                f"Expected 'replace_idx[1]' to be an iterator, got {type(rgb_iter).__name__} "
                f"instead"
            )
        if not isinstance(replace_idx, Sequence):
            replace_idx = {replace_idx}
        else:
            replace_idx = set(replace_idx)
        valid_idx_range = range(3)
        if any(idx_diff := replace_idx.difference(valid_idx_range)):
            raise ValueError(f"Invalid replacement indices: {idx_diff}")
        if replace_idx == set(valid_idx_range):
            raise ValueError(f"All 3 indexes selected for replacement: {replace_idx=}")
    else:
        rgb_iter = None
        replace_idx = []
    return replace_idx, rgb_iter


def _init_gradient_color_vec(
    num: SupportsIndex,
    start: Int3Tuple | Float3Tuple,
    step: SupportsIndex,
    stop: Int3Tuple | Float3Tuple,
):
    def convert_bounds(rgb: Int3Tuple):
        if all(0 <= n <= 255 for n in rgb):
            return rgb2hsl(rgb)
        raise ValueError

    start, stop = tuple(map(convert_bounds, (start, stop)))
    start_h, start_s, start_l = start
    stop_h, stop_s, stop_l = stop
    if num:
        num_samples = num
    else:
        abs_h = abs(stop_h - start_h)
        h_diff = min(abs_h, 360 - abs_h)
        dist = math.sqrt(h_diff**2 + (stop_s - start_s) ** 2 + (stop_l - start_l) ** 2)
        num_samples = max(int(dist / float(step)), 1)
    color_vec = [
        np.linspace(*bounds, num=num_samples, dtype=float)
        for bounds in zip(start, stop)
    ]
    color_vec = list(zip(*color_vec))
    return color_vec


def hsl_gradient(
    start: Int3Tuple | Float3Tuple,
    stop: Int3Tuple | Float3Tuple,
    step: SupportsIndex,
    num: SupportsIndex = None,
    ncycles: int | float = float('inf'),
    replace_idx: tuple[SupportsIndex | Iterable[SupportsIndex], Iterator[Color]] = None,
    dtype: type[Color] | Callable[[Int3Tuple], int] = Color,
):
    replace_idx, rgb_iter = _resolve_replacement_indices(replace_idx)
    while abs(float(step)) < 1:
        step *= 10
    color_vec = _init_gradient_color_vec(num, start, step, stop)
    color_iter = iter(color_vec)
    type_map: dict[type[Color | int], ...] = {
        Color: lambda x: x.rgb,
        int: lambda x: int2rgb(x),
    }
    get_rgb_iter_idx: Callable[[Color | int, SupportsIndex], int] = (
        lambda x, ix: rgb2hsl(type_map[type(x)](x))[ix]
    )
    next_rgb_iter = None
    prev_output = None
    while ncycles > 0:
        try:
            cur_iter = next(color_iter)
            if cur_iter != prev_output:
                for idx in replace_idx:
                    try:
                        next_rgb_iter = next(rgb_iter)
                        cur_iter = list(cur_iter)
                        cur_iter[idx] = get_rgb_iter_idx(next_rgb_iter, idx)
                    except StopIteration:
                        raise GeneratorExit
                    except KeyError:
                        raise TypeError(
                            f"Expected iterator to return "
                            f"{repr(Color.__qualname__)} or {repr(int.__qualname__)}, "
                            f"got {repr(type(next_rgb_iter).__qualname__)} instead"
                        ) from None
                output = hsl2rgb(cast(Float3Tuple, cur_iter))
                if callable(dtype):
                    output = dtype(output)
                yield output
            prev_output = cur_iter
        except StopIteration:
            ncycles -= 1
            color_vec.reverse()
            color_iter = iter(color_vec)
        except GeneratorExit:
            break


def rgb_luma_transform(
    rgb: Int3Tuple,
    start: SupportsIndex = None,
    num: SupportsIndex = 50,
    step: SupportsIndex = 1,
    cycle: bool | Literal['wave'] = False,
    ncycles: int | float = float('inf'),
    gradient: Int3Tuple = None,
    dtype: type[Color] = None,
) -> Iterator[Int3Tuple | int | Color]:
    if dtype is None:
        ret_type = tuple
    elif issubclass(dtype, int):
        ret_type = lambda x: dtype(rgb2int(x))
    is_cycle = bool(cycle is not False)
    is_oscillator = cycle == 'wave'
    if is_oscillator:
        ncycles *= 2
    h, s, luma = rgb2hsl(rgb)
    luma_linspace = [*np.linspace(start=0, stop=1, num=num)][::step]
    if start:
        start = min(max(float(start), 0), 1)
        luma = min(luma_linspace, key=lambda x: abs(x - start))
        start_idx = luma_linspace.index(luma)
        remaining_indices = luma_linspace[start_idx:]
        luma_iter = iter(remaining_indices)
    else:
        luma_iter = iter(luma_linspace)

    def _generator():
        nonlocal luma_iter, ncycles
        if step == 0:
            yield rgb
            return
        prev_output = None
        while ncycles > 0:
            try:
                output = hsl2rgb((h, s, next(luma_iter)))
                if output != prev_output:
                    yield ret_type(output)
                prev_output = output
            except StopIteration as STOP_IT:
                if not is_cycle:
                    raise STOP_IT
                ncycles -= 1
                if is_oscillator:
                    luma_linspace.reverse()
                luma_iter = iter(luma_linspace)

    if gradient is not None:
        _gradient = hsl_gradient(
            start=rgb, stop=gradient, step=step, num=num, replace_idx=(2, _generator())
        )
        return iter(_gradient)
    return iter(_generator())
