__all__ = [
    'ANSI_4BIT_RGB',
    'ansi_4bit_to_rgb',
    'ansi_8bit_to_rgb',
    'int2rgb',
    'hexstr2rgb',
    'hsl2rgb',
    'hsv2rgb',
    'is_u24',
    'lab2rgb',
    'lab2xyz',
    'nearest_ansi_4bit_rgb',
    'nearest_ansi_8bit_rgb',
    'rgb2int',
    'rgb2hexstr',
    'rgb2hsl',
    'rgb2hsv',
    'rgb2lab',
    'rgb2xyz',
    'rgb_diff',
    'rgb_to_ansi_8bit',
    'xyz2lab',
    'xyz2rgb',
]

import typing as tp
from typing import Literal as L

import numpy as np

from .._typing import Float3Tuple, Int3Tuple, RGBVectorLike, ShapedNDArray

def is_u24(value, *, strict: bool = False) -> bool: ...
def hexstr2rgb(s: str, /) -> Int3Tuple: ...
def rgb2hexstr(rgb: RGBVectorLike, /) -> str: ...
def rgb2int(rgb: RGBVectorLike, /) -> int: ...
def int2rgb(x: int, /) -> Int3Tuple: ...

M_RGB2XYZ: tp.Final[ShapedNDArray[tuple[L[3], L[3]], np.float64]]
M_XYZ2RGB: tp.Final[ShapedNDArray[tuple[L[3], L[3]], np.float64]]
REFWT: tp.Final[ShapedNDArray[tuple[L[3]], np.float64]]
EPS: tp.Final[float]
LIN: tp.Final[float]

def xyz2rgb[_Shape: tuple[int, ...]](
    xyz: ShapedNDArray[_Shape, np.floating], /
) -> ShapedNDArray[_Shape, np.uint8]: ...
def rgb2xyz[_Shape: tuple[int, ...]](
    rgb: ShapedNDArray[_Shape, np.uint8], /
) -> ShapedNDArray[_Shape, np.float64]: ...

@tp.overload
def xyz2lab[_Shape: tuple[int, ...]](
    xyz: ShapedNDArray[_Shape, np.floating], /
) -> ShapedNDArray[_Shape, np.float64]: ...
@tp.overload
def xyz2lab(xyz: Float3Tuple, /) -> ShapedNDArray[tuple[L[3]], np.float64]: ...

@tp.overload
def lab2xyz[_Shape: tuple[int, ...]](
    lab: ShapedNDArray[_Shape, np.floating], /
) -> ShapedNDArray[_Shape, np.float64]: ...
@tp.overload
def lab2xyz(lab: Float3Tuple, /) -> ShapedNDArray[tuple[L[3]], np.float64]: ...

@tp.overload
def hsl2rgb[_Shape: tuple[int, ...]](
    hsl: ShapedNDArray[_Shape, np.floating], /
) -> ShapedNDArray[_Shape, np.uint8]: ...
@tp.overload
def hsl2rgb(hsl: Float3Tuple, /) -> ShapedNDArray[tuple[L[3]], np.uint8]: ...

@tp.overload
def rgb2hsl[_Shape: tuple[int, ...]](
    rgb: ShapedNDArray[_Shape, np.integer], /
) -> ShapedNDArray[_Shape, np.float32]: ...
@tp.overload
def rgb2hsl(rgb: Int3Tuple, /) -> ShapedNDArray[tuple[L[3]], np.float32]: ...

@tp.overload
def hsv2rgb[_Shape: tuple[int, ...]](
    hsv: ShapedNDArray[_Shape, np.floating], /
) -> ShapedNDArray[_Shape, np.uint8]: ...
@tp.overload
def hsv2rgb(hsv: Float3Tuple, /) -> ShapedNDArray[tuple[L[3]], np.uint8]: ...

@tp.overload
def rgb2hsv[_Shape: tuple[int, ...]](
    rgb: ShapedNDArray[_Shape, np.integer], /
) -> ShapedNDArray[_Shape, np.float32]: ...
@tp.overload
def rgb2hsv(rgb: Int3Tuple, /) -> ShapedNDArray[tuple[L[3]], np.float32]: ...

@tp.overload
def lab2rgb[_Shape: tuple[int, ...]](
    lab: ShapedNDArray[_Shape, np.floating], /
) -> ShapedNDArray[_Shape, np.uint8]: ...
@tp.overload
def lab2rgb(lab: Float3Tuple, /) -> ShapedNDArray[tuple[L[3]], np.uint8]: ...

@tp.overload
def rgb2lab[_Shape: tuple[int, ...]](
    rgb: ShapedNDArray[_Shape, np.uint8], /
) -> ShapedNDArray[_Shape, np.float64]: ...
@tp.overload
def rgb2lab(rgb: Int3Tuple, /) -> ShapedNDArray[tuple[L[3]], np.float64]: ...

@tp.overload
def rgb_diff[_Shape: tuple[int, ...]](
    rgb1: ShapedNDArray[_Shape, np.integer], rgb2: ShapedNDArray[_Shape, np.integer], /
) -> ShapedNDArray[_Shape, np.uint8]: ...
@tp.overload
def rgb_diff(
    rgb1: Int3Tuple, rgb2: Int3Tuple, /
) -> ShapedNDArray[tuple[L[3]], np.uint8]: ...

ANSI_4BIT_RGB: tp.Final[
    tuple[
        tuple[L[0],   L[0],   L[0]],
        tuple[L[170], L[0],   L[0]],
        tuple[L[0],   L[170], L[0]],
        tuple[L[170], L[85],  L[0]],
        tuple[L[0],   L[0],   L[170]],
        tuple[L[170], L[0],   L[170]],
        tuple[L[0],   L[170], L[170]],
        tuple[L[170], L[170], L[170]],
        tuple[L[85],  L[85],  L[85]],
        tuple[L[255], L[85],  L[85]],
        tuple[L[85],  L[255], L[85]],
        tuple[L[255], L[255], L[85]],
        tuple[L[85],  L[85],  L[255]],
        tuple[L[255], L[85],  L[255]],
        tuple[L[85],  L[255], L[255]],
        tuple[L[255], L[255], L[255]],
    ]   # fmt: skip
]

def ansi_4bit_to_rgb(value: int, /) -> Int3Tuple: ...

ANSI_4BIT_RGB_LUT: tp.Final[ShapedNDArray[tuple[L[32], L[32], L[32], L[3]], np.uint8]]

@tp.overload
def nearest_ansi_4bit_rgb[_Shape: tuple[int, ...]](
    rgb: ShapedNDArray[_Shape, np.integer], /
) -> ShapedNDArray[_Shape, np.uint8]: ...
@tp.overload
def nearest_ansi_4bit_rgb(rgb: Int3Tuple, /) -> Int3Tuple: ...

@tp.overload
def nearest_ansi_8bit_rgb[_Shape: tuple[int, ...]](
    rgb: ShapedNDArray[_Shape, np.integer], /
) -> ShapedNDArray[_Shape, np.uint8]: ...
@tp.overload
def nearest_ansi_8bit_rgb(rgb: Int3Tuple, /) -> Int3Tuple: ...

@tp.overload
def ansi_8bit_to_rgb[_Shape: tuple[int, ...]](
    value: ShapedNDArray[_Shape, np.integer], /
) -> ShapedNDArray[tuple[*_Shape, L[3]], np.uint8]: ...
@tp.overload
def ansi_8bit_to_rgb(value: int, /) -> Int3Tuple: ...

@tp.overload
def rgb_to_ansi_8bit(rgb: Int3Tuple, /) -> int: ...
@tp.overload
def rgb_to_ansi_8bit[_D1: int](
    rgb: ShapedNDArray[tuple[_D1, L[3]], np.integer], /
) -> ShapedNDArray[tuple[_D1], np.uint8]: ...
@tp.overload
def rgb_to_ansi_8bit[_D1: int, _D2: int](
    rgb: ShapedNDArray[tuple[_D1, _D2, L[3]], np.integer], /
) -> ShapedNDArray[tuple[_D1, _D2], np.uint8]: ...
@tp.overload
def rgb_to_ansi_8bit(
    rgb: np.typing.NDArray[np.integer], /
) -> np.typing.NDArray[np.uint8]: ...
