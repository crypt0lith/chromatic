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
from functools import lru_cache
from types import MappingProxyType as mappingproxy
from typing import Final, Literal as L, SupportsInt, TypeGuard

import numpy as np

from .._typing import Float3Tuple, Int3Tuple, RGBPixel, RGBVectorLike, ShapedNDArray


@lru_cache
def _supports_int(typ: type, /) -> TypeGuard[type[SupportsInt]]:
    return issubclass(typ, SupportsInt)


def is_u24(value, *, strict: bool = False) -> bool:
    """Check if value is an unsigned 24-bit integer.

    Parameters
    ---------
    value
        Input number
    strict : bool
        Whether to return False or raise ValueError on failure

    Raises
    ------
    ValueError
        Raised when `strict=True` and value is not u24
    """
    if _supports_int(value.__class__):
        if 0 <= int(value) < (1 << 24):
            return True
        elif not strict:
            return False
    raise ValueError(f"{value!r} is not u24")


def hexstr2rgb(s: str, /) -> Int3Tuple:
    n = len(s)
    if n % 4 == 0:  # trunc alpha
        n *= 3
        n //= 4
        s = s[:n]
    if n == 3:  # rgb -> rrggbb
        s = ''.join(c * 2 for c in s)
    x = int(s, 16)
    if not 0 <= x < (1 << 24):
        raise ValueError(f"{x:#x} is not u24")
    return int2rgb(x)


def rgb2hexstr(rgb: RGBVectorLike, /) -> str:
    return "%02x%02x%02x" % tuple(rgb)


def rgb2int(rgb: RGBVectorLike, /) -> int:
    r, g, b = map(int, rgb)
    return r << 16 | g << 8 | b


def int2rgb(x: int, /) -> Int3Tuple:
    x = int(x) & 0xFFFFFF
    return (x >> 16) & 0xFF, (x >> 8) & 0xFF, x & 0xFF


# fmt: off
# sRGB RGB to XYZ [M]
M_RGB2XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]],
    dtype=np.float64
)
M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)

# D65 reference white
REFWT = M_RGB2XYZ.sum(axis=1)

M_RGB2XYZ.flags.writeable   = \
M_XYZ2RGB.flags.writeable   = \
REFWT.flags.writeable       = False
# fmt: on

EPS = 216 / 24389
LIN = (1 / 3) * (6 / 29) ** -2


def xyz2rgb(xyz, /):
    out = np.clip(np.asarray(xyz, dtype=np.float64) @ M_XYZ2RGB.T, 0.0, 1.0)
    return np.rint(out * 255).astype(np.uint8)


def rgb2xyz(rgb, /):
    return (np.asarray(rgb, dtype=np.float64) / 255.0) @ M_RGB2XYZ.T


def xyz2lab(xyz, /):
    arr = np.asarray(xyz, dtype=np.float64)
    shape = arr.shape
    arr = np.atleast_2d(arr)
    n = arr / REFWT
    f = np.where(n > EPS, np.cbrt(n), LIN * n + (16 / 116))
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack((L, a, b), axis=-1).reshape(shape)


def lab2xyz(lab, /):
    arr = np.asarray(lab, dtype=np.float64)
    shape = arr.shape
    arr = np.atleast_2d(arr)
    L, a, b = arr[..., 0], arr[..., 1], arr[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    f = np.stack((fx, fy, fz), axis=-1)
    f3 = f**3
    n = np.where(f3 > EPS, f3, (f - (16 / 116)) / LIN)
    return (n * REFWT).reshape(shape)


def hsl2rgb(hsl, /):
    arr = np.asarray(hsl, dtype=np.float32)
    shape = arr.shape
    arr = np.atleast_2d(arr)
    h, s, L = arr[..., 0], arr[..., 1], arr[..., 2]
    C = (1.0 - np.abs(2.0 * L - 1.0)) * s
    h6 = h * 6.0
    i = np.floor(h6).astype(int)
    X = C * (1.0 - np.abs(h6 % 2.0 - 1.0))
    m = L - C / 2.0
    xs = np.stack([C + m, X + m, m], axis=-1)
    P = np.array(
        [[0, 1, 2],
         [1, 0, 2],
         [2, 0, 1],
         [2, 1, 0],
         [1, 2, 0],
         [0, 2, 1]]
    )   # fmt: skip
    out = np.take_along_axis(xs, P[i % 6], axis=-1)
    return np.rint(out * 255).astype(np.uint8).reshape(shape)


def rgb2hsl(rgb, /):
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    shape = arr.shape
    arr = np.atleast_2d(arr)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    m = np.min(arr, axis=-1)
    v = np.max(arr, axis=-1)
    C = v - m
    L = (v + m) / 2.0
    s = np.zeros_like(v)
    denom = 1.0 - np.abs(2.0 * L - 1.0)
    ok = denom != 0
    s[ok] = C[ok] / denom[ok]
    nz = C != 0
    rmax = (v == r) & nz
    gmax = (v == g) & nz
    bmax = (v == b) & nz
    h = np.zeros_like(v)
    h[rmax] = ((g[rmax] - b[rmax]) / C[rmax]) % 6
    h[gmax] = ((b[gmax] - r[gmax]) / C[gmax]) + 2
    h[bmax] = ((r[bmax] - g[bmax]) / C[bmax]) + 4
    h = (h / 6.0) % 1.0
    return np.stack((h, s, L), axis=-1).reshape(shape)


def hsv2rgb(hsv, /):
    arr = np.asarray(hsv, dtype=np.float32)
    shape = arr.shape
    arr = np.atleast_2d(arr)
    h, s, v = arr[..., 0], arr[..., 1], arr[..., 2]
    h6 = h * 6.0
    i = np.floor(h6).astype(int)
    f = h6 - i
    xs = np.stack([v, v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)], axis=-1)
    P = np.array(
        [[0, 3, 1],
         [2, 0, 1],
         [1, 0, 3],
         [1, 2, 0],
         [3, 1, 0],
         [0, 1, 2]]
    )   # fmt: skip
    out = np.take_along_axis(xs, P[i % 6], axis=-1)
    return np.rint(out * 255).astype(np.uint8).reshape(shape)


def rgb2hsv(rgb, /):
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    shape = arr.shape
    arr = np.atleast_2d(arr)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    m = np.min(arr, axis=-1)
    v = np.max(arr, axis=-1)
    C = v - m
    ok = v != 0
    s = np.zeros_like(v)
    s[ok] = C[ok] / v[ok]
    nz = C != 0
    rmax = (v == r) & nz
    gmax = (v == g) & nz
    bmax = (v == b) & nz
    h = np.zeros_like(v)
    h[rmax] = ((g[rmax] - b[rmax]) / C[rmax]) % 6
    h[gmax] = ((b[gmax] - r[gmax]) / C[gmax]) + 2
    h[bmax] = ((r[bmax] - g[bmax]) / C[bmax]) + 4
    h = (h / 6.0) % 1.0
    return np.stack((h, s, v), axis=-1).reshape(shape)


def lab2rgb(lab, /):
    return xyz2rgb(lab2xyz(lab))


def rgb2lab(rgb, /):
    return xyz2lab(rgb2xyz(rgb))


def rgb_diff(rgb1, rgb2, /):
    return lab2rgb((rgb2lab(rgb1) + rgb2lab(rgb2)) / 2)


ANSI_4BIT_RGB = (
    (0,   0,   0),      # black
    (170, 0,   0),      # red
    (0,   170, 0),      # green
    (170, 85,  0),      # yellow
    (0,   0,   170),    # blue
    (170, 0,   170),    # magenta
    (0,   170, 170),    # cyan
    (170, 170, 170),    # white
    (85,  85,  85),     # bright black (grey)
    (255, 85,  85),     # bright red
    (85,  255, 85),     # bright green
    (255, 255, 85),     # bright yellow
    (85,  85,  255),    # bright blue
    (255, 85,  255),    # bright magenta
    (85,  255, 255),    # bright cyan
    (255, 255, 255),    # bright white
)   # fmt: skip


def ansi_4bit_to_rgb(value: int, /):
    offset = 0
    if value > 37:
        if value <= 47:
            offset -= 10
        elif value <= 97:
            offset += 8
        else:
            offset -= 2
    value %= 30
    value += offset
    return ANSI_4BIT_RGB[value]


def _4b_lookup() -> ShapedNDArray[tuple[L[32], L[32], L[32], L[3]], np.uint8]:
    def rgb_dist(rgb, ansi):
        r_mean = (rgb[:, 0:1] + ansi[:, 0]) / 2
        r_diff = (rgb[:, 0:1] - ansi[:, 0]) * (2 + r_mean / 256)
        g_diff = (rgb[:, 1:2] - ansi[:, 1]) * 4
        b_diff = (rgb[:, 2:3] - ansi[:, 2]) * (2 + (255 - r_mean) / 256)
        return r_diff**2 + g_diff**2 + b_diff**2

    rgb_4b_arr = np.asarray(ANSI_4BIT_RGB)
    shape = (32,) * 3
    quants = np.indices(shape).reshape(3, -1).T
    nearest_colors = rgb_4b_arr[np.argmin(rgb_dist(quants * 8, rgb_4b_arr), axis=1)]
    lut = nearest_colors.reshape(*shape, 3).astype(np.uint8)
    lut.flags.writeable = False
    return lut  # type: ignore


ANSI_4BIT_RGB_LUT = _4b_lookup()


def nearest_ansi_4bit_rgb(rgb, /):
    if is_3tuple := (rgb.__class__ is tuple and len(rgb) == 3):
        r, g, b = (min(x >> 3, 32) for x in rgb)
    else:
        arr = np.asarray(rgb, dtype=np.uint8) >> 3
        if not arr.shape or arr.shape[-1] != 3:
            raise ValueError
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    out = ANSI_4BIT_RGB_LUT[r, g, b]
    return tuple(out.tolist()) if is_3tuple else out


def nearest_ansi_8bit_rgb(value, /):
    return ansi_8bit_to_rgb(rgb_to_ansi_8bit(value))


def ansi_8bit_to_rgb(value, /):
    arr = np.atleast_1d(value).astype(np.uint8)
    out = np.empty((*arr.shape, 3), dtype=np.uint8)
    mask = np.ones(arr.shape, dtype=np.bool_)
    if np.any(ansi_16c := arr < 16):
        out[ansi_16c] = np.take(ANSI_4BIT_RGB, arr[ansi_16c], axis=0)
        mask &= ~ansi_16c
    if np.any(colorcube := (arr >= 16) & (arr < 232)):
        c = arr[colorcube] - 16
        out[colorcube, 0] = (c // 36) * 51
        out[colorcube, 1] = (c % 36 // 6) * 51
        out[colorcube, 2] = (c % 6) * 51
        mask &= ~colorcube
    out[mask] = (8 + (arr[mask] - 232) * 10)[:, None]
    return tuple(out[0].tolist()) if np.isscalar(value) else out


def rgb_to_ansi_8bit(rgb, /) -> int | ShapedNDArray[tuple[int, ...], np.uint8]:
    is_3tuple = rgb.__class__ is tuple and len(rgb) == 3
    arr = np.asarray(rgb, dtype=np.uint8)
    out = np.zeros(arr.shape[:-1], dtype=np.uint8)
    mask = np.ones(arr.shape[:-1], dtype=np.bool_)
    if np.any(grey := (arr[..., 1:] == arr[..., 0:1]).all(axis=-1)):
        c = arr[..., 0]
        r_lo = grey & (c < 8)
        r_hi = grey & (c > 248)
        mid = grey & ~(r_lo | r_hi)
        out[r_lo] = 16
        out[r_hi] = 231
        out[mid] = np.rint((c[mid] - 8) / 247 * 24) + 232
        mask &= ~grey
    rest = np.rint(arr[mask] / 255 * 5)
    r, g, b = rest[..., 0], rest[..., 1], rest[..., 2]
    out[mask] = 16 + (36 * r) + (6 * g) + b
    return out.item() if is_3tuple else out
