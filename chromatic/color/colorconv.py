__all__ = [
    'ANSI_4BIT_RGB',
    'ansi_4bit_to_rgb',
    'ansi_8bit_to_rgb',
    'hex2rgb',
    'hexstr2rgb',
    'hsl2rgb',
    'hsv2rgb',
    'is_hex_rgb',
    'lab2rgb',
    'lab2xyz',
    'nearest_ansi_4bit_rgb',
    'nearest_ansi_8bit_rgb',
    'rgb2hex',
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

from operator import mul, truediv
from typing import Final, Literal, SupportsInt, cast

import numpy as np

from .._typing import Float3Tuple, FloatSequence, Int3Tuple, RGBPixel, RGBVectorLike, ShapedNDArray


def is_hex_rgb(value, *, strict: bool = False):
    if issubclass(type(value), SupportsInt):
        if 0x0 <= int(value) <= 0xFFFFFF:
            return True
        elif not strict:
            return False
    raise TypeError(f"{value!r} is not a valid RGB color") from None


def hexstr2rgb(__str: str) -> Int3Tuple:
    if is_hex_rgb(value := int(__str, 16), strict=True):
        return hex2rgb(value)


def rgb2hexstr(rgb: RGBVectorLike) -> str:
    r, g, b = rgb
    return f'{r:02x}{g:02x}{b:02x}'


def rgb2hex(rgb: RGBVectorLike) -> int:
    r, g, b = map(int, rgb)
    return r << 16 | g << 8 | b


def hex2rgb(value: int) -> Int3Tuple:
    return (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF


def xyz2lab(xyz: FloatSequence) -> Float3Tuple:
    x, y, z = (
        n ** (1 / 3) if n > 0.008856 else (7.787 * n) + (16 / 116)
        for n in map(truediv, xyz, (95.047, 100.0, 108.883))
    )
    L = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)
    return L, a, b


def lab2xyz(lab: FloatSequence) -> Float3Tuple:
    L, a, b = lab
    y = (L + 16) / 116
    x = a / 500 + y
    z = y - b / 200
    x, y, z = map(
        mul,
        (95.047, 100.0, 108.883),
        map(lambda n: (lambda c: c if c > 0.008856 else (n - 16 / 116) / 7.787)(n**3), (x, y, z)),
    )
    return x, y, z


M_RGB2XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]],
    dtype=np.float64
)  # fmt: skip
M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)


def rgb2xyz(rgb: RGBPixel) -> Float3Tuple:
    x, y, z = M_RGB2XYZ @ (np.array(rgb, dtype=np.float64) / 255.0)
    return x, y, z


def xyz2rgb(xyz: ShapedNDArray[tuple[Literal[3]], np.float64]) -> Int3Tuple:
    r, g, b = (np.clip(M_XYZ2RGB @ np.array(xyz, dtype=np.float64), 0.0, 1.0) * 255.0).astype(int)
    return r, g, b


def hsl2rgb(hsl: FloatSequence) -> Int3Tuple:
    h, s, L = hsl
    h = (h / 360) % 1
    if h < 0:
        h += 1
    r = g = b = L
    v = (L * (1.0 + s)) if L <= 0.5 else (L + s - L * s)
    if v > 0:
        m = L + L - v
        sv = (v - m) / v
        h *= 6.0
        sextant = int(h)
        fract = h - sextant
        vsf = v * sv * fract
        mid1 = m + vsf
        mid2 = v - vsf
        if sextant == 0:
            r, g, b = v, mid1, m
        elif sextant == 1:
            r, g, b = mid2, v, m
        elif sextant == 2:
            r, g, b = m, v, mid1
        elif sextant == 3:
            r, g, b = m, mid2, v
        elif sextant == 4:
            r, g, b = mid1, m, v
        elif sextant == 5:
            r, g, b = v, m, mid2
        r, g, b = (round(x * 255) for x in (r, g, b))
    else:
        r, g, b = (round(L * 255) for _ in range(3))
    return r, g, b


def rgb2hsl(rgb: RGBVectorLike) -> Float3Tuple:
    r, g, b = (x / 255.0 for x in rgb)
    m, v = sorted([r, g, b])[::2]
    L = (m + v) / 2
    h = s = 0
    if L > 0:
        vm = v - m
        s = vm / (v + m) if L <= 0.5 else vm / (2 - v - m)
        if vm > 0:
            r2 = (v - r) / vm
            g2 = (v - g) / vm
            b2 = (v - b) / vm
            if v == r:
                h = b2 - g2
            elif v == g:
                h = 2 + r2 - b2
            else:
                h = 4 + g2 - r2
            h = (h / 6) % 1
    return (360 * h, s, L)


def hsv2rgb(hsv: FloatSequence) -> Int3Tuple:
    h, s, v = hsv
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 0:
        h += 360
    h %= 360
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    r, g, b = (int(round((i + m) * 255)) for i in (r, g, b))
    return r, g, b


def rgb2hsv(rgb: RGBVectorLike) -> Float3Tuple:
    r, g, b = (x / 255.0 for x in rgb)
    m, v = sorted([r, g, b])[::2]
    delta = v - m
    if delta == 0:
        h = 0
    elif v == r:
        h = (g - b) / delta % 6
    elif v == g:
        h = (b - r) / delta + 2
    else:
        h = (r - g) / delta + 4
    h *= 60
    if h < 0:
        h += 360
    s = 0 if v == 0 else delta / v
    return h, s, v


def lab2rgb(lab: FloatSequence) -> Int3Tuple:
    xyz = lab2xyz(lab)
    return xyz2rgb(np.array(xyz, dtype=np.float64))


def rgb2lab(rgb: RGBVectorLike) -> Float3Tuple:
    xyz = rgb2xyz(rgb)
    return xyz2lab(xyz)


ANSI_4BIT_RGB: Final[list[Int3Tuple]] = [
    (0, 0, 0),  # black
    (170, 0, 0),  # red
    (0, 170, 0),  # green
    (170, 85, 0),  # yellow
    (0, 0, 170),  # blue
    (170, 0, 170),  # magenta
    (0, 170, 170),  # cyan
    (170, 170, 170),  # white
    (85, 85, 85),  # bright black (grey)
    (255, 85, 85),  # bright red
    (85, 255, 85),  # bright green
    (255, 255, 85),  # bright yellow
    (85, 85, 255),  # bright blue
    (255, 85, 255),  # bright magenta
    (85, 255, 255),  # bright cyan
    (255, 255, 255),  # bright white
]


def ansi_4bit_to_rgb(value: int):
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


def _4b_lookup():
    def rgb_dist(rgb, ansi):
        r_mean = (rgb[:, 0:1] + ansi[:, 0]) / 2
        r_diff = (rgb[:, 0:1] - ansi[:, 0]) * (2 + r_mean / 256)
        g_diff = (rgb[:, 1:2] - ansi[:, 1]) * 4
        b_diff = (rgb[:, 2:3] - ansi[:, 2]) * (2 + (255 - r_mean) / 256)
        return r_diff**2 + g_diff**2 + b_diff**2

    rgb_4b_arr = np.asarray(ANSI_4BIT_RGB)
    quants = np.stack(
        np.meshgrid(*np.repeat(np.arange(32).reshape([1, -1]), 3, 0), indexing='ij'), axis=-1
    ).reshape([-1, 3])
    rgb_colors = quants * 8
    nearest_colors = rgb_4b_arr[np.argmin(rgb_dist(rgb_colors, rgb_4b_arr), axis=1)]
    table = {
        tuple(map(int, color)): tuple(map(int, nearest_colors[i])) for i, color in enumerate(quants)
    }
    return cast(dict[Int3Tuple, Int3Tuple], table)


ANSI_4BIT_RGB_MAP = _4b_lookup()


def _quantize_rgb(rgb: RGBVectorLike):
    r, g, b = rgb
    return min(r >> 3, 0x1F), min(g >> 3, 0x1F), min(b >> 3, 0x1F)


def nearest_ansi_4bit_rgb(value: RGBVectorLike) -> Int3Tuple:
    return ANSI_4BIT_RGB_MAP[_quantize_rgb(value)]


def nearest_ansi_8bit_rgb(value: RGBVectorLike) -> Int3Tuple:
    try:
        return ansi_8bit_to_rgb(rgb_to_ansi_8bit(value))
    except ValueError:
        raise ValueError(f"invalid RGB value: {value!r}") from None


def ansi_8bit_to_rgb(value: int):
    if 0 <= value < 16:
        return ANSI_4BIT_RGB[value]
    elif value < 232:
        value -= 16
        return value // 36 * 51, (value % 36 // 6) * 51, (value % 6) * 51
    elif value <= 255:
        grey = 8 + (value - 232) * 10
        return grey, grey, grey
    raise ValueError(f"expected an unsigned 8-bit integer, got {value}")


def rgb_to_ansi_8bit(rgb: RGBVectorLike) -> int:
    if len(set(rgb)) == 1:
        c = rgb[0]
        if c < 8:
            return 16
        if c > 248:
            return 231
        return round((c - 8) / 247 * 24) + 232
    r, g, b = (round((x / 255) * 5) for x in rgb)
    return 16 + (36 * r) + (6 * g) + b


def rgb_diff(rgb1: Int3Tuple, rgb2: Int3Tuple) -> Int3Tuple:
    lab1, lab2 = map(rgb2lab, (rgb1, rgb2))
    return lab2rgb([(i + j) / 2 for i, j in zip(lab1, lab2)])
