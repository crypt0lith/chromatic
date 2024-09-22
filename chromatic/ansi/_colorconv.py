__all__ = [
    'ANSI_4BIT_RGB',
    'lab2xyz',
    'rgb2hsv',
    'ansi_4bit_to_rgb',
    'ansi_8bit_to_rgb',
    'hex2rgb',
    'hsl2rgb',
    'hsv2rgb',
    'lab2rgb',
    'nearest_ansi_4bit_rgb',
    'nearest_ansi_8bit_rgb',
    'rgb2hex',
    'rgb2hsl',
    'rgb2lab',
    'rgb2xyz',
    'rgb_to_ansi_8bit',
    'xyz2lab',
    'xyz2rgb'
]

import math
from typing import Final

import numpy as np

from chromatic._typing import Float3Tuple, FloatSequence, Int3Tuple, RGBVector


def rgb2hex(rgb: RGBVector) -> int:
    if isinstance(rgb, np.ndarray):
        rgb = tuple(map(int, rgb))
    r, g, b = rgb
    return (r << 16) + (g << 8) + b


def hex2rgb(value: int) -> Int3Tuple:
    r = (value >> 16) & 0xFF
    g = (value >> 8) & 0xFF
    b = value & 0xFF
    return (r, g, b)


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


def nearest_ansi_4bit_rgb(value: RGBVector) -> Int3Tuple:
    if isinstance(value, np.ndarray):
        value = tuple(map(int, value))
    if value in ANSI_4BIT_RGB:
        return value

    def _dist_4b(a: Int3Tuple):
        return math.dist(a, value)

    try:
        return min(ANSI_4BIT_RGB, key=_dist_4b)
    except ValueError:
        raise ValueError(
            f"invalid RGB value: {value!r}") from None


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


def nearest_ansi_8bit_rgb(value: RGBVector) -> Int3Tuple:
    try:
        return ansi_8bit_to_rgb(rgb_to_ansi_8bit(value))
    except ValueError:
        raise ValueError(
            f"invalid RGB value: {value!r}") from None


def ansi_8bit_to_rgb(value: int):
    if 0 <= value < 16:
        return ANSI_4BIT_RGB[value]
    elif value < 232:
        value -= 16
        return value // 36 * 51, (value % 36 // 6) * 51, (value % 6) * 51
    elif value <= 255:
        grey = 8 + (value - 232) * 10
        return grey, grey, grey
    r = range(256)
    raise ValueError(
        f"Value {value} not in {(r.start, r.stop)}")


def rgb_to_ansi_8bit(rgb: RGBVector) -> int:
    r, g, b = rgb
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return round(((r - 8) / 247) * 24) + 232
    return 16 + (36 * round(r / 255 * 5)) + (6 * round(g / 255 * 5)) + round(b / 255 * 5)


def xyz2lab(xyz: Float3Tuple | FloatSequence) -> Float3Tuple:
    x, y, z = xyz
    x /= 95.047
    y /= 100.0
    z /= 108.883
    x = x ** (1 / 3) if x > 0.008856 else (7.787 * x) + (16 / 116)
    y = y ** (1 / 3) if y > 0.008856 else (7.787 * y) + (16 / 116)
    z = z ** (1 / 3) if z > 0.008856 else (7.787 * z) + (16 / 116)
    l = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)
    return (l, a, b)


def lab2xyz(lab: Float3Tuple | FloatSequence) -> Float3Tuple:
    l, a, b = lab
    y = (l + 16) / 116
    x = a / 500 + y
    z = y - b / 200
    x = 95.047 * (x ** 3 if x ** 3 > 0.008856 else (x - 16 / 116) / 7.787)
    y = 100.0 * (y ** 3 if y ** 3 > 0.008856 else (y - 16 / 116) / 7.787)
    z = 108.883 * (z ** 3 if z ** 3 > 0.008856 else (z - 16 / 116) / 7.787)
    return (x, y, z)


def xyz2rgb(xyz: Float3Tuple | FloatSequence) -> Int3Tuple:
    x, y, z = [v / 100.0 for v in xyz]
    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570
    r = 1.055 * (r ** (1 / 2.4)) - 0.055 if r > 0.0031308 else 12.92 * r
    g = 1.055 * (g ** (1 / 2.4)) - 0.055 if g > 0.0031308 else 12.92 * g
    b = 1.055 * (b ** (1 / 2.4)) - 0.055 if b > 0.0031308 else 12.92 * b
    r = max(0, min(255, int(r * 255)))
    g = max(0, min(255, int(g * 255)))
    b = max(0, min(255, int(b * 255)))
    return (r, g, b)


def rgb2xyz(rgb: RGBVector) -> Float3Tuple:
    r, g, b = [x / 255.0 for x in rgb]
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    return (x * 100.0, y * 100.0, z * 100.0)


def hsl2rgb(hsl: Float3Tuple | FloatSequence) -> Int3Tuple:
    h, sl, l = (hsl[0] / 360) % 1, hsl[1], hsl[2]
    if h < 0:
        h += 1
    r = g = b = l
    v = (l * (1.0 + sl)) if l <= 0.5 else (l + sl - l * sl)
    if v > 0:
        m = l + l - v
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
    return (int(r * 255), int(g * 255), int(b * 255))


def rgb2hsl(rgb: RGBVector) -> Float3Tuple:
    r, g, b = [x / 255.0 for x in rgb]
    v = max(r, g, b)
    m = min(r, g, b)
    l = (m + v) / 2
    h = s = 0
    if l > 0:
        vm = v - m
        s = vm
        if s > 0:
            s /= (l <= 0.5) * (v + m) or (2 - v - m)
            r2 = (v - r) / vm
            g2 = (v - g) / vm
            b2 = (v - b) / vm
            if v == r:
                h = (g == m) * (5.0 + b2) + (g != m) * (1.0 - g2)
            elif v == g:
                h = (b == m) * (1.0 + r2) + (b != m) * (3.0 - b2)
            else:
                h = (r == m) * (3.0 + g2) + (r != m) * (5.0 - r2)
            h /= 6
    return (360 * (h % 1), s, l)


def hsv2rgb(hsv: Float3Tuple | FloatSequence) -> Int3Tuple:
    h, s, v = hsv
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if 0 <= h < 60:
        r_prime, g_prime, b_prime = c, x, 0
    elif 60 <= h < 120:
        r_prime, g_prime, b_prime = x, c, 0
    elif 120 <= h < 180:
        r_prime, g_prime, b_prime = 0, c, x
    elif 180 <= h < 240:
        r_prime, g_prime, b_prime = 0, x, c
    elif 240 <= h < 300:
        r_prime, g_prime, b_prime = x, 0, c
    else:
        r_prime, g_prime, b_prime = c, 0, x
    return int((r_prime + m) * 255), int((g_prime + m) * 255), int((b_prime + m) * 255)


def rgb2hsv(rgb: RGBVector) -> Float3Tuple:
    r, g, b = [x / 255.0 for x in rgb]
    v = max(r, g, b)
    m = min(r, g, b)
    delta = v - m
    if delta == 0:
        h = 0
    elif v == r:
        h = ((g - b) / delta) % 6
    elif v == g:
        h = (b - r) / delta + 2
    else:
        h = (r - g) / delta + 4
    h *= 60
    if h < 0:
        h += 360
    s = 0 if v == 0 else delta / v
    return (h, s, v)


def lab2rgb(lab: Float3Tuple | FloatSequence) -> Int3Tuple:
    xyz = lab2xyz(lab)
    return xyz2rgb(xyz)


def rgb2lab(rgb: RGBVector) -> Float3Tuple:
    xyz = rgb2xyz(rgb)
    return xyz2lab(xyz)
