import numpy as np
import pytest

from chromatic.color.colorconv import (
    ANSI_4BIT_RGB,
    ansi_4bit_to_rgb,
    ansi_8bit_to_rgb,
    hexstr2rgb,
    hsl2rgb,
    hsv2rgb,
    int2rgb,
    is_u24,
    lab2lch,
    lab2rgb,
    lab2xyz,
    lch2lab,
    lch2rgb,
    lerp_lch,
    nearest_ansi_4bit_rgb,
    nearest_ansi_8bit_rgb,
    rgb2hexstr,
    rgb2hsl,
    rgb2hsv,
    rgb2int,
    rgb2lab,
    rgb2lch,
    rgb2xyz,
    rgb_diff,
    rgb_to_ansi_8bit,
    xyz2lab,
    xyz2rgb,
)

# Curated to hit distinct code paths: black/white, primaries, secondaries,
# mid grey, and the near-black (<8) / near-white (>248) grey branches used by
# the 8-bit greyscale ramp, plus a few arbitrary chromatic colors.
SAMPLES = [
    (0, 0, 0),
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 128, 128),
    (5, 5, 5),
    (250, 250, 250),
    (10, 20, 30),
    (200, 100, 50),
    (3, 200, 199),
]


@pytest.fixture(params=SAMPLES, ids=[f"{r:02x}{g:02x}{b:02x}" for r, g, b in SAMPLES])
def rgb(request):
    return request.param


def _rgb(x):
    return tuple(np.asarray(x).tolist())


@pytest.mark.parametrize(
    "fwd,back",
    [
        pytest.param(rgb2hsv, hsv2rgb, id="hsv"),
        pytest.param(rgb2hsl, hsl2rgb, id="hsl"),
        pytest.param(rgb2xyz, xyz2rgb, id="xyz"),
        pytest.param(rgb2lab, lab2rgb, id="lab"),
        pytest.param(rgb2lch, lch2rgb, id="lch"),
    ],
)
def test_rgb_roundtrip_is_lossless(fwd, back, rgb):
    assert _rgb(back(fwd(rgb))) == rgb


@pytest.mark.parametrize(
    "encode,decode",
    [
        pytest.param(rgb2int, int2rgb, id="int"),
        pytest.param(rgb2hexstr, hexstr2rgb, id="hexstr"),
    ],
)
def test_rgb_codec_roundtrip(encode, decode, rgb):
    assert decode(encode(rgb)) == rgb


@pytest.mark.parametrize(
    "fn,arg,expected",
    [
        pytest.param(rgb2hsv, (255, 0, 0), (0.0, 1.0, 1.0), id="hsv-red"),
        pytest.param(rgb2hsl, (255, 0, 0), (0.0, 1.0, 0.5), id="hsl-red"),
        pytest.param(rgb2lab, (255, 255, 255), (100.0, 0.0, 0.0), id="lab-white"),
        pytest.param(rgb2lab, (0, 0, 0), (0.0, 0.0, 0.0), id="lab-black"),
    ],
)
def test_known_conversions(fn, arg, expected):
    # round-trip proves fwd/back are inverses; anchors prove they're correct.
    assert np.allclose(np.asarray(fn(arg)), expected, atol=1e-4)


@pytest.mark.parametrize(
    "s,expected",
    [
        pytest.param("ff0000", (255, 0, 0), id="6-char"),
        pytest.param("f00", (255, 0, 0), id="3-char-expand"),
        pytest.param("ff000080", (255, 0, 0), id="8-char-alpha-trunc"),
    ],
)
def test_hexstr2rgb_formats(s, expected):
    assert hexstr2rgb(s) == expected


def test_hexstr2rgb_rejects_out_of_range():
    with pytest.raises(ValueError):
        hexstr2rgb("1000000")


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param(0, True, id="min"),
        pytest.param((1 << 24) - 1, True, id="max"),
        pytest.param(1 << 24, False, id="over"),
        pytest.param(-1, False, id="negative"),
    ],
)
def test_is_u24(value, expected):
    assert is_u24(value) is expected


def test_is_u24_strict_raises():
    with pytest.raises(ValueError):
        is_u24(1 << 24, strict=True)


@pytest.mark.parametrize("i", range(8))
def test_ansi_4bit_maps_standard_and_bright(i):
    assert ansi_4bit_to_rgb(30 + i) == ANSI_4BIT_RGB[i]
    assert ansi_4bit_to_rgb(90 + i) == ANSI_4BIT_RGB[8 + i]


def test_nearest_ansi_4bit_returns_palette_color(rgb):
    assert nearest_ansi_4bit_rgb(rgb) in ANSI_4BIT_RGB


def test_nearest_ansi_8bit_is_idempotent(rgb):
    once = nearest_ansi_8bit_rgb(rgb)
    assert nearest_ansi_8bit_rgb(once) == once


def test_rgb_to_ansi_8bit_in_range(rgb):
    code = rgb_to_ansi_8bit(rgb)
    assert isinstance(code, int)
    assert 0 <= code <= 255


def test_rgb_diff_of_equal_is_identity(rgb):
    assert _rgb(rgb_diff(rgb, rgb)) == rgb


def test_lerp_lch_hits_endpoints():
    a = rgb2lch((255, 0, 0))
    b = rgb2lch((0, 0, 255))
    out = np.asarray(lerp_lch(a, b, num=8))
    assert out.shape == (8, 3)
    assert np.allclose(out[0], a)
    assert np.allclose(out[-1], b)


# --- stub overload conformance: dtype, and the scalar-vs-array return contract ---

_RGB = (10, 20, 30)
_XYZ = rgb2xyz(_RGB)
_LAB = rgb2lab(_RGB)
_LCH = rgb2lch(_RGB)
_HSV = rgb2hsv(_RGB)
_HSL = rgb2hsl(_RGB)


@pytest.mark.parametrize(
    "call,dtype",
    [
        pytest.param(lambda: rgb2hsv(_RGB), np.float32, id="rgb2hsv"),
        pytest.param(lambda: rgb2hsl(_RGB), np.float32, id="rgb2hsl"),
        pytest.param(lambda: rgb2xyz(_RGB), np.float64, id="rgb2xyz"),
        pytest.param(lambda: rgb2lab(_RGB), np.float64, id="rgb2lab"),
        pytest.param(lambda: rgb2lch(_RGB), np.float64, id="rgb2lch"),
        pytest.param(lambda: xyz2lab(_XYZ), np.float64, id="xyz2lab"),
        pytest.param(lambda: lab2xyz(_LAB), np.float64, id="lab2xyz"),
        pytest.param(lambda: lab2lch(_LAB), np.float64, id="lab2lch"),
        pytest.param(lambda: lch2lab(_LCH), np.float64, id="lch2lab"),
        pytest.param(lambda: xyz2rgb(_XYZ), np.uint8, id="xyz2rgb"),
        pytest.param(lambda: hsv2rgb(_HSV), np.uint8, id="hsv2rgb"),
        pytest.param(lambda: hsl2rgb(_HSL), np.uint8, id="hsl2rgb"),
        pytest.param(lambda: lab2rgb(_LAB), np.uint8, id="lab2rgb"),
        pytest.param(lambda: lch2rgb(_LCH), np.uint8, id="lch2rgb"),
        pytest.param(lambda: rgb_diff(_RGB, (40, 50, 60)), np.uint8, id="rgb_diff"),
    ],
)
def test_scalar_output_dtype_and_shape(call, dtype):
    out = call()
    assert out.dtype == dtype
    assert out.shape == (3,)


@pytest.mark.parametrize(
    "fwd,back",
    [
        pytest.param(rgb2hsv, hsv2rgb, id="hsv"),
        pytest.param(rgb2hsl, hsl2rgb, id="hsl"),
        pytest.param(rgb2xyz, xyz2rgb, id="xyz"),
        pytest.param(rgb2lab, lab2rgb, id="lab"),
        pytest.param(rgb2lch, lch2rgb, id="lch"),
    ],
)
def test_batch_preserves_leading_shape(fwd, back):
    batch = np.asarray(SAMPLES, dtype=np.uint8)
    assert back(fwd(batch)).shape == batch.shape


@pytest.mark.parametrize(
    "fn", [nearest_ansi_4bit_rgb, nearest_ansi_8bit_rgb], ids=["4bit", "8bit"]
)
def test_nearest_scalar_returns_tuple(fn):
    out = fn((200, 10, 10))
    assert isinstance(out, tuple)
    assert len(out) == 3


def test_nearest_4bit_batch_returns_uint8_array():
    out = nearest_ansi_4bit_rgb(np.array([[200, 10, 10], [5, 5, 5]], dtype=np.uint8))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.shape == (2, 3)


def test_ansi_8bit_to_rgb_scalar_returns_tuple_array_returns_ndarray():
    assert isinstance(ansi_8bit_to_rgb(5), tuple)
    out = ansi_8bit_to_rgb(np.array([5, 200]))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.shape == (2, 3)


def test_rgb_to_ansi_8bit_scalar_returns_int():
    assert isinstance(rgb_to_ansi_8bit((10, 20, 30)), int)


@pytest.mark.parametrize(
    "shape_in,shape_out",
    [pytest.param((2, 3), (2,), id="1d"), pytest.param((2, 2, 3), (2, 2), id="2d")],
)
def test_rgb_to_ansi_8bit_batch_shape(shape_in, shape_out):
    out = rgb_to_ansi_8bit(np.zeros(shape_in, dtype=np.uint8))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.shape == shape_out


@pytest.mark.parametrize(
    "shape_in,shape_out",
    [
        pytest.param((3,), (8, 3), id="vector"),
        pytest.param((2, 3), (2, 8, 3), id="1d-batch"),
        pytest.param((2, 2, 3), (2, 2, 8, 3), id="2d-batch"),
    ],
)
def test_lerp_lch_shape_overloads(shape_in, shape_out):
    out = lerp_lch(np.zeros(shape_in), np.ones(shape_in), num=8)
    assert np.asarray(out).shape == shape_out
