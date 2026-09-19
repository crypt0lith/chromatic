import numpy as np
import pytest

# opencv (a transitive dependency of the image pipeline) needs system GL
# libraries, which are not guaranteed to be present on minimal machines.
pytest.importorskip("cv2", reason="opencv requires system OpenGL libraries")

from PIL import Image  # noqa: E402

from chromatic import (  # noqa: E402
    ansi2img,
    ansi_quantize,
    ansify,
    ascii2img,
    color_chain,
    contrast_stretch,
    equalize_white_point,
    img2ansi,
    img2ascii,
    reshape_ansi,
    sort_glyphs,
)
from chromatic.image import scale_saturation  # noqa: E402

FACTOR = 40
CHAR_SET = " .:-=+*#%@"


@pytest.fixture(scope="module")
def rgb_array():
    # deterministic image with a gradient and structured noise, so the glyph and
    # color mappings hit a realistic spread of values
    rng = np.random.default_rng(0xC0FFEE)
    y, x = np.mgrid[0:96, 0:96]
    base = np.stack([x * 2, y * 2, (x + y)], axis=-1) % 256
    noise = rng.integers(0, 48, size=base.shape)
    return ((base + noise) % 256).astype(np.uint8)


@pytest.fixture(scope="module")
def image(rgb_array):
    return Image.fromarray(rgb_array)


@pytest.fixture(scope="module")
def ansi_array(image):
    return img2ansi(image, factor=FACTOR, char_set=CHAR_SET, outarray=True)


@pytest.fixture(scope="module")
def ascii_str(image):
    return img2ascii(image, factor=FACTOR, char_set=CHAR_SET)


def test_img2ascii_shape_matches_factor(ascii_str):
    lines = ascii_str.split("\n")
    assert len(lines[0]) == FACTOR
    assert set(ascii_str) <= set(CHAR_SET + "\n")


def test_img2ansi_array_roundtrips_to_chain(ansi_array):
    assert ansi_array.dtype == color_chain.dtype
    assert isinstance(color_chain.fromarray(ansi_array), color_chain)


def test_ansi2img_renders_image(ansi_array):
    assert isinstance(ansi2img(ansi_array), Image.Image)


def test_ascii2img_renders_image(ascii_str):
    assert isinstance(ascii2img(ascii_str), Image.Image)


@pytest.mark.parametrize("ansi_type", ["4b", "8b"])
def test_ansi_quantize_preserves_shape(rgb_array, ansi_type):
    assert ansi_quantize(rgb_array, ansi_type).shape == rgb_array.shape


# ---- benchmarks ----


@pytest.mark.parametrize("factor", [40, 80], ids=["f40", "f80"])
def test_bench_img2ascii(benchmark, image, factor):
    benchmark(lambda: img2ascii(image, factor=factor, char_set=CHAR_SET))


@pytest.mark.parametrize("ansi_type", ["4b", "8b", "24b"])
def test_bench_img2ansi(benchmark, image, ansi_type):
    benchmark(
        lambda: img2ansi(
            image, factor=FACTOR, char_set=CHAR_SET, ansi_type=ansi_type, outarray=True
        )
    )


def test_bench_img2ansi_equalized(benchmark, image):
    benchmark(
        lambda: img2ansi(
            image, factor=FACTOR, char_set=CHAR_SET, equalize=True, outarray=True
        )
    )


def test_bench_img2ansi_to_chain(benchmark, image):
    benchmark(lambda: img2ansi(image, factor=FACTOR, char_set=CHAR_SET))


def test_bench_ansi2img(benchmark, ansi_array):
    benchmark(lambda: ansi2img(ansi_array))


def test_bench_ascii2img(benchmark, ascii_str):
    benchmark(lambda: ascii2img(ascii_str))


def test_bench_ansify(benchmark, image):
    benchmark(lambda: ansify(image, factor=FACTOR, char_set=CHAR_SET))


@pytest.mark.parametrize("ansi_type", ["4b", "8b"])
def test_bench_ansi_quantize(benchmark, rgb_array, ansi_type):
    benchmark(lambda: ansi_quantize(rgb_array, ansi_type))


def test_bench_contrast_stretch(benchmark, rgb_array):
    benchmark(lambda: contrast_stretch(rgb_array))


def test_bench_equalize_white_point(benchmark, rgb_array):
    benchmark(lambda: equalize_white_point(rgb_array))


def test_bench_scale_saturation(benchmark, rgb_array):
    benchmark(lambda: scale_saturation(rgb_array.copy(), 1.5))


def test_bench_sort_glyphs(benchmark):
    benchmark(lambda: sort_glyphs(CHAR_SET, "vga437"))


def test_bench_reshape_ansi(benchmark, image):
    src = str(img2ansi(image, factor=FACTOR, char_set=CHAR_SET))
    benchmark(lambda: reshape_ansi(src, (16, FACTOR)))
