__all__ = [
    'ansi2img',
    'ansi_quantize',
    'ansify',
    'ascii2img',
    'contrast_stretch',
    'equalize_white_point',
    'get_font_key',
    'get_font_object',
    'img2ansi',
    'img2ascii',
    'otsu_mask',
    'read_ans',
    'render_ans',
    'render_font_char',
    'render_font_str',
    'reshape_ansi',
    'scale_saturation',
    'shuffle_char_set',
]

import collections.abc as abc
import enum
import os
import random
import re
import time
import typing as tp
from functools import lru_cache
from math import ceil
from shutil import get_terminal_size

import cv2 as cv
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from .. import _typing as _tp
from ..color import core
from ..color.colorconv import nearest_ansi_4bit_rgb, nearest_ansi_8bit_rgb
from ..color.palette import rgb_dispatch
from ..data import userfont as uf


def get_font_key(font: PIL.ImageFont.FreeTypeFont):
    """Obtain a unique tuple pair from a FreeTypeFont object.

    Parameters
    ----------
    font : FreeTypeFont
        The FreeTypeFont object from which to derive a key.

    Returns
    -------
    tuple[str, str]
        A tuple containing the font family and font name.

    Raises
    ------
    ValueError
        If the font key cannot be generated due to missing fields.
    """
    font = get_font_object(font)
    font_key = font.getname()
    if not all(font_key):
        missing = []
        s = 'font %s'
        if font_key[0] is None:
            missing.append(f"{s % 'name'!r}")
        if font_key[-1] is None:
            missing.append(f"{s % 'family'!r}")
        raise ValueError(
            f"Unable to generate font key due to missing fields {' and '.join(missing)}: "
            f"{font_key}"
        )
    return font_key


@tp.overload
def get_font_object(
    font: _tp.FontArgType, *, retpath: tp.Literal[False] = False
) -> PIL.ImageFont.FreeTypeFont: ...


@tp.overload
def get_font_object(font: _tp.FontArgType, *, retpath: tp.Literal[True]) -> str: ...


@tp.overload
def get_font_object(
    font: _tp.FontArgType, *, retpath: bool
) -> PIL.ImageFont.FreeTypeFont | str: ...


@lru_cache
def get_font_object(
    font: _tp.FontArgType, *, retpath: bool = False
) -> PIL.ImageFont.FreeTypeFont | str:
    """Return a FreeTypeFont object or its filepath.

    The result is cached to prevent FreeType from consuming excessive resources.

    Parameters
    ----------
    font : FontArgType
        FreeTypeFont, UserFont, or string.

    retpath : bool, optional
        Return filepath instead of FreeTypeFont object

    Returns
    -------
    FreeTypeFont or str
        FreeTypeFont object, or filepath (if `retpath=True`).

    Raises
    ------
    TypeError
        If the input type is unsupported.
    """

    if retpath:
        return (
            getattr(font.path, 'name', os.fspath(font.path))
            if isinstance(font, PIL.ImageFont.FreeTypeFont)
            else get_font_object(get_font_object(font), retpath=True)
        )
    else:
        match font:
            case PIL.ImageFont.FreeTypeFont():
                return font
            case uf.UserFont():
                return font.to_truetype()
            case str() if font in uf.userfonts:
                return get_font_object(uf.userfonts[font])
            case str() | os.PathLike():
                return PIL.ImageFont.truetype(font, 24)
    raise TypeError(
        f"Expected {PIL.ImageFont.FreeTypeFont.__name__!r} or pathlike object, "
        f"got {type(font).__name__!r} object instead"
    )


def shuffle_char_set(chars: abc.Iterable[str]):
    """Flatten `chars` into a list and return the randomly shuffled string.

    Parameters
    ----------
    chars : Iterable[str]
        Iterable of characters (or strings, which will be flattened).

    Returns
    -------
    str

    Raises
    ------
    TypeError
        If `chars` is not iterable, or contains non-strings.
    """
    xs = list(c for s in chars for c in s)
    random.shuffle(xs)
    return ''.join(xs)


def render_font_str(s: str, /, font: _tp.FontArgType):
    """Render a string as an image using the specified font.

    Parameters
    ----------
    s : str
        The string to render.

    font : FontArgType
        The font to use for rendering.

    Returns
    -------
    ImageType
        An image of the rendered string.

    Raises
    ------
    ValueError
        If the string is empty.
    """
    s = s.expandtabs(4)
    font = get_font_object(font)
    if len(s) > 1:
        lines = s.splitlines()
        maxlen = max(map(len, lines))
        stacked = np.vstack(
            [
                np.hstack(
                    [
                        np.array(render_font_char(c, font=font), dtype=np.uint8)
                        for c in line
                    ]
                )
                for line in map(lambda x: f'{x:<{maxlen}}', lines)
            ]
        )
        return PIL.Image.fromarray(stacked)
    return render_font_char(s, font)


def render_font_char(
    c: str,
    /,
    font: _tp.FontArgType,
    size=(24, 24),
    fill: _tp.Int3Tuple = (0xFF, 0xFF, 0xFF),
):
    """Render a one-character string as an image.

    Parameters
    ----------
    c : str
        Character to be rendered.

    font : FontArgType
        Font to use for rendering.

    size : tuple[int, int]
        Size of the bounding box to use for the output image, in pixels.

    fill : tuple[int, int, int]
        The color to fill the character.

    Returns
    -------
    Image :
        The character rendered in the given font.

    Raises
    ------
        ValueError : If the input string is longer than a single character.
    """
    if len(c) > 1:
        raise ValueError(f"expected a character, but string of length {len(c)} found")
    img = PIL.Image.new('RGB', size=size)
    draw = PIL.ImageDraw.Draw(img)
    font_obj = get_font_object(font)
    bbox = draw.textbbox((0, 0), c, font=font_obj)
    x_offset, y_offset = (
        (size[i] - (bbox[i + 2] - bbox[i])) // 2 - bbox[i] for i in range(2)
    )
    draw.text((x_offset, y_offset), c, font=font_obj, fill=fill)
    return img


def get_rgb_array(img: str | os.PathLike[str] | _tp.RGBImageLike, /):
    """Convert an input image into an RGB array.

    Parameters
    ----------
    img : str | PathLike[str] | RGBImageLike
        Input image or path to the image.

    Returns
    -------
    RGBArray

    Raises
    ------
    ValueError
        If the image format is invalid.

    TypeError
        If the input is not a valid image or path.
    """
    if isinstance(img, (str, os.PathLike)):
        x = cv.imread(os.fspath(img))
        if x is None:
            raise ValueError
        img = x
    if not _is_rgb_array(img):
        if _is_image(img):
            img = img.convert('RGB')
        elif _is_array(img):
            if img.ndim == 2:
                img = cv.cvtColor(img[:, :, 0], cv.COLOR_GRAY2RGB)
            elif img.ndim == 4:
                img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
            else:
                raise ValueError(f"unexpected array shape: {img.shape!r}")
        else:
            err = TypeError(
                _tp.type_error_msg(img, os.PathLike, PIL.Image.Image, np.ndarray)
            )
            raise err
        img = np.asarray(img, dtype=np.uint8)
    return img


def ansi_quantize(img: _tp.RGBArray, ansi_type: core.AnsiColorParam):
    """Color-quantize an RGB array into ANSI 4-bit or 8-bit color space.

    Parameters
    ----------
    img : RGBArray
        Input image in RGB format.

    ansi_type : type[ansicolor4Bit | ansicolor8Bit]
        ANSI color format to map the quantized image to.

    Raises
    ------
    TypeError
        If `ansi_type` is not ``ansi_color_4Bit`` or ``ansi_color_8Bit``.

    Returns
    -------
    quantized : RGBArray
        The image with RGB values transformed into ANSI color space.
    """
    ansi_type = core.get_ansi_type(ansi_type)
    if ansi_type is core.ansicolor4Bit:
        img = nearest_ansi_4bit_rgb(img)
    elif ansi_type is core.ansicolor8Bit:
        img = nearest_ansi_8bit_rgb(img)
    return img


def equalize_white_point(img: _tp.RGBArray) -> _tp.RGBArray:
    """Apply histogram equalization to the L-channel (lightness) in LAB color space.

    Parameters
    ----------
    img : RGBArray

    Returns
    -------
    eq_img : RGBArray

    See Also
    --------
    contrast_stretch
    """
    lab_img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    Lc, Ac, Bc = cv.split(lab_img)
    Lc_eq = cv.equalizeHist(Lc)
    lab_eq_img = cv.merge((Lc_eq, Ac, Bc))
    eq_img = cv.cvtColor(lab_eq_img, cv.COLOR_LAB2RGB)
    return eq_img


def contrast_stretch(
    img: _tp.RGBArray, percentile: tuple[int, int] = (2, 98)
) -> _tp.RGBArray:
    """Rescale the intensities of an RGB image using linear contrast stretching.

    Balances contrast across both lightness and color.

    Parameters
    ----------
    img : RGBArray
    percentile : tuple[int, int], optional

    Returns
    -------
    eq_img : RGBArray

    See Also
    --------
    equalize_white_point
    """
    imin, imax = np.percentile(img, percentile)
    out_dtype = img.dtype
    if issubclass(dt := img.dtype.type, np.integer):
        info = np.iinfo(dt)
        omin, omax = info.min, info.max
    elif issubclass(dt, np.inexact):
        omin, omax = -1, 1
    elif dt is np.bool_:
        omin, omax = False, True
    else:
        omin, omax = imin, imax
    omin, omax = map(float, (omin, omax))
    if imin >= 0:
        omin = 0.0
    img = np.clip(img, imin, imax)
    if imin != imax:
        img = (img - imin) / (imax - imin)
        return (img * (omax - omin) + omin).astype(out_dtype)
    else:
        return np.clip(img, omin, omax).astype(out_dtype)


def scale_saturation(
    img: _tp.RGBArray, alpha: tp.Optional[float] = None
) -> _tp.RGBArray:
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    img[:, :, 1] = cv.convertScaleAbs(img[:, :, 1], alpha=alpha or 1.0)
    img[:] = cv.cvtColor(img, cv.COLOR_HSV2RGB)
    return img


def _get_asciidraw_vars(
    img: str | os.PathLike[str] | _tp.RGBImageLike, font: _tp.FontArgType, /
):
    return get_rgb_array(img), get_font_object(font)


def _get_bbox_shape(font: PIL.ImageFont.FreeTypeFont, /):
    return font.getbbox(' ')[2:]


@tp.overload
def img2ascii(
    img: str | os.PathLike[str] | _tp.RGBImageLike,
    /,
    font: _tp.FontArgType = ...,
    factor: int = ...,
    char_set: tp.Optional[str] = ...,
    sort_glyphs: bool | tp.Literal[-1] = ...,
    *,
    ret_img: tp.Literal[False] = False,
) -> str: ...


@tp.overload
def img2ascii(
    img: str | os.PathLike[str] | _tp.RGBImageLike,
    /,
    font: _tp.FontArgType = ...,
    factor: int = ...,
    char_set: tp.Optional[str] = ...,
    sort_glyphs: bool | tp.Literal[-1] = ...,
    *,
    ret_img: tp.Literal[True],
) -> tuple[
    _tp.ShapedNDArray[tuple[int, int], np.str_],
    _tp.ShapedNDArray[tuple[int, int, tp.Literal[3]], np.uint8],
]: ...


def img2ascii(  # type: ignore
    img,
    /,
    font=uf.VGA437,
    factor=200,
    char_set=None,
    sort_glyphs=True,
    *,
    ret_img=False,
):
    """Convert an image to a multiline ASCII string.

    Parameters
    ----------
    img : str | os.PathLike[str] | RGBImageLike
        Base image being converted to ASCII.

    font : FontArgType
        Font to use for glyph comparisons and representation.

    factor : int
        Length of each line in characters per line in `output_str`. Affects level of detail.

    char_set : Iterable[str], optional
        Characters to be mapped to greyscale values of 'img'.

    sort_glyphs : {True, False, ``-1``}
        Specifies to sort `char_set` or leave it unsorted before mapping to greyscale.

        Glyph bitmasks obtained from 'font' are compared when sorting the string.

        ``-1`` specifies reverse sorting order.

    ret_img : bool, default=False
        Specifies to return both the output string and original RGB array.

        Used by ``img2ansi`` to lazily obtain the base ASCII chars and original RGB array.

    Returns
    -------
    output_str : str
        Characters from `char_set` mapped to the input image, as a multi-line string.

    Raises
    ------
    TypeError
        If `char_set` is of an unexpected type.

    See Also
    --------
    ascii2img : Render an ASCII string as an image.
    """
    rgb, font = _get_asciidraw_vars(img, font)
    assert isinstance(rgb, np.ndarray)
    grey: _tp.MatrixLike[np.uint8] = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    shape = grey.shape
    img_aspect = shape[-1] / shape[0]
    ch, cw = _get_bbox_shape(font)
    char_aspect = ceil(cw / ch)
    out_h = int(factor / img_aspect / char_aspect)
    out_w = factor
    blur = grey.astype(np.float64) / 255.0
    if (sy := (shape[0] / out_h - 1) / 2) > 0:
        blur = cv.filter2D(
            blur,
            -1,
            cv.getGaussianKernel(2 * int(4.0 * sy + 0.5) + 1, sy),
            borderType=cv.BORDER_REFLECT_101,
        )
    if (sx := (shape[1] / out_w - 1) / 2) > 0:
        blur = cv.filter2D(
            blur,
            -1,
            cv.getGaussianKernel(2 * int(4.0 * sx + 0.5) + 1, sx).T,
            borderType=cv.BORDER_REFLECT_101,
        )
    grey = (
        cv.resize(blur, (out_w, out_h), interpolation=cv.INTER_LINEAR) * 255.0
    ).astype(np.uint8)
    if char_set is None:
        if font.path is uf.VGA437:
            from ._curses import cp437_printable

            char_set = cp437_printable()
        else:
            from ._curses import ascii_printable

            char_set = ascii_printable()
        if not sort_glyphs:
            char_set = shuffle_char_set(char_set)
    if sort_glyphs:
        from ._glyph import sort_glyphs as glyph_sort

        char_set = glyph_sort(char_set, font, reverse=not ~int(sort_glyphs))
    chars = np.frombuffer(char_set.encode("utf-32-le"), dtype="<U1")
    if not chars.size:
        raise ValueError("empty charset")
    interp = chars[np.rint(grey / 255 * (chars.size - 1)).astype(np.intp)]
    if ret_img is True:
        return interp, rgb
    newlines = np.zeros((interp.shape[0], 1), dtype="<U1")
    newlines[:-1] = "\n"
    interp = np.concatenate((interp, newlines), axis=1)
    return "".join(interp.flat)


@tp.overload
def img2ansi(
    img: str | os.PathLike[str] | _tp.RGBImageLike,
    /,
    font: _tp.FontArgType = ...,
    factor: int = ...,
    char_set: tp.Optional[str] = ...,
    sort_glyphs: bool | tp.Literal[-1] = ...,
    ansi_type: tp.Optional[core.AnsiColorParam] = ...,
    equalize: bool | tp.Literal['white_point'] = ...,
    bg: tp.Optional[_tp.Int3Tuple | str] = ...,
    *,
    outarray: tp.Literal[False] = False,
) -> core.color_chain: ...


@tp.overload
def img2ansi(
    img: str | os.PathLike[str] | _tp.RGBImageLike,
    /,
    font: _tp.FontArgType = ...,
    factor: int = ...,
    char_set: tp.Optional[str] = ...,
    sort_glyphs: bool | tp.Literal[-1] = ...,
    ansi_type: tp.Optional[core.AnsiColorParam] = ...,
    equalize: bool | tp.Literal['white_point'] = ...,
    bg: tp.Optional[_tp.Int3Tuple | str] = ...,
    *,
    outarray: tp.Literal[True],
) -> _tp.ShapedNDArray[tuple[int, int], np.void]: ...


@rgb_dispatch('bg')
def img2ansi(
    img: str | os.PathLike[str] | _tp.RGBImageLike,
    /,
    font: _tp.FontArgType = uf.VGA437,
    factor: int = 200,
    char_set: tp.Optional[str] = None,
    sort_glyphs: bool | tp.Literal[-1] = True,
    ansi_type: tp.Optional[core.AnsiColorParam] = None,
    equalize: bool | tp.Literal['white_point'] = False,
    bg: tp.Optional[_tp.Int3Tuple | str] = None,
    *,
    outarray=False,
):
    """Convert an image to an ANSI array.

    Parameters
    ----------
    img : str | os.PathLike[str] | RGBImageLike
        Base image or path to image being convert into ANSI.

    font : FontArgType
        Font to use for glyph comparisons and representation.

    factor : int
        Length of each line in characters per line in `output_str`. Affects level of detail.

    char_set : str, optional
        The literal string or sequence of strings to use for greyscale interpolation and
        visualization.

        If None (default), the character set will be determined based on the 'font' parameter.

    sort_glyphs : {True, False, ``-1``}
        Specifies to sort `char_set` or leave it unsorted before mapping to greyscale.

        Glyph bitmasks obtained from 'font' are compared when sorting the string.

        ``-1`` specifies reverse sorting order.

    ansi_type : AnsiColorParam
        ANSI color format to map the RGB values to.

        Can be 4-bit, 8-bit, or 24-bit ANSI color space.

        If 4-bit or 8-bit, the RGB array will be color-quantized into ANSI color space;

        if 24-bit, uses the RGB colors of the image;

        if `None` (default), uses default ANSI type (4-bit or 8-bit, depending on the system).

    equalize : {True, False, 'white_point'}
        Apply contrast equalization to the input image.

        If True, performs contrast stretching;

        if 'white_point', applies white-point equalization.

    bg : sequence of ints or RGBArray
        Background color

    outarray : bool, default=False
        If True, an ndarray is returned instead of a color_chain object.

    Returns
    -------
    ansi_array : `color_chain` or ``ndarray[tuple[int, int], dtype[void]]``
        The ANSI-converted image.

    Raises
    ------
    ValueError
        If `bg` cannot be coerced into a ``Color`` object.

    TypeError
        If `ansi_type` is not a valid ANSI type.

    See Also
    --------
    ansi2img : Render an ANSI array as an image.
    img2ascii : Used to obtain the base ASCII characters.
    """
    if bg is None:
        pass
    elif not (isinstance(bg, tuple) and len(bg) == 3):
        raise TypeError
    s, rgb = img2ascii(img, font, factor, char_set, sort_glyphs, ret_img=True)
    h, w = s.shape
    if equalize is True:
        rgb = contrast_stretch(rgb)
    elif equalize == "white_point":
        rgb = equalize_white_point(rgb)
    ansi_type = core.get_ansi_type(ansi_type)
    rgb = ansi_quantize(rgb, ansi_type=ansi_type)
    with (
        PIL.Image.fromarray(rgb, mode='RGB') as img,
        img.resize((w, h), resample=PIL.Image.Resampling.LANCZOS) as resized,
    ):
        out = np.zeros(s.shape, dtype=core.color_chain.dtype)
        out["char"] = s
        out["rgb"][..., 0, 0] = ansi_type.typecode
        out["rgb"][..., 0, 1:] = np.asarray(resized, dtype=np.uint8)
        if bg is not None:
            out["rgb"][..., 1, 0] = ansi_type.typecode
            out["rgb"][..., 1, 1:] = bg
        return out if outarray is True else core.color_chain.fromarray(out)


@rgb_dispatch('fg', 'bg')
def ascii2img(
    s: str,
    /,
    font: _tp.FontArgType = uf.VGA437,
    font_size=16,
    *,
    fg: _tp.Int3Tuple | str = (0, 0, 0),
    bg: _tp.Int3Tuple | str = (0xFF, 0xFF, 0xFF),
):
    """Render a literal string as an image.

    Parameters
    ----------
    s : str
        The ASCII string to convert into an image.

    font : FontArgType
        Font to use for rendering the ASCII characters.

    font_size : int
        Font size in pixels for the rendered ASCII characters.

    fg : tuple[int, int, int]
        Foreground (text) color.

    bg : tuple[int, int, int]
        Background color.

    Returns
    -------
    ascii_img : Image
        A `PIL.Image.Image` object of the rendered ASCII string.

    See Also
    --------
    img2ascii : Convert an image into an ASCII string.
    """
    font = PIL.ImageFont.truetype(get_font_object(font, retpath=True), font_size)
    lines = s.split('\n')
    n_rows, n_cols = map(len, (lines, lines[0]))
    cw, ch = _get_bbox_shape(font)
    iw, ih = (int(i * j) for i, j in zip((cw, ch), (n_cols, n_rows)))
    r, g, b = tuple(map(int, bg))
    img = PIL.Image.new('RGB', (iw, ih), (r, g, b))
    draw = PIL.ImageDraw.Draw(img)
    y_offset = 0
    for line in lines:
        draw.text((0, y_offset), line, font=font, fill=fg)
        y_offset += ch
    return img


@rgb_dispatch('fg_default', 'bg_default')
def ansi2img(
    arr: (
        _tp.ShapedNDArray[tuple[int, int], np.void]
        | core.color_chain
        | list[core.color_chain]
        | list[list[core.ColorStr]]
    ),
    /,
    font: _tp.FontArgType = uf.VGA437,
    font_size=16,
    *,
    fg_default: _tp.Int3Tuple | _tp.TupleOf4[int] | str = (170, 170, 170),
    bg_default: _tp.Int3Tuple | _tp.TupleOf4[int] | str = (0, 0, 0),
):
    """Render an ANSI array as an image.

    Parameters
    ----------
    ansi_array : list[list[ColorStr]]
        A 2D list of ``ColorStr`` objects

    font : FontArgType
        Font to render the ANSI strings with.

    font_size : int
        Font size in pixels.

    fg_default : tuple[int, int, int] | tuple[int, int, int, int]
        Default foreground color of rendered text.

    bg_default : tuple[int, int, int] | tuple[int, int, int, int]
        Default background color of rendered text, and the fill color of the base canvas.

    Returns
    -------
    ansi_img : Image
        The rendered ANSI array as an `PIL.Image.Image` object.

    Raises
    ------
    ValueError
        If the input ANSI array is empty.

    See Also
    --------
    img2ansi : Create an ANSI array from an input image, font, and character set.
    """
    if isinstance(arr, core.color_chain):
        arr = arr.term_array()
    elif not isinstance(arr, np.ndarray):
        arr = np.asarray(
            [core.color_chain(x) for x in arr], dtype=core.color_chain.dtype
        )
    if not arr.size:
        raise ValueError("input array is empty")

    font = PIL.ImageFont.truetype(get_font_object(font, retpath=True), font_size)
    bbox_h = _get_bbox_shape(font)[-1]
    widths = np.asarray(
        [[font.getbbox(c)[2] for c in x["char"]] for x in arr], dtype=np.uint32
    )

    iw = widths.sum(axis=1).max().item()
    ih = round(arr.shape[0] * bbox_h)

    channels = [fg_default, bg_default]
    rgba = False

    for c in channels:
        x = len(c)
        if x == 4:
            rgba = True
        elif x != 3:
            raise ValueError
    if rgba:
        mode = "RGBA"
        rgba_descr = arr.dtype.descr.copy()
        rgb_field = rgba_descr[-1]
        subarr_shape = rgb_field[-1]
        subarr_shape = subarr_shape[:-1] + (subarr_shape[-1] + 1,)
        rgba_descr[-1] = rgb_field[:-1] + (subarr_shape,)
        arr = arr.astype(rgba_descr)
        arr["rgb"][..., 0, -1] = 0xFF
    else:
        mode = "RGB"

    for i, fill in enumerate(channels):
        mask = arr["rgb"][..., i, 0] == 0
        arr["rgb"][mask, i, 0] = 1
        arr["rgb"][mask, i, 1 : len(fill) + 1] = fill

    img = PIL.Image.new(mode, (iw, ih), bg_default)
    draw = PIL.ImageDraw.Draw(img)
    y_offset = 0
    for y in range(arr.shape[0]):
        x_offset = 0
        for x in range(arr.shape[1]):
            width = widths[y, x]
            item = arr[y, x]
            fg, bg = (tuple(ch) if ans else None for [ans, *ch] in item["rgb"].tolist())
            if bg is not None:
                draw.rectangle(
                    (x_offset, y_offset, x_offset + width, y_offset + bbox_h), fill=bg
                )
            draw.text((x_offset, y_offset), item["char"], font=font, fill=fg)
            x_offset += width
        y_offset += bbox_h
    return img


def ansify(
    img: str | os.PathLike[str] | _tp.RGBImageLike,
    /,
    font: _tp.FontArgType = uf.VGA437,
    font_size: int = 16,
    *,
    factor: int = 200,
    char_set: tp.Optional[str] = None,
    sort_glyphs: bool | tp.Literal[-1] = True,
    ansi_type: tp.Optional[core.AnsiColorParam] = None,
    equalize: bool | tp.Literal['white_point'] = False,
    fg: _tp.Int3Tuple | str = (170, 170, 170),
    bg: _tp.Int3Tuple | str = (0, 0, 0),
):
    ansi_type = core.get_ansi_type(ansi_type)
    return ansi2img(
        img2ansi(
            img,
            font,
            factor=factor,
            char_set=char_set,
            ansi_type=ansi_type,
            sort_glyphs=sort_glyphs,
            equalize=equalize,
            bg=bg,
            outarray=True,
        ),
        font,
        font_size=font_size,
        fg_default=fg,
        bg_default=bg,
    )


def _is_array(obj: tp.Any, /) -> tp.TypeGuard[np.ndarray]:
    return isinstance(obj, np.ndarray)


def _is_rgb_array(obj: tp.Any, /) -> tp.TypeGuard[_tp.RGBArray]:
    return _is_array(obj) and obj.ndim == 3 and np.issubdtype(obj.dtype, np.uint8)


def _is_image(obj: tp.Any, /) -> tp.TypeGuard[PIL.Image.Image]:
    return isinstance(obj, PIL.Image.Image)


@lru_cache(maxsize=1)
def cursor_or_sgr_pattern():
    sgr_re = core.sgr_pattern().pattern.removeprefix(r'\x1b\[')
    return re.compile(
        r"(?:"
        r"\x1b\[(?:"
        r"(?P<cursor>\d*[A-G]|\d*(?:;\d*)?H)"
        f"|(?P<sgr>{sgr_re})"
        r")"
        r"|(?P<carriage_return>\r)"
        r")?(?P<text>[^\x1b\r]*)"
    )


class ReshapeAnsiFlag(enum.IntFlag):
    BOLD_COLORS = enum.auto()
    """SGR code `1` enables 'bright' foreground colors (IBM VGA convention)"""
    BOLD_FONT = enum.auto()
    """Keep the bold bit after promotion (does nothing without `BOLD_COLORS`)"""
    RESET_BOLD_AND_FAINT = enum.auto()
    """SGR code `22` clears bold bit"""
    ICE_COLORS = enum.auto()
    """SGR code `5` enables 'bright' background colors (iCE colors)"""


def _sgr_state_updater[_T: core.SgrSequence](flags: int, /) -> abc.Callable[[_T], _T]:
    """Return a callable mapping each SGR escape to the accumulated SGR state."""
    fg: bytes | None = None
    bg: bytes | None = None
    other: dict[bytes, None] = {}
    bold_fg = bold_bg = False
    cache: dict[tuple[bytes, ...], _T] = {}

    def update(sgr: _T, /) -> _T:
        nonlocal fg, bg, bold_fg, bold_bg
        is_reset = False
        for p in sgr:
            if p.is_reset():
                fg = bg = None
                other.clear()
                is_reset = True
                bold_fg = bold_bg = False
            elif p.is_color():
                v = p.value
                if v.kind() == "fg":
                    fg = v
                else:
                    bg = v
            elif p == b"1":
                bold_fg = True
            elif p == b"5":
                bold_bg = bold_bg or flags & ReshapeAnsiFlag.ICE_COLORS
            elif p == b"22":
                bold_fg = bold_fg and not flags & ReshapeAnsiFlag.RESET_BOLD_AND_FAINT
            elif p == b"39":
                fg = None
            elif p == b"49":
                bg = None
            else:
                other[p.value] = None
        out = []
        if is_reset:
            out.append(b"0")
        f, b = fg, bg
        if bold_fg:
            if flags & ReshapeAnsiFlag.BOLD_COLORS:
                if flags & ReshapeAnsiFlag.BOLD_FONT:
                    out.append(b"1")
                if f is None:
                    f = b"97"
                elif f.isdigit() and 30 <= (x := int(f)) <= 37:
                    f = b"%d" % (x + 60)
                elif ReshapeAnsiFlag.BOLD_FONT & ~flags:
                    out.append(b"1")
            else:
                out.append(b"1")
        if bold_bg:
            if flags & ReshapeAnsiFlag.ICE_COLORS:
                if b is None:
                    b = b"107"
                elif b.isdigit() and 40 <= (x := int(b)) <= 47:
                    b = b"%d" % (x + 60)
                else:
                    out.append(b"5")
            else:
                out.append(b"5")
        if f is not None:
            out.append(f)
        if b is not None:
            out.append(b)
        out.extend(other)
        key = tuple(out)
        try:
            return cache[key]
        except KeyError:
            return cache.setdefault(key, sgr.__class__(out))

    return update


def reshape_ansi(s: str, /, shape: tuple[int, int], flags=0) -> core.color_chain:
    """Return the string padded for a grid with dims `shape`.

    The output string represents a terminal render after stateful transitions
    have been applied.

    Cursor codes and `'\\r'` are consumed and resolved to character emplacements,
    and null character cells are translated to whitespace (0x20).

    Parameters
    ----------
    s : str

    shape : tuple[int, int]
        Shape of the output as (width, height). Must be 2D.

    flags : int, default=0
        Additonal flags for state transitions. See ``ReshapeAnsiFlag`` for more
        info.

    Returns
    -------
    out : str
        Reshaped string with ANSI escape state transitions applied.
    """
    w, h = shape
    total = w * h

    chars = np.zeros(total, dtype="<U1")
    sgr_ids = np.full(total, -1, dtype=np.intp)

    pos = y = x = 0

    def move(i: int):
        nonlocal pos, y, x
        pos = min(max(i, 0), total - 1)
        y, x = divmod(pos, w)

    cursor_code: dict[str, abc.Callable[[int], None]] = {
        "A": lambda n: move(max(0, y - n) * w + x),
        "B": lambda n: move(min(h - 1, y + n) * w + x),
        "C": lambda n: move(y * w + min(w - 1, x + n)),
        "D": lambda n: move(y * w + max(0, x - n)),
        "E": lambda n: move(min(h - 1, y + n) * w),
        "F": lambda n: move(max(0, y - n) * w),
        "G": lambda n: move(y * w + min(w - 1, max(0, n - 1))),
        "H": move,
    }
    cursor_crlf: dict[str, abc.Callable[[], None]] = {
        "\r": lambda: move(y * w),
        "\n": lambda: move(min(h - 1, y + 1) * w),
    }

    finditer = cursor_or_sgr_pattern().finditer
    update_sgr = _sgr_state_updater(flags)
    sgr_buf: list[core.SgrSequence] = [update_sgr(core.SgrSequence())]
    seen = False
    for line in s.split("\n"):
        for m in finditer(line):
            if cg := m["cursor"]:
                nums, code = cg[:-1], cg[-1]
                if code == "H":
                    y_, x_ = (max(0, int(n or 1) - 1) for n in nums.partition(";")[::2])
                    n = min(y_, h - 1) * w + min(x_, w - 1)
                else:
                    n = int(nums or 1)
                cursor_code[code](n)
            elif m["carriage_return"]:
                cursor_crlf["\r"]()
            elif sgr_s := m["sgr"]:
                sgr_buf.append(update_sgr(core.SgrSequence(sgr_s[:-1].encode())))
                seen = True
            if text := m["text"]:
                count = min(len(text), total - pos)
                span = slice(pos, pos + count)
                chars[span] = [*text[:count]]
                sgr_ids[span] = len(sgr_buf) - 1
                move(pos + count)
        cursor_crlf["\n"]()

    chars[~chars.astype(np.bool_)] = " "
    if not seen:
        return core.color_chain("\n".join(map("".join, chars.reshape(h, w))))

    # sgr state ffill
    src = np.where(sgr_ids >= 0, np.arange(total), 0)
    np.maximum.accumulate(src, out=src)
    cell_ids = sgr_ids[src]

    keys: dict[bytes, int] = {}
    i2k = np.empty(len(sgr_buf) + 1, dtype=np.intp)
    i2k[0] = -1
    for i, sgr in enumerate(sgr_buf):
        i2k[i + 1] = keys.setdefault(bytes(sgr), len(keys))
    cell_keys = i2k[cell_ids + 1]

    out = []
    prev_key = None
    was_esc = was_str = paired = False
    for r in range(h):
        lo = r * w
        r_keys = cell_keys[lo : lo + w]
        r_chars = chars[lo : lo + w]
        starts = np.flatnonzero(np.r_[True, r_keys[1:] != r_keys[:-1]])
        for i, start in enumerate(starts):
            stop = starts[i + 1] if i + 1 < starts.size else w
            key = int(r_keys[start])
            if key != prev_key and key >= 0:
                out.append(sgr_buf[int(cell_ids[lo + start])])
                was_esc = True
                paired = False
            prev_key = key
            s = "".join(r_chars[start:stop])
            if was_esc:
                out[-1] = [out[-1], s]
                was_esc = False
                paired = True
            elif paired:
                out[-1][1] += s
            elif was_str:
                out[-1] += s
            else:
                out.append(s)
            was_str = True
        if r < h - 1:
            if paired:
                out[-1][1] += "\n"
            elif was_str:
                out[-1] += "\n"
            else:
                out.append("\n")
            was_str = True
    return core.color_chain(
        x if isinstance(x, (core.SgrSequence, str)) else tuple(x) for x in out
    )


class ANSiFlag(enum.IntFlag):
    ICE_COLORS = 0b000001
    """Use iCE Color (non-blink mode)"""
    LS_8_PX = 0b000010
    """Use 8 pixel letter-spacing variant of the font"""
    LS_9_PX = 0b000100
    """Use 9 pixel letter-spacing variant of the font"""
    AR_LEGACY = 0b001000
    """Image assumes aspect ratio of legacy device (stretching required)"""
    AR_MODERN = 0b010000
    """Image assumes aspect ratio of modern device (no stretching)"""


def _parse_sauce(rec: bytes, /) -> dict[str, tp.Any]:
    if not rec.startswith(b"SAUCE"):
        raise ValueError
    import struct

    keys, fields = zip(
        ("id", "5s"),
        ("version", "2s"),
        ("title", "35s"),
        ("author", "20s"),
        ("group", "20s"),
        ("date", "8s"),
        ("filesize", "I"),
        ("datatype", "B"),
        ("filetype", "B"),
        ("tinfo1", "H"),
        ("tinfo2", "H"),
        ("tinfo3", "H"),
        ("tinfo4", "H"),
        ("comments", "B"),
        ("tflags", "B"),
        ("tinfos", "22s"),
    )
    values = struct.unpack("".join(["<", *fields]), rec)
    return {
        k: v.rstrip(b"\0 ").decode("cp437") if isinstance(v, bytes) else v
        for k, v in zip(keys, values)
    }


class _AnsiFileKwargs(tp.TypedDict, total=False):
    filetype: int
    date: time.struct_time
    columns: tp.Required[int]
    lines: tp.Required[int]
    comments: str | int
    ansiflags: tp.Required[int]
    fontname: tp.Required[str | None]


def read_ans(
    buf: tp.BinaryIO, /, fallback: tuple[int, int] | None = None
) -> tuple[str, _AnsiFileKwargs]:
    fallback = get_terminal_size() if fallback is None else get_terminal_size(fallback)
    buf.seek(-128, 2)
    if buf.read(5) == b"SAUCE":
        buf.seek(-5, 1)
        d = _parse_sauce(buf.read())
        if d["filetype"] > 2:
            raise ValueError(
                "unexpected filetype from SAUCE record "
                "(not ASCII or ANSi): {filetype}".format_map(d)
            )   # fmt: skip
        del d["id"], d["version"]
        d["date"] = time.strptime(d["date"], "%Y%m%d")
        d["columns"] = d.pop("tinfo1") or fallback.columns
        d["lines"] = d.pop("tinfo2") or fallback.lines
        del d["tinfo3"], d["tinfo4"]
        if n_comments := d["comments"]:
            buf.seek(-sum([128, n_comments * 64, 5]), 2)
            if buf.read(5) != b"COMNT":
                d["comments"] = False
            else:
                comments = []
                for _ in range(n_comments):
                    comment = buf.read(64).rstrip(b"\0 ").decode("cp437")
                    comments.append(comment)
                d["comments"] = "\n".join(comments)
        d["ansiflags"] = ANSiFlag(d.pop("tflags"))
        d["fontname"] = d.pop("tinfos") or None
        buf.seek(0)
        size = d["filesize"]
        content = buf.read(size)
    else:
        buf.seek(0)
        content = buf.read()
        d = {
            "columns": fallback.columns,
            "lines": fallback.lines,
            "ansiflags": 0,
            "fontname": None,
        }
    return content.decode("cp437").rstrip("\x1a"), d  # type: ignore[return-type]


def render_ans(
    buf: tp.BinaryIO,
    /,
    fallback: tuple[int, int] | None = None,
    font: _tp.FontArgType | None = None,
    font_size: int = 16,
    *,
    bg_default: _tp.Int3Tuple | _tp.TupleOf4[int] | str = (0, 0, 0),
) -> PIL.Image.Image:
    """Return an image render of an ANS file.

    Parameters
    ----------
    s : str
        Literal ANSI text.

    fallback : tuple[int, int]
        ``(columns, lines)`` of the ANS file, if no SAUCE record is present.

        Defaults to ``shutil.get_terminal_size()``

    font : FontArgType
        Font to draw the image. Overrides SAUCE record if present.

    font_size : int
        Font size in pixels.

    bg_default : tuple[int, int, int] or tuple[int, int, int, int]
        Background color to use as a fallback when ANSI SGR has none.
    """
    content, d = read_ans(buf, fallback=fallback)
    if fallback is None:
        fallback = d["columns"], d["lines"]
    if font is None:
        if (fontname := d["fontname"]) and fontname in uf.userfonts:
            font = uf.userfonts[fontname]
        elif (fontname or "").startswith(("IBM VGA", "IBM EGA")):
            font = uf.VGA437
        else:
            font = uf.DEFAULT_FONT
    flags = ReshapeAnsiFlag.BOLD_COLORS
    if d["ansiflags"] & ANSiFlag.ICE_COLORS:
        flags |= ReshapeAnsiFlag.ICE_COLORS
    norm = reshape_ansi(content, fallback, flags)
    arr = [
        [core.ColorStr(f"{sgr}{s}") for sgr, s in line] for line in norm.splitlines()
    ]
    return ansi2img(arr, font, font_size, bg_default=bg_default)


def otsu_mask(
    img: tp.Union[PIL.Image.Image, _tp.MatrixLike[np.uint8]],
) -> _tp.MatrixLike[np.uint8]:
    if type(img) is not np.ndarray:
        img = np.uint8(img)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    return cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
