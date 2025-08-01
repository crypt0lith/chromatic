from __future__ import annotations

__all__ = [
    'AnsiImage',
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
    'scaled_hu_moments',
    'shuffle_char_set',
    'to_sgr_array',
]

import math
import os
import random
import re
from functools import lru_cache, partial
from io import TextIOWrapper
from os import PathLike
from typing import TYPE_CHECKING, cast, overload

import cv2 as cv
import numpy as np
import skimage as ski
from PIL import Image, ImageDraw
from PIL.Image import Image as ImageType
from PIL.ImageFont import FreeTypeFont, truetype
from sklearn.cluster import DBSCAN

from ..color.colorconv import nearest_ansi_4bit_rgb, nearest_ansi_8bit_rgb
from ..color.core import (
    Color,
    ColorStr,
    DEFAULT_ANSI,
    SGR_RESET,
    SgrSequence,
    ansicolor24Bit,
    ansicolor4Bit,
    ansicolor8Bit,
    get_ansi_type,
    sgr_re_pattern,
)
from ..color.palette import rgb_dispatch
from ..data import UserFont, userfont

if TYPE_CHECKING:
    from _typeshed import SupportsRead
    from typing import (
        Any,
        AnyStr,
        Callable,
        Generator,
        Iterable,
        Iterator,
        Literal,
        Optional,
        Self,
        Sequence,
        TypeAlias,
        TypeGuard,
        Union,
    )
    from ..color.core import AnsiColorParam, AnsiColorType
    from .._typing import (
        FontArgType,
        GreyscaleArray,
        GreyscaleGlyphArray,
        Int3Tuple,
        MatrixLike,
        RGBArray,
        RGBImageLike,
        ShapedNDArray,
        TupleOf2,
    )

    LiteralDigit: TypeAlias = Sequence[
        Literal['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    ]


def get_font_key(font: FreeTypeFont):
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
    return cast(tuple[str, str], font_key)


@overload
def get_font_object(
    font: FontArgType, *, retpath: Literal[False] = False
) -> FreeTypeFont: ...


@overload
def get_font_object(font: FontArgType, *, retpath: Literal[True]) -> AnyStr: ...


@overload
def get_font_object(font: FontArgType, *, retpath: bool) -> FreeTypeFont | AnyStr: ...


@lru_cache
def get_font_object(
    font: FontArgType, *, retpath: bool = False
) -> FreeTypeFont | AnyStr:
    """Return a FreeTypeFont object or its filepath.

    The result is cached to prevent FreeType from overloading resources.

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
            if isinstance(font, FreeTypeFont)
            else get_font_object(get_font_object(font), retpath=True)
        )
    else:
        match font:
            case FreeTypeFont():
                return font
            case UserFont():
                return font.to_truetype()
            case str() if font in userfont:
                return get_font_object(userfont[font])
            case str() | PathLike():
                return truetype(font, 24)
    raise TypeError(
        f"Expected {FreeTypeFont.__name__!r} or pathlike object, "
        f"got {type(font).__name__!r} object instead"
    )


def shuffle_char_set(chars: Iterable[str]):
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


def render_font_str(__s: str, font: FontArgType):
    """Render a string as an image using the specified font.

    Parameters
    ----------
    __s : str
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
    __s = __s.expandtabs(4)
    font = get_font_object(font)
    if len(__s) > 1:
        lines = __s.splitlines()
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
        return Image.fromarray(stacked)
    return render_font_char(__s, font)


def render_font_char(
    __c: str, font: FontArgType, size=(24, 24), fill: Int3Tuple = (0xFF, 0xFF, 0xFF)
):
    """Render a one-character string as an image.

    Parameters
    ----------
    __c : str
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
    if len(__c) > 1:
        raise ValueError(
            f"{render_font_char.__name__}() expected a character, "
            f"but string of length {len(__c)} found"
        )
    img = Image.new('RGB', size=size)
    draw = ImageDraw.Draw(img)
    font_obj = get_font_object(font)
    bbox = draw.textbbox((0, 0), __c, font=font_obj)
    x_offset, y_offset = (
        (size[i] - (bbox[i + 2] - bbox[i])) // 2 - bbox[i] for i in range(2)
    )
    draw.text((x_offset, y_offset), __c, font=font_obj, fill=fill)
    return img


def get_rgb_array(__img: str | PathLike[str] | RGBImageLike):
    """Convert an input image into an RGB array.

    Parameters
    ----------
    __img : RGBImageLike | PathLike[str] | str
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
    if hasattr(__img, '__fspath__'):
        img = ski.io.imread(os.fspath(__img))
    elif isinstance(__img, str):
        img = ski.io.imread(__img)
    else:
        img = __img
    if not _is_rgb_array(img):
        if _is_image(img):
            img = img.convert('RGB')
        elif _is_array(img):
            conv = dict.get(
                {
                    2: lambda im: cv.cvtColor(im[:, :, 0], cv.COLOR_GRAY2RGB),
                    4: lambda im: cv.cvtColor(im, cv.COLOR_RGBA2RGB),
                },
                img.ndim,
            )
            if conv is None:
                raise ValueError(f"unexpected array shape: {img.shape!r}")
            img = conv(img)
        else:
            from .._typing import type_error_msg

            err = TypeError(type_error_msg(img, PathLike, ImageType, np.ndarray))
            raise err
        img = np.uint8(img)
    return img


def _rgb_transform2vec(pyfunc: Callable[[Int3Tuple], Int3Tuple]) -> np.ufunc:
    return np.frompyfunc(lambda *rgb: pyfunc(rgb), 3, 3)


def _apply_rgb_ufunc(img: RGBArray, rgb_ufunc: np.ufunc) -> RGBArray:
    return np.uint8(rgb_ufunc(*np.moveaxis(img, -1, 0))).transpose(1, 2, 0)


_ANSI_QUANTIZERS = {
    t: partial(_apply_rgb_ufunc, rgb_ufunc=_rgb_transform2vec(f))
    for t, f in zip(
        (ansicolor4Bit, ansicolor8Bit), (nearest_ansi_4bit_rgb, nearest_ansi_8bit_rgb)
    )
}


def ansi_quantize(
    img: RGBArray,
    ansi_type: type[ansicolor4Bit | ansicolor8Bit],
    *,
    equalize: bool | Literal['white_point'] = True,
):
    """Color-quantize an RGB array into ANSI 4-bit or 8-bit color space.

    Parameters
    ----------
    img : RGBArray
        Input image in RGB format.

    ansi_type : type[ansicolor4Bit | ansicolor8Bit]
        ANSI color format to map the quantized image to.

    equalize : {True, False, 'white_point'}
        Apply contrast equalization before ANSI color quantization.
        If True, performs contrast stretching;
        if 'white_point', applies white-point equalization.

    Raises
    ------
    TypeError
        If `ansi_type` is not ``ansi_color_4Bit`` or ``ansi_color_8Bit``.

    Returns
    -------
    quantized : RGBArray
        The image with RGB values transformed into ANSI color space.
    """
    try:
        quantizer = _ANSI_QUANTIZERS[ansi_type]
    except KeyError:
        from .._typing import unionize, type_error_msg

        err = TypeError(
            type_error_msg(
                ansi_type,
                context=f"ansi_type={type[unionize(_ANSI_QUANTIZERS.keys())]}",
            )
        )
        raise err from None
    if eq_f := dict.get(
        {True: contrast_stretch, 'white_point': equalize_white_point}, equalize
    ):
        img = eq_f(img)
    if img.size > 1024**2:  # downsize for faster quantization
        w, h, _ = img.shape
        new_w, new_h = (int(x * 768 / max(w, h)) for x in (w, h))
        img = np.array(
            Image.fromarray(img, mode='RGB').resize(
                (new_h, new_w), resample=Image.Resampling.LANCZOS
            )
        )
    return quantizer(img)


def equalize_white_point(img: RGBArray) -> RGBArray:
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


def contrast_stretch(img: RGBArray, percentile: tuple[int, int] = (2, 98)) -> RGBArray:
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
    lo, hi = np.percentile(img, percentile)
    return cast(..., ski.exposure.rescale_intensity(cast(..., img), in_range=(lo, hi)))


def scale_saturation(img: RGBArray, alpha: float = None) -> RGBArray:
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    img[:, :, 1] = cv.convertScaleAbs(img[:, :, 1], alpha=alpha or 1.0)
    img[:] = cv.cvtColor(img, cv.COLOR_HSV2RGB)
    return img


def _get_asciidraw_vars(__img: str | PathLike[str] | RGBImageLike, __font: FontArgType):
    img = get_rgb_array(__img)
    font = get_font_object(__font)
    return img, font


def _get_bbox_shape(__font: FreeTypeFont):
    return cast(tuple[float, float], __font.getbbox(' ')[2:])


@overload
def img2ascii(
    __img: RGBImageLike | str | PathLike[str],
    __font: FontArgType = ...,
    factor: int = ...,
    char_set: Optional[str] = ...,
    sort_glyphs: bool | type[reversed] = ...,
    *,
    ret_img: Literal[False] = False,
) -> str: ...


@overload
def img2ascii(
    __img: RGBImageLike | str | PathLike[str],
    __font: FontArgType = ...,
    factor: int = ...,
    char_set: Optional[str] = ...,
    sort_glyphs: bool | type[reversed] = ...,
    *,
    ret_img: Literal[True],
) -> tuple[str, RGBArray]: ...


def img2ascii(
    __img: RGBImageLike | PathLike[str] | str,
    __font: FontArgType = 'arial.ttf',
    factor: int = 100,
    char_set: Iterable[str] = None,
    sort_glyphs: bool | type[reversed] = True,
    *,
    ret_img: bool = False,
) -> str | tuple[str, RGBArray]:
    """Convert an image to a multiline ASCII string.

    Parameters
    ----------
    __img : RGBImageLike | PathLike[str] | str
        Base image being converted to ASCII.

    __font : FontArgType
        Font to use for glyph comparisons and representation.

    factor : int
        Length of each line in characters per line in `output_str`. Affects level of detail.

    char_set : Iterable[str], optional
        Characters to be mapped to greyscale values of `__img`.

    sort_glyphs : {True, False, ``reversed``}
        Specifies to sort `char_set` or leave it unsorted before mapping to greyscale.
        Glyph bitmasks obtained from `__font` are compared when sorting the string.
        ``reversed`` specifies reverse sorting order.

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
    img, font = _get_asciidraw_vars(__img, __font)
    greyscale: MatrixLike[np.uint8] = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img_shape = greyscale.shape
    img_aspect = img_shape[-1] / img_shape[0]
    ch, cw = _get_bbox_shape(font)
    char_aspect = math.ceil(cw / ch)
    new_height = int(factor / img_aspect / char_aspect)
    greyscale = ski.transform.resize(greyscale, (new_height, factor))
    if char_set is None:
        from ._curses import ascii_printable, cp437_printable

        chars_getter = dict.get(
            {userfont['vga437']: cp437_printable}, __font, ascii_printable
        )
        char_set = shuffle_char_set(chars_getter())
    elif type(char_set) is not str:
        char_set = ''.join(char_set)
    if sort_glyphs in {True, reversed}:
        from ._glyph_proc import sort_glyphs as glyph_sort

        char_set = glyph_sort(char_set, font, reverse=(sort_glyphs is reversed))
    maxlen = len(char_set) - 1
    interp_charset = np.frompyfunc(lambda x: char_set[int(x * maxlen)], 1, 1)
    ascii_str = '\n'.join(map(''.join, interp_charset(greyscale)))
    if ret_img:
        return ascii_str, img
    return ascii_str


@rgb_dispatch(('bg',))
def img2ansi(
    __img: RGBImageLike | PathLike[str] | str,
    __font: FontArgType = 'arial.ttf',
    factor: int = 100,
    char_set: Iterable[str] = None,
    ansi_type: AnsiColorParam = DEFAULT_ANSI,
    sort_glyphs: bool | type[reversed] = True,
    equalize: bool | Literal['white_point'] = True,
    bg: Color | Int3Tuple | str = (0, 0, 0),
):
    """Convert an image to an ANSI array.

    Parameters
    ----------
    __img : RGBImageLike | PathLike[str] | str
        Base image or path to image being convert into ANSI.

    __font : FontArgType
        Font to use for glyph comparisons and representation.

    factor : int
        Length of each line in characters per line in `output_str`. Affects level of detail.

    char_set : Iterable[str], optional
        The literal string or sequence of strings to use for greyscale interpolation and
        visualization.
        If None (default), the character set will be determined based on the `__font` parameter.

    ansi_type : AnsiColorParam
        ANSI color format to map the RGB values to.
        Can be 4-bit, 8-bit, or 24-bit ANSI color space.
        If 4-bit or 8-bit, the RGB array will be color-quantized into ANSI color space;
        if 24-bit, colors are sourced from the base RGB array;
        if None (default), uses default ANSI type (4-bit or 8-bit, depending on the system).

    sort_glyphs : {True, False, ``reversed``}
        Specifies to sort `char_set` or leave it unsorted before mapping to greyscale.
        Glyph bitmasks obtained from `__font` are compared when sorting the string.
        ``reversed`` specifies reverse sorting order.

    equalize : {True, False, 'white_point'}
        Apply contrast equalization to the input image.
        If True, performs contrast stretching;
        if 'white_point', applies white-point equalization.

    bg : sequence of ints or RGBArray
        Background color to use for all ``ColorStr`` objects in the array.

    Returns
    -------
    ansi_array : list[list[ColorStr]]
        The ANSI-converted image, as an array of ``ColorStr`` objects.

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
    if ansi_type is not DEFAULT_ANSI:
        ansi_type = get_ansi_type(ansi_type)
    bg_wrapper = ColorStr('%s', bg=bg, ansi_type=ansi_type, reset=False)
    base_ascii, color_arr = img2ascii(
        __img, __font, factor, char_set, sort_glyphs, ret_img=True
    )
    lines = base_ascii.splitlines()
    h, w = map(len, (lines, lines[0]))
    if ansi_type is not ansicolor24Bit:
        color_arr = ansi_quantize(color_arr, ansi_type=ansi_type, equalize=equalize)
    elif eq_func := dict.get(
        {True: contrast_stretch, 'white_point': equalize_white_point}, equalize
    ):
        color_arr = eq_func(color_arr)
    color_arr = Image.fromarray(color_arr, mode='RGB').resize(
        (w, h), resample=Image.Resampling.LANCZOS
    )
    xs = []
    for i in range(h):
        x = []
        for j in range(w):
            char = lines[i][j]
            fg_color = Color.from_rgb(color_arr.getpixel([j, i]))
            if j > 0 and x[-1].fg == fg_color:
                x[-1] += char
            else:
                x.append(ColorStr.recolor(bg_wrapper % char, fg=fg_color))
        xs.append(x)
    return xs


@rgb_dispatch(('fg', 'bg'))
def ascii2img(
    __ascii: str,
    __font: FontArgType = 'arial.ttf',
    font_size=24,
    *,
    fg: Int3Tuple | str = (0, 0, 0),
    bg: Int3Tuple | str = (0xFF, 0xFF, 0xFF),
):
    """Render a literal string as an image.

    Parameters
    ----------
    __ascii : str
        The ASCII string to convert into an image.

    __font : FontArgType
        Font to use for rendering the ASCII characters.

    font_size : int
        Font size in pixels for the rendered ASCII characters.

    fg : tuple[int, int, int]
        Foreground (text) color.

    bg : tuple[int, int, int]
        Background color.

    Returns
    -------
    ascii_img : ImageType
        An Image object of the rendered ASCII string.

    See Also
    --------
    img2ascii : Convert an image into an ASCII string.
    """
    font = truetype(get_font_object(__font, retpath=True), font_size)
    lines = __ascii.split('\n')
    n_rows, n_cols = map(len, (lines, lines[0]))
    cw, ch = _get_bbox_shape(font)
    iw, ih = (int(i * j) for i, j in zip((cw, ch), (n_cols, n_rows)))
    img = Image.new('RGB', (iw, ih), tuple(map(int, bg)))
    draw = ImageDraw.Draw(img)
    y_offset = 0
    for line in lines:
        draw.text((0, y_offset), line, font=font, fill=fg)
        y_offset += ch
    return img


@rgb_dispatch(('fg_default', 'bg_default'))
def ansi2img(
    __ansi_array: list[list[ColorStr]],
    __font: FontArgType = 'arial.ttf',
    font_size=24,
    *,
    fg_default: Int3Tuple | str = (170, 170, 170),
    bg_default: Int3Tuple | str | Literal['auto'] = 'auto',
):
    """Render an ANSI array as an image.

    Parameters
    ----------
    __ansi_array : list[list[ColorStr]]
        A 2D list of ``ColorStr`` objects

    __font : FontArgType
        Font to render the ANSI strings with.

    font_size : int
        Font size in pixels.

    fg_default : tuple[int, int, int]
        Default foreground color of rendered text.

    bg_default : tuple[int, int, int]
        Default background color of rendered text, and the fill color of the base canvas.

    Returns
    -------
    ansi_img : ImageType
        The rendered ANSI array as an ``Image`` object.

    Raises
    ------
    ValueError
        If the input ANSI array is empty.

    See Also
    --------
    img2ansi : Create an ANSI array from an input image, font, and character set.
    """
    if not (n_rows := len(__ansi_array)):
        raise ValueError('ANSI string input is empty')
    font = truetype(get_font_object(__font, retpath=True), font_size)
    row_height = _get_bbox_shape(font)[-1]
    max_row_width = max(
        sum(font.getbbox(color_str.base_str)[2] for color_str in row)
        for row in __ansi_array
    )
    iw, ih = map(int, (max_row_width, n_rows * row_height))
    input_fg = fg_default
    if auto := bg_default == 'auto':
        input_bg = bg_default = None
    else:
        input_bg = bg_default
    img = Image.new('RGB', (iw, ih), cast(tuple[float, ...], bg_default))
    draw = ImageDraw.Draw(img)
    y_offset = 0
    for row in __ansi_array:
        x_offset = 0
        for color_str in row:
            text_width = font.getbbox(color_str.base_str)[2]
            if getattr(color_str, '_sgr').is_reset():
                fg_default = None
                bg_default = input_bg
            if fg_color := getattr(color_str.fg, 'rgb', fg_default):
                fg_default = fg_color
            if bg_color := getattr(color_str.bg, 'rgb', bg_default):
                if auto:
                    bg_default = bg_color
            draw.rectangle(
                [x_offset, y_offset, x_offset + text_width, y_offset + row_height],
                fill=bg_color or (0, 0, 0),
            )
            draw.text(
                (x_offset, y_offset),
                color_str.base_str,
                font=font,
                fill=fg_color or input_fg,
            )
            x_offset += text_width
        y_offset += row_height
    return img


def ansify(
    __img: RGBImageLike | PathLike[str] | str,
    __font: FontArgType = userfont['vga437'],
    font_size: int = 16,
    *,
    factor: int = 200,
    char_set: Iterable[str] = None,
    sort_glyphs: bool | type[reversed] = True,
    ansi_type: AnsiColorParam = DEFAULT_ANSI,
    equalize: bool | Literal['white_point'] = True,
    fg: Int3Tuple | str = (170, 170, 170),
    bg: Int3Tuple | str | Literal['auto'] = (0, 0, 0),
):
    return ansi2img(
        img2ansi(
            __img,
            __font,
            factor=factor,
            char_set=char_set,
            ansi_type=ansi_type,
            sort_glyphs=sort_glyphs,
            equalize=equalize,
            bg=bg,
        ),
        __font,
        font_size=font_size,
        fg_default=fg,
        bg_default=bg,
    )


def _is_array(__obj: Any) -> TypeGuard[np.ndarray]:
    return isinstance(__obj, np.ndarray)


def _is_rgb_array(__obj: Any) -> TypeGuard[RGBArray]:
    return _is_array(__obj) and __obj.ndim == 3 and np.issubdtype(__obj.dtype, np.uint8)


def _is_greyscale_array(__obj: Any) -> TypeGuard[GreyscaleArray]:
    return (
        _is_array(__obj) and __obj.ndim == 2 and np.issubdtype(__obj.dtype, np.float64)
    )


def _is_greyscale_glyph(__obj: Any) -> TypeGuard[GreyscaleGlyphArray]:
    return _is_greyscale_array(__obj) and __obj.shape == (24, 24)


def _is_image(__obj: Any) -> TypeGuard[ImageType]:
    return isinstance(__obj, ImageType)


def _is_rgb_image(__obj: Any) -> TypeGuard[ImageType]:
    return _is_image(__obj) and __obj.mode == 'RGB'


def _is_rgb_imagelike(__obj: Any) -> TypeGuard[Union[RGBArray, ImageType]]:
    return _is_rgb_array(__obj) or _is_rgb_image(__obj)


def _is_csi_param(__c: str) -> TypeGuard[Literal[';'] | LiteralDigit]:
    return __c == ';' or __c.isdigit()


@lru_cache(maxsize=None)
def sgr_span_re_pattern():
    sgr_re = sgr_re_pattern().pattern.removeprefix(r'\x1b\[')
    return re.compile(
        rf"(?:\x1b\[(?:(?P<cursor>\d*[A-G]|\d*(?:;\d*)?H)|(?P<sgr>{sgr_re})))?(?P<text>[^\x1b]*)"
    )


def _sub_bold_colors(lines: Iterable[str]) -> Iterator[str]:
    """Yield lines with bold foreground colors normalized to their ESC[9(n)m variants.

    Previous colors are also forwarded at each SGR position, if they are not overridden.
    """
    type TupleOf5[_T] = tuple[_T, _T, _T, _T, _T]
    ansi_16c_fg_std = range(30, 38)
    ansi_16c_fg_bold = range(90, 98)

    def sub(m: re.Match):
        nonlocal bold_bit, prev_colors

        params: list[Int3Tuple | TupleOf5[int] | int] = []
        nums = map(int, m[0].removeprefix('\x1b[').removesuffix('m').split(';'))
        for i in nums:
            if i in {38, 48}:
                j = next(nums)
                if j == 5:
                    extended_param = (i, j, next(nums))
                elif j == 2:
                    extended_param = (i, j, *(next(nums) for _ in range(3)))
                else:
                    raise ValueError("invalid ansi color")
                params.append(extended_param)
                continue
            elif i == 0:
                bold_bit = False
                prev_colors.clear()
            elif i == 1:
                bold_bit = True
                for k, v in prev_colors.items():
                    if len(v) == 1 and v[0] in ansi_16c_fg_std:
                        prev_colors[k][0] += 60
            elif i == 22:
                bold_bit = False
                for k, v in prev_colors.items():
                    if len(v) == 1 and v[0] in ansi_16c_fg_bold:
                        prev_colors[k][0] -= 60
            elif bold_bit and i in ansi_16c_fg_std:
                i += 60
            params.append(i)
        sgr = SgrSequence(
            i
            for xs in [prev_colors.values(), params]
            for x in xs
            for i in ([x] if isinstance(x, int) else x)
        )
        for p in sgr:
            if p.is_color():
                prev_colors |= dict.fromkeys(
                    p._value._rgb_dict.keys(),
                    [int(x) for x in p._value.split(b';') if x],
                )
        return f"{sgr}"

    for line in lines:
        bold_bit = False
        prev_colors = {}
        yield sgr_re_pattern().sub(sub, line)


def reshape_ansi(__str: str, w: int, h: int) -> str:

    def cursor() -> Generator[tuple[int, int], tuple[int, int], None]:
        idx, total = 0, h * w
        while idx < total:
            jmp = yield divmod(idx, w)
            if jmp:
                idx = jmp[0] * w + jmp[1]
            else:
                idx += 1

    def write_cell(content: str, *, incr=False):
        nonlocal x, y
        if content == '\n':
            x = min(x + 1, h - 1)
            y = 0
            cur.send((x, y))
            return
        elif arr[x][y] == '\x00':
            arr[x][y] = content
        else:
            arr[x][y] += content
        if incr:
            x, y = next(cur)

    arr = [['\x00'] * w for _ in range(h)]
    cur = cursor()
    x, y = next(cur)
    for line in _sub_bold_colors(__str.splitlines()):
        for m in sgr_span_re_pattern().finditer(line + '\n'):
            if cg := m['cursor']:
                write_cell(' ')
                param, code = cg[:-1], cg[-1]
                if code == 'H':
                    if ';' in param:
                        x, y = (int(i or 1) - 1 for i in param.partition(';')[::2])
                    else:
                        x = int(param or 1) - 1
                        y = 0
                else:
                    n = int(param or 1)
                    match code:
                        case 'A':
                            x = max(0, x - n)
                        case 'B':
                            x = min(h - 1, x + n)
                        case 'C':
                            y = min(w - 1, y + n)
                        case 'D':
                            y = max(0, y - n)
                        case 'E':
                            x += n
                            y = 0
                        case 'F':
                            x -= n
                            y = 0
                        case 'G':
                            y = n - 1
                cur.send((x, y))
            elif sgr := m['sgr']:
                write_cell(f"\x1b[{sgr}")
            for ch in m['text']:
                write_cell(ch, incr=True)

    out_lines = []
    any_sgr_re = re.compile(r"\x1b\[\d*(?:;\d*)*[A-HJ-KS-Tmf]")
    for row in arr:
        out_row = ''.join(cell.translate({0: ' '}) for cell in row)
        lhs, rhs = map(' '.__mul__, divmod(w - len(any_sgr_re.sub('', out_row)), 2))
        out_lines.append(f"{lhs}{out_row}{rhs}")

    return '\n'.join(out_lines)


@lru_cache
def to_sgr_array(__str: str, ansi_type: AnsiColorParam = DEFAULT_ANSI):
    ansi_typ = get_ansi_type(ansi_type)
    new_cs = partial(ColorStr, ansi_type=ansi_typ, reset=False)
    xs = []
    for line in _sub_bold_colors(__str.splitlines()):
        x = []
        for m in sgr_span_re_pattern().finditer(line):
            text = m["text"]
            if m["sgr"]:
                sgr = SgrSequence(map(int, m["sgr"].removesuffix('m').split(';')))
                cs = new_cs(f"{sgr}{text}")
            else:
                cs = new_cs(text)
            x.append(cs)
        xs.append(x)
    return xs


def render_ans(
    __str: str,
    shape: TupleOf2[int],
    font: FontArgType = userfont['vga437'],
    font_size: int = 16,
    *,
    bg_default: Int3Tuple | str | Literal['auto'] = (0, 0, 0),
) -> ImageType:
    """Create an image from a literal ANSI string.

    Parameters
    ----------
    __str : str
        Literal ANSI text.

    shape : tuple[int, int]
        (width, height) of the expected output, in ASCII characters.

    font : FontArgType
        Font to use when rendering the image.

    font_size : int
        Font size in pixels.

    bg_default : tuple[int, int, int] | Literal['auto']
        Background color to use as a fallback when ANSI SGR has none.
        'auto' will determine background color dynamically.
    """
    return ansi2img(
        to_sgr_array(reshape_ansi(__str, *shape)),
        font,
        font_size,
        bg_default=bg_default,
    )


def read_ans(__buf: SupportsRead[str] | TextIOWrapper[str]) -> str:
    """Interpret a text buffer as an .ANS file and return the content as a string.

    Extends 'cp437' translation if `__buf.encoding='cp437'`, and truncates any SAUCE metadata.
    Otherwise, this is just a text file read operation.
    """

    content = __buf.read().translate({0: ' '})
    if ~(sauce_idx := content.rfind('\x1aSAUCE00')):
        content = content[:sauce_idx]
    if hasattr(__buf, 'encoding') and __buf.encoding == 'cp437':
        from ._curses import translate_cp437

        content = translate_cp437(content, ignore=(0x0A, 0x1A, 0x1B))
    return content


class AnsiImage:
    from shutil import get_terminal_size

    @classmethod
    def open[AnyStr: (
        str,
        bytes,
    )](
        cls,
        fp: int | PathLike[AnyStr] | AnyStr,
        shape: TupleOf2[int] = get_terminal_size(),
        encoding: str = 'cp437',
        ansi_type: AnsiColorParam = DEFAULT_ANSI,
    ) -> Self:
        """Construct an `AnsiImage` object from a text file.

        Parameters
        ----------
        fp : int or PathLike[AnyStr] or AnyStr
            File descriptor or filepath to ANSI file.

        shape : tuple[int, int]
            Dimensions of ANSI image (width, height).

        encoding : str='cp437'
            File encoding.

        ansi_type : AnsiColorParam
            ANSI color format.

        Returns
        -------
        AnsiImage
        """
        inst = super().__new__(cls)
        inst._ansi_format = get_ansi_type(ansi_type)
        inst.file = open(fp, mode='r', encoding=encoding or None)
        inst._shape = shape
        return inst

    def _getvalue(self):
        attr_names = set(vars(self))
        attr_names &= {'file', 'data'}
        attr_name = attr_names.pop()
        assert not attr_names, "ambiguous value attribute"
        if attr_name == 'file':
            file: TextIOWrapper[str] = self.__dict__.pop(attr_name)
            setattr(
                self,
                'data',
                to_sgr_array(
                    reshape_ansi(read_ans(file), *self.shape),
                    ansi_type=self.ansi_format,
                ),
            )
            file.close()
        return self.data

    @property
    def ansi_format(self) -> AnsiColorType:
        return self._ansi_format

    @property
    def height(self):
        return self.shape[1]

    @property
    def width(self):
        return self.shape[0]

    @property
    def shape(self):
        return self._shape

    def render(
        self, font: FontArgType = userfont['vga437'], font_size: int = 16, **kwargs
    ) -> ImageType:
        return ansi2img(self._getvalue(), font, font_size, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'file') and not self.file.closed:
            self.file.close()

    def __init__(
        self, arr: list[list[ColorStr]], *, ansi_type: AnsiColorParam = DEFAULT_ANSI
    ):
        h = len(arr)
        w = len(arr[0]) if h > 0 else 0
        if h > 1:
            for wv in arr[1:]:
                if len(wv) == w:
                    continue
                raise ValueError("inhomogenous shape")
        self._shape = w, h
        self.data = arr
        self._ansi_format = get_ansi_type(ansi_type)

    def __str__(self) -> str:
        attr_name = f"_{type(self).__name__}__str"
        if hasattr(self, attr_name) and getattr(self, attr_name)[-1] == self.shape:
            return getattr(self, attr_name)[0]
        lines = []
        for line in self._getvalue():
            if line:
                buffer = []
                initial = line[0]
                for s in line[1:]:
                    if s.ansi_partition()[::2] == initial.ansi_partition()[::2]:
                        initial = initial.replace(
                            initial.base_str, initial.base_str + s.base_str
                        )
                    else:
                        buffer.append(initial)
                        initial = s
                buffer.append(initial)
                lines.append(''.join(buffer))
        setattr(self, attr_name, ('\n'.join(lines) + SGR_RESET.decode(), self.shape))
        return getattr(self, attr_name)[0]


def otsu_mask(img: Union[ImageType, MatrixLike[np.uint8]]) -> MatrixLike[np.uint8]:
    if type(img) is not np.ndarray:
        img = np.uint8(img)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    return cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


def _canny_edges[_SCT](arr: MatrixLike[_SCT]) -> MatrixLike[_SCT]:
    return ski.feature.canny(
        arr, sigma=0.1, low_threshold=0.1, high_threshold=0.2, use_quantiles=False
    )


def scaled_hu_moments(arr: ShapedNDArray[TupleOf2[int], np.uint8]):
    if set.isdisjoint({0, 0xFF}, np.unique_values(arr)):
        arr = otsu_mask(arr)
    hms = cv.HuMoments(cv.moments(arr)).ravel()
    nz = hms.nonzero()
    out = np.zeros_like(hms)
    out[nz] = -np.sign(hms[nz]) * np.log10(np.abs(hms[nz]))
    return out


def approx_gridlike(
    __fp: PathLike[str] | str,
    shape: TupleOf2[int],
    font: FontArgType = userfont['vga437'],
):
    from ._curses import cp437_printable

    def _get_grid_indices(arr: np.ndarray) -> list[TupleOf2[slice]]:
        regions = ski.measure.regionprops(ski.measure.label(_canny_edges(arr)))
        area_bboxes = np.zeros([np.shape(regions)[0]])
        bboxes = np.int32([area_bboxes] * 4).T
        for n, region in enumerate(regions):
            bboxes[n], area_bboxes[n] = region.bbox, region.area_bbox
        bboxes = bboxes[area_bboxes < np.std(area_bboxes) * 2]
        r, c = zip(np.min(bboxes[:, :2], axis=0), np.max(bboxes[:, 2:], axis=0), shape)
        h, w = map(round, ((x[1] - x[0]) / x[-1] for x in (r, c)))
        rr = r[0] + np.asarray(rs := range(r[-1])) * h
        cc = c[0] + np.asarray(cs := range(c[-1])) * w
        out: Any = [
            np.index_exp[rr[rx] : (rr + h)[rx], cc[cx] : (cc + w)[cx]]
            for rx in rs
            for cx in cs
        ]
        return out

    def _normalize_cell(arr: np.ndarray):
        cell = np.zeros(cell_shape, dtype=np.uint8)
        coords = np.argwhere(arr)
        if coords.size == 0:
            return cell
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = arr[y0:y1, x0:x1]
        dy, dx = cropped.shape
        ys, xs = map(lambda t, d: (t - d) // 2, cell_shape, (dy, dx))
        cell[ys : ys + dy, xs : xs + dx] = cropped
        return cell

    with Image.open(__fp).convert('L') as grey:
        thresh = otsu_mask(grey)

    grid_indices = _get_grid_indices(thresh)
    cell_shape = thresh[grid_indices[0]].shape
    clustered_grid = np.reshape(
        getattr(
            DBSCAN(eps=0.5, min_samples=2, metric='euclidean').fit(
                np.array(
                    [
                        scaled_hu_moments(thresh[ind])  # type: ignore[arg-type]
                        for ind in grid_indices
                    ]
                )
            ),
            'labels_',
        ),
        shape,
    )
    char_grid = np.full_like(clustered_grid, ' ', dtype=np.str_)
    glyph_map = {
        c: otsu_mask(render_font_char(c, font, size=cell_shape[::-1]).convert('L'))
        for c in cp437_printable()
    }
    for u_indices in map(clustered_grid.__eq__, np.unique_values(clustered_grid)):
        u_slice = thresh[
            grid_indices[
                next(idx for (idx, v) in enumerate(np.ravel(u_indices)) if v is True)
            ]
        ]
        char_grid[u_indices] = min(
            glyph_map,
            key=lambda k: ski.metrics.mean_squared_error(
                *map(_normalize_cell, (glyph_map[k], u_slice))
            ),
        )

    return AnsiImage([list(map(ColorStr, r)) for r in char_grid])
