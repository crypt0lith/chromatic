from __future__ import annotations

__all__ = [
    'ansi2img',
    'ansify',
    'ansi_quantize',
    'ascii2img',
    'contrast_stretch',
    'equalize_white_point',
    'get_font_key',
    'get_font_object',
    'img2ansi',
    'img2ascii',
    'read_ans',
    'render_ans',
    'render_font_char',
    'render_font_str',
    'reshape_ansi',
    'scale_saturation',
    'shuffle_char_set',
    '_scaled_hu_moments',
    'to_sgr_array',
    'AnsiImage',
    '_otsu_mask',
]

import math
import os.path
import random
from collections.abc import Mapping
from functools import lru_cache, partial
from os import PathLike
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Self,
    Sequence,
    TYPE_CHECKING,
    TypeAlias,
    TypeGuard,
    Union,
    cast,
    overload,
)

import cv2 as cv
import numpy as np
import skimage as ski
from PIL import Image, ImageDraw
from PIL.Image import Image as ImageType
from PIL.ImageFont import FreeTypeFont, truetype
from numpy import float64, issubdtype, ndarray, uint8
from sklearn.cluster import DBSCAN

from .._typing import (
    FontArgType,
    GreyscaleArray,
    GreyscaleGlyphArray,
    Int3Tuple,
    MatrixLike,
    RGBArray,
    RGBImageLike,
    TupleOf2,
    type_error_msg,
)
from ..color.colorconv import nearest_ansi_4bit_rgb, nearest_ansi_8bit_rgb
from ..color.core import (
    AnsiColorParam,
    AnsiColorType,
    Color,
    ColorStr,
    DEFAULT_ANSI,
    SGR_RESET,
    SgrSequence,
    ansicolor24Bit,
    ansicolor4Bit,
    ansicolor8Bit,
    get_ansi_type,
)
from ..color.palette import rgb_dispatch
from ..data import UserFont

if TYPE_CHECKING:
    from _typeshed import AnyStr_co


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
def get_font_object(font: FontArgType, *, retpath: Literal[True]) -> str: ...


@lru_cache
def get_font_object(
    font: FontArgType, *, retpath: bool = False
) -> Union[FreeTypeFont, str]:
    """Obtain a FreeTypeFont object or its filepath from various input types.

    Parameters
    ----------
    font : FontArgType
        The font input, which could be a FreeTypeFont object, UserFont, string, or integer.

    retpath : bool, optional
        If True, returns the font's filepath instead of the font object.

    Returns
    -------
    FreeTypeFont or str
        The FreeTypeFont object or its filepath, depending on the `retpath` parameter.

    Raises
    ------
    FileNotFoundError
        If the font file cannot be located.

    TypeError
        If the input type is unsupported.
    """

    def path2obj(__path: AnyStr_co):
        return truetype(__path, 24)

    if retpath:
        out_f = lambda x: x
    else:
        out_f = path2obj
    if isinstance(font, FreeTypeFont):
        return font.path if retpath else font
    if hasattr(font, '__fspath__'):
        return out_f(font)
    if isinstance(font, UserFont):
        return out_f(font.path)
    if font in set(UserFont._value2member_map_):
        return out_f(UserFont(font).path)
    if isinstance(font, str):
        if font in set(UserFont._member_names_):
            return out_f(UserFont[font].path)
        try:
            font_obj = path2obj(font)
            return font_obj.path if retpath else font_obj
        except OSError:
            raise FileNotFoundError(font) from None
    raise TypeError(
        f"Expected {FreeTypeFont.__qualname__!r}, got {type(font).__qualname__!r} instead"
    )


def shuffle_char_set(char_set: Iterable[str], *xi: *tuple[Optional[slice | int], ...]):
    """Randomly shuffle characters in a given character set.

    Parameters
    ----------
    char_set : Iterable[str]
        Iterable containing characters.

    *xi : *(slice | int | None)
        Specifies slices to take from the shuffled result.

    Returns
    -------
    str
        The shuffled string of characters.

    Raises
    ------
    TypeError
        If `char_set` is not iterable.

    ValueError
        If the number of varargs exceeds `[start[, stop[, step]]]`.
    """
    if not isinstance(char_set, Iterable):
        raise TypeError(
            f"Expected 'char_set' to be iterable type, "
            f"got {type(char_set).__qualname__!r} instead"
        )
    if xi:
        if (n_args := len(xi)) > 3:
            raise ValueError(f"Unexpected extra args: expected max 3, got {n_args}")
        if n_args == 1:
            xi = xi[0]
            vt = type(xi)
            if vt not in {int, slice}:
                raise TypeError(f"Unexpected arg type: {vt.__qualname__!r}")
            if vt is int:
                xi = slice(xi)
        else:
            good_types = {int, type(None)}
            if bad_types := set(map(type, xi)) - good_types:
                err = (
                    type_error_msg(
                        ', '.join(sorted(repr(t.__qualname__) for t in bad_types)),
                        *good_types,
                        obj_repr=True,
                    )
                    .removeprefix('expected')
                    .lstrip()
                )
                raise ValueError(f"Multiple args must be {err}")
            xi = slice(*xi)
    else:
        xi = slice(0, None)
    char_list = list(char_set)
    random.shuffle(char_list)
    return ''.join(char_list)[xi]


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
    __s = __s.translate({ord('\t'): ' ' * 4})
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

    font : FreeTypeFont | UserFont | int | str
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
        img = ski.io.imread(__img.__fspath__())
    elif isinstance(__img, str):
        img = ski.io.imread(__img)
    else:
        img = __img
    if not _is_rgb_array(img):
        if _is_image(img):
            img = img.convert('RGB')
        elif _is_array(img):
            conv = {
                2: lambda im: cv.cvtColor(im[:, :, 0], cv.COLOR_GRAY2RGB),
                4: lambda im: cv.cvtColor(im, cv.COLOR_RGBA2RGB),
            }.get(img.ndim)
            if conv is None:
                raise ValueError(f"unexpected array shape: {img.shape!r}")
            img = conv(img)
        else:
            raise TypeError(type_error_msg(img, PathLike, ImageType, ndarray))
        img = uint8(img)
    return img


def _rgb_transform2vec(pyfunc: Callable[[Int3Tuple], Int3Tuple]):
    return np.frompyfunc(lambda *rgb: pyfunc(rgb), 3, 3)


def _apply_rgb_ufunc(img: RGBArray, rgb_ufunc: np.ufunc) -> RGBArray:
    return uint8(rgb_ufunc(*np.moveaxis(img, -1, 0))).transpose(1, 2, 0)


_ANSI_QUANTIZERS = {
    t: partial(_apply_rgb_ufunc, rgb_ufunc=_rgb_transform2vec(f))
    for (t, f) in zip(
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
        quantizer = _ANSI_QUANTIZERS.get(ansi_type)
    except TypeError:
        quantizer = None
    if quantizer is None:
        from .._typing import unionize

        context = "{}=type[{}]".format(
            f"{ansi_type=}".partition('=')[0],
            ' | '.join(
                getattr(x, '__name__', f"{x}")
                for x in unionize(_ANSI_QUANTIZERS.keys()).__args__
            ),
        )
        raise TypeError(type_error_msg(ansi_type, context=context))
    if eq_f := {True: contrast_stretch, 'white_point': equalize_white_point}.get(
        equalize
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

    Enhances contrast while preserving color, ideal for pronounced light/dark effects.

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


def contrast_stretch(img: RGBArray) -> RGBArray:
    """Rescale the intensities of an RGB image using linear contrast stretching.

    Provides subtle, balanced contrast enhancement across both lightness and color.

    Parameters
    ----------
    img : RGBArray

    Returns
    -------
    eq_img : RGBArray

    See Also
    --------
    equalize_white_point
    """
    p2, p98 = np.percentile(img, (2, 98))
    return cast(
        RGBArray, ski.exposure.rescale_intensity(cast(..., img), in_range=(p2, p98))
    )


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
    greyscale: MatrixLike[uint8] = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img_shape = greyscale.shape
    img_aspect = img_shape[-1] / img_shape[0]
    ch, cw = _get_bbox_shape(font)
    char_aspect = math.ceil(cw / ch)
    new_height = int(factor / img_aspect / char_aspect)
    greyscale = ski.transform.resize(greyscale, (new_height, factor))
    if char_set is None:
        from ._curses import ascii_printable, cp437_printable

        cursed_fonts = {UserFont.IBM_VGA_437_8X16: cp437_printable}
        char_set = shuffle_char_set(cursed_fonts.get(__font, ascii_printable)())
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


@rgb_dispatch
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
    bg_wrapper = ColorStr('{}', color_spec={'bg': bg}, ansi_type=ansi_type, reset=False)
    base_ascii, color_arr = img2ascii(
        __img, __font, factor, char_set, sort_glyphs, ret_img=True
    )
    lines = base_ascii.splitlines()
    h, w = map(len, (lines, lines[0]))
    if ansi_type is not ansicolor24Bit:
        color_arr = ansi_quantize(color_arr, ansi_type=ansi_type, equalize=equalize)
    elif eq_func := {True: contrast_stretch, 'white_point': equalize_white_point}.get(
        equalize
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
                x.append(bg_wrapper.format(char).recolor(fg=fg_color))
        xs.append(x)
    return xs


@rgb_dispatch
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


@rgb_dispatch
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
            if getattr(color_str, '_sgr_').is_reset():
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
    __font: FontArgType = UserFont.IBM_VGA_437_8X16,
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


def _is_array(__obj: Any) -> TypeGuard[ndarray]:
    return isinstance(__obj, ndarray)


def _is_rgb_array(__obj: Any) -> TypeGuard[RGBArray]:
    return _is_array(__obj) and __obj.ndim == 3 and issubdtype(__obj.dtype, uint8)


def _is_greyscale_array(__obj: Any) -> TypeGuard[GreyscaleArray]:
    return _is_array(__obj) and __obj.ndim == 2 and issubdtype(__obj.dtype, float64)


def _is_greyscale_glyph(__obj: Any) -> TypeGuard[GreyscaleGlyphArray]:
    return _is_greyscale_array(__obj) and __obj.shape == (24, 24)


def _is_image(__obj: Any) -> TypeGuard[ImageType]:
    return isinstance(__obj, ImageType)


def _is_rgb_image(__obj: Any) -> TypeGuard[ImageType]:
    return _is_image(__obj) and __obj.mode == 'RGB'


def _is_rgb_imagelike(__obj: Any) -> TypeGuard[Union[RGBArray, ImageType]]:
    return _is_rgb_array(__obj) or _is_rgb_image(__obj)


_LiteralDigitStr: TypeAlias = Sequence[
    Literal['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
]


def _is_csi_param(__c: str) -> TypeGuard[Literal[';'] | _LiteralDigitStr]:
    return __c == ';' or __c.isdigit()


def reshape_ansi(__str: str, h: int, w: int):
    if len(__str.splitlines()) == h and all(
        sum(map(len, x)) == w for x in to_sgr_array(__str)
    ):
        return __str
    arr = [['\x00'] * w for _ in range(h)]
    offsets = dict.fromkeys(range(h), 0)
    flat_iter = (divmod(idx, w) for idx in range(h * w))
    str_len = len(__str)
    j = 0

    def _increment(__c: str = ' '):
        nonlocal x, y
        arr[x][y] += __c
        offsets[x] += 1
        try:
            x, y = next(flat_iter)
        except StopIteration:
            pass

    try:
        x, y = next(flat_iter)
        while j < str_len:
            if __str[j : (i := j + 2)] == '\x1b[':
                j = i
                while _is_csi_param(c := __str[j]):
                    j += 1
                params = __str[i:j]
                if c == 'C':
                    for _ in range(int(params)):
                        _increment()
                elif c == 'm':
                    arr[x][y] += str(SgrSequence(list(map(int, params.split(';')))))
            elif (c := __str[j]) == '\n':
                while y < w - 1:
                    _increment()
                x, y = next(flat_iter)
            else:
                _increment(c)
            j += 1
    except StopIteration:
        pass
    return '\n'.join(
        ' ' * (w - offsets[idx]) + ''.join(c for c in row if c != '\x00')
        for idx, row in enumerate(arr)
    )


@lru_cache(maxsize=None)
def sgr_span_re_pattern():
    import re
    from ..color.core import sgr_re_pattern

    sgr_re = sgr_re_pattern().pattern
    return re.compile(rf"(?P<params>{sgr_re})?(?P<text>[^\x1b]*)")


@lru_cache
def to_sgr_array(__str: str, ansi_type: AnsiColorParam = DEFAULT_ANSI):
    ansi_typ = get_ansi_type(ansi_type)
    new_cs = partial(ColorStr, ansi_type=ansi_typ, reset=False)
    sgr: SgrSequence
    x = []
    for line in __str.splitlines():
        xs = []
        ansi_ctx = {}
        prev: ColorStr | None = None
        for m in sgr_span_re_pattern().finditer(line):
            if m["params"]:
                params, text = (
                    m["params"].removeprefix('\x1b[').removesuffix('m'),
                    m["text"],
                )
                sgr = SgrSequence(map(int, params.split(';')))
                cs = new_cs(text, sgr)
                if sgr.is_color():
                    prev = cs
            else:
                cs = new_cs(m["text"])
                sgr = getattr(cs, '_sgr_')
            if sgr.is_reset():
                ansi_ctx.clear()
            for k in set.intersection({b'39', b'49'}, sgr.values()):
                del ansi_ctx['fg' if k == b'39' else 'bg']
            if sgr.is_color():
                for k in sgr.rgb_dict.keys():
                    ansi_ctx[k] = sgr.get_color(k)
            if sgr.has_bright_colors:
                ansi_ctx['bright'] = True
            elif ansi_ctx.get('bright') or (prev is not None and b'1' in sgr.values()):
                if sgr.is_color():
                    new_sgr = SgrSequence(
                        b'1;%s' % v if type(v) is ansicolor4Bit else v
                        for v in sgr.values()
                    )
                    for k in new_sgr.rgb_dict.keys():
                        ansi_ctx[k] = new_sgr.get_color(k)
                    prev = cs = new_cs(cs.base_str, color_spec=new_sgr)
                elif prev:
                    if not ansi_ctx.get('bright'):
                        ansi_ctx['bright'] = True
                    new_sgr = SgrSequence(
                        v
                        for vs in (
                            sgr.values(),
                            (p._value_ for p in getattr(prev, '_sgr_') if p.is_color()),
                        )
                        for v in vs
                    )
                    prev = cs = new_cs(cs.base_str, new_sgr)
            xs.append(cs.as_ansi_type(ansi_typ))
        x.append(xs)
    return x


def render_ans(
    __str: str,
    shape: TupleOf2[int],
    font: FontArgType = UserFont.IBM_VGA_437_8X16,
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
        (height, width) of the expected output, in ASCII characters.

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


def read_ans[AnyStr: (
    str,
    bytes,
)](__path: PathLike[AnyStr] | AnyStr, encoding='cp437') -> str:
    """Read a .ANS file and return the content as a string.

    Extends code page 437 translation if `encoding='cp437'`, and truncates any SAUCE metadata.
    Otherwise, the function is just a wrapped text file read operation.
    """

    with open(__path, mode='r', encoding=encoding) as f:
        content = f.read().translate({0: ' '})
    if ~(sauce_idx := content.rfind('\x1aSAUCE00')):
        content = content[:sauce_idx]
    if encoding == 'cp437':
        from ._curses import translate_cp437

        content = translate_cp437(content, ignore=(0x0A, 0x1A, 0x1B))
    return content


class AnsiImage:

    @classmethod
    def open[AnyStr: (
        str,
        bytes,
    )](
        cls,
        fp: PathLike[AnyStr] | AnyStr,
        shape: TupleOf2[int],
        encoding='cp437',
        ansi_type: AnsiColorParam = DEFAULT_ANSI,
    ) -> Self:
        """Construct an ``AnsiImage`` object from a readable file.

        Parameters
        ----------
        fp : PathLike[AnyStr] or AnyStr
            Filepath to the ANSI file.

        shape : tuple[int, int]
            Dimensions of the ANSI image (height, width).

        encoding : str='cp437'
            Encoding of the ANSI file.

        ansi_type : AnsiColorParam
            ANSI color format.

        Returns
        -------
        AnsiImage
        """
        inst = super().__new__(cls)
        inst._ansi_format_ = get_ansi_type(ansi_type)
        inst.fp = os.path.abspath(fp)
        inst.data = to_sgr_array(
            reshape_ansi(read_ans(inst.fp, encoding=encoding), shape[0], shape[1]),
            ansi_type=inst._ansi_format_,
        )
        return inst

    @property
    def ansi_format(self) -> AnsiColorType:
        return self._ansi_format_

    @property
    def height(self):
        return len(self.data[0]) if self.data else 0

    @property
    def width(self):
        return len(self.data)

    @property
    def shape(self):
        return self.height, self.width

    def render(
        self,
        font: FontArgType = UserFont.IBM_VGA_437_8X16,
        font_size: int = 16,
        bg_default=None,
        **kwargs,
    ) -> ImageType:
        return ansi2img(
            self.data, font, font_size, bg_default=bg_default or 'auto', **kwargs
        )

    def translate(self, __table: Mapping[int, str | int | None]):
        if not __table:
            return self
        table = {
            k: (
                v
                if v not in frozenset(x for c in ' \t\n\r\v\f' for x in (c, ord(c)))
                else ' '
            )
            for (k, v) in __table.items()
            if k != ord('\n')
        }
        data = self.data
        for row in range(self.height):
            for col in range(self.width):
                data[row][col] = data[row][col].translate(table)
        return type(self)(data, ansi_type=self.ansi_format)

    def __init__(
        self, arr: list[list[ColorStr]], *, ansi_type: AnsiColorParam = DEFAULT_ANSI
    ):
        self.data = arr
        self._ansi_format_ = get_ansi_type(ansi_type)
        self.fp = None

    def __str__(self) -> str:
        cls = type(self)
        attr_name = f"_{cls.__name__}__str"
        if hasattr(self, attr_name) and getattr(self, attr_name)[-1] == self.shape:
            return getattr(self, attr_name)[0]
        lines = []
        for line in self.data:
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


def _otsu_mask(img: MatrixLike[np.uint8] | ImageType) -> MatrixLike[np.uint8]:
    if type(img) is not np.ndarray:
        img = np.uint8(img)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    return cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


def _canny_edges(arr: MatrixLike) -> MatrixLike:
    return ski.feature.canny(
        arr, sigma=0.1, low_threshold=0.1, high_threshold=0.2, use_quantiles=False
    )


def _scaled_hu_moments(arr: MatrixLike):
    if {0, 255}.isdisjoint(np.unique_values(arr)):
        arr = _otsu_mask(arr)
    hms = cv.HuMoments(cv.moments(arr)).ravel()
    nz = hms.nonzero()
    out = np.zeros_like(hms)
    out[nz] = -np.sign(hms[nz]) * np.log10(np.abs(hms[nz]))
    return out


def approx_gridlike(
    __fp: PathLike[str] | str,
    shape: TupleOf2[int],
    font: FontArgType = UserFont.IBM_VGA_437_8X16,
):
    def _get_grid_indices(arr: np.ndarray):
        regions = ski.measure.regionprops(ski.measure.label(_canny_edges(arr)))
        area_bboxes = np.zeros([np.shape(regions)[0]])
        bboxes = np.int32([area_bboxes] * 4).T
        for n, region in enumerate(regions):
            bboxes[n], area_bboxes[n] = region.bbox, region.area_bbox
        bboxes = bboxes[area_bboxes < np.std(area_bboxes) * 2]
        r, c = cast(
            TupleOf2[Int3Tuple],
            zip(np.min(bboxes[:, :2], axis=0), np.max(bboxes[:, 2:], axis=0), shape),
        )
        h, w = map(round, ((x[1] - x[0]) / x[-1] for x in (r, c)))
        rr = r[0] + np.asarray(rs := range(r[-1])) * h
        cc = c[0] + np.asarray(cs := range(c[-1])) * w
        return cast(
            list[TupleOf2[slice]],
            [
                np.index_exp[rr[rx] : (rr + h)[rx], cc[cx] : (cc + w)[cx]]
                for rx in rs
                for cx in cs
            ],
        )

    with Image.open(__fp).convert('L') as grey:
        thresh = _otsu_mask(np.array(grey))
    from ._curses import cp437_printable

    grid_indices = _get_grid_indices(thresh)
    cell_shape = thresh[grid_indices[0]].shape
    clustered_grid = np.reshape(
        getattr(
            DBSCAN(eps=0.5, min_samples=2, metric='euclidean').fit(
                np.array([_scaled_hu_moments(thresh[ind]) for ind in grid_indices])
            ),
            'labels_',
        ),
        shape,
    )
    char_grid = np.full_like(clustered_grid, ' ', dtype=np.str_)
    glyph_map = {
        c: _otsu_mask(render_font_char(c, font, size=cell_shape[::-1]).convert('L'))
        for c in cp437_printable()
    }

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

    return AnsiImage([[ColorStr(s) for s in r] for r in char_grid])
