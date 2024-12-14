from __future__ import annotations

__all__ = ['get_glyph_masks', 'ttf_extract_codepoints', 'sort_glyphs']

from os import PathLike
from typing import Literal, overload, Sequence, Union

import numpy as np
from fontTools.ttLib import TTFont
from numpy import float64, uint8
from scipy.ndimage import distance_transform_edt

from ._array import otsu_mask
from ._curses import ascii_printable
from .._typing import FontArgType, GlyphArray, GlyphBitmask, ShapedNDArray


@overload
def get_glyph_masks(
    __font: FontArgType,
    char_set: Sequence[str] = ...,
    dist_transform: Literal[False] = False,
) -> dict[str, GlyphBitmask]: ...


@overload
def get_glyph_masks(
    __font: FontArgType,
    char_set: Sequence[str] = ...,
    dist_transform: Literal[True] = ...,
) -> dict[str, GlyphArray[float64]]: ...


@overload
def get_glyph_masks(
    __font: FontArgType, char_set: Sequence[str] = ..., dist_transform: bool = ...
) -> dict[str, GlyphArray[Union[uint8, float64]]]: ...


def get_glyph_masks(
    __font: FontArgType, char_set: Sequence[str] = None, dist_transform: bool = False
) -> dict[str, GlyphArray[Union[uint8, float64]]]:
    from ._array import get_font_object, render_font_char

    char_set = char_set or ascii_printable()
    font = get_font_object(__font)

    def _get_threshold(__c: str):
        out = otsu_mask(render_font_char(__c, font).convert('L'))
        if dist_transform is True:
            return distance_transform_edt(out)
        return out

    space = _get_threshold(' ')
    non_printable = _get_threshold('�')
    glyph_masks = {}
    for char in set(char_set):
        thresh = _get_threshold(char)
        if np.array_equal(thresh, non_printable):
            thresh = space
        glyph_masks[char] = thresh
    return glyph_masks


def sort_glyphs(__s: str, font: FontArgType, reverse: bool = False):
    def _sum_mask(item: tuple[str, np.ndarray]):
        return item[0], np.sum(item[1])

    return ''.join(
        char
        for (char, value) in sorted(
            map(_sum_mask, get_glyph_masks(font, __s, dist_transform=True).items()),
            key=lambda x: x[1],
            reverse=reverse,
        )
        if value > 0 or char == ' '
    )


def ttf_extract_codepoints(
    __fp: PathLike[str] | str | FontArgType, **kwargs
) -> ShapedNDArray[tuple[int], np.uint16]:
    if not hasattr(__fp, '__fspath__'):
        if isinstance(__fp, int):
            from ..data import UserFont

            __fp = UserFont(__fp)
        if hasattr(__fp, 'path'):
            __fp = __fp.path

    codepoints = set()
    with TTFont(__fp, **kwargs) as font:
        for table in font['cmap'].tables:
            codepoints |= table.cmap.keys()

    return np.sort(
        np.array([i for i in codepoints if chr(i).isprintable()], dtype='<u2')
    )
