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
    all_chars = list(__s)
    mapping = {}
    for c, arr in get_glyph_masks(font, __s, dist_transform=True).items():
        v = np.sum(arr)
        if v <= 0 and c != ' ':
            continue
        mapping[c] = v
    return ''.join(
        sorted(
            filter(mapping.__contains__, all_chars),
            key=mapping.__getitem__,
            reverse=reverse,
        )
    )


def ttf_extract_codepoints(
    __fp: FontArgType | PathLike[str], **kwargs
) -> ShapedNDArray[tuple[int], np.uint16]:
    codepoints = set()
    with TTFont(__fp, **kwargs) as font:
        for table in font['cmap'].tables:
            codepoints |= table.cmap.keys()
    arr = np.array([i for i in codepoints if chr(i).isprintable()], dtype='<u2')
    return np.sort(arr)  # type: ignore[arg-type]
