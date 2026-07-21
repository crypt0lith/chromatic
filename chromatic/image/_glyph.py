__all__ = ["get_glyph_masks", "ttf_extract_codepoints", "sort_glyphs"]

import collections.abc as abc
import os
import typing as tp
from typing import Literal as L

import numpy as np
from fontTools.ttLib import TTFont
from scipy.ndimage import distance_transform_edt

from .. import _typing as _tp
from ._array import otsu_mask
from ._curses import ascii_printable


@tp.overload
def get_glyph_masks(
    font: _tp.FontArgType, /, char_set: abc.Sequence[str] | None = ...
) -> dict[str, _tp.GlyphArray[np.uint8]]: ...


@tp.overload
def get_glyph_masks(
    font: _tp.FontArgType,
    /,
    char_set: abc.Sequence[str] | None = ...,
    *,
    dist_transform: L[False],
) -> dict[str, _tp.GlyphArray[np.uint8]]: ...


@tp.overload
def get_glyph_masks(
    font: _tp.FontArgType,
    /,
    char_set: abc.Sequence[str] | None = ...,
    *,
    dist_transform: L[True],
) -> dict[str, _tp.GlyphArray[np.float64]]: ...


def get_glyph_masks(
    font: _tp.FontArgType,
    /,
    char_set: abc.Sequence[str] | None = None,
    *,
    dist_transform: bool = False,
):
    from ._array import get_font_object, render_font_char

    char_set = char_set or ascii_printable()
    font = get_font_object(font)

    def _get_threshold(c: str, /):
        out = otsu_mask(render_font_char(c, font).convert("L"))
        if dist_transform is True:
            return distance_transform_edt(out)
        return out

    space = _get_threshold(" ")
    non_printable = _get_threshold("�")
    glyph_masks = {}
    for char in set(char_set):
        thresh = _get_threshold(char)
        if np.array_equal(thresh, non_printable):
            thresh = space
        glyph_masks[char] = thresh
    return glyph_masks


def sort_glyphs(s: str, /, font: _tp.FontArgType, reverse: bool = False):
    all_chars = list(s)
    mapping = {}
    for c, arr in get_glyph_masks(font, s, dist_transform=True).items():
        v = np.sum(arr)
        if v <= 0 and c != " ":
            continue
        mapping[c] = v
    return "".join(
        sorted(
            filter(mapping.__contains__, all_chars),
            key=mapping.__getitem__,
            reverse=reverse,
        )
    )


def ttf_extract_codepoints(
    fp: str | os.PathLike[str], /, **kwargs
) -> _tp.ShapedNDArray[tuple[int], np.uint32]:
    with TTFont(fp, **kwargs) as font:
        codepoints = {i for table in font["cmap"].tables for i in table.cmap}
    arr = np.array([i for i in codepoints if chr(i).isprintable()], dtype="<u4")
    return np.sort(arr)  # type: ignore[arg-type]
