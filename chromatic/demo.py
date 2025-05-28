import functools
import math
import os
import sys
import time
from os import PathLike
from pathlib import PurePath
from types import FunctionType
from typing import Callable


def escher_dragon_ascii():
    """Displays the image-to-ASCII transform of 'Dragon' by M.C. Escher."""
    from chromatic.ascii import ascii2img, img2ascii
    from chromatic.data import UserFont, escher

    input_img = escher()
    font = UserFont.IBM_VGA_437_8X16
    char_set = r"  ._-~+<vX♦'^Vx>|πΦ0Ω#$║╫"

    ascii_str = img2ascii(
        input_img, font, factor=240, char_set=char_set, sort_glyphs=True
    )

    ascii_img = ascii2img(ascii_str, font, font_size=16, fg='white', bg='black')

    ascii_img.show()


def escher_dragon_256color():
    """Displays the image-to-ANSI transform of 'Dragon' by M.C. Escher in 8-bit color."""
    from chromatic.ascii import ansi2img, img2ansi
    from chromatic.data import UserFont, escher

    input_img = escher()
    font = UserFont.IBM_VGA_437_8X16

    ansi_array = img2ansi(input_img, font, factor=240, ansi_type='8b', equalize=True)

    ansi_img = ansi2img(ansi_array, font, font_size=16)

    ansi_img.show()


def butterfly_16color():
    """Displays image-to-ANSI transform of 'Spider Lily & Papilio xuthus' in 4-bit color.

    Good ol' C-x M-c M-butterfly...
    """
    from chromatic.color import ansicolor4Bit
    from chromatic.ascii import ansi2img, img2ansi
    from chromatic.data import UserFont, butterfly

    input_img = butterfly()

    font = UserFont.IBM_VGA_437_8X16

    char_set = r"'·,•-_→+<>ⁿ*%⌂7√Iï∞πbz£9yîU{}1αHSw♥æ?GX╕╒éà⌡MF╝╩ΘûÇƒQ½☻Å¶┤▄╪║▒█"

    ansi_array = img2ansi(
        input_img, font, factor=200, char_set=char_set, ansi_type=ansicolor4Bit
    )

    ansi_img = ansi2img(ansi_array, font, font_size=16)

    ansi_img.show()


def butterfly_truecolor():
    """Displays the image-to-ANSI transform of 'Spider Lily & Papilio xuthus' in 24-bit color."""
    from chromatic.ascii import ansi2img, img2ansi
    from chromatic.data import UserFont, butterfly

    input_img = butterfly()

    font = UserFont.IBM_VGA_437_8X16

    ansi_array = img2ansi(
        input_img, font, factor=200, ansi_type='24b', equalize='white_point'
    )

    ansi_img = ansi2img(ansi_array, font, font_size=16)

    ansi_img.show()


def butterfly_randcolor():
    from chromatic.ascii import ansi2img, img2ansi
    from chromatic.color import randcolor, rgb2hsv, hsv2rgb, Color
    from chromatic.data import UserFont, butterfly

    input_img = butterfly()

    font = UserFont.IBM_VGA_437_8X16

    ansi_array = img2ansi(
        input_img, font, factor=200, ansi_type='8b', equalize='white_point'
    )

    for row in range(len(ansi_array)):
        for idx, cs in enumerate(ansi_array[row]):
            if (fg := cs.fg) is not None:
                _, _, v = rgb2hsv(fg.rgb)
                h, s, _ = rgb2hsv(randcolor().rgb)
                ansi_array[row][idx] = cs.recolor(fg=Color.from_rgb(hsv2rgb((h, s, v))))

    ansi_img = ansi2img(ansi_array, font, font_size=16)

    ansi_img.show()


def goblin_virus_truecolor():
    """`G-O-B-L-I-N VIRUS <https://imgur.com/n0Mng2P>`__"""
    from chromatic.ascii import ansi2img, img2ansi
    from chromatic.data import UserFont, goblin_virus

    input_img = goblin_virus()

    font = UserFont.IBM_VGA_437_8X16

    char_set = r'  .-|_⌐¬^:()═+<>v≥≤«*»x└┘π╛╘┴┐┌┬╧╚╙X╒╜╨#0╓╝╩╤╥│╔┤├╞╗╦┼╪║╟╠╫╣╬░▒▓█▄▌▐▀'

    ansi_array = img2ansi(
        input_img, font, factor=200, char_set=char_set, ansi_type='24b', equalize=False
    )

    ansi_img = ansi2img(ansi_array, font, font_size=16)

    ansi_img.show()


def named_colors():
    from chromatic.color.palette import named_color_idents, ColorNamespace
    from chromatic.color.colorconv import rgb2hsv, rgb2lab

    print(f"{'.'.join([ColorNamespace.__module__, ColorNamespace.__name__])}:")
    named = named_color_idents()
    whites = [0]
    for idx, n in enumerate(named):
        hsv = rgb2hsv(n.fg.rgb)
        if all(
            map(lambda i, x: math.isclose(hsv[i], x, abs_tol=0.16), (-1, 1), (1, 0))
        ):
            if idx - whites[-1] < 4:
                whites.pop()
            whites.append(idx)
    whites.append(-1)
    buffer = []
    for start, stop in zip(whites, whites[1:]):
        xs = sorted(
            named[start + 1 if start else None : stop + 1 if ~stop else None],
            key=lambda x: rgb2lab(x.fg.rgb),
        )
        buffer.append(xs)
    for ln in buffer:
        print(' | '.join(map(str, ln)))


def color_table():
    """Print foreground / background combinations in each ANSI format.

    A handful of stylistic SGR parameters are displayed as well.
    """
    from chromatic.color import (
        ColorStr,
        SgrParameter,
        ansicolor24Bit,
        ansicolor4Bit,
        ansicolor8Bit,
    )
    from chromatic.color.palette import ColorNamespace

    color_ns = ColorNamespace()
    ansi_types = [ansicolor4Bit, ansicolor8Bit, ansicolor24Bit]
    colors = [
        color_ns.BLACK,
        color_ns.WHITE,
        color_ns.RED,
        color_ns.ORANGE,
        color_ns.YELLOW,
        color_ns.GREEN,
        color_ns.BLUE,
        color_ns.INDIGO,
        color_ns.PURPLE,
    ]
    colors_dict = {v.name.title(): v for v in colors}
    spacing = max(map(len, colors_dict)) + 1
    fg_colors = [
        ColorStr(
            f"{c.name.title(): ^{spacing}}",
            color_spec=dict(fg=c),
            ansi_type=ansicolor24Bit,
        )
        for c in colors
    ]
    bg_colors = [ColorStr().recolor(bg=None)] + [
        c.recolor(fg=None, bg=c.fg) for c in fg_colors
    ]
    print(
        '|'.join(
            f"{'%dbit' % n: {'>' if n > 9 else '^'}{spacing - 1}}" for n in (4, 8, 24)
        )
    )
    suffix = '\x1b[0m' if sys.stdout.isatty() else ''
    for row in fg_colors:
        for col in bg_colors:
            for typ in ansi_types:
                print(row.as_ansi_type(typ).recolor(bg=col.bg), end=suffix)
        print()
    print('\nstyles:')
    print()
    style_params = [
        SgrParameter.BOLD,
        SgrParameter.ITALICS,
        SgrParameter.CROSSED_OUT,
        SgrParameter.ENCIRCLED,
        SgrParameter.SINGLE_UNDERLINE,
        SgrParameter.DOUBLE_UNDERLINE,
        SgrParameter.NEGATIVE,
    ]
    for style in style_params:
        print(
            ColorStr('.'.join([SgrParameter.__qualname__, style.name])).update_sgr(
                style
            ),
            end=suffix + (' ' * 4),
        )
    print()


def glyph_comparisons(__output_dir: str | PathLike[str] = None):
    from skimage.metrics import mean_squared_error
    from numpy import ndarray
    from chromatic.ascii import cp437_printable
    from chromatic import get_glyph_masks
    from chromatic.data import UserFont
    from random import choices as get_random

    def _find_best_matches(
        glyph_masks1: dict[str, ndarray], glyph_masks2: dict[str, ndarray]
    ) -> dict[str, str]:
        best_matches = {}
        for char1, mask1 in glyph_masks1.items():
            best_char = None
            best_score = float('inf')
            for char2, mask2 in glyph_masks2.items():
                score = mean_squared_error(mask1, mask2)
                if score < best_score:
                    best_score = score
                    best_char = char2
            best_matches[char1] = best_char
        return best_matches

    if __output_dir and not os.path.isdir(__output_dir):
        raise NotADirectoryError(__output_dir)
    user_fonts = [pair := (UserFont.IBM_VGA_437_8X16, UserFont.CONSOLAS), pair[::-1]]
    trans_table = str.maketrans({']': None, '0': ' ', '[': ' '})
    char_set = cp437_printable()
    separator = '#' * 100
    for font1, font2 in user_fonts:
        glyph_masks_1 = get_glyph_masks(font1, char_set, dist_transform=True)
        glyph_masks_2 = get_glyph_masks(font2, char_set, dist_transform=True)
        best_matches_ = _find_best_matches(glyph_masks_1, glyph_masks_2)
        txt = ''.join(
            '->'.center(32, ' ')
            .join(['{}'] * 2)
            .format(
                f"{font1.name}"
                f"[{input_char!r}, {input_char.encode('unicode_escape').decode()!r}]",
                f"{font2.name}"
                f"[{matched_char!r}, {matched_char.encode('unicode_escape').decode()!r}]",
            )
            .center(100, ' ')
            + '\n\n'
            + '\n'.join(
                ''.join(z).translate(trans_table)
                for z in zip(
                    f'{glyph_masks_1[input_char].astype(int)}\n'.splitlines(),
                    f'{glyph_masks_2[matched_char].astype(int)}\n'.splitlines()[1:],
                )
            )
            + separator.join(['\n'] * 2)
            for input_char, matched_char in best_matches_.items()
        )
        if __output_dir is not None:
            fname = (
                PurePath(__output_dir)
                / f"{'_to_'.join(font.name.lower() for font in (font1, font2))}.txt"
            )
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(txt)
        else:
            for glyph in get_random(txt.split(separator), k=len(char_set) // 2):
                print(separator + glyph)


class _time_wrapper[**P, R]:

    def __init__(self, func: Callable[P, R] | FunctionType | type = None):
        self.func = func
        if self.func is not None:
            functools.update_wrapper(self, self.func)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if self.func is not None:
            start = time.perf_counter()
            result = self.func(*args, **kwargs)
            stop = time.perf_counter()
            print(f"Total execution time: {self._delta(start, stop)}")
            return result
        else:
            self.func = args[0]
            functools.update_wrapper(self, self.func)
            return self

    @staticmethod
    def _delta(start: float, stop: float) -> str:
        delta = stop - start
        mag, fmt = min(
            [(1, 's'), (1e-3, 'ms'), (1e-6, 'μs'), (1e-9, 'ns'), (1e-12, 'ps')],
            key=lambda x: abs(math.log10(x[0]) - math.log10(delta)),
        )
        delta *= 1 / mag
        return f"{delta:.3f} {fmt}"


def main():
    demo_globals = dict(globals())
    demo_globals.pop('main')
    from inspect import getargs

    global_func_enum = dict(
        enumerate(
            sorted(
                k
                for k, v in demo_globals.items()
                if isinstance(v, FunctionType) and not v.__name__.startswith('_')
            )
        )
    )
    safe_funcs = {-1: exit}
    choices = [f'[{x[0]}]: {x[1].name}' for x in safe_funcs.items()]
    names = []
    for k, v in global_func_enum.items():
        if not any(getargs(demo_globals[v].__code__)):
            if safe_funcs.get(k - 1) is None:
                k_val = list(safe_funcs).pop() + 1
            else:
                k_val = k
            safe_funcs[k_val] = globals()[v]
            choices.append(f"[{k_val}]: {v}")
            names.append(v)

    def _check_user_input(user_key: str):
        if user_key.strip('-').isdigit():
            if (k := int(user_key)) in safe_funcs:
                return k
        if (s := user_key.strip().replace(' ', '_').casefold()) in names:
            return next(i for i, v in enumerate(names) if v == s)
        return

    selection = None
    if len(sys.argv) > 1:
        key = sys.argv[1]
        if key.casefold() == '-h'.casefold():
            docstrings = (
                '{}:\n\t{}'.format(
                    n,
                    '\n'.join(
                        ln for ln in (globals()[n].__doc__ or '...').splitlines() if ln
                    ).strip(),
                )
                for n in names
            )
            print(
                '\n'
                + '\n\n'.join(['Run one of the following demo functions:', *docstrings])
            )
            exit()
        selection = _check_user_input(key)
    if selection is None:
        print('\n'.join(choices))
    while selection not in safe_funcs:
        try:
            selection = _check_user_input(input(f"Select a demo function:\t"))
        except ValueError:
            pass
        except KeyboardInterrupt:
            exit()
    try:
        if selection == -1:
            exit()
        print(f"Running {names[selection]!r}...\n")
    except KeyError:
        pass
    _time_wrapper(safe_funcs[selection])()


if __name__ == '__main__':
    main()
