import os
import sys
from os import PathLike
from pathlib import PurePath


def escher_dragon_ascii():
    """
    Render and display an image-to-ASCII transform of 'Dragon' by M.C. Escher.
    """
    from chromatic.ascii import ascii2img, img2ascii
    from chromatic.data import UserFont, escher

    input_img = escher()
    font = UserFont.IBM_VGA_437_8X16
    char_set = r"  ._-~+<vX♦'^Vx>|πΦ0Ω#$║╫"

    ascii_str = img2ascii(
        input_img,
        font,
        factor=240,
        char_set=char_set,
        sort_glyphs=True)

    ascii_img = ascii2img(
        ascii_str,
        font,
        font_size=16,
        fg=(255, 255, 255),
        bg=(0, 0, 0))

    ascii_img.show()


def escher_dragon_256color():
    """
    Render and display an in 8-bit color image-to-ANSI transform of 'Dragon' by M.C. Escher.
    """
    from chromatic.ascii import ansi2img, img2ansi
    from chromatic.data import UserFont, escher

    input_img = escher()
    font = UserFont.IBM_VGA_437_8X16

    ansi_array = img2ansi(
        input_img,
        font,
        factor=240,
        ansi_type='8b',
        equalize=True)

    ansi_img = ansi2img(
        ansi_array,
        font,
        font_size=16)

    ansi_img.show()


def butterfly_16color():
    """
    Render and display an image-to-ANSI transform of 'Spider Lily & Papilio xuthus' in 4-bit color.
    Good ol' C-x M-c M-butterfly...
    """
    from chromatic.ansi import ansicolor4Bit
    from chromatic.ascii import ansi2img, img2ansi
    from chromatic.data import UserFont, butterfly

    input_img = butterfly()

    font = UserFont.IBM_VGA_437_8X16

    char_set = r"'·,•-_→+<>ⁿ*%⌂7√Iï∞πbz£9yîU{}1αHSw♥æ?GX╕╒éà⌡MF╝╩ΘûÇƒQ½☻Å¶┤▄╪║▒█"

    ansi_array = img2ansi(
        input_img,
        font,
        factor=200,
        char_set=char_set,
        ansi_type=ansicolor4Bit)

    ansi_img = ansi2img(
        ansi_array,
        font,
        font_size=16)

    ansi_img.show()


def butterfly_truecolor():
    """
    Render and display an image-to-ANSI transform of 'Spider Lily & Papilio xuthus' in 24-bit color.
    """
    from chromatic.ascii import ansi2img, img2ansi
    from chromatic.data import UserFont, butterfly

    input_img = butterfly()

    font = UserFont.IBM_VGA_437_8X16

    ansi_array = img2ansi(
        input_img,
        font,
        factor=200,
        ansi_type='24b',
        equalize='white_point')

    ansi_img = ansi2img(
        ansi_array,
        font,
        font_size=16)

    ansi_img.show()


def goblin_virus_truecolor():
    """
    G-O-B-L-I-N VIRUS
    https://imgur.com/n0Mng2P
    """
    from chromatic.ascii import ansi2img, img2ansi
    from chromatic.data import UserFont, goblin_virus

    input_img = goblin_virus()

    font = UserFont.IBM_VGA_437_8X16

    char_set = r'  .-|_⌐¬^:()═+<>v≥≤«*»x└┘π╛╘┴┐┌┬╧╚╙X╒╜╨#0╓╝╩╤╥│╔┤├╞╗╦┼╪║╟╠╫╣╬░▒▓█▄▌▐▀'

    ansi_array = img2ansi(
        input_img,
        font,
        factor=200,
        char_set=char_set,
        ansi_type='24b',
        equalize=False)

    ansi_img = ansi2img(
        ansi_array,
        font,
        font_size=16)

    ansi_img.show()


def named_colors():
    from chromatic.ansi.palette import display_named_colors, ColorNamespace

    color_ns_typ = type(ColorNamespace)
    print(f"{'.'.join([color_ns_typ.__module__, color_ns_typ.__qualname__.lstrip('_')])}:")
    named = display_named_colors()
    color_count = len(named)
    get_padding = lambda x: float(len(x) * 1.5).__ceil__()
    row_len = float(color_count ** (1 / 2)).__floor__()
    spacer = ' :: '
    arr = []
    it = iter(named)
    while color_count >= row_len:
        s = ''
        for _ in range(row_len):
            color_count -= 1
            color = next(it)
            s += f"{color:^{get_padding(color)}}\x1b[0m{spacer}"
        arr.append(s.removesuffix(spacer))
    arr.append(spacer.join(f"{x:^{get_padding(x)}}\x1b[0m" for x in it))
    print('\n'.join(arr))


def color_table():
    """
    Print out monochromatic + ROYGBIV foreground / background combinations in each ANSI format.
    A handful of stylistic SGR parameters are displayed as well.
    """
    from chromatic.ansi import (
        ColorStr,
        SgrParameter,
        ansicolor24Bit,
        ansicolor4Bit,
        ansicolor8Bit
    )
    from chromatic.ansi.palette import ColorNamespace

    ansi_types = [ansicolor4Bit, ansicolor8Bit, ansicolor24Bit]
    colors = [
        ColorNamespace.BLACK,
        ColorNamespace.WHITE,
        ColorNamespace.RED,
        ColorNamespace.ORANGE,
        ColorNamespace.YELLOW,
        ColorNamespace.GREEN,
        ColorNamespace.BLUE,
        ColorNamespace.INDIGO,
        ColorNamespace.PURPLE]
    colors_dict = {v.name.title(): v for v in colors}
    spacing = max(map(len, colors_dict)) + 1
    fg_colors = [ColorStr(
        f"{c.name.title(): ^{spacing}}",
        color_spec=dict(fg=c),
        ansi_type=ansicolor24Bit)
        for c in colors]
    bg_colors = [
                    ColorStr().recolor(bg=None)
                ] + [c.recolor(fg=None, bg=c.fg) for c in fg_colors]
    pad = spacing - 1
    print(
        '|'.join(
            [f"{'4bit': ^{pad}}",
             f"{'8bit': ^{pad}}",
             f"{'24bit': >{pad}}"]))
    for row in fg_colors:
        for col in bg_colors:
            for typ in ansi_types:
                print(row.as_ansi_type(typ).recolor(bg=col.bg), end='\x1b[0m')
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
        SgrParameter.NEGATIVE]
    for style in style_params:
        print(
            ColorStr('.'.join([SgrParameter.__qualname__, style.name]))
            .update_sgr(style),
            end='\x1b[0m' + (' ' * 4))
    print()


def glyph_comparisons(__output_dir: str | PathLike[str] = None):
    from skimage.metrics import mean_squared_error
    from numpy import ndarray
    from chromatic.ascii import get_glyph_masks, cp437_printable
    from chromatic.data import UserFont
    from random import choices as get_random

    def _find_best_matches(glyph_masks1: dict[str, ndarray],
                           glyph_masks2: dict[str, ndarray]) -> dict[str, str]:
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
        raise NotADirectoryError(
            __output_dir)
    user_fonts = [(UserFont.IBM_VGA_437_8X16, UserFont.CONSOLAS)[::x] for x in (1, -1)]
    trans_table = str.maketrans({']': None, '0': ' ', '[': ' '})
    char_set = cp437_printable()
    separator = '#' * 100
    for font1, font2 in user_fonts:
        glyph_masks_1 = get_glyph_masks(font1, char_set, dist_transform=True)
        glyph_masks_2 = get_glyph_masks(font2, char_set, dist_transform=True)
        best_matches_ = _find_best_matches(glyph_masks_1, glyph_masks_2)
        txt = ''.join(
            '->'.center(32, ' ').join(['{}'] * 2).format(
                f"{font1.name}"
                f"[{input_char!r}, {input_char.encode('unicode_escape').decode()!r}]",
                f"{font2.name}"
                f"[{matched_char!r}, {matched_char.encode('unicode_escape').decode()!r}]")
            .center(100, ' ')
            + '\n\n'
            + '\n'.join(
                ''.join
                (z).translate(trans_table)
                for z in zip(
                    f'{glyph_masks_1[input_char].astype(int)}\n'.splitlines(),
                    f'{glyph_masks_2[matched_char].astype(int)}\n'.splitlines()[1:]))
            + separator.join(['\n'] * 2) for input_char, matched_char in best_matches_.items())
        if __output_dir is not None:
            fname = (PurePath(__output_dir) /
                     f"{'_to_'.join(font.name.lower() for font in (font1, font2))}.txt")
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(txt)
        else:
            for glyph in get_random(txt.split(separator), k=len(char_set) // 2):
                print(separator + glyph)


def main():
    demo_globals = dict(globals())
    demo_globals.pop('main')
    from types import FunctionType
    from inspect import getargs

    global_func_enum = dict(
        enumerate(sorted(k for k, v in demo_globals.items() if isinstance(v, FunctionType))))
    safe_funcs = {}
    safe_funcs[-1] = exit
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
            print(
                '\nRun one of the following demo functions:\n\n'
                f"{'\n'.join(f'{n}: {globals()[n].__doc__}' for n in names)}")
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
    safe_funcs[selection]()


if __name__ == '__main__':
    main()
