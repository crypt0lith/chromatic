import random
import sys
from string import ascii_letters

from chromatic import ColorStr, ansicolor4Bit, ansicolor8Bit, ansicolor24Bit
from chromatic.color.core import randcolor

if not __package__:
    from pathlib import Path

    parents = Path(__file__).resolve().parents[1::-1]
    __package__ = ".".join(p.stem for p in parents)
    sys.path.insert(0, str(parents[0].parent))

from . import cprofile_wrapper


def _rand_color_str_array(n_rows=10, n_cols=10):
    size = n_rows * n_cols
    bit_str = ""
    while "1" not in set(bit_str):
        rand_bits = random.getrandbits(size)
        bit_str = f"{rand_bits:0{size}b}"
    rand_bin = list(map(lambda x: bool(int(x)), bit_str))
    random.shuffle(rand_bin)
    rand_bin_iter = iter(rand_bin)
    printable_chars = list(ascii_letters)
    output = []
    for _ in range(n_rows):
        current = []
        for _ in range(n_cols):
            char = random.choice(printable_chars) if next(rand_bin_iter) else None
            current.append(
                ColorStr(char, randcolor(), ansi_type=ansicolor24Bit) if char else " "
            )
        output.append(
            "{}{}{}{}".format(
                *map(
                    "".join,
                    (
                        current,
                        *(
                            [
                                c.as_ansi_type(t) if isinstance(c, ColorStr) else c
                                for c in current
                            ]
                            for t in (ansicolor4Bit, ansicolor8Bit, ansicolor24Bit)
                        ),
                    ),
                )
            )
        )
    return "\n".join(output)


def main():
    cprofile_wrapper(number=1000)(_rand_color_str_array)()


if __name__ == "__main__":
    sys.exit(main())
