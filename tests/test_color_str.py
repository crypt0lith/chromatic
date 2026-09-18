import random
import unittest
from string import ascii_letters

from chromatic import (
    Color,
    ColorStr,
    SgrParameter,
    ansicolor4Bit,
    ansicolor8Bit,
    ansicolor24Bit,
    colorbytes,
)
from chromatic.color.colorconv import (
    ansi_4bit_to_rgb,
    ansi_8bit_to_rgb,
    nearest_ansi_4bit_rgb,
    rgb_to_ansi_8bit,
)
from chromatic.color.core import randcolor
from chromatic.color.palette import ColorNamespace

ANSI_4BIT_RGB: list[tuple[int, int, int]] = [
    (0, 0, 0),  # black
    (170, 0, 0),  # red
    (0, 170, 0),  # green
    (170, 85, 0),  # yellow
    (0, 0, 170),  # blue
    (170, 0, 170),  # magenta
    (0, 170, 170),  # cyan
    (170, 170, 170),  # white
    (85, 85, 85),  # bright black (grey)
    (255, 85, 85),  # bright red
    (85, 255, 85),  # bright green
    (255, 255, 85),  # bright yellow
    (85, 85, 255),  # bright blue
    (255, 85, 255),  # bright magenta
    (85, 255, 255),  # bright cyan
    (255, 255, 255),  # bright white
]


# noinspection PyTypeChecker
class TestAnsiColorBytes(unittest.TestCase):

    def test_colorbytes_init(self):
        self.assertIsInstance(colorbytes(b"\x1b[38;5;46m"), colorbytes)
        self.assertIsInstance(ansicolor4Bit(b"\x1b[31m"), ansicolor4Bit)
        self.assertIsInstance(ansicolor8Bit(b"\x1b[38;5;46m"), ansicolor8Bit)
        self.assertIsInstance(ansicolor24Bit(b"\x1b[38;2;255;0;0m"), ansicolor24Bit)

    def test_colorbytes_bad_input(self):
        with self.assertRaises(ValueError):
            ansicolor4Bit(b"\x1b[invalidm")

        with self.assertRaises(TypeError):
            ansicolor4Bit("not_bytes")

    def test_ansi_4bit_to_rgb(self):
        for value, expected_rgb in enumerate(ANSI_4BIT_RGB):
            self.assertEqual(ansi_4bit_to_rgb(value), expected_rgb)

    def test_nearest_ansi_4bit_color(self):
        for _ in range(100):
            r, g, b = randcolor().rgb
            nearest_color = nearest_ansi_4bit_rgb((r, g, b))
            self.assertIsInstance(nearest_color, tuple)
            self.assertEqual(len(nearest_color), 3)

    def test_rgb_to_ansi_8bit(self):
        for _ in range(100):
            rgb = randcolor().rgb
            ansi_code = rgb_to_ansi_8bit(rgb)
            self.assertIsInstance(ansi_code, int)
            self.assertTrue(0 <= ansi_code <= 255)

    def test_ansi_8bit_to_rgb(self):
        for value in range(256):
            rgb = ansi_8bit_to_rgb(value)
            self.assertIsInstance(rgb, tuple)
            self.assertEqual(len(rgb), 3)

    def test_color_class(self):
        for _ in range(100):
            color = randcolor()
            r, g, b = color.rgb
            hex_value = (r << 16) + (g << 8) + b
            color_from_hex = Color(hex_value)
            self.assertEqual(color_from_hex.rgb, (r, g, b))


class TestColorStr(unittest.TestCase):

    def test_color_instances(self):
        cs = ColorStr("Red text", fg=Color(0xFF0000))
        self.assertEqual(cs.fg.rgb, (255, 0, 0))

        cs_bg = ColorStr("Blue background text", bg=Color(0x0000FF))
        self.assertEqual(cs_bg.bg.rgb, (0, 0, 255))

    def test_rgb_tuple(self):
        cs = ColorStr("Red text", fg=(255, 0, 0))
        self.assertEqual(cs.fg.rgb, (255, 0, 0))

        cs_bg = ColorStr("Blue background text", bg=(0, 0, 255))
        self.assertEqual(cs_bg.bg.rgb, (0, 0, 255))

    def test_int_hex(self):
        cs = ColorStr("Red text", fg=0xFF0000)
        self.assertEqual(cs.fg.rgb, (255, 0, 0))

        cs_bg = ColorStr("Blue background text", bg=0x0000FF)
        self.assertEqual(cs_bg.bg.rgb, (0, 0, 255))

    def test_ansi_bytes(self):
        cs = ColorStr(b"\x1b[38;5;46mGreen text\x1b[0m", encoding="utf-8")
        self.assertEqual(cs.fg.rgb, (0, 255, 0))
        self.assertEqual(cs.base_str, "Green text")

        cs_bg = ColorStr(b"\x1b[48;5;46mGreen background\x1b[0m", encoding="utf-8")
        self.assertIn("bg", cs_bg.rgb_dict)
        self.assertEqual(cs_bg.bg.rgb, (0, 255, 0))

    def test_ansi_str(self):
        cs = ColorStr("\x1b[38;5;46mGreen text\x1b[0m")
        self.assertEqual(cs.fg.rgb, (0, 255, 0))
        self.assertEqual(cs.base_str, "Green text")

        cs_bg = ColorStr("\x1b[48;5;46mGreen background")
        self.assertIn("bg", cs_bg.rgb_dict)
        self.assertEqual(cs_bg.bg.rgb, (0, 255, 0))

    def test_color_dict(self):
        cs = ColorStr("Red on Blue text", fg=Color(0xFF0000), bg=Color(0x0000FF))
        self.assertEqual(cs.fg.rgb, (255, 0, 0))
        self.assertEqual(cs.bg.rgb, (0, 0, 255))

    def test_mixed_color_spec(self):
        cs = ColorStr("Red on Green text", fg=Color(0xFF0000), bg=(0, 255, 0))
        self.assertEqual(cs.fg.rgb, (255, 0, 0))
        self.assertEqual(cs.bg.rgb, (0, 255, 0))

        cs_mixed = ColorStr("Blue on Yellow text", fg=0x0000FF, bg=(255, 255, 0))
        self.assertEqual(cs_mixed.fg.rgb, (0, 0, 255))
        self.assertEqual(cs_mixed.bg.rgb, (255, 255, 0))

    def test_fuzz_constructors(self):
        for _ in range(100):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            random_text = "".join(
                random.choices(ascii_letters, k=random.randint(5, 15))
            )

            cs_color = ColorStr(random_text, fg=Color.from_rgb((r, g, b)))
            self.assertEqual(cs_color.fg.rgb, (r, g, b))

            cs_rgb = ColorStr(random_text, fg=(r, g, b))
            self.assertEqual(cs_rgb.fg.rgb, (r, g, b))

            hex_value = (r << 16) + (g << 8) + b
            cs_int = ColorStr(random_text, fg=hex_value)
            self.assertEqual(cs_int.fg.rgb, (r, g, b))

    def test_update_sgr(self):
        cs = ColorStr(ansi_type="4b", reset=False)
        self.assertEqual((cs.base_str, cs.ansi), ("", b""))

        red_fg = cs.add_sgr_param(SgrParameter.RED_BRIGHT_FG)
        self.assertEqual([ansicolor4Bit(b"91")], list(red_fg._sgr.values()))
        red_fg += "iadd"
        self.assertEqual([ansicolor4Bit(b"91")], list(red_fg._sgr.values()))
        self.assertEqual(red_fg.base_str, "iadd")


if __name__ == "__main__":
    unittest.main()
