![image](/logo/logo.PNG)

[![image](https://img.shields.io/pypi/v/chromatic-python)](https://pypi.org/project/chromatic-python/)
![image](https://img.shields.io/pypi/pyversions/chromatic-python)
[![image](https://static.pepy.tech/badge/chromatic-python)](https://pepy.tech/projects/chromatic-python)
[![image](https://mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

Chromatic is a library for processing and transforming ANSI escape sequences (colored terminal text).

It offers a collection of algorithms and types for a variety of use cases:	
- Image-to-ASCII / Image-to-ANSI conversions.
- ANSI art rendering, with support for user-defined fonts.
- A `ColorStr` type which enables precise low-level control over ANSI-escaped strings through a convenient interface.
- [colorama](https://github.com/tartley/colorama/)-style wrappers (`Fore`, `Back`, `Style`).
- Parametrization of ANSI color bit formats, allowing arbitrary conversion between 16-color, 256-color, and true-color (RGB) colorspace on any object implementing a `colorbytes` buffer.
- Et Cetera 😲

### Usage
#### ColorStr
```python
from chromatic import ColorStr

base_str = 'hello world'

red_fg = ColorStr(base_str, 0xFF0000)

assert red_fg.base_str == base_str
assert red_fg.rgb_dict == {'fg': (0xFF, 0, 0)}
assert red_fg.ansi == b'\x1b[38;5;196m'
```

`ColorStr` can handle different signatures for `color_spec`:
```python
from chromatic import ColorStr

assert all(
	ColorStr('*', cs) == ColorStr('*', 0xFF0000)
    for cs in [b'\x1b[38;5;196m', b'\xff\x00\x00', (0xFF, 0, 0), {'fg': 0xFF0000}]
)
```

The ANSI color format can be given as an argument, or returned by `ColorStr.as_ansi_type()` as a new instance.
```python
from chromatic import ColorStr, ansicolor24Bit, ansicolor4Bit

truecolor = ColorStr('*', 0xFF0000, ansi_type=ansicolor24Bit)
a_16color = truecolor.as_ansi_type(ansicolor4Bit)

# each ansi color format has an alias that can be used in place of the type object
assert a_16color == truecolor.as_ansi_type('4b')

assert truecolor.ansi_format is ansicolor24Bit and truecolor.ansi == b'\x1b[38;2;255;0;0m'
assert a_16color.ansi_format is ansicolor4Bit and a_16color.ansi == b'\x1b[31m'
```

Adding and removing specific ANSI codes from the escape sequence:
```python
import chromatic as cm

boring_str = cm.ColorStr('hello world')

assert boring_str.ansi == b''

bold_str = boring_str + cm.SgrParameter.BOLD
assert bold_str.ansi == b'\x1b[1m'

# use ColorStr.update_sgr() to remove and add SGR values
unbold_str = bold_str.update_sgr(cm.SgrParameter.BOLD)
assert unbold_str == boring_str
assert bold_str == unbold_str.update_sgr(cm.SgrParameter.BOLD)
```

#### Image-to-ANSI conversion

Converting an image into an array of ANSI-escaped characters, then rendering the ANSI array as another image:
```python
from chromatic.color import ansicolor4Bit
from chromatic.ascii import ansi2img, img2ansi
from chromatic.data import UserFont, butterfly

input_img = butterfly()

font = UserFont.IBM_VGA_437_8X16

# by default, `char_set` would be sorted based on the relative weight of glyphs in the font
# but because `sort_glyphs` is set to False, `char_set` will be directly mapped to the image brightness
#            | <- index 0 is the 'darkest'
char_set = r"'·,•-_→+<>ⁿ*%⌂7√Iï∞πbz£9yîU{}1αHSw♥æ?GX╕╒éà⌡MF╝╩ΘûÇƒQ½☻Å¶┤▄╪║▒█"
#                                            index -1 is the 'brightest' -> |

ansi_array = img2ansi(input_img, font, sort_glyphs=False, char_set=char_set, ansi_type=ansicolor4Bit, factor=200)

# ansi2img() returns a PIL.Image.Image object
ansi_img = ansi2img(ansi_array, font, font_size=16)
ansi_img.show()
```

### Installation
Install the package using `pip`:
```bash
pip install chromatic-python
```
