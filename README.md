![image](https://raw.githubusercontent.com/crypt0lith/chromatic/master/banner.png)

[![image](https://img.shields.io/pypi/v/chromatic-python)](https://pypi.org/project/chromatic-python/)
![image](https://img.shields.io/pypi/pyversions/chromatic-python)
[![image](https://static.pepy.tech/badge/chromatic-python)](https://pepy.tech/projects/chromatic-python)
[![image](https://mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

Chromatic is a library for ANSI art image processing and colored terminal text.

It offers a collection of algorithms and types for a variety of use cases:

- Image-to-ASCII / Image-to-ANSI conversions.
- ANSI art rendering, with support for user-defined fonts.
- A `ColorStr` type for low-level control over ANSI SGR strings.
- [colorama](https://github.com/tartley/colorama/)-style wrappers (`Fore`, `Back`, `Style`).
- Conversion between 16-color, 256-color, and true-color (RGB) ANSI colorspace via the `colorbytes` type.
- Et Cetera 😲

### Usage

#### `color_chain`

A `color_chain` is a printable sequence of colored text that round-trips with a NumPy array:

```python
from chromatic import color_chain, ColorStr

cc = color_chain(
    [ColorStr("hello", fg=0xFF0000), ColorStr(" world", fg=0x00FF00)], ansi_type="8b"
)

# parsed from a raw SGR string — no ansi_type needed
cc2 = color_chain("\x1b[38;5;196mhello\x1b[38;5;46m world")

assert cc == cc2

# red "hello", green " world"
print(cc)

# a structured ndarray of dtype [('char', '<U1'), ('sgr', '<u8'), ('rgb', 'u1', (2, 4))]
arr = cc.term_array()

assert str(color_chain.fromarray(arr)) == str(cc)
```

#### `img2ansi`

`img2ansi` reads an image and returns a `color_chain`:

```python
from chromatic import img2ansi
from chromatic.data import userfonts

font = userfonts["vga437"]

# the image, as ANSI art
cc = img2ansi("input.png", font, factor=200)
print(cc)

# or the raw ndarray
arr = img2ansi("input.png", font, factor=200, outarray=True)
```

#### `ansi2img`

`ansi2img` renders a `color_chain` (or array) back to an image:

```python
from chromatic import img2ansi, ansi2img
from chromatic.data import userfonts

font = userfonts["vga437"]

cc = img2ansi("input.png", font, factor=200)

# a PIL.Image.Image
img = ansi2img(cc, font, font_size=16)
img.show()
```

#### `ansify`

`ansify` runs both steps at once, returning the rendered image with its array on `Image.info`:

```python
from chromatic import ansify
from chromatic.data import userfonts

font = userfonts["vga437"]

img = ansify("input.png", font, font_size=16, factor=200)
img.show()

# the ndarray it rendered from
arr = img.info["ansi_array"]
```

#### Animated images

An animated image becomes a list of `color_chain` frames, one per frame — so a GIF plays straight to the terminal:

```python
import sys
import time

from chromatic import img2ansi
from chromatic.data import userfonts

font = userfonts["vga437"]

# list[color_chain], one per frame
frames = img2ansi("input.gif", font, factor=200)

for cc in frames:
    # redraw each frame in place
    sys.stdout.write(f"\x1b[H{cc}")
    time.sleep(0.1)
```

`ansify` renders the same GIF back to an animated image, and `chromatic image ansify input.gif --stdout` plays it in the terminal for you.

#### `ColorStr`

`ColorStr` is a `str` subclass that carries its SGR color as metadata; the escape codes only surface when you render it:

```python
from chromatic import ColorStr

cs = ColorStr("hello world", fg=0xFF0000, ansi_type="8b")

assert isinstance(cs, str)

# the codes aren't part of the string itself
assert cs.base_str == "hello world"
assert len(cs) == len("hello world")

# they surface when you print / str() it
assert str(cs) == "\x1b[38;5;196mhello world\x1b[0m"
assert cs.ansi == b"\x1b[38;5;196m"
```

### Installation

Install the package using `pip`:

```shell
pip install chromatic-python
```

### Credits

Banner artwork: [main rules by Crasher (2002)](https://16colo.rs/pack/galza-14/CRS-MAIN.ANS)
