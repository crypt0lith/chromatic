# CLI

Installing `chromatic-python` also installs a `chromatic` command:

```shell
chromatic --help
```

The command is split into two groups:

- [`chromatic image`](image.md) — convert images into ANSI art. Render the
  result to an image file, stream the escape-coded text to a terminal, or dump
  the intermediate array as a NumPy file (and read it back). Animated inputs are
  supported, with GIF/WEBP fallbacks and in-terminal playback.
- [`chromatic font`](font.md) — manage the user-font registry: register a
  TrueType font under a name, edit its attributes, list what is registered,
  rename or delete an entry, or mark one as the default.

## The font registry

Image conversion needs a font to compare glyphs against. Fonts are looked up by
name from a registry backed by a directory on disk. The `chromatic font`
subcommands maintain that registry; the `--font-dir` option (or the
`CHROMATIC_FONTS` environment variable) selects which directory to use.

A font marked as the default is exposed to the library as
`chromatic.DEFAULT_FONT`.
