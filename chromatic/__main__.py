import collections.abc as abc
import os
import sys
import typing as tp
from collections import ChainMap
from inspect import signature


def parse_args():
    import argparse as ap

    class SetEnvAction(ap.Action):
        def __init__(self, *args, **kwargs):
            if kwargs.get("nargs") not in ("?", None):
                raise ValueError(
                    "ambiguous 'nargs' for env setter: {nargs!r}".format_map(kwargs)
                )
            env = kwargs.pop("env")
            if not isinstance(env, str):
                raise TypeError
            self.env = env
            super().__init__(*args, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            if self.nargs == "?" and values is None:
                setattr(namespace, self.dest, values)
            elif not isinstance(values, str):
                parser.error(
                    "{.__class__.__name__!r} values must be {.__name__!r}, "
                    "got {.__class__.__name__!r}".format(self, str, values)
                )
            else:
                vars(namespace)[self.dest] = os.environ[self.env] = values

    class EnvHelpFormatter(ap.ArgumentDefaultsHelpFormatter):
        def _get_help_string(self, action):
            out = super()._get_help_string(action)
            if out is not None and isinstance(action, SetEnvAction):
                out += " (env: {}={})".format(
                    *(lambda s: (s, os.environ.get(s, "")))(action.env)
                )
            return out

    new_base_parser = lambda *args, **kwargs: ap.ArgumentParser(
        *args,
        **ChainMap(
            kwargs,
            dict(add_help=False, allow_abbrev=False, argument_default=ap.SUPPRESS),
        ),
    )

    def init_font_dir_env_parser():
        parser = new_base_parser()
        group = parser.add_argument_group("env options")
        group.add_argument(
            "--font-dir",
            metavar="DIRECTORY",
            action=SetEnvAction,
            env="CHROMATIC_FONTS",
            help="path to font directory",
        )
        return parser

    font_dir_env_base = init_font_dir_env_parser()

    if os.path.isfile(sys.argv[0]) and os.path.samefile(sys.argv[0], __file__):
        top_parser = ap.ArgumentParser(prog=__package__)
    else:
        top_parser = ap.ArgumentParser()

    cmd_subparsers = top_parser.add_subparsers(dest="cmd", required=True)

    def init_font_subcmds(parser: ap.ArgumentParser):
        subcmds = parser.add_subparsers(dest="subcmd", required=True)
        mut_by_name_base = new_base_parser()
        mut_by_name_base.add_argument(
            dest="name", metavar="NAME", help="name of the registered font"
        )
        uf_attr_base = new_base_parser()
        uf_attr_opts = uf_attr_base.add_argument_group("font attribute options")
        uf_attr_opts.add_argument(
            "--size", dest="size", type=int, help="size in pixels"
        )
        uf_attr_opts.add_argument(
            "--index", dest="index", type=int, help="which font face to load"
        )
        uf_attr_opts.add_argument(
            "--encoding",
            dest="encoding",
            metavar="ENC",
            help="which font encoding to use",
        )
        uf_attr_opts.add_argument(
            "--default",
            dest="is_default",
            action=ap.BooleanOptionalAction,
            help="whether to set this font as the new default font",
        )
        subcmds.add_parser(
            "delete",
            parents=[font_dir_env_base, mut_by_name_base],
            formatter_class=EnvHelpFormatter,
            help="delete a user font from the registry",
        )
        subcmd_p_edit = subcmds.add_parser(
            "edit",
            parents=[font_dir_env_base, uf_attr_base, mut_by_name_base],
            formatter_class=EnvHelpFormatter,
            help="edit the attributes of a registered user font",
        )
        subcmd_p_list = subcmds.add_parser(
            "list",
            parents=[font_dir_env_base],
            formatter_class=EnvHelpFormatter,
            help="list all currently registered user fonts",
        )
        subcmd_p_register = subcmds.add_parser(
            "register",
            parents=[font_dir_env_base, uf_attr_base],
            formatter_class=EnvHelpFormatter,
            help="add a new font to the registry",
        )
        subcmd_p_rename = subcmds.add_parser(
            "rename",
            parents=[font_dir_env_base, mut_by_name_base],
            formatter_class=EnvHelpFormatter,
            help="rename a registered user font",
        )
        subcmds.add_parser(
            "set-default",
            parents=[font_dir_env_base, mut_by_name_base],
            formatter_class=EnvHelpFormatter,
            help=f"""\
            set a user font as the default font.
            it will be accessible as `{__package__}.DEFAULT_FONT`""",
        )
        p_edit_font_opts = subcmd_p_edit.add_argument_group(
            "font options", argument_default=ap.SUPPRESS
        )
        p_edit_font_opts.add_argument(
            "--font", dest="font", metavar="FILE", help="path to truetype font file"
        )
        p_list_fmt_opts = subcmd_p_list.add_argument_group("formatting options")
        p_list_fmt_opts.add_argument(
            "--json",
            dest="json",
            action="store_true",
            help="emit entries in JSON format",
        )
        subcmd_p_register.add_argument(
            dest="font", metavar="FONT", help="path to truetype font file"
        )
        subcmd_p_register.add_argument(
            dest="name",
            metavar="NAME",
            nargs="?",
            default=ap.SUPPRESS,
            help="name of the registered font",
        )
        p_register_fs_opts = subcmd_p_register.add_argument_group("filesystem options")
        p_register_fs_opts.add_argument(
            "-s",
            "--symbolic",
            dest="symlink",
            action="store_true",
            help="""\
            create a symlink instead of copying the file.
            like `ln -s TARGET LINK_NAME`, this will fail if the link name already exists""",
        )
        subcmd_p_rename.add_argument(
            dest="newname",
            metavar="NEWNAME",
            help="new name to give the registered font",
        )

    def init_image_subcmds(parser: ap.ArgumentParser):
        subcmds = parser.add_subparsers(dest="subcmd", required=True)
        ansify_base = new_base_parser()
        font_opts = ansify_base.add_argument_group("font options")
        font_opts.add_argument(
            "--font",
            dest="font",
            help="name of user font, or path to truetype font file",
        )
        font_opts.add_argument(
            "--size",
            dest="font_size",
            type=int,
            metavar="SIZE",
            help="font size, in pixels",
        )

        from .color.palette import ColorNamespace

        color_ns = ColorNamespace.asdict()

        def color_parser(allow_rgba=False):
            def parsed_color(
                s: str, /
            ) -> tuple[int, int, int] | tuple[int, int, int, int]:
                try:
                    return color_ns[s.strip().replace(" ", "_").upper()]
                except KeyError:
                    pass
                values = tuple(bytes.fromhex(s))
                if len(values) == 3 or (allow_rgba and len(values) == 4):
                    return values
                raise ValueError

            return parsed_color

        color_opts = ansify_base.add_argument_group("color defaults")
        color_opts.add_argument(
            "--fg",
            dest="fg_default",
            metavar="COLOR",
            type=color_parser(allow_rgba=True),
            help="default foreground color. can be hex rgb(a) value or color name",
        )
        color_opts.add_argument(
            "--bg",
            dest="bg_default",
            metavar="COLOR",
            type=color_parser(allow_rgba=True),
            help="default background color. can be hex rgb(a) value or color name",
        )

        from_img_base = new_base_parser()

        colorize_opts = from_img_base.add_argument_group("colorization options")
        colorize_opts.add_argument(
            "-a",
            "--ansi-type",
            metavar="ANSITYPE",
            choices=("4b", "8b", "24b"),
            help="""\
            which ansi colorspace to use for color quantization.
            since truecolor is rgb, '24b' effectively means 'original colors'
            (choices: %(choices)s)""",
        )
        equalize_opts = colorize_opts.add_mutually_exclusive_group()
        equalize_opts.add_argument(
            "--equalize",
            dest="equalize",
            action=ap.BooleanOptionalAction,
            help="""\
            whether to apply contrast stretch equalization before image-to-ansi conversion""",
        )
        equalize_opts.add_argument(
            "--white-point",
            dest="equalize",
            action="store_const",
            const="white_point",
            help="apply white-point equalization before image-to-ansi conversion",
        )
        colorize_opts.add_argument(
            "--input-bg",
            dest="bg",
            metavar="COLOR",
            type=color_parser(),
            default=None,
            help="""\
            initial background (canvas) color,
            injected as an attribute on the intermediate text object itself.
            can be hex rgb value or color name""",
        )
        glyph_opts = from_img_base.add_argument_group("glyph options")
        glyph_opts.add_argument(
            "--factor",
            dest="factor",
            metavar="N",
            type=int,
            help="""\
            scaling factor for image width to %(metavar)s columns per output line.
            analogous to level-of-detail""",
        )
        char_set_opts = glyph_opts.add_mutually_exclusive_group()
        char_set_opts.add_argument(
            "-c",
            "--chars",
            dest="char_set",
            metavar="CHARS",
            help="""\
            charset where %(metavar)s is a string containing chars to use for glyph selection.
            a char can appear more than once, which affects frequency distribution""",
        )

        from .image._curses import ascii_printable, cp437_printable

        char_set_preset_help = (
            "use printable characters from {} character encoding as charset"
        ).format
        char_set_opts.add_argument(
            "--ascii",
            dest="char_set",
            action="store_const",
            const=ascii_printable(),
            help=char_set_preset_help("ASCII"),
        )
        char_set_opts.add_argument(
            "--latin1",
            dest="char_set",
            help=char_set_preset_help("Latin-1"),
            action="store_const",
            const="".join(filter(str.isprintable, bytes(range(256)).decode("latin1"))),
        )
        char_set_opts.add_argument(
            "--cp437",
            dest="char_set",
            action="store_const",
            const=cp437_printable(),
            help=char_set_preset_help("code page 437"),
        )
        char_set_opts.add_argument(
            "--cp1252",
            dest="char_set",
            help=char_set_preset_help("Windows-1252"),
            action="store_const",
            const="".join(
                filter(
                    str.isprintable, bytes(range(256)).decode("cp1252", errors="ignore")
                )
            ),
        )

        glyph_sort_opts = glyph_opts.add_mutually_exclusive_group()
        glyph_sort_opts.add_argument(
            "--sort",
            dest="sort_glyphs",
            action=ap.BooleanOptionalAction,
            help="""\
            whether to sort glyphs based on their perceived 'luminance'.
            for example, ('.' -> '#') is (dark -> light)""",
        )
        glyph_sort_opts.add_argument(
            "-r",
            "--reverse",
            dest="sort_glyphs",
            action="store_const",
            const=reversed,
            help="""\
            sort glyphs in reverse order.
            flips the luminance mapping to (light -> dark)""",
        )

        def save_img_callback(path: str | os.PathLike[str] | None = None, /):

            # deferred exception handler
            # on write error, fall-through to Image.show()
            # then re-raise after in-memory image is opened

            from functools import wraps

            def defer_exc[**P, R](
                f: abc.Callable[P, abc.Generator[Exception, tp.Any, R]], /
            ):
                @wraps(f)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                    err = None
                    it = f(*args, **kwargs)
                    try:
                        while True:
                            err = next(it)
                    except StopIteration as stop:
                        return stop.value
                    finally:
                        if err is not None:
                            raise err

                return wrapper

            import PIL.Image

            @defer_exc
            def callback(ns: ap.Namespace, img: PIL.Image.Image):
                if path is not None:
                    try:
                        img.save(path)
                    except Exception as e:
                        yield e
                    else:
                        if getattr(ns, "show", False):
                            with PIL.Image.open(path) as f:
                                f.show()
                        return path
                dump_stdout = hasattr(ns, "dumpfile") and ns.dumpfile.fileno() == 1
                if getattr(ns, "show", not dump_stdout):
                    img.show()

            return callback

        def save_img_to_dir(dirname: str, /):
            from datetime import datetime
            from pathlib import Path

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            outfile = Path(dirname, f"{__package__}_{timestamp}.png")
            return save_img_callback(outfile)

        output_opts_base = new_base_parser()
        output_opts = output_opts_base.add_argument_group("output options")
        outfile_opts = output_opts.add_mutually_exclusive_group()
        outfile_opts.add_argument(
            "-O",
            "--outfile",
            metavar="FILE",
            dest="outfile_callback",
            type=save_img_callback,
            help="save the image to %(metavar)s",
        )
        outfile_opts.add_argument(
            "-o",
            "--output-dir",
            metavar="DIRECTORY",
            dest="outfile_callback",
            type=save_img_to_dir,
            help="save the image as a png file in %(metavar)s",
        )
        output_opts.add_argument(
            "--show",
            action=ap.BooleanOptionalAction,
            help="whether to show the image in the system's image viewer",
        )
        dumpfile_opts = output_opts.add_mutually_exclusive_group()
        dumpfile_opts.add_argument(
            "-d",
            "--dump-text",
            metavar="FILE",
            dest="dumpfile",
            type=ap.FileType("wb"),
            help="write the ansified image text to %(metavar)s",
        )
        dumpfile_opts.add_argument(
            "--stdout",
            dest="dumpfile",
            action="store_const",
            const=sys.stdout.buffer,
            help="write the ansified image text to stdout",
        )
        output_opts_base.set_defaults(_outfile_callback=save_img_callback())
        subcmd_p_ansify = subcmds.add_parser(
            "ansify",
            parents=[font_dir_env_base, ansify_base, from_img_base, output_opts_base],
            formatter_class=EnvHelpFormatter,
        )
        subcmd_p_ansify.add_argument(
            dest="img",
            metavar="IMAGEFILE",
            help="input image",
        )

    font_cmd_subparser = cmd_subparsers.add_parser("font")
    image_cmd_subparser = cmd_subparsers.add_parser("image")

    init_font_subcmds(font_cmd_subparser)
    init_image_subcmds(image_cmd_subparser)

    return top_parser.parse_args()


def _call_from_ns[R](f: abc.Callable[..., R], /, ns, **kwargs) -> R:
    params = signature(f).parameters
    kwargs = ChainMap(kwargs, vars(ns))
    f_args, f_kwargs = [], {}
    for k, p in params.items():
        if k not in kwargs:
            continue
        if p.kind == 0:
            f_args.append(kwargs[k])
        else:
            f_kwargs[k] = kwargs[k]
    return f(*f_args, **f_kwargs)

def handle_image(ns):
    vars(ns).setdefault("outfile_callback", ns._outfile_callback)
    match ns.subcmd:
        case "ansify":
            from .image import img2ansi, ansi2img

            ansi_array = _call_from_ns(img2ansi, ns)
            if hasattr(ns, "dumpfile"):
                ns.dumpfile.writelines(f"{s}\n".encode() for s in map("".join, ansi_array))
            img = _call_from_ns(ansi2img, ns, ansi_array=ansi_array)
            try:
                outpath = ns.outfile_callback(ns, img)
            except Exception as e:
                print(f"[-] error: {e}", file=sys.stderr)
                return -1
            if outpath is not None:
                print(f"{outpath}")
            return
        case _:
            raise ValueError(f"invalid subcommand: {ns.subcmd!r}")


def font_list(ns):
    from dataclasses import asdict

    from .data import userfont as uf

    json_mode = getattr(ns, "json", False)
    if sys.stdout.isatty() and not json_mode:
        from collections import defaultdict

        from . import ColorStr

        table = defaultdict(list)
        widths = defaultdict(int)
        justs = {}
        anno = uf._userfont_dict_struct().annotations
        for name, obj in uf.userfonts.items():
            is_default = obj is uf.DEFAULT_FONT
            for k, v in dict(name=name, **asdict(obj)).items():
                justs.setdefault(k, ">" if anno.get(k) is int else "<")
                parts = [f"{v}"]
                if k in ("is_default", "_base_dir"):
                    parts[0] = ColorStr(parts[0], fg=0x6C6C6C)
                    if k == "is_default":
                        parts[0] = parts[0].lower()
                        if is_default:
                            parts.append(ColorStr("default", fg=0xEC143C))
                elif k == "name":
                    parts[0] = ColorStr(parts[0]).bold()
                table[k].append(parts)
                width = sum(map(len, parts))
                width += (len(parts) - 1) * 2
                if widths[k] < width:
                    widths[k] = width
        out = defaultdict(list)
        for k, (col, just, width) in zip(
            table, zip(table.values(), justs.values(), widths.values())
        ):
            k = k.upper().strip("_")
            width = max(width, len(k), 8)
            idx = 0 if just == ">" else -1
            for cell in col:
                if len(cell) > 1:
                    diff = width - len(cell[~idx])
                    cell[0] = cell[0].remove_reset()
                else:
                    diff = 0
                cell[idx] = f"{cell[idx]: {just}{width - diff}}"
            header = ColorStr(f"\x1b[37m{k}").faint().underline()
            out[f"{header: {just}{width}}"].extend(map(", ".join, col))
        headers, cols = zip(*out.items())
        outlines = map(("\xa0" * 2).join, [headers, *zip(*cols)])
    else:
        out = []
        for name, obj in uf.userfonts.items():
            d: dict[str, tp.Any] = dict(name=name, **asdict(obj))
            d.update(font=str(d.pop("_base_dir").joinpath(d["font"])))
            out.append(d)
        if json_mode:
            import json

            outlines = map(json.dumps, out)
        else:
            outlines = ("\t".join(map(str, d.values())) for d in out)
    return sys.stdout.writelines(f"{line}\n" for line in outlines)


def handle_font(ns):
    match ns.subcmd:
        case "list":
            return font_list(ns)
        case ("register" | "edit") as sc:
            keys = {"size", "index", "encoding", "is_default"}
            match sc:
                case "register":
                    from .data.userfont import register_userfont as f

                    args = [ns.font]
                    keys.update(("font_dir", "name", "symlink"))
                case "edit":
                    from .data.userfont import edit_userfont as f

                    args = [ns.name]
                    keys.add("font")
                case _:
                    raise RuntimeError("unreachable")
            kwargs = {}
            for k in vars(ns).keys() & keys:
                kwargs[k] = getattr(ns, k)
            return f(*args, **kwargs)
        case ("rename" | "set-default" | "delete") as sc:
            args = [ns.name]
            match sc:
                case "rename":
                    from .data.userfont import rename_userfont as f

                    args.append(ns.newname)
                case "set-default":
                    from .data.userfont import set_default_userfont as f

                case "delete":
                    from .data.userfont import delete_userfont as f

                case _:
                    raise RuntimeError("unreachable")
            return f(*args)
        case _:
            raise ValueError(f"invalid subcommand: {ns.subcmd!r}")


def main():
    ns = parse_args()
    match ns.cmd:
        case "font":
            return handle_font(ns)
        case "image":
            return handle_image(ns)


if __name__ == "__main__":
    sys.exit(main())
