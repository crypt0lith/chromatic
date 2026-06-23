import os
import sys
import typing as tp


def parse_args():
    import argparse

    class SetEnvAction(argparse.Action):
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

    class EnvHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
        def _get_help_string(self, action):
            out = super()._get_help_string(action)
            if out is not None and isinstance(action, SetEnvAction):
                out += " (env: {}={})".format(
                    *(lambda s: (s, os.environ.get(s, "")))(action.env)
                )
            return out

    parser_kw = {
        "allow_abbrev": False,
        "argument_default": argparse.SUPPRESS,
        "formatter_class": EnvHelpFormatter,
    }

    new_base_parser = lambda *args, **kwargs: argparse.ArgumentParser(
        *args,
        **dict(
            add_help=False,
            allow_abbrev=False,
            argument_default=argparse.SUPPRESS,
            **kwargs,
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
        top_parser = argparse.ArgumentParser(prog=__package__)
    else:
        top_parser = argparse.ArgumentParser()

    cmd_subparsers = top_parser.add_subparsers(dest="cmd", required=True)

    def init_font_subcmds(parser: argparse.ArgumentParser):
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
            action=argparse.BooleanOptionalAction,
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
            help=(f"""\
                set a user font as the default font.
                it will be accessible as `{__package__}.DEFAULT_FONT`"""),
        )
        p_edit_font_opts = subcmd_p_edit.add_argument_group(
            "font options", argument_default=argparse.SUPPRESS
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
            dest="name", metavar="NAME", nargs="?", help="name of the registered font"
        )
        subcmd_p_rename.add_argument(
            dest="newname",
            metavar="NEWNAME",
            help="new name to give the registered font",
        )

    def init_image_subcmds(parser: argparse.ArgumentParser):
        subcmds = parser.add_subparsers(dest="subcmd", required=True)
        pass

    font_cmd_subparser = cmd_subparsers.add_parser("font")
    image_cmd_subparser = cmd_subparsers.add_parser("image")

    init_font_subcmds(font_cmd_subparser)
    init_image_subcmds(image_cmd_subparser)

    return top_parser.parse_args()


def handle_image(ns):
    match ns.subcmd:
        case _:
            raise NotImplementedError(f"subcommand {ns.subcmd!r} not yet implemented")


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
        anno = uf._userfont_dict_struct()[-1]
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
            d = dict(name=name, **asdict(obj))
            d.update(font=str(d.pop("_base_dir").joinpath(d["font"])))
            out.append(d)
        if json_mode:
            import json

            outlines = map(json.dumps, out)
        else:
            outlines = ["\t".join(map(str, d.values())) for d in out]

    return print(*outlines, sep="\n")


def handle_font(ns):
    match ns.subcmd:
        case "list":
            return font_list(ns)
        case "register":
            from .data.userfont import register_userfont

            kwargs = {
                k: v
                for k, v in vars(ns).items()
                if k
                in {
                    "font_dir",
                    "name",
                    "size",
                    "index",
                    "encoding",
                    "is_default",
                    "symlink",
                }
            }
            return register_userfont(ns.font, **kwargs)
        case "delete":
            from .data.userfont import delete_userfont

            return delete_userfont(ns.name)
        case _:
            raise NotImplementedError(f"subcommand {ns.subcmd!r} not yet implemented")


def main():
    ns = parse_args()
    match ns.cmd:
        case "font":
            return handle_font(ns)
        case "image":
            return handle_image(ns)


if __name__ == "__main__":
    sys.exit(main())
