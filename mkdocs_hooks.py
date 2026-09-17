import argparse
import os

from mkdocs.structure.files import File


def _subparsers(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _help_block(parser):
    return "```text\n" + parser.format_help().rstrip() + "\n```"


def _render_group(name, group):
    lines = [f"# `chromatic {name}`", "", _help_block(group), ""]
    for sub_name, sub in _subparsers(group).items():
        lines += [f"## `chromatic {name} {sub_name}`", "", _help_block(sub), ""]
    return "\n".join(lines).rstrip() + "\n"


def on_files(files, config):
    import chromatic.__main__ as main

    COLUMNS = "96"
    parser = main.Parser(prog=main.__package__)
    prev = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = COLUMNS
    try:
        for name, group in _subparsers(parser).items():
            content = _render_group(name, group)
            files.append(File.generated(config, f"cli/{name}.md", content=content))
    finally:
        del os.environ["COLUMNS"]
        if prev is not None:
            os.environ["COLUMNS"] = prev
    return files
