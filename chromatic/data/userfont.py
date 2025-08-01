import json
import os
import os.path as osp
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import AnyStr, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypedDict, Required

    class _UserFontDict(TypedDict, total=False):
        font: Required[str]
        size: int
        index: int
        encoding: str


os.environ.setdefault("CHROMATIC_DATADIR", osp.dirname(__file__))
os.environ["CHROMATIC_FONTDIR"] = osp.join(os.environ["CHROMATIC_DATADIR"], "fonts")
if not osp.exists(os.environ["CHROMATIC_FONTDIR"]):
    os.mkdir(os.environ["CHROMATIC_FONTDIR"])

_TRUETYPE_EXT = frozenset({'.ttf', '.ttc'})


@dataclass(frozen=True, slots=True)
class UserFont:
    font: str
    size: int = field(default=24, kw_only=True)
    index: int = field(default=0, kw_only=True)
    encoding: str = field(default='', kw_only=True)

    def __hash__(self):
        return hash((type(self), self.font, self.size, self.index, self.encoding))

    def __fspath__(self):
        return osp.realpath(
            osp.join(os.environ["CHROMATIC_DATADIR"], self.font), strict=True
        )

    def to_truetype(self):
        from PIL.ImageFont import truetype

        return truetype(self, self.size, self.index, self.encoding)


userfont: MappingProxyType[str, UserFont] = {}.items().mapping


def _load_userfonts():
    global userfont

    userfont_json = osp.join(os.environ["CHROMATIC_DATADIR"], "userfont.json")
    if osp.exists(userfont_json):
        with open(userfont_json, mode='rb') as f:
            data = json.load(f)
        userfont = {k: UserFont(**v) for k, v in data.items()}.items().mapping
    return userfont


def _update_userfonts(*items: 'tuple[str, _UserFontDict]'):
    if items:
        userfont_json = osp.join(os.environ["CHROMATIC_DATADIR"], "userfont.json")
        if osp.exists(userfont_json):
            with open(userfont_json, mode='rb') as rf:
                current = json.load(rf)
        else:
            current = {}
        current.update(items)
        with open(userfont_json, mode='w') as wf:
            json.dump(current, wf, indent='\t', sort_keys=True)  # type: ignore[arg-type]
        return _load_userfonts()
    return userfont


def register_userfont(
    fp: AnyStr | os.PathLike[AnyStr],
    *,
    name: str = None,
    size: int = None,
    index: int = None,
    encoding: str = None,
    symlink=False,
    copy=False,
):
    metadata = {
        k: v
        for k, v in locals().items()
        if (k in {'size', 'index', 'encoding'} and v is not None)
    }
    name = name or osp.splitext(osp.basename(fp))[0]
    if not isinstance(name, str):
        raise TypeError(
            f"expected 'name' to be 'str', got {type(name).__name__!r} instead"
        )
    fp = osp.realpath(fp, strict=True)
    if not osp.isfile(fp):
        raise ValueError(f"not a file: {fp!r}")
    if osp.splitext(fp)[1] not in _TRUETYPE_EXT:
        raise ValueError(f"file is not valid truetype font {tuple(_TRUETYPE_EXT)}")
    if osp.dirname(fp) != os.environ["CHROMATIC_FONTDIR"] and (symlink or copy):
        loc = osp.join(os.environ["CHROMATIC_FONTDIR"], osp.basename(fp))
        if symlink:
            os.symlink(fp, loc)
        else:
            chunksize = 0xFFFF + 1
            with open(fp, mode='rb') as rf, open(loc, mode='wb') as wf:
                while chunk := rf.read(chunksize):
                    wf.write(chunk)
        fp = loc
    fp = osp.relpath(fp, os.environ["CHROMATIC_DATADIR"])
    return _update_userfonts((name, {'font': fp} | metadata))


def _validate_default_font(name='vga437'):
    from ._fetchers import filehash, _fetch_remote

    if name in userfont and (
        filehash(userfont[name])
        == "a8c767fa925624d28d9879c3a03a86204f78bce4decda0a206fd152bdd906c94"
    ):
        return
    filename = f"{name}.ttf"
    relpath = f"fonts/{filename}"
    out_path = osp.join(os.environ["CHROMATIC_FONTDIR"], filename)
    register_userfont(_fetch_remote(relpath, out_path), name=name)


def _init_userfonts():
    _load_userfonts()
    if not userfont:
        for fname in os.listdir(os.environ["CHROMATIC_FONTDIR"]):
            if osp.splitext(fname)[1] in _TRUETYPE_EXT:
                register_userfont(osp.join(os.environ["CHROMATIC_FONTDIR"], fname))
    _validate_default_font()


_init_userfonts()
