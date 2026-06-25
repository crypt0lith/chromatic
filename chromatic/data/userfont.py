__all__ = [
    "DEFAULT_FONT",
    "UserFont",
    "VGA437",
    "delete_userfont",
    "edit_userfont",
    "register_userfont",
    "rename_userfont",
    "set_default_userfont",
    "unregister_userfont",
    "userfonts",
]
import json
import os
import sys
import typing as tp
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType as mappingproxy

_TRUETYPE_EXT = frozenset({'.ttf', '.ttc'})
_ROOT_FONT_DIR = Path(__file__).parent / "fonts"
_ROOT_FONT_KEY = "vga437"
_DEFAULT_FONT = None


@dataclass(frozen=True, slots=True, repr=False)
class UserFont:
    font: str
    size: int = field(default=24, kw_only=True)
    index: int = field(default=0, kw_only=True)
    encoding: str = field(default='', kw_only=True)
    is_default: bool = field(default=False, kw_only=True, compare=False)
    _base_dir: Path = field(init=False, compare=False)

    def __post_init__(self):
        font_path = Path(self.font)
        if not font_path.is_absolute():
            raise ValueError
        if not font_path.is_file():
            raise FileNotFoundError(f"{font_path}")
        object.__setattr__(self, "font", font_path.name)
        object.__setattr__(self, "_base_dir", font_path.parent)
        if self.is_default:
            global _DEFAULT_FONT
            _DEFAULT_FONT = self

    def __hash__(self):
        return hash((type(self), self.font, self.size, self.index, self.encoding))

    def __fspath__(self):
        return os.fspath(self._base_dir.joinpath(self.font).resolve(strict=True))

    def to_truetype(self):
        from PIL.ImageFont import truetype

        return truetype(self, self.size, self.index, self.encoding)


_userfonts = dict[str, UserFont]()
userfonts = mappingproxy(_userfonts)
DEFAULT_FONT: UserFont


class _UserfontDict(tp.TypedDict, total=False):
    font: tp.Required[str]
    size: int
    index: int
    encoding: str
    is_default: bool


class _TypedDictStruct(tp.NamedTuple):
    required: frozenset[str]
    optional: frozenset[str]
    all_keys: frozenset[str]
    annotations: mappingproxy[str, type[tp.Any]]

    @classmethod
    def from_typeddict(
        cls, typ: type[tp.TypedDict], /  # type: ignore[valid-type]
    ) -> tp.Self:
        required = typ.__required_keys__
        optional = typ.__optional_keys__
        all_keys = frozenset(required | optional)
        annotations = mappingproxy(tp.get_type_hints(typ))
        return cls(required, optional, all_keys, annotations)

    def match(self, obj: Mapping[str, tp.Any], /) -> bool:
        if not obj.keys() <= self.all_keys:
            return False
        for k in self.required:
            if not (k in obj and isinstance(obj[k], self.annotations[k])):
                return False
        for k in self.optional:
            if k in obj and not isinstance(obj[k], self.annotations[k]):
                return False
        return True


@lru_cache(maxsize=1)
def _userfont_dict_struct():
    return _TypedDictStruct.from_typeddict(_UserfontDict)


def _is_userfont_dict(obj: dict, /) -> tp.TypeGuard[_UserfontDict]:
    return _userfont_dict_struct().match(obj)


def _load_userfonts(userfont_json: Path) -> dict[str, _UserfontDict]:
    fname = "userfont.json"
    if userfont_json.name != fname:
        raise ValueError
    if not userfont_json.exists():
        raise FileNotFoundError(f"{userfont_json}")
    with userfont_json.open("rb") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise TypeError
    for v in d.values():
        if not isinstance(v, dict):
            raise TypeError
        if not _is_userfont_dict(v):
            raise ValueError
    return d


def _load_userfonts_from_dir(font_dir: Path, sync=False) -> dict[str, UserFont] | None:
    if not font_dir.is_dir():
        raise NotADirectoryError(f"{font_dir}")
    userfont_json = font_dir / "userfont.json"
    try:
        d = _load_userfonts(userfont_json)
    except FileNotFoundError:
        return
    nonexistent = set()
    for k, v in d.items():
        font_abspath = os.path.normpath(font_dir.joinpath(v["font"]).absolute())
        if not os.path.isfile(font_abspath):
            if sync and not os.path.exists(font_abspath):
                nonexistent.add(k)
                continue
            raise FileNotFoundError(font_abspath)
        v.update(font=font_abspath)
    if sync and nonexistent:
        for k in nonexistent:
            del d[k]
        if not d:
            _try_delete_file(userfont_json)
            return
    return {k: UserFont(**v) for k, v in d.items()}


def _dump_userfonts(
    mapping: dict[str, dict[str, tp.Any]], /, font_dir: str | os.PathLike[str]
):
    font_dir = Path(font_dir)
    userfont_json = font_dir / "userfont.json"
    d = {}
    if userfont_json.exists():
        d.update(_load_userfonts(userfont_json))
    for k, v in mapping.items():
        if not isinstance(v, dict):
            raise TypeError
        if not _is_userfont_dict(v):
            raise ValueError
        d[k] = v
    with userfont_json.open("w") as f:
        json.dump(d, f, indent="\t", sort_keys=True)


def _try_delete_file(path: str | os.PathLike[str]) -> bool:
    path = Path(path)
    if os.name != "nt" and not os.access(path.parent.absolute(), os.W_OK | os.X_OK):
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def _get_font_dir():
    try:
        if d := os.environ["CHROMATIC_FONTS"]:
            return Path(d)
    except KeyError:
        pass
    home = Path.home()
    if os.name == "nt":
        d = os.environ.get("LOCALAPPDATA") or home.joinpath("AppData", "Local")
    else:
        d = os.environ.get("XDG_DATA_HOME") or home.joinpath(".local", "share")
    return Path(d).joinpath(__name__.partition(".")[0], "fonts")


class _RegisterUserfontKwargs(tp.TypedDict, total=False):
    name: str
    size: int
    index: int
    encoding: str
    is_default: bool
    symlink: bool


def register_userfont(
    fp: str | os.PathLike[str],
    font_dir: str | os.PathLike[str] | None = None,
    **kwargs: tp.Unpack[_RegisterUserfontKwargs],
):
    fp = Path(fp)
    if not fp.is_file():
        raise FileNotFoundError(f"{fp}")
    if fp.suffix.lower() not in _TRUETYPE_EXT:
        raise ValueError("not a truetype font file: %r" % str(fp))
    if font_dir is None:
        font_dir = _get_font_dir()
    else:
        font_dir = Path(font_dir)
    font_dir.mkdir(parents=True, exist_ok=True)
    if not font_dir.is_absolute():
        font_dir = Path(os.path.normpath(font_dir.absolute()))
    if font_dir.samefile(_ROOT_FONT_DIR):
        caller_file = Path(sys._getframe(1).f_code.co_filename)
        if not (caller_file.is_file() and caller_file.samefile(__file__)):
            import warnings

            warnings.warn(
                "you are writing to the root font directory. "
                "files added here will likely be deleted "
                "the next time you update this package.",
                UserWarning,
            )
    name = fp.stem
    metadata_fields = {"size": int, "index": int, "encoding": str, "is_default": bool}
    metadata = {}
    symlink = False
    typ_err_msg = (
        "expected {!r} to be {.__name__}, got type {.__class__.__name__!r} instead"
    ).format
    for k, v in kwargs.items():
        if k == "name":
            if not isinstance(v, str):
                err = typ_err_msg(k, str, v)
                raise TypeError(err)
            name = v
        elif k in metadata_fields:
            expected_t = metadata_fields[k]
            if not isinstance(v, expected_t):
                err = typ_err_msg(k, expected_t, v)
                raise TypeError(err)
            metadata[k] = v
        elif k == "symlink":
            symlink = bool(v)
        else:
            raise ValueError(f"unexpected keyword argument: {k!r}")
    if not fp.parent.samefile(font_dir):
        loc = font_dir / fp.name
        if symlink:
            loc.symlink_to(os.path.normpath(fp.absolute()))
        else:
            with fp.open("rb") as rf, loc.open("wb") as wf:
                chunksize = 0xFFFF + 1
                while chunk := rf.read(chunksize):
                    wf.write(chunk)
        fp = loc
    metadata["font"] = os.path.abspath(fp)
    _userfonts[name] = UserFont(**metadata)
    _dump_userfonts({name: metadata | {"font": fp.name}}, font_dir)


def unregister_userfont(name: str, /, delete=False):
    if name == _ROOT_FONT_KEY:
        raise ValueError(f"cannot unregister root default font: {name!r}")
    try:
        obj = _userfonts.pop(name)
    except KeyError as err:
        raise ValueError(f"invalid font: {name!r}") from err
    base_dir = obj._base_dir
    userfont_json = base_dir.joinpath("userfont.json")
    with userfont_json.open("r") as f:
        d = json.load(f)
    del d[name]
    if d:
        with userfont_json.open("w") as f:
            json.dump(d, f, indent="\t", sort_keys=True)
    else:
        # safe try-delete userfont.json because it is empty
        _try_delete_file(userfont_json)
    if delete is True:
        # unsafe unlink font file when explicitly passed
        base_dir.joinpath(obj.font).unlink()


def delete_userfont(name: str, /):
    return unregister_userfont(name, delete=True)


def rename_userfont(name: str, newname: str, /):
    if name == _ROOT_FONT_KEY:
        raise ValueError(f"cannot rename root default font: {name!r}")
    if name not in _userfonts:
        raise ValueError(f"invalid font: {name!r}")
    if name == newname:
        return
    userfont_json = _userfonts[name]._base_dir / "userfont.json"
    with userfont_json.open("r") as f:
        d = json.load(f)
    d[newname] = d.pop(name)
    with userfont_json.open("w") as f:
        json.dump(d, f, indent="\t", sort_keys=True)
    _userfonts[newname] = _userfonts.pop(name)


def _userfont_asdict(obj: UserFont, /):
    d = asdict(obj)
    d.update(font=str(d.pop("_base_dir").joinpath(d.pop("font"))))
    return d


class _EditUserfontKwargs(_UserfontDict):
    font: tp.NotRequired[str]  # type: ignore[misc]


def edit_userfont(name: str, /, **kwargs: tp.Unpack[_EditUserfontKwargs]):
    if name not in _userfonts:
        raise ValueError(f"invalid font: {name!r}")
    if not kwargs:
        return
    all_keys, *_, anno = _userfont_dict_struct()
    if not kwargs.keys() <= all_keys:
        raise ValueError("unexpected keys: {!r}".format(kwargs.keys() - all_keys))
    for k, typ in anno.items():
        if k not in kwargs:
            continue
        v = kwargs[k]
        if not isinstance(v, typ):
            err = str.format(
                "expected {!r} to be {.__name__!r}, got {.__class__.__name__!r} instead",
                k,
                typ,
                v,
            )
            raise TypeError(err)
        if k != "font":
            continue
        if name == _ROOT_FONT_KEY:
            raise ValueError(f"cannot change filepath of root default font: {name!r}")
        p = Path(v).absolute()
        if not p.is_file():
            raise FileNotFoundError(v)
        new_pardir = p.parent
        current_pardir = _userfonts[name]._base_dir
        if not new_pardir.samefile(current_pardir):
            err = str.format(
                "invalid filepath {!r}: "
                "parent directory does not match registered parent directory ({} != {})",
                v,
                new_pardir,
                current_pardir,
            )
            raise ValueError(err)
        kwargs[k] = str(p)
    obj = _userfonts[name]
    d = _userfont_asdict(obj)
    d.update(kwargs)
    new_obj = _userfonts[name] = UserFont(**d)
    d.update(font=new_obj.font)
    _dump_userfonts({name: d}, obj._base_dir)
    if name == _ROOT_FONT_KEY:
        global VGA437
        VGA437 = new_obj


def set_default_userfont(name: str, /):
    if _DEFAULT_FONT is None:
        edit_userfont(name, is_default=True)
        return
    current = next(k for k, v in _userfonts.items() if v is _DEFAULT_FONT)
    edit_userfont(current, is_default=False)
    try:
        edit_userfont(name, is_default=True)
    except Exception:
        edit_userfont(current, is_default=True)
        raise


def _fetch_default_font():
    from ._fetchers import _fetch_remote

    name = _ROOT_FONT_KEY
    fname = f"{name}.ttf"
    out_path = str(_ROOT_FONT_DIR / fname)
    _ROOT_FONT_DIR.mkdir(exist_ok=True)
    out_file = _fetch_remote(f"{_ROOT_FONT_DIR.name}/{fname}", out_path)
    register_userfont(out_file, _ROOT_FONT_DIR, name=name, is_default=True)


def _validate_default_font():
    from ._fetchers import filehash

    name = _ROOT_FONT_KEY
    if name in userfonts and (
        filehash(userfonts[name])
        == "a8c767fa925624d28d9879c3a03a86204f78bce4decda0a206fd152bdd906c94"
    ):
        return
    return _fetch_default_font()


def _init_default_font():
    if _ROOT_FONT_DIR.exists():
        d = _load_userfonts_from_dir(_ROOT_FONT_DIR, sync=True)
        if d is not None:
            _userfonts.update(d)
            return _validate_default_font()
        default_font_fname = f"{_ROOT_FONT_KEY}.ttf"
        default_font_path = _ROOT_FONT_DIR.joinpath(default_font_fname)
        if default_font_path.exists():
            register_userfont(
                default_font_path, _ROOT_FONT_DIR, name=_ROOT_FONT_KEY, is_default=True
            )
            return _validate_default_font()
    _fetch_default_font()


def _init_user_fonts():
    user_font_dir = _get_font_dir()
    if not user_font_dir.exists():
        return
    d = _load_userfonts_from_dir(user_font_dir, sync=True)
    if d is None:
        return
    if _ROOT_FONT_KEY in d:
        del d[_ROOT_FONT_KEY]
    _userfonts.update(d)


_init_default_font()
VGA437 = userfonts[_ROOT_FONT_KEY]
_init_user_fonts()


def __getattr__(name, /):
    if name == "DEFAULT_FONT":
        return _DEFAULT_FONT or VGA437
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
