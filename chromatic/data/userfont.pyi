__all__ = [
    "DEFAULT_FONT",
    "UserFont",
    "VGA437",
    "delete_userfont",
    "register_userfont",
    "unregister_userfont",
    "userfonts",
]
import os
import typing as tp
from dataclasses import dataclass, field
from types import MappingProxyType

from PIL.ImageFont import FreeTypeFont

@dataclass(frozen=True, slots=True, repr=False)
class UserFont:
    font: str
    size: int = field(default=24, kw_only=True)
    index: int = field(default=0, kw_only=True)
    encoding: str = field(default='', kw_only=True)
    is_default: bool = field(default=False, kw_only=True, compare=False)
    def __post_init__(self) -> None: ...
    def __hash__(self) -> int: ...
    def __fspath__(self) -> str: ...
    def to_truetype(self) -> FreeTypeFont: ...

userfonts: tp.Final[MappingProxyType[str, UserFont]]

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
) -> None: ...
def unregister_userfont(name: str, /, delete: bool = False) -> None: ...
def delete_userfont(name: str, /) -> None: ...

VGA437: tp.Final[UserFont]
DEFAULT_FONT: tp.Final[UserFont]
