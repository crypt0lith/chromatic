from os import PathLike
from types import MappingProxyType
from typing import Final

from PIL.ImageFont import FreeTypeFont

class UserFont:
    font: str
    size: int = ...
    index: int = ...
    encoding: str = ...
    is_default: bool = ...

    def __init__(
        self,
        font: str,
        *,
        size: int = ...,
        index: int = ...,
        encoding: str = ...,
        is_default: bool = ...,
    ): ...
    def __fspath__(self) -> str: ...
    def to_truetype(self) -> FreeTypeFont: ...

userfonts: Final[MappingProxyType[str, UserFont]]

def register_userfont(
    fp: str | PathLike[str],
    font_dir: str | PathLike[str] | None = None,
    *,
    name: str = ...,
    size: int = ...,
    index: int = ...,
    encoding: str = ...,
    is_default: bool = ...,
    symlink: bool = False,
) -> None: ...
def unregister_userfont(name: str, /, delete=False) -> None: ...
def delete_userfont(name: str, /) -> None: ...

VGA437: Final[UserFont]
DEFAULT_FONT: Final[UserFont]
