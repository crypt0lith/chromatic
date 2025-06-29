from os import PathLike
from typing import AnyStr
from types import MappingProxyType

from PIL.ImageFont import FreeTypeFont

class UserFont:
    font: str
    size: int = ...
    index: int = ...
    encoding: str = ...

    def __init__(self, font: str, *, size: int = ..., index: int = ..., encoding: str = ...): ...
    def __fspath__(self) -> str: ...
    def to_truetype(self) -> FreeTypeFont: ...

userfont: MappingProxyType[str, UserFont]

def register_userfont(
    fp: AnyStr | PathLike[AnyStr],
    *,
    name: str = ...,
    size: int = ...,
    index: int = ...,
    encoding: str = ...,
    symlink: bool = True,
    copy: bool = False,
) -> UserFont: ...
