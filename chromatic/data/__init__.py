__all__ = [
    'DEFAULT_FONT',
    'UserFont',
    'VGA437',
    'butterfly',
    'escher',
    'goblin_virus',
    'register_userfont',
    'userfonts',
]

from ._fetchers import _load
from .userfont import VGA437, UserFont, register_userfont, userfonts


def butterfly():
    return _load("butterfly.jpg")


def escher():
    return _load("escher.png")


def goblin_virus():
    return _load("goblin_virus.png")


DEFAULT_FONT: UserFont


def __dir__():
    return __all__[:]


def __getattr__(name, /):
    if name == "DEFAULT_FONT":
        from .userfont import DEFAULT_FONT

        return DEFAULT_FONT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
