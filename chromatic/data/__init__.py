__all__ = ["DEFAULT_FONT", "UserFont", "VGA437", "register_userfont", "userfonts"]
__dir__ = __all__.copy

from .userfont import VGA437, UserFont, register_userfont, userfonts

DEFAULT_FONT: UserFont


def __getattr__(name, /):
    if name == "DEFAULT_FONT":
        from .userfont import DEFAULT_FONT

        return DEFAULT_FONT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
