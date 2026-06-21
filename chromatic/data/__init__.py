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

import os
import os.path as osp

if "CHROMATIC_DATADIR" not in os.environ:
    os.environ["CHROMATIC_DATADIR"] = osp.dirname(__file__)

from ._fetchers import _load
from .userfont import DEFAULT_FONT, VGA437, UserFont, register_userfont, userfonts


def __dir__():
    return __all__[:]


def butterfly():
    return _load("butterfly.jpg")


def escher():
    return _load("escher.png")


def goblin_virus():
    return _load("goblin_virus.png")
