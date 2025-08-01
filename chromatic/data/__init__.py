__all__ = [
    'UserFont',
    'butterfly',
    'escher',
    'goblin_virus',
    'register_userfont',
    'userfont',
]

import os
import os.path as osp

if "CHROMATIC_DATADIR" not in os.environ:
    os.environ["CHROMATIC_DATADIR"] = osp.dirname(__file__)

from ._fetchers import _load
from .userfont import userfont, register_userfont, UserFont


def __dir__():
    return __all__[:]


def butterfly():
    return _load("butterfly.jpg")


def escher():
    return _load("escher.png")


def goblin_virus():
    return _load("goblin_virus.png")
