from . import _array, _curses
from ._array import *
from ._curses import *

__all__ = list(set(_array.__all__) | set(_curses.__all__))
