__all__ = [
    'ctrl',
    'isprint',
    'ControlCharacter',
    'alt',
    'ascii_printable',
    'cp437_printable',
    'isctrl',
    'translate_cp437',
    'unctrl'
]

from enum import IntEnum
from types import MappingProxyType
from typing import Iterable, Iterator, overload, Union


class ControlCharacter(IntEnum):
    NUL = 0x00  # ^@
    SOH = 0x01  # ^A
    STX = 0x02  # ^B
    ETX = 0x03  # ^C
    EOT = 0x04  # ^D
    ENQ = 0x05  # ^E
    ACK = 0x06  # ^F
    BEL = 0x07  # ^G
    BS = 0x08  # ^H
    TAB = 0x09  # ^I
    HT = 0x09  # ^I
    LF = 0x0a  # ^J
    NL = 0x0a  # ^J
    VT = 0x0b  # ^K
    FF = 0x0c  # ^L
    CR = 0x0d  # ^M
    SO = 0x0e  # ^N
    SI = 0x0f  # ^O
    DLE = 0x10  # ^P
    DC1 = 0x11  # ^Q
    DC2 = 0x12  # ^R
    DC3 = 0x13  # ^S
    DC4 = 0x14  # ^T
    NAK = 0x15  # ^U
    SYN = 0x16  # ^V
    ETB = 0x17  # ^W
    CAN = 0x18  # ^X
    EM = 0x19  # ^Y
    SUB = 0x1a  # ^Z
    ESC = 0x1b  # ^[
    FS = 0x1c  # ^\
    GS = 0x1d  # ^]
    RS = 0x1e  # ^^
    US = 0x1f  # ^_
    DEL = 0x7f  # delete
    NBSP = 0xa0  # non-breaking hard space
    SP = 0x20  # space


CP437_TRANS_TABLE = MappingProxyType(
    {
        0: None, 1: 0x263a, 2: 0x263b, 3: 0x2665, 4: 0x2666, 5: 0x2663, 6: 0x2660, 7: 0x2022,
        8: 0x25d8, 9: 0x25cb, 10: 0x25d9, 11: 0x2642, 12: 0x2640, 13: 0x266a, 14: 0x266b,
        15: 0x263c, 16: 0x25ba, 17: 0x25c4, 18: 0x2195, 19: 0x203c, 20: 0xb6, 21: 0xa7,
        22: 0x25ac, 23: 0x21a8, 24: 0x2191, 25: 0x2193, 0x1a: 0x2192, 0x1b: 0x2190,
        0x1c: 0x221f, 0x1d: 0x2194, 0x1e: 0x25b2, 0x1f: 0x25bc, 0x7f: 0x2302, 0xa0: None
    })


@overload
def translate_cp437[_T: (int, str)](
    __x: str,
    *,
    ignore: Union[_T, Iterable[_T]] = ...
) -> str:
    ...


@overload
def translate_cp437[_T: (int, str)](
    __iter: Iterable[str],
    *,
    ignore: Union[_T, Iterable[_T]] = ...
) -> Iterator[str]:
    ...


def translate_cp437(
    __x: Union[str, Iterable[str]],
    *,
    ignore: Union[int, Iterable[int]] = None
) -> Union[str, Iterator[str]]:
    keys_view = set(CP437_TRANS_TABLE.keys())
    if ignore is not None:
        if isinstance(ignore, Iterable):
            keys_view.difference_update(ignore)
        else:
            keys_view.discard(ignore)
    trans_table = {k: v for (k, v) in
                   CP437_TRANS_TABLE.items()
                   if k in keys_view}
    if not isinstance(__x, str):
        return iter(map(lambda s: str.translate(s, trans_table), __x))
    return __x.translate(trans_table)


def cp437_printable():
    """Return a string containing all graphical characters in code page 437"""
    return translate_cp437(bytes(range(256)).decode(encoding='cp437'))


def ascii_printable():
    return bytes(range(32, 127)).decode(encoding='ascii')


def _ctoi(c: Union[str, int]):
    if isinstance(c, str):
        return ord(c)
    else:
        return c


def isprint(c: Union[str, int]):
    return 32 <= _ctoi(c) <= 126


def isctrl(c: Union[str, int]):
    return 0 <= _ctoi(c) < 32


def ctrl(c: Union[str, int]):
    if isinstance(c, str):
        return chr(_ctoi(c) & 0x1f)
    else:
        return _ctoi(c) & 0x1f


def alt(c: Union[str, int]):
    if isinstance(c, str):
        return chr(_ctoi(c) | 0x80)
    else:
        return _ctoi(c) | 0x80


def unctrl(c: Union[str, int]):
    bits = _ctoi(c)
    if bits == 0x7f:
        rep = '^?'
    elif isprint(bits & 0x7f):
        rep = chr(bits & 0x7f)
    else:
        rep = '^' + chr(((bits & 0x7f) | 0x20) + 0x20)
    if bits & 0x80:
        return '!' + rep
    return rep
