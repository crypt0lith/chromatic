__all__ = ['ControlCharacter', 'cp437_translate', 'cp437_printable', 'ascii_printable', 'isctrl', 'isprint', 'alt',
           'ctrl', 'unctrl']

from enum import IntEnum
from typing import Callable, Iterable, Iterator, overload


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


CP437_TRANS_TABLE = lambda: str.maketrans(
    dict(
        zip(
            sorted(c for c in ControlCharacter if c is not ControlCharacter.SP),
            [None, 0x263a, 0x263b, 0x2665, 0x2666, 0x2663, 0x2660, 0x2022, 0x25d8, 0x25cb, 0x25d9, 0x2642, 0x2640,
             0x266a, 0x266b, 0x263c, 0x25ba, 0x25c4, 0x2195, 0x203c, 0xb6, 0xa7, 0x25ac, 0x21a8, 0x2191, 0x2193, 0x2192,
             0x2190, 0x221f, 0x2194, 0x25b2, 0x25bc, 0x2302, None])))


@overload
def cp437_translate[_T: int | str](__x: str, *, ignore: _T | Iterable[_T] = ...) -> str:
    ...


@overload
def cp437_translate[_T: int | str](__iter: Iterable[str], *, ignore: _T | Iterable[_T] = ...) -> Iterator[str]:
    ...


def cp437_translate(__x: str | Iterable[str],
                    *,
                    ignore: int | str | Iterable[int | str] = None) -> str | Iterator[str]:
    translation_table = CP437_TRANS_TABLE()
    if ignore:
        if issubclass(vt := type(ignore), str | int):
            translation_table.pop(ignore if issubclass(vt, int) else ord(ignore))
        else:
            for char in ignore:
                if isinstance(char, str):
                    key = ord(char)
                else:
                    key = char
                translation_table.pop(key)
    if not isinstance(__x, str):
        f: Callable[[str], str] = lambda s: str.translate(s, translation_table)
        return iter(map(f, __x))
    return __x.translate(translation_table)


def cp437_printable():
    """Return a string containing all graphical characters in code page 437"""
    return cp437_translate(bytes(range(256)).decode(encoding='cp437'))


def ascii_printable():
    return bytes(range(32, 127)).decode(encoding='ascii')


def _ctoi(c: str | int):
    if isinstance(c, str):
        return ord(c)
    else:
        return c


def isprint(c: str | int):
    return 32 <= _ctoi(c) <= 126


def isctrl(c: str | int):
    return 0 <= _ctoi(c) < 32


def ctrl(c: str | int):
    if isinstance(c, str):
        return chr(_ctoi(c) & 0x1f)
    else:
        return _ctoi(c) & 0x1f


def alt(c: str | int):
    if isinstance(c, str):
        return chr(_ctoi(c) | 0x80)
    else:
        return _ctoi(c) | 0x80


def unctrl(c: str | int):
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
