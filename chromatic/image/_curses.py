__all__ = [
    'ctrl',
    'isprint',
    'ControlCharacter',
    'alt',
    'ascii_printable',
    'cp437_printable',
    'isctrl',
    'translate_cp437',
    'unctrl',
]

from enum import IntEnum
from types import MappingProxyType
from typing import Iterable, Iterator, overload


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
    LF = 0x0A  # ^J
    NL = 0x0A  # ^J
    VT = 0x0B  # ^K
    FF = 0x0C  # ^L
    CR = 0x0D  # ^M
    SO = 0x0E  # ^N
    SI = 0x0F  # ^O
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
    SUB = 0x1A  # ^Z
    ESC = 0x1B  # ^[
    FS = 0x1C  # ^\
    GS = 0x1D  # ^]
    RS = 0x1E  # ^^
    US = 0x1F  # ^_
    DEL = 0x7F  # delete
    NBSP = 0xA0  # non-breaking hard space
    SP = 0x20  # space


CP437_TRANS_TABLE = MappingProxyType(
    {
        0x0: None, 0x1: 0x263A, 0x2: 0x263B, 0x3: 0x2665, 0x4: 0x2666,
        0x5: 0x2663, 0x6: 0x2660, 0x7: 0x2022, 0x8: 0x25D8, 0x9: 0x25CB,
        0xA: 0x25D9, 0xB: 0x2642, 0xC: 0x2640, 0xD: 0x266A, 0xE: 0x266B,
        0xF: 0x263C, 0x10: 0x25BA, 0x11: 0x25C4, 0x12: 0x2195, 0x13: 0x203C,
        0x14: 0xB6, 0x15: 0xA7, 0x16: 0x25AC, 0x17: 0x21A8, 0x18: 0x2191, 0x19: 0x2193,
        0x1A: 0x2192, 0x1B: 0x2190, 0x1C: 0x221F, 0x1D: 0x2194, 0x1E: 0x25B2, 0x1F: 0x25BC,
        0x7F: 0x2302, 0xA0: None  # fmt: skip
    }
)


@overload
def translate_cp437[_T: (int, str)](
    __x: str, *, ignore: _T | Iterable[_T] = ...
) -> str: ...


@overload
def translate_cp437[_T: (int, str)](
    __iter: Iterable[str], *, ignore: _T | Iterable[_T] = ...
) -> Iterator[str]: ...


def translate_cp437(
    __x: str | Iterable[str], *, ignore: int | Iterable[int] = None
) -> str | Iterator[str]:
    keys_view = set(CP437_TRANS_TABLE.keys())
    if ignore is not None:
        if isinstance(ignore, Iterable):
            keys_view.difference_update(ignore)
        else:
            keys_view.discard(ignore)
    trans_table = {k: v for (k, v) in CP437_TRANS_TABLE.items() if k in keys_view}
    if not isinstance(__x, str):
        return map(lambda s: str.translate(s, trans_table), __x)
    return __x.translate(trans_table)


def cp437_printable():
    """Return a string containing all graphical characters in code page 437"""
    return translate_cp437(bytes(range(256)).decode(encoding='cp437'))


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
        return chr(_ctoi(c) & 0x1F)
    else:
        return _ctoi(c) & 0x1F


def alt(c: str | int):
    if isinstance(c, str):
        return chr(_ctoi(c) | 0x80)
    else:
        return _ctoi(c) | 0x80


def unctrl(c: str | int):
    bits = _ctoi(c)
    if bits == 0x7F:
        rep = '^?'
    elif isprint(bits & 0x7F):
        rep = chr(bits & 0x7F)
    else:
        rep = '^' + chr(((bits & 0x7F) | 0x20) + 0x20)
    if bits & 0x80:
        return '!' + rep
    return rep
