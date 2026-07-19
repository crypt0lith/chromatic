__all__ = [
    'ControlCharacter',
    'alt',
    'ascii_printable',
    'backtrans_cp437',
    'cp437_printable',
    'ctrl',
    'isctrl',
    'isprint',
    'translate_cp437',
    'unctrl',
]

from enum import IntEnum
from types import MappingProxyType
from typing import Iterable


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
        0x01: 0x263A,   0x02: 0x263B,   0x03: 0x2665,   0x04: 0x2666,
        0x05: 0x2663,   0x06: 0x2660,   0x07: 0x2022,   0x08: 0x25D8,
        0x09: 0x25CB,   0x0A: 0x25D9,   0x0B: 0x2642,   0x0C: 0x2640,
        0x0D: 0x266A,   0x0E: 0x266B,   0x0F: 0x263C,   0x10: 0x25BA,
        0x11: 0x25C4,   0x12: 0x2195,   0x13: 0x203C,   0x14: 0xB6,
        0x15: 0xA7,     0x16: 0x25AC,   0x17: 0x21A8,   0x18: 0x2191,
        0x19: 0x2193,   0x1A: 0x2192,   0x1B: 0x2190,   0x1C: 0x221F,
        0x1D: 0x2194,   0x1E: 0x25B2,   0x1F: 0x25BC,   0x7F: 0x2302,
    }   # fmt: skip
)


def translate_cp437(x: str, /, ignore: Iterable[int] = ()) -> str:
    """Translate control chars (0x1-0x1F, 0x7F) into cp437 graphical chars"""
    keys = CP437_TRANS_TABLE.keys() - ignore
    return x.translate({k: CP437_TRANS_TABLE[k] for k in keys})


def backtrans_cp437(x: str, /, keys: Iterable[int] | None = None) -> str:
    """Translate cp437 graphical chars back into control chars"""
    return x.translate(
        {v: k for k, v in CP437_TRANS_TABLE.items()}
        if keys is None
        else {CP437_TRANS_TABLE[k]: k for k in keys}
    )


def cp437_printable():
    """Return a string containing all graphical characters in code page 437"""
    return translate_cp437(bytes([*range(1, 0x20), *range(0x21, 0xFF)]).decode('cp437'))


def ascii_printable():
    return bytes(range(32, 127)).decode('ascii')


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
