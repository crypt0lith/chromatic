__all__ = ["ascii_printable", "backtrans_cp437", "cp437_printable", "translate_cp437"]

from types import MappingProxyType
from typing import Iterable

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
    return translate_cp437(bytes([*range(1, 0x20), *range(0x21, 0xFF)]).decode("cp437"))


def ascii_printable():
    return bytes(range(32, 127)).decode("ascii")
