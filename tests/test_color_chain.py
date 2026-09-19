import numpy as np
import pytest

from chromatic import ColorStr, color_chain


@pytest.fixture
def segments():
    return [
        ColorStr("red", fg=(255, 0, 0), ansi_type="24b"),
        ColorStr("blue", fg=(0, 0, 255), ansi_type="24b"),
    ]


@pytest.fixture
def chain(segments):
    return color_chain(segments)


def _text(cc):
    return "".join(s for _, s in cc)


def test_empty_chain_is_empty():
    cc = color_chain()
    assert len(cc) == 0
    assert not cc


def test_len_and_bool(chain):
    assert len(chain) == 2
    assert chain


def test_getitem_index_is_sgr_str_pair(chain):
    sgr, s = chain[0]
    assert s == "red"
    assert sgr.fg == (255, 0, 0)


def test_getitem_slice_is_list(chain):
    assert isinstance(chain[:1], list)
    assert len(chain[:1]) == 1


def test_iterates_segment_pairs(chain):
    assert [s for _, s in chain] == ["red", "blue"]


def test_equality(chain, segments):
    assert chain == color_chain(segments)
    assert chain != color_chain(segments[:1])
    assert chain != "not a chain"


def test_str_renders_text_with_escapes(chain):
    assert _text(chain) == "redblue"
    assert "\x1b" in str(chain)


def test_call_wraps_with_reset(chain):
    assert chain("!").endswith("!\x1b[0m")
    assert chain().endswith("\x1b[0m")


def test_str_roundtrip(chain):
    assert color_chain(str(chain)) == chain


def test_add_str_appends(chain):
    out = chain + "X"
    assert isinstance(out, color_chain)
    assert str(out).endswith("X")


def test_radd_str_prepends(chain):
    out = "X" + chain
    assert isinstance(out, color_chain)
    assert str(out).startswith("X")


def test_add_chain_concatenates(chain, segments):
    out = chain + color_chain(segments[:1])
    assert isinstance(out, color_chain)
    assert len(out) == len(chain) + 1
    assert _text(out) == "redbluered"


def test_insert_accepts_str():
    cc = color_chain()
    cc.insert(0, "hi")
    assert len(cc) == 1
    assert cc[0][1] == "hi"


def test_append_and_delete(chain):
    n = len(chain)
    chain.append(ColorStr("!", fg=(0, 255, 0), ansi_type="24b"))
    assert len(chain) == n + 1
    del chain[0]
    assert len(chain) == n
    assert _text(chain) == "blue!"


def test_setitem_rejects_non_pair(chain):
    with pytest.raises(TypeError):
        chain[0] = "not a pair"


@pytest.mark.parametrize("ansi_type,alias", [("4b", "4b"), ("8b", "8b"), (None, "24b")])
def test_ansi_type_coerces_segment_colors(ansi_type, alias):
    red = ColorStr("red", fg=(255, 0, 0), ansi_type="24b")
    cc = color_chain([red], ansi_type=ansi_type)
    assert cc[0][0].ansi_type().alias == alias


def test_splitlines_splits_and_carries_color():
    src = ColorStr("a\nb\nc", fg=(255, 0, 0), ansi_type="24b")
    lines = color_chain([src]).splitlines()
    assert len(lines) == 3
    assert all(isinstance(x, color_chain) for x in lines)
    assert [_text(x) for x in lines] == ["a", "b", "c"]
    assert all(x[0][0].fg == (255, 0, 0) for x in lines)


def test_shrink_compacts_degenerate_parse():
    cc = color_chain("\x1b[1m\x1b[3m\x1b[31mhi\x1b[0m\x1b[0m")
    raw = [(bytes(sgr), s) for sgr, s in cc]
    cc.shrink()
    compacted = [(bytes(sgr), s) for sgr, s in cc]
    assert compacted == [(b"\x1b[1;3;31m", "hi"), (b"\x1b[0m", "")]
    assert compacted != raw


def test_array_has_structured_dtype(chain):
    assert np.asarray(chain).dtype.names == ("char", "sgr", "rgb")


def test_array_rejects_copy_false(chain):
    with pytest.raises(ValueError):
        chain.__array__(copy=False)


def test_array_rejects_foreign_dtype(chain):
    with pytest.raises(TypeError):
        chain.__array__(dtype=np.dtype("U1"))


def test_fromarray_roundtrip(chain):
    assert color_chain.fromarray(np.asarray(chain)) == chain


def test_term_array_default_shape(chain):
    out = chain.term_array()
    assert out.shape == (1, len(_text(chain)))


@pytest.mark.parametrize("shape", [(3, 5), (2, 10)])
def test_term_array_explicit_shape(chain, shape):
    assert chain.term_array(shape).shape == shape


def test_term_array_rejects_non_2d(chain):
    with pytest.raises(ValueError):
        chain.term_array((3,))


def test_term_array_fillchar_pads(chain):
    row = chain.term_array((1, 10), fillchar=".")["char"][0].tolist()
    assert "".join(row) == "redblue..."


def test_term_array_empty_is_zero_shape():
    assert color_chain().term_array().shape == (0, 0)
