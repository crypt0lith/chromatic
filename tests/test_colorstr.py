import pytest

from chromatic import ColorStr, SgrParameter


@pytest.fixture
def fg():
    return (1, 2, 3)


@pytest.fixture
def colored(fg):
    return ColorStr("hello", fg=fg)


@pytest.fixture(
    params=[
        pytest.param(ColorStr("abc", fg=(0, 255, 0)), id="from-args"),
        pytest.param(ColorStr("\x1b[38;5;46mabc\x1b[0m"), id="from-ansi"),
    ]
)
def green_abc(request):
    return request.param


def test_len_is_visible_length(green_abc):
    assert len(green_abc) == 3


def test_base_str_strips_ansi(green_abc):
    assert green_abc.base_str == "abc"
    assert "\x1b" not in green_abc.base_str


def test_str_renders_with_escapes(green_abc):
    assert "\x1b" in str(green_abc)


def test_never_equal_to_bare_str():
    plain = ColorStr("x")
    assert plain.fg is None
    assert plain != "x"
    assert "x" != plain


def test_equal_iff_same_text_and_color():
    a = ColorStr("x", fg=(1, 2, 3))
    assert a == ColorStr("x", fg=(1, 2, 3))
    assert a != ColorStr("x", fg=(9, 9, 9))
    assert a != ColorStr("y", fg=(1, 2, 3))


def test_hash_consistent_with_equality():
    a = ColorStr("x", fg=(1, 2, 3))
    b = ColorStr("x", fg=(1, 2, 3))
    c = ColorStr("x", fg=(9, 9, 9))
    assert hash(a) == hash(b)
    assert len({a, b, c}) == 2


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda s: s.upper(), id="upper"),
        pytest.param(lambda s: s.replace("l", "L"), id="replace"),
        pytest.param(lambda s: s[1:], id="slice"),
        pytest.param(lambda s: s + "!", id="concat"),
        pytest.param(lambda s: s * 2, id="repeat"),
        pytest.param(lambda s: s.center(7), id="center"),
        pytest.param(lambda s: s.split("e"), id="split"),
    ],
)
def test_transforms_match_str_and_preserve_fg(colored, fg, op):
    got = op(colored)
    want = op(colored.base_str)
    got = got if isinstance(got, list) else [got]
    want = want if isinstance(want, list) else [want]
    assert [p.base_str for p in got] == want
    assert all(isinstance(p, ColorStr) and p.fg.rgb == fg for p in got)


@pytest.mark.parametrize(
    "ansi_type,expected",
    [
        pytest.param("4b", b"\x1b[31m", id="4bit"),
        pytest.param("8b", b"\x1b[38;5;196m", id="8bit"),
        pytest.param("24b", b"\x1b[38;2;255;0;0m", id="24bit"),
    ],
)
def test_pure_red_encodes_per_type(ansi_type, expected):
    assert ColorStr("x", fg=(255, 0, 0), ansi_type=ansi_type).ansi == expected


def test_as_ansi_type_preserves_color():
    base = ColorStr("x", fg=(10, 20, 30), ansi_type="24b")
    narrowed = base.as_ansi_type("4b")
    assert narrowed.ansi != base.ansi
    assert narrowed.fg.rgb == (10, 20, 30)


def test_rendered_form_reparses_to_same_color():
    cs = ColorStr("hello", fg=(255, 0, 0))
    assert ColorStr(str(cs)).fg.rgb == (255, 0, 0)


def test_add_sgr_param_emits_code():
    cs = ColorStr(ansi_type="4b", reset=False).add_sgr_param(SgrParameter.RED_BRIGHT_FG)
    assert cs.ansi == b"\x1b[91m"


@pytest.mark.parametrize(
    "cs,expected",
    [
        pytest.param(ColorStr("x"), b"\x1b[1m", id="no-color"),
        pytest.param(ColorStr("x", fg=(1, 2, 3)), b"\x1b[38;5;16;1m", id="with-fg"),
    ],
)
def test_bold_injects_sgr_1(cs, expected):
    assert cs.bold().ansi == expected


@pytest.mark.parametrize(
    "fg_in,inverted", [((0, 0, 0), (255, 255, 255)), ((255, 0, 0), (0, 255, 255))]
)
def test_invert_complements_color(fg_in, inverted):
    assert (~ColorStr("hi", fg=fg_in)).fg.rgb == inverted


@pytest.mark.parametrize(
    "styled",
    [
        pytest.param(lambda cs: cs, id="plain"),
        pytest.param(lambda cs: cs.bold(), id="bold"),
    ],
)
def test_strip_style_keeps_color(styled):
    assert styled(ColorStr("x", fg=(1, 2, 3))).strip_style().fg.rgb == (1, 2, 3)


@pytest.mark.parametrize(
    "value,result",
    [
        pytest.param((256, 0, 0), (0, 0, 0), id="tuple-overflow-wraps"),
        pytest.param((-1, 0, 0), (255, 0, 0), id="tuple-negative-wraps"),
        pytest.param((1.5, 2, 3), (1, 2, 3), id="tuple-float-truncates"),
        pytest.param(0x1000000, (0, 0, 0), id="int-above-24bit-masks"),
        pytest.param(-1, (255, 255, 255), id="int-negative-masks"),
    ],
)
def test_fg_input_is_not_range_checked(value, result):
    assert ColorStr("x", fg=value).fg.rgb == result


@pytest.mark.parametrize("bad", [(1, 2), (1, 2, 3, 4)], ids=["too-short", "too-long"])
def test_wrong_length_tuple_raises(bad):
    with pytest.raises(TypeError):
        ColorStr("x", fg=bad)


# ---- benchmarks ----


@pytest.fixture
def subject():
    return ColorStr("chromatic", fg=(200, 100, 50), bg=(10, 20, 30), ansi_type="24b")


def test_bench_construct(benchmark):
    benchmark(
        lambda: ColorStr(
            "chromatic", fg=(200, 100, 50), bg=(10, 20, 30), ansi_type="24b"
        )
    )


@pytest.mark.parametrize("ansi_type", ["4b", "8b", "24b"])
def test_bench_as_ansi_type(benchmark, subject, ansi_type):
    benchmark(lambda: subject.as_ansi_type(ansi_type))


def test_bench_recolor(benchmark, subject):
    benchmark(lambda: subject.recolor(fg=(0, 255, 0), bg=(255, 0, 255)))


def test_bench_invert(benchmark, subject):
    benchmark(lambda: ~subject)


def test_bench_strip_style(benchmark, subject):
    styled = subject.bold().italicize()
    benchmark(styled.strip_style)


def test_bench_add_sgr_param(benchmark, subject):
    benchmark(lambda: subject.add_sgr_param(SgrParameter.SINGLE_UNDERLINE))


def test_bench_remove_sgr_param(benchmark, subject):
    styled = subject.bold()
    benchmark(lambda: styled.remove_sgr_param(SgrParameter.BOLD))


def test_bench_style_toggles(benchmark, subject):
    benchmark(lambda: subject.bold().italicize().underline().negative())


def test_bench_color_properties(benchmark, subject):
    benchmark(lambda: (subject.ansi, subject.rgb_dict, subject.ansi_partition()))


@pytest.mark.parametrize("ansi_type", ["4b", "8b", "24b"])
def test_bench_parse_sgr(benchmark, ansi_type):
    raw = str(ColorStr("chromatic", fg=(200, 100, 50), ansi_type=ansi_type).bold())
    benchmark(lambda: ColorStr(raw))


def test_bench_render_str(benchmark, subject):
    styled = subject.bold().underline()
    benchmark(lambda: str(styled))


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda s: s.upper(), id="upper"),
        pytest.param(lambda s: s.replace("a", "A"), id="replace"),
        pytest.param(lambda s: s.center(64, "."), id="center"),
        pytest.param(lambda s: s.split("a"), id="split"),
        pytest.param(lambda s: s + s, id="concat"),
    ],
)
def test_bench_str_transforms(benchmark, subject, op):
    benchmark(lambda: op(subject))


def test_bench_iter_chars(benchmark, subject):
    benchmark(lambda: [c.base_str for c in subject])


def test_bench_hash(benchmark, subject):
    benchmark(lambda: hash(subject))
