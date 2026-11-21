"""Table.__getitem__ with a time slice copies recorded rows instead of interpolating
them; the result must match the interpolate-everything reference bit for bit."""

import geometry as g
import numpy as np
import pytest

from flightdata import State
from flightdata.base.table.table import _init_misc_fields

N = 40


def _build(full: bool = True, seed: int = 1) -> State:
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.02, 0.2, N))
    q = rng.normal(size=(N, 4))
    kw = {}
    if full:
        kw = dict(
            vel=g.Point(rng.normal(size=(N, 3))),
            rvel=g.Point(rng.normal(size=(N, 3))),
            acc=g.Point(rng.normal(size=(N, 3))),
        )
    return State(
        g.Time.from_t(t),
        g.Point(rng.normal(size=(N, 3)) * 100),
        g.Quaternion(q / np.linalg.norm(q, axis=1, keepdims=True)),
        **kw,
    )


def _labelled(st: State) -> State:
    names = np.array(["e0"] * 10 + ["e1"] * 12 + ["e2"] * (N - 22))
    return st.label(element=names)


def reference(st: State, sli: slice) -> State:
    """The pre-change implementation of slice indexing."""
    start = sli.start if sli.start is not None else st.t[0]
    istart = np.searchsorted(st.t, start, "left")
    stop = sli.stop if sli.stop is not None else st.t[-1]
    istop = np.searchsorted(st.t, stop, "right") - 1
    times = []
    if start >= st.t[0] and start <= st.t[-1] and st.t[istart] != start:
        times.append(st._get_interpolator("time")(start))
    times.append(st.time[istart : istop + 1])
    if stop >= st.t[0] and stop <= st.t[-1] and st.t[istop] != stop:
        times.append(st._get_interpolator("time")(stop))
    t = g.Time.concatenate(times).t
    t = np.where(t < 0, len(st) + t, t)
    return st.interpolate(t).label(st.labels.slice(np.min(t), np.max(t)))


def assert_identical(new: State, ref: State):
    assert len(new) == len(ref)
    assert new.labels == ref.labels
    for name in ["time", "pos", "att", "vel", "rvel", "acc"]:
        assert np.array_equal(getattr(new, name).data, getattr(ref, name).data), name
    assert np.array_equal(new.dt, ref.dt)


@pytest.fixture(scope="module")
def st():
    return _build()


# (start, stop) as functions of the recorded times; None leaves the end open
CASES = {
    "both_recorded": (lambda t: t[5], lambda t: t[20]),
    "frac_start": (lambda t: (t[7] + t[8]) / 2, lambda t: t[20]),
    "frac_stop": (lambda t: t[5], lambda t: (t[20] + t[21]) / 2),
    "both_frac": (lambda t: t[3] + 0.01, lambda t: t[30] - 0.01),
    "open_ends": (None, None),
    "open_start_frac_stop": (None, lambda t: (t[7] + t[8]) / 2),
    "frac_start_open_stop": (lambda t: (t[7] + t[8]) / 2, None),
    "single_interior_row": (lambda t: t[5] - 0.001, lambda t: t[5] + 0.001),
    "single_recorded_row": (lambda t: t[5], lambda t: t[5]),
    "no_interior_row": (lambda t: t[10] + 0.1 * (t[11] - t[10]), lambda t: t[10] + 0.6 * (t[11] - t[10])),
    "whole_table": (lambda t: t[0], lambda t: t[-1]),
}


@pytest.mark.parametrize("case", CASES)
def test_time_slice_matches_reference(st: State, case):
    f0, f1 = CASES[case]
    sli = slice(f0(st.t) if f0 else None, f1(st.t) if f1 else None)
    assert_identical(st[sli], reference(st, sli))


def test_row_slice_path_is_used(st: State, monkeypatch):
    calls = []
    orig = State._row_slice
    monkeypatch.setattr(State, "_row_slice", lambda self, *a: calls.append(1) or orig(self, *a))
    st[st.t[5] + 0.001 : st.t[20]]
    assert calls


def test_labelled_slices(st: State):
    lab = _labelled(st)
    for key in ["e1", ["e0", "e1"], ["e1", "e2"]]:
        new = lab.element[key]
        assert len(new.labels.element.labels) >= 1
        assert_identical(new, reference(lab, slice(new.t[0], new.t[-1])))


def test_labelled_fractional_slice(st: State):
    lab = _labelled(st)
    sli = slice(lab.t[8] + 0.013, lab.t[25] + 0.021)
    assert_identical(lab[sli], reference(lab, sli))


def test_lazy_constructs_derived_equal():
    st = _build(full=False)
    assert st._vel is None and st._acc is None
    for sli in [slice(st.t[4], st.t[30]), slice(st.t[4] + 0.01, st.t[30] + 0.01)]:
        new, ref = st[sli], reference(st, sli)
        assert new._vel is None and new._rvel is None and new._acc is None
        assert_identical(new, ref)


def test_slice_does_not_share_memory(st: State):
    sli = st[st.t[5] : st.t[20]]
    before = {n: getattr(st, n).data.copy() for n in ["time", "pos", "att"]}
    for n in before:
        arr = getattr(sli, n).data
        assert not np.shares_memory(arr, getattr(st, n).data)
        arr[:] = 0
    for n, d in before.items():
        assert np.array_equal(getattr(st, n).data, d)


@pytest.mark.parametrize(
    "sli",
    [
        lambda t: slice(t[-1] + 1, t[-1] + 2),
        lambda t: slice(t[0] - 2, t[0] - 1),
        lambda t: slice(t[5], t[-1] + 5),
        lambda t: slice(t[0] - 1, t[5]),
        lambda t: slice(t[8], t[3]),
    ],
    ids=["after", "before", "overrun_stop", "underrun_start", "reversed"],
)
def test_out_of_range_matches_old_behaviour(st: State, sli):
    sli = sli(st.t)
    try:
        ref = reference(st, sli)
    except Exception as e:
        with pytest.raises(type(e)):
            st[sli]
    else:
        assert_identical(st[sli], ref)


def _spline_state() -> State:
    t = np.linspace(0, 10, 200)
    th = t * 0.5
    att = g.Quaternion.from_euler(g.Point(np.c_[0.2 * np.sin(th), 0.1 * th, 0.3 * th]))
    pos = g.Point(np.c_[10 * t, 5 * np.sin(th), 2 * np.cos(th)])
    flat = State(g.Time.from_t(t), pos, att)
    return flat.create_splines()


def test_spline_state_uses_fallback_and_matches_reference():
    st = _spline_state()
    assert st.splines is not None
    assert not st._row_slice_supported()
    for sli in [slice(1.0, 5.0), slice(1.013, 5.021), slice(None, 4.4)]:
        assert_identical(st[sli], reference(st, sli))


def test_replace_keeps_splines_and_labels():
    st = _labelled(_build())
    new = st.replace()
    assert new.splines is None and new.labels == st.labels
    assert_identical(new, st)
    sp = _spline_state()
    sp = sp.label(element=np.array(["a"] * 100 + ["b"] * 100))
    new = sp.replace()
    assert new.splines is sp.splines and new.labels == sp.labels
    assert np.array_equal(new.pos.data, sp.pos.data)
    assert np.array_equal(new.acc.data, sp.acc.data)


def test_init_misc_fields():
    assert _init_misc_fields(State) == ("splines",)
    assert _init_misc_fields(State) is _init_misc_fields(State)
