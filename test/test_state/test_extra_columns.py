"""Extra columns: per-sample values derived from a whole State that its slices carry.

A value computed over the whole state (for example a rate smoothed over a manoeuvre) must
read the same at a given time from any slice, so nothing downstream recomputes it from a
slice with different edge effects. Array-backed and spline-backed states take different
slicing paths, so both are checked.
"""

import geometry as g
import numpy as np
from pytest import approx, fixture, raises

from flightdata import State
from flightdata.base.table.table import TableError

SPEED = 25.0
RADIUS = 50.0


def _loop() -> State:
    duration = 2 * np.pi * RADIUS / SPEED
    time = g.Time.from_t(np.linspace(0, duration, int(duration * 25)))
    initial = State.from_transform(
        g.Transformation.zero(), vel=g.PX(SPEED), rvel=g.PY(SPEED / RADIUS)
    )
    return initial.fill(time)


@fixture(params=["array", "spline"])
def st(request) -> State:
    loop = _loop()
    if request.param == "spline":
        loop = loop.create_splines()
    # a column equal to the time, so interpolated values are easy to check
    return loop.with_extra(tt=loop.t.copy(), vec=np.column_stack([loop.t, -loop.t]))


def _check(sub: State):
    assert set(sub.extra) == {"tt", "vec"}
    assert len(sub.extra["tt"]) == len(sub)
    assert sub.extra["tt"] == approx(sub.t)
    assert sub.extra["vec"].shape == (len(sub), 2)
    assert sub.extra["vec"][:, 1] == approx(-sub.t)


def test_time_slice_off_sample(st: State):
    _check(st[st.t[10] + 0.013 : st.t[40] - 0.007])


def test_time_slice_on_sample(st: State):
    _check(st[st.t[10] : st.t[40]])


def test_iloc(st: State):
    _check(st.iloc[5:30])
    _check(st.iloc[7])
    _check(st.iloc[np.array([3, 4, 9])])


def test_interpolate(st: State):
    _check(st.interpolate(np.linspace(st.t[2], st.t[50], 17)))


def test_labels_and_elements(st: State):
    labelled = st.label(element=np.where(st.t < st.t[len(st) // 2], "a", "b"))
    _check(labelled)
    _check(labelled.element.a)
    _check(labelled.element.b)
    _check(labelled.remove_labels())


def test_splice(st: State):
    """As DownGrade.select does: splice a state with fractional-index rows of itself."""
    _check(State.splice([st, st.iloc[np.array([3.5, 10.25, 20])]]))


def test_stack_and_concatenate(st: State):
    a, b = st.iloc[:20], st.iloc[20:40]
    _check(State.concatenate([a, b]))
    stacked = State.stack([a, b])
    assert len(stacked.extra["tt"]) == len(stacked)


def test_shift_keeps_rows(st: State):
    shifted = st.replace(time=st.time + 3.0)
    assert shifted.extra["tt"] == approx(st.extra["tt"])


def test_kinematic_replace_drops(st: State):
    """Extra columns are derived from the kinematics, so changing them drops the columns."""
    assert st.replace(pos=st.pos * 2).extra == {}
    if st.splines is None:  # State.scale does not handle spline-backed states
        assert st.scale(2).extra == {}
    assert st.move(g.Transformation(g.PX(10), g.Q0())).extra == {}


def test_copy_is_independent(st: State):
    c = st.copy()
    c.extra["tt"][0] = 99.0
    assert st.extra["tt"][0] != 99.0


def test_length_checked(st: State):
    with raises(TableError):
        st.with_extra(bad=np.zeros(len(st) + 1))


def test_not_serialised(st: State):
    assert State.from_dict(st.to_dict()).extra == {}


def test_plain_state_has_none():
    assert _loop().extra == {}
    assert _loop().iloc[3:9].extra == {}
