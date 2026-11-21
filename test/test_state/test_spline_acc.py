"""Spline states must follow the State.acc convention (kinematic + 1 g upwards)
whether or not the log supplied accelerometer data."""

import geometry as g
import numpy as np
from pytest import approx, fixture, mark

from flightdata import State
from flightdata.ardupilot import Field, StateData
from flightdata.state.spline_interpolation import SplineState

SPEED = 25.0
RADIUS = 50.0


@fixture(scope="module")
def loop() -> State:
    """A full vertical loop flown at constant speed, sampled at 25 Hz."""
    duration = 2 * np.pi * RADIUS / SPEED
    time = g.Time.from_t(np.linspace(0, duration, int(duration * 25)))
    initial = State.from_transform(
        g.Transformation.zero(), vel=g.PX(SPEED), rvel=g.PY(SPEED / RADIUS)
    )
    return initial.fill(time)


def _state_data(loop: State, with_vel: bool, with_acc: bool) -> StateData:
    t = loop.time.t
    return StateData(
        Field(t, loop.att),
        None,
        Field(t, loop.pos),
        Field(t, loop.vel) if with_vel else None,
        Field(t, loop.acc) if with_acc else None,
    )


def _middle(values: np.ndarray) -> np.ndarray:
    n = len(values)
    return values[n // 10 : -n // 10]


@mark.parametrize("with_vel", [True, False], ids=["independent", "smoothing"])
@mark.parametrize("from_dict", [False, True], ids=["built", "from_dict"])
def test_derived_acc_includes_gravity(loop: State, with_vel: bool, from_dict: bool):
    splines = SplineState.build(_state_data(loop, with_vel, with_acc=False))
    if from_dict:
        splines = SplineState.from_dict(splines.to_dict())
    st = State(loop.time, splines=splines)

    kinematic = np.gradient(loop.wvel.data, loop.time.t, axis=0)
    np.testing.assert_allclose(
        _middle(st.wacc.data), _middle(kinematic + [0, 0, 9.81]), atol=0.5
    )

    centripetal = _middle(abs(st.acc_zero_g))
    np.testing.assert_allclose(centripetal, SPEED**2 / RADIUS, rtol=0.05)

    radius = _middle(abs(st.arc_centre()))
    np.testing.assert_allclose(radius, RADIUS, rtol=0.05)


def test_derived_acc_matches_measured_acc(loop: State):
    derived = State(
        loop.time, splines=SplineState.build(_state_data(loop, True, with_acc=False))
    )
    measured = State(
        loop.time, splines=SplineState.build(_state_data(loop, True, with_acc=True))
    )
    np.testing.assert_allclose(
        _middle(derived.acc.data), _middle(measured.acc.data), atol=0.3
    )


def test_spline_measure_radius(loop: State):
    st = State(
        loop.time, splines=SplineState.build(_state_data(loop, True, with_acc=False))
    )
    axis = loop[0].att.transform_point(g.PY(1))
    middle = st[float(st.t[len(st) // 10]) : float(st.t[-len(st) // 10])]
    assert middle.measure_radius(loop[0].att, axis) == approx(RADIUS, rel=0.03)
