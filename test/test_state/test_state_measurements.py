
import geometry as g
import numpy as np
from pytest import approx

from flightdata import State

from ..conftest import flight, origin, state


def test_direction(state):
    direcs = state.direction()
    assert isinstance(direcs, np.ndarray)

def test_curvature():
    t = g.Time.from_t(np.linspace(0, 1, 11))
    st0 = State.from_transform(vel=g.PX(10), rvel=g.PY(1)).fill(t)
    curvature = abs(st0.curvature(g.PY(1)))
    assert curvature==approx(curvature[0])



def test_boundary_measure_rate_reduces_as_window_widens():
    st0 = State.from_transform(vel=g.PX(30)).extrapolate(0.2)
    st1 = st0[-1].extrapolate(0.2).superimpose_roll(2*np.pi)
    st2 = st1[-1].copy(rvel=g.P0()).extrapolate(0.2)
    st = State.stack([st0,st1,st2], "element", ["l1", "r", "l2"])

    rate, dmr1, dmr2 = st.boundary_measure("measure_rate", 0.199, 0.401, st0[0].att, g.PY(), 0.01)
    
    assert dmr1 > 0 
    assert dmr2 < 0
    pass



def test_boundary_measure_length(state: State):
    iatt = state[0].att.closest_principal()

    m, dm1, dm2 = state.boundary_measure(
        "measure_length", state.t[0], state.t[0] + 3.0, iatt, None
    ) 
    assert m == approx(
        abs(state.interpolate(state.t[0] + 3.0).x[0] - state.iloc[0].x[0]), rel=1e-2
    )


def test_measure_rate(state: State):

    t0 = state.element.e_1_0.t[0]

    iatt = state.element.e_1_0[0].att.closest_principal()

    t0 = state.element.e_1_0.t[0]
    t1 = state.element.e_1_0.t[-1]

    assert (
        (state.boundary_measure("measure_rate", state, iatt, t0, t1 + 0.001) ** 2)[0]
        < (state.boundary_measure("measure_rate", state, iatt, t0, t1) ** 2)[0]
    )
    assert (
        (state.boundary_measure("measure_rate", state, iatt, t0, t1 - 0.001) ** 2)[0]
        > (state.boundary_measure("measure_rate", state, iatt, t0, t1) ** 2)[0]
    )


def test_measure_duration(state: State):
    t0 = state.t[0]
    t1 = state.t[-1]

    assert state.boundary_measure("measure_duration", state, t0, t1, None, None)[0] == approx(t1 - t0, rel=1e-2)


def test_estimate_wind(state: State):
    wind = state.estimate_wind()
    assert isinstance(wind[0], g.Point)