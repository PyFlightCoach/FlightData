
from flightdata import State
from ..conftest import flight, origin, state
from pytest import approx, fixture
import geometry as g
import numpy as np




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
