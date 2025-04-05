from tuning import curvature_lowpass
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
