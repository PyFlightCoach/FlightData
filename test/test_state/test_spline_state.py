from pytest import fixture

from flightdata import State
from flightdata.state.spline_interpolation import RSpline, SplineState, TSpline

from ..conftest import origin





@fixture(scope="session")
def state(origin) -> State:
    return State.read_bin("test/data/p23.BIN", origin)



def test_state_construction(state: State):
    assert isinstance(state, State)
    assert isinstance(state.splines, SplineState)
    assert state._vel is None
    assert state._acc is None
    assert state._rvel is None


