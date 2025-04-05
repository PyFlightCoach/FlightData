import numpy as np
from flightdata import State, Environment, Flow

from pytest import mark, fixture
from geometry import Transformation, Point, P0, Euler, PY, PX
from ..conftest import state


def test_to_track(state: State):
    
    jst = state.to_track()

    assert isinstance(jst, State)

    env = Environment.from_constructs(state.time)
    flw_body = Flow.from_body(state, env)
    flw_judge = Flow.from_body(jst, env)

    
    #this wont reduce alpha and beta to zero as velocity comes from IMU,
    #but should be order of magnitude smaller
    assert np.nanmean(np.abs(flw_body.alpha / flw_judge.alpha)) > 1000
    assert np.nanmean(np.abs(flw_body.beta / flw_judge.beta)) > 1000


@fixture
def sl_wind_axis():
    return State.from_transform(
        Transformation(P0(), Euler(0, np.pi, 0)),
        vel=PX(30)
    ).extrapolate(10)
    

def test_to_track_sim(sl_wind_axis: State):
    body_axis = sl_wind_axis.superimpose_angles(Point(0, np.radians(20), 0))  
    
    judging = body_axis.to_track()
    assert judging.pos == sl_wind_axis.pos
    np.testing.assert_almost_equal(sl_wind_axis.att.data, judging.att.data, 0)

    np.testing.assert_almost_equal(
        judging.vel.data,
        Point(30,0,0).tile(len(judging)).data
    )

@mark.skip
def test_track_to_wind(sl_wind_axis: State):
        
    env = Environment.from_constructs(
        sl_wind_axis.time, 
        wind=PY(30*np.tan(np.radians(10)), len(sl_wind_axis))
    )

    wind = sl_wind_axis.track_to_wind(env)

    np.testing.assert_array_almost_equal(wind.att.data, sl_wind_axis.att.data)



