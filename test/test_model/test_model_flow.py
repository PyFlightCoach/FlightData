from pytest import approx, fixture
from flightdata import *
from pytest import approx
import numpy as np
from geometry import *
from ..conftest import state, flight, origin

@fixture
def environment(flight, origin):

    return Environment.from_flight(flight, origin)

def test_build(state, environment):
    flows = Flow.build(state, environment)
    assert np.mean(flows.alpha) == approx(0.0, abs=1)

@fixture
def sl_wind_axis():
    return State.from_transform(
        Transformation(P0(), Euler(0, np.pi, 0)),
        vel=PX(30)
    ).extrapolate(10)

def test_alpha_only_0_wind(sl_wind_axis):
    body_axis = sl_wind_axis.superimpose_angles(Point(0, np.radians(20), 0))  
    env = Environment.from_constructs(sl_wind_axis.time)
    flw = Flow.build(body_axis, env)
    assert flw.alpha == approx(np.full(len(flw), np.radians(20)))


def test_alpha_beta_0_wind(sl_wind_axis):
    stability_axis = sl_wind_axis.superimpose_angles(Point(0, 0, -np.radians(10)))
    body_axis = stability_axis.superimpose_angles(Point(0, np.radians(20), 0))
    env = Environment.from_constructs(sl_wind_axis.time)
    flw = Flow.build(body_axis, env)
    assert np.degrees(flw.alpha) == approx(np.full(len(flw), 20))
    assert np.degrees(flw.beta) == approx(np.full(len(flw), 10))


def test_zero_wind_assumption(state):
    env = Environment.from_constructs(state.time)
    flow = Flow.build(state, env)
    ab = flow.data.loc[:, ["alpha", "beta"]]
    
    