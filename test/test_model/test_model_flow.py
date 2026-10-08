import geometry as g
import numpy as np
from pytest import approx, fixture, mark

from flightdata import Environment, Flight, Flow, Origin, State

from ..conftest import flight, origin, state


@fixture
def environment(flight: Flight, origin: Origin):
    return Environment.from_flight(flight, origin)

def test_from_body(state: State, environment: Environment):
    flows = Flow.from_state(state, environment)
    assert np.mean(flows.alpha) == approx(0.0, abs=1)

@fixture
def sl_wind_axis():
    return State.from_transform(
        g.Transformation(g.P0(), g.Euler(0, np.pi, 0)),
        vel=g.PX(30)
    ).extrapolate(10)

@mark.skip
def test_alpha_only_0_wind(sl_wind_axis: State):
    body_axis = sl_wind_axis.superimpose_angles(g.Point(0, np.radians(20), 0))  
    env = Environment(sl_wind_axis.time)
    flw = Flow.from_state(body_axis, env)
    assert flw.alpha == approx(np.full(len(flw), np.radians(20)))

@mark.skip
def test_alpha_beta_0_wind(sl_wind_axis: State):
    stability_axis = sl_wind_axis.superimpose_angles(g.Point(0, 0, -np.radians(10)))
    body_axis = stability_axis.superimpose_angles(g.Point(0, np.radians(20), 0))
    env = Environment(sl_wind_axis.time)
    flw = Flow.from_state(body_axis, env)
    assert np.degrees(flw.alpha) == approx(np.full(len(flw), 20))
    assert np.degrees(flw.beta) == approx(np.full(len(flw), 10))


def test_zero_wind_assumption(state: State):
    env = Environment(state.time)
    flow = Flow.from_state(state, env)
    ab = flow.df.loc[:, ["alpha", "beta"]]
    
    