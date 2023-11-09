from pytest import approx, fixture
from flightanalysis import *
from pytest import approx
import numpy as np
from geometry import *


@fixture
def environments(flight, st):
    wmodel = WindModelBuilder.uniform(1.0, 20.0)([np.pi, 1.0])
    return Environment.build(flight, st, wmodel)

def test_build(st, environments):
    flows = Flow.build(st, environments)
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


def test_zero_wind_assumption(st):
    env = Environment.from_constructs(st.time)
    flow = Flow.build(st, env)
    ab = flow.data.loc[:, ["alpha", "beta"]]
    
    