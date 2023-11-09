from pytest import approx, fixture
from flightdata import Environment, WindModelBuilder, Flow, cold_draft, Coefficients, State
from pytest import approx
import numpy as np



@fixture
def environments(flight, st):
    wmodel = WindModelBuilder.uniform(1.0, 20.0)([np.pi, 1.0])
    return Environment.build(flight, st, wmodel)

@fixture
def flows(st, environments):
    return Flow.build(st, environments)

def test_build(st, flows):
    flows = Coefficients.build(st, flows, cold_draft)

    pass


