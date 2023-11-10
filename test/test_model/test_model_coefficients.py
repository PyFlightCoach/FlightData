from pytest import approx, fixture
from flightdata import Environment, WindModelBuilder, Flow, cold_draft, Coefficients, State
from pytest import approx
import numpy as np
from ..conftest import flight, state


@fixture
def environments(flight, state):
    wmodel = WindModelBuilder.uniform(1.0, 20.0)([np.pi, 1.0])
    return Environment.build(flight, state, wmodel)

@fixture
def flows(state, environments):
    return Flow.build(state, environments)

def test_build(state, flows):
    coeffs = Coefficients.build(state, flows.q, cold_draft)
    assert isinstance(coeffs, Coefficients)
    pass


