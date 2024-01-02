from pytest import approx, fixture
from flightdata import Environment, WindModelBuilder, Flow, Coefficients, State
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

