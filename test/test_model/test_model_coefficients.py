from pytest import approx, fixture
from flightanalysis.environment import Environment
from flightanalysis.environment.wind import WindModelBuilder
from flightanalysis.model.flow import Flow
from pytest import approx
import numpy as np
from flightanalysis.model.constants import cold_draft
from flightanalysis.model.coefficients import Coefficients
from flightanalysis.state import State


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


