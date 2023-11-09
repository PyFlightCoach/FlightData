from pytest import fixture

from flightanalysis.environment import Environment
from flightanalysis.environment.wind import WindModelBuilder
import numpy as np
import pandas as pd
from geometry import Point



@fixture
def wmodel():
    return WindModelBuilder.uniform(1.0, 20.0)([np.pi, 5.0])


def test_build(flight, st, wmodel):
    env = Environment.build(flight, st, wmodel)

    assert isinstance(env.data, pd.DataFrame)

    assert isinstance(env.wind, Point)
    assert isinstance(env[20], Environment)
    