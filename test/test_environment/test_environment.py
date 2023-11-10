from pytest import fixture

from flightdata import Environment, WindModelBuilder
import numpy as np
import pandas as pd
from geometry import Point
from ..conftest import state


@fixture
def wmodel():
    return WindModelBuilder.uniform(1.0, 20.0)([np.pi, 5.0])


def test_build(flight, state, wmodel):
    env = Environment.build(flight, state, wmodel)

    assert isinstance(env.data, pd.DataFrame)

    assert isinstance(env.wind, Point)
    assert isinstance(env[20], Environment)
    

