import numpy as np
import pandas as pd
from geometry import Point
from numpy.testing import assert_allclose
from pytest import fixture

from flightdata import Environment, Flight

from ..conftest import flight


def test_from_flight(flight: Flight):
    env = Environment.from_flight(flight)

    assert isinstance(env.df, pd.DataFrame)

    assert isinstance(env.wind, Point)
    assert isinstance(env[20], Environment)
    assert_allclose(env.rho, 1.2, rtol=0.2)

