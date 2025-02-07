from pytest import fixture

from flightdata import Environment, Flight
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from geometry import Point
from ..conftest import flight



def test_from_flight(flight: Flight):
    env = Environment.from_flight(flight)

    assert isinstance(env.data, pd.DataFrame)

    assert isinstance(env.wind, Point)
    assert isinstance(env[20], Environment)
    assert_allclose(env.rho, 1.2, rtol=0.2)

