from flightdata import State
from ..conftest import flight, origin, state
from pytest import approx, fixture

import numpy as np




def test_direction(state):
    direcs = state.direction()
    assert isinstance(direcs, np.ndarray)
