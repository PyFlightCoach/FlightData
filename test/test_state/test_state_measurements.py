from flightanalysis.state.state import State

from flightanalysis import State

from ..conftest import flight, box
from .conftest import state
from pytest import approx, fixture

import numpy as np




def test_direction(state):
    direcs = state.direction()
    assert isinstance(direcs, np.ndarray)
