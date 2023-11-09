import pytest
from flightanalysis.controls import Controls, cold_draft_controls, Actuators
from flightdata import Flight
import numpy as np
import pandas as pd

@pytest.fixture
def flight():
    return Flight.from_json("tests/data/p23_flight.json")


def test_init(flight):
    cont = Controls.build(flight, cold_draft_controls)

    assert isinstance(cont.surfaces, Actuators)
    


