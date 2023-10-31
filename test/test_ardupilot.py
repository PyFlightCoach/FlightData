from pytest import fixture

from flightdata import Flight

@fixture
def fl():
    return Flight.from_log("test/test_inputs/00000137.BIN")

def test_data(fl):
    assert isinstance(fl,Flight)
    pass
