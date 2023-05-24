from pytest import fixture

from flightdata import Flight

@fixture
def fl():
    return Flight.from_log("test/00000150.BIN")

def test_data(fl):
    pass
