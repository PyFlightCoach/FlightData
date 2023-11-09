from pytest import fixture
from flightdata import Flight
from flightanalysis import Box


@fixture(scope="session")
def flight():
    return Flight.from_json('test/data/p23_flight.json')


@fixture(scope="session")
def box():
    return Box.from_f3a_zone('test/data/p23_box.f3a')


