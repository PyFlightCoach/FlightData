from pytest import fixture
from flightdata import Flight, Origin
from flightdata import State


@fixture(scope="session")
def flight():
    return Flight.from_json('test/data/p23_flight.json')


@fixture(scope="session")
def origin():
    return Origin.from_f3a_zone('test/data/p23_box.f3a')


@fixture(scope="session")
def state(flight, origin) -> State:
    return State.from_flight(flight, origin)