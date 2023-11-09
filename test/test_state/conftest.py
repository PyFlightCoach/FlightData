from pytest import fixture
from ..conftest import flight, origin
from flightdata import State


@fixture(scope="session")
def state(flight, origin) -> State:
    return State.from_flight(flight, origin)