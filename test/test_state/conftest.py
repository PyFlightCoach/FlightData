from pytest import fixture
from ..conftest import flight, box
from flightanalysis import State


@fixture(scope="session")
def state(flight, box) -> State:
    return State.from_flight(flight, box)