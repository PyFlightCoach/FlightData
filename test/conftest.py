from pytest import fixture
from flightdata import Flight, Origin, State, fcj

@fixture(scope='session')
def fcjson():
    return Flight.from_fc_json(fcj.FCJ.model_validate_json(open('test/data/p23_fc.json', 'r').read()))

@fixture(scope="session")
def flight():
    return Flight.from_json('test/data/p23_flight.json')


@fixture(scope="session")
def origin():
    return Origin.from_f3a_zone('test/data/p23_box.f3a')


@fixture(scope="session")
def state(flight, origin) -> State:
    return State.from_flight(flight, origin)