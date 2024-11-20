from flightdata import Flight, State, Origin, BinData, fcj
from pytest import approx, fixture
from geometry import Transformation, PX, PY, P0, Time
from geometry.checks import assert_almost_equal
import numpy as np
from time import sleep, time
from json import load
from ..conftest import state, flight, origin, fcjson

def test_extrapolate():
    initial = State.from_transform(
        Transformation(),
        vel=PX(30)
    )

    extrapolated = initial.extrapolate(10)
    assert extrapolated.x[-1] == approx(300)
    
    assert len(extrapolated) == 250
    assert_almost_equal(extrapolated.pos[0], initial.pos)


def test_extrapolate_rot():
    initial = State.from_transform(
        Transformation(),
        vel=PX(30),
        rvel=PY(2*np.pi/10)
    )

    extrapolated = initial.extrapolate(10)
    
    assert_almost_equal(
        extrapolated.pos[-2], 
        P0(),
        0
    )
    


def test_from_flight(flight, state):
    assert len(state.data) == len(flight.data)
    assert not np.any(np.isnan(state.pos.data))
    assert state.z.mean() > 0

def test_from_flight_pos(flight: Flight, state: State, origin: Origin):
    fl2 = flight.copy()
    fl2.primary_pos_source = 'position'
    st2 = State.from_flight(fl2, origin)
    #pd.testing.assert_frame_equal(state.data, st2.data)
    assert st2.z.mean() > 0

def test_fc_json(fcjson: fcj.FCJ):
    fl = Flight.from_fc_json(fcjson)
    origin = Origin.from_fcjson_parameters(fcjson.parameters)
    st = State.from_flight(fl, origin)
    assert st.z.mean() > 0


def test_stack_singles():
    start=time()
    st=State.from_constructs(Time(time(), 0))
    
    for _ in range(10):
        sleep(0.01)
        st=st.append(State.from_constructs(Time.from_t(0)), "now")
        

    assert time()-start == approx(st.duration, abs=1e-2)

def test_fill():
    _t = Time.from_t(np.linspace(0, 1, 11))
    st0 = State.from_transform(Transformation.zero(), vel=PX(10))
    st = st0.fill(_t)
    assert len(st) == 11
    assert st.pos.x[0] == approx(0)
    assert st.pos.x[-1] == approx(10)
    
def test_fill_vart():
    _dt = np.full(11, 0.1)
    _dt[::2] = 0.11
    _t = Time.from_t(np.cumsum(_dt))
    q = 2*np.pi/10
    st0 = State.from_transform(Transformation.zero(), vel=PX(10), rvel=PY(q))
    st = st0.fill(_t)
    np.testing.assert_array_equal(st.q, np.full(11, q))
    



@fixture(scope='session')
def bindata():
    return BinData.parse_json(load(open('test/data/web_bin_parse.json', 'r')))

@fixture(scope='session')
def flbd(bindata: BinData):
    return Flight.from_log(bindata)


def test_st_from_bindata(flbd: State):
    st = State.from_flight(flbd)
    assert isinstance(st, State)
