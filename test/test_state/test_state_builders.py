from flightdata import Flight, State, Origin, BinData
from schemas import fcj
from pytest import approx, fixture, mark
import geometry as g
from geometry import checks

import numpy as np
from time import sleep, time
from json import load
from ..conftest import state, flight, origin, fcjson


def test_extrapolate():
    initial = State.from_transform(g.Transformation(), vel=g.PX(30))

    assert isinstance(initial, State)
    extrapolated = initial.extrapolate(10)
    assert extrapolated.x[-1] == approx(300)

    assert len(extrapolated) == 250
    checks.assert_almost_equal(extrapolated.pos[0], initial.pos)


def test_extrapolate_rot():
    initial = State.from_transform(
        g.Transformation(g.Euler(np.pi, 0, np.pi / 2)),
        vel=g.PX(30),
        rvel=g.Point(2 * np.pi / 10, 2 * np.pi / 10, 0),
    )

    extrapolated = initial.extrapolate(10)
    checks.assert_almost_equal(extrapolated.pos[-1], g.P0(), 0)

#    checkst = State.from_constructs(
#        extrapolated.time, extrapolated.pos, extrapolated.att
#    )

#    checks.assert_almost_equal(checkst.vel, extrapolated.vel)
#    checks.assert_almost_equal(checkst.rvel, extrapolated.rvel)
#    checks.assert_almost_equal(checkst.acc, extrapolated.acc)


def test_from_flight(flight: Flight, state: State):
    assert len(state.data) == len(flight.data)
    assert not np.any(np.isnan(state.pos.data))
    assert state.z.mean() > 0


@mark.skip
def test_from_flight_pos(flight: Flight, state: State, origin: Origin):
    fl2 = flight.copy()
    fl2.primary_pos_source = "position"
    st2 = State.from_flight(fl2, origin)
    # pd.testing.assert_frame_equal(state.data, st2.data)
    assert st2.z.mean() > 0


@mark.skip
def test_fc_json(fcjson: fcj.FCJ):
    fl = Flight.from_fc_json(fcjson)
    origin = Origin.from_fcjson_parameters(fcjson.parameters)
    st = State.from_flight(fl, origin)

    assert st.z.mean() > 0



def test_fill():
    t = g.Time.from_t(np.linspace(0, 1, 11))
    st0 = State.from_transform(vel=g.PX(10))
    st = st0.fill(t)
    assert len(st) == 11
    assert st.pos.x[0] == approx(0)
    assert st.pos.x[-1] == approx(10)

def test_fill_zero_v():
    t = g.Time.from_t(np.linspace(0, 1, 11))
    st0 = State.from_transform(vel=g.P0(), rvel=g.P0())
    st = st0.fill(t)
    assert len(st) == 11
    assert st.pos.x[0] == approx(0)
    assert st.pos.x[-1] == approx(0)

def test_fill_vart():
    _dt = np.full(11, 0.1)
    _dt[::2] = 0.2
    _t = g.Time.from_t(np.cumsum(_dt))
    q = 2 * np.pi / 10
    st0 = State.from_transform(g.Transformation.zero(), vel=g.PX(10), rvel=g.PY(q))
    st = st0.fill(_t)
    np.testing.assert_array_equal(st.q, np.full(11, q))
    assert st.zero_g_acc().z == approx(-6.28318531)


@fixture(scope="session")
def bindata():
    return BinData.parse_json(load(open("test/data/web_bin_parse.json", "r")))


@fixture(scope="session")
def flbd(bindata: BinData):
    return Flight.from_log(bindata)


def test_st_from_bindata(flbd: State):
    st = State.from_flight(flbd)
    assert isinstance(st, State)



def test_st_slice():
    st = State.from_transform(vel=g.PX(30), rvel=g.PY(np.pi/4)).fill(g.Time.from_t(np.arange(5)))
    #st.plot().show(nmodels=100, scale=1)

    att0 = st.interpolate(1).att
    att1 = st.interpolate(1.5).att
    att2 = st.interpolate(2).att

    d1 = g.Quaternion.body_axis_rates(att0, att1)
    d2 = g.Quaternion.body_axis_rates(att1, att2)
    dtot = g.Quaternion.body_axis_rates(att0, att2)

    assert dtot.y == approx(d1.y + d2.y)
    assert d1.y == approx(d2.y)

    pass



    st_sliced = st[0.5:3.5]
#
    st_sliced.plot().show(nmodels=100, scale=1)
