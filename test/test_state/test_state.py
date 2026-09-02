from json import load
from pathlib import Path

import geometry as g
import numpy as np
import pandas as pd
from geometry.checks import assert_almost_equal, assert_equal
from pytest import fixture, mark

from flightdata import State
from flightdata.state.alignment import Alignment


@fixture
def st():
    return State(
            time=g.Time.uniform(0.4, 5),
            pos=g.PX(np.arange(5)),
            att=g.upright().tile(5),
        )

def test_basic_initialization_lazy_attributes(st: State):
    assert st._vel is None
    assert_almost_equal(st.vel, g.PX(10, 5))
    assert st._vel is not None

    assert_almost_equal(st.rvel, g.P0(5))
    assert_almost_equal(st.acc, g.PZ(-9.81, 5))

    assert np.shares_memory(st.pos.data, st.x) 

def test_to_from_numpy(st: State):   
    st2 = State.from_numpy(st.to_numpy(generate=False))
    assert_equal(st.pos, st2.pos)
    assert st._vel is None
    
    st2 = State.from_numpy(st.to_numpy(generate=True))
    assert_equal(st.pos, st2.pos)
    assert st.vel is not None


def test_from_transform():
    st = State.from_transform(g.Transformation())
    assert st.vel.x == 0

    st = State.from_transform(g.Transformation(), vel=g.PX(20))
    assert st.vel.x == 20


def test_to_from_df(st: State):
    df = st.to_dataframe()
    st2 = State.from_df(df)
    assert_equal(st.pos, st2.pos)


def test_from_old_dict():
    with Path("test/data/old_state.json").open() as fp:
        data = load(fp)

    df = pd.DataFrame.from_dict(data).set_index("t", drop=False)

    st = State.from_dict(data)

    assert len(st.manoeuvre.sql) == len(df.loc[df.manoeuvre == "sql"]) 
    assert len(st.manoeuvre.hSqL) == len(df.loc[df.manoeuvre == "hSqL"]) + 1

    assert len(st.labels) == 1
    assert len(st.labels.manoeuvre) == 2
    assert len(st.manoeuvre.sql.labels) == 1

    assert len(st.manoeuvre.hSqL.element.entry_line) == 4


def test_to_old_dict():
    st = State.from_transform(vel=g.PX(20)).extrapolate(0.5).label(element="e1")
    data = st.to_dict(True)
    assert isinstance(data, list)
    assert "t" in data[0]
    assert "element" in data[0]

def test_to_new_dict_data_only():
    st = State.from_transform(vel=g.PX(20)).extrapolate(0.5).label(element="e1")
    data = st.to_dict(legacy=False, include_data=True)
    assert isinstance(data, dict)
    assert "data" in data
    assert "labels" in data

def test_to_from_new_dict_data_only():
    st = State.from_transform(vel=g.PX(20)).extrapolate(0.5).label(element="e1")
    data = st.to_dict(legacy=False, include_data=False)
    st2 = State.from_dict(data)
    assert st.almost_equal(st2)
    assert st.labels == st2.labels

def test_to_from_new_dict_with_splines():
    st = State.from_transform(vel=g.PX(20)).extrapolate(5).label(element="e1")
    ss = st.create_splines()
    data = ss.to_dict(legacy=False, include_data=True)

    ss2 = State.from_dict(data)
    assert_almost_equal(ss.pos, ss2.pos)
    assert_almost_equal(ss.vel, ss2.vel)
    assert_almost_equal(ss.acc, ss2.acc)
    assert_almost_equal(ss.rvel, ss2.rvel)
    assert_almost_equal(ss.att, ss2.att)

def test_slice_with_splines():
    st = State.from_transform(vel=g.PX(20)).extrapolate(5).label(element="e1")
    ss = st.create_splines()

    sliced = ss[1:4]

    assert sliced.splines is ss.splines


@mark.skip
def test_align():
    st0 = State.from_transform(g.Transformation(g.Euler(np.pi, 0, 0)), vel=g.PX(30)).extrapolate(2)
    st1 = st0[-1].copy(rvel=g.PY(0.5)).extrapolate(2)
    st2 = st1[-1].copy(rvel=g.P0()).extrapolate(2)
    template = State.stack([st0, st1, st2], "element",["e1", "e2", "e3"])

    st1b = st0[-1].copy(rvel=g.PY(0.5)).extrapolate(4)
    st2b = st1b[-1].copy(rvel=g.P0()).extrapolate(3)
    flown = State.stack([st0, st1b, st2b], "element", ["e1", "e2", "e3"])
    res = Alignment.align(flown.remove_labels(), template)


    assert flown.labels == res.aligned.labels

    pass


def test_align_resample():
    from flightanalysis.elements import Elements, Line, Loop
    itrans = g.Transformation()
    tp = Elements([Line("l1", 30, 30, 0), Line("l2", 30, 30, np.pi), Line("l3", 30, 30, 0)]).create_templates(itrans)
    fl = Elements([Line("l1", 30, 70, 0), Line("l2", 30, 70, np.pi), Line("l3", 30, 70, 0)]).create_templates(itrans)
    tp = State.stack(tp, "element")
    fl = State.stack(fl, "element").remove_labels()
    aligmnent = Alignment.align(fl, tp)

    assert len(aligmnent.aligned) == len(fl)



def test_resample():
    st = State.from_transform(vel=g.PX(30)).extrapolate(1)
    st = st.resample(0.1)
    assert len(st) == 10

@mark.skip
def test_state_interpolate():
    with Path("test/data/st_interpolation_testing.json").open() as fp:
        st = State.from_dict(load(fp))

    inst = st.iloc[np.linspace(0, 2, 20)]

    np.testing.assert_array_almost_equal(inst.t, np.linspace(st.t[0], st.t[2], 20))


    assert_almost_equal(st.iloc[0.5].pos, (st.pos[0] + st.pos[1]) / 2)

    assert_almost_equal(st.iloc[0.5].vel, (st.vel[0] + st.vel[1]) / 2)
    assert_almost_equal(st.iloc[0.5].acc, (st.acc[0] + st.acc[1]) / 2)
    assert_almost_equal(st.iloc[0.5].rvel, (st.rvel[0] + st.rvel[1]) / 2)

    assert_almost_equal(st.iloc[0.5].wvel, (st.wvel[0] + st.wvel[1]) / 2, 0.01)
    assert_almost_equal(st.iloc[0.5].wacc, (st.wacc[0] + st.wacc[1]) / 2, 0.01)
    
    pass