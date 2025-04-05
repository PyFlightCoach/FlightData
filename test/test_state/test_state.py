from pytest import mark
from flightdata import State
import geometry as g
from pathlib import Path
from json import load
import pandas as pd
import numpy as np

from flightdata.state import align

def test_from_constructs():
    st = State.from_constructs(
        time=g.Time(5, 1 / 30),
        pos=g.Point.zeros(),
        att=g.Quaternion.from_euler(g.Point.zeros()),
    )
    assert st.pos == g.Point.zeros()


def test_from_transform():
    st = State.from_transform(g.Transformation())
    assert st.vel.x == 0

    st = State.from_transform(g.Transformation(), vel=g.PX(20))
    assert st.vel.x == 20


def test_from_old_dict():
    data = load(Path("test/data/old_state.json").open())

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

def test_to_new_dict():
    st = State.from_transform(vel=g.PX(20)).extrapolate(0.5).label(element="e1")
    data = st.to_dict()
    assert isinstance(data, dict)
    assert "data" in data
    assert "labels" in data

def test_to_from_new_dict():
    st = State.from_transform(vel=g.PX(20)).extrapolate(0.5).label(element="e1")
    data = st.to_dict()
    st2 = State.from_dict(data)
    assert st.data.equals(st2.data)
    assert st.labels == st2.labels

@mark.skip
def test_align():
    st0 = State.from_transform(g.Transformation(g.Euler(np.pi, 0, 0)), vel=g.PX(30)).extrapolate(2)
    st1 = st0[-1].copy(rvel=g.PY(0.5)).extrapolate(2)
    st2 = st1[-1].copy(rvel=g.P0()).extrapolate(2)
    template = State.stack([st0, st1, st2], "element",["e1", "e2", "e3"])

    st1b = st0[-1].copy(rvel=g.PY(0.5)).extrapolate(4)
    st2b = st1b[-1].copy(rvel=g.P0()).extrapolate(3)
    flown = State.stack([st0, st1b, st2b], "element", ["e1", "e2", "e3"])
    res = align(flown.remove_labels(), template)


    assert flown.labels == res.aligned.labels

    pass

def test_resample():
    st = State.from_transform(vel=g.PX(30)).extrapolate(1)
    