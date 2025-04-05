import pytest
from flightdata import Table
import numpy as np
import pandas as pd
from geometry import Time

from pytest import fixture, mark
from flightdata.base.table import Slicer, Label, LabelGroup, LabelGroups


@fixture
def df():
    df = pd.DataFrame(np.linspace(0, 5, 6), columns=["t"])
    return df.set_index("t", drop=False)


@fixture
def tab(df):
    return Table(df, False)


@fixture
def tab_full(df):
    return Table.build(df, fill=True)


def test_table_init(tab_full: Table):
    np.testing.assert_array_equal(tab_full.data.columns, ["t", "dt"])


def test_table_init_junk_cols(df: pd.DataFrame):
    df = df.assign(junk=6)
    tab = Table.build(df)
    assert len(tab.data.columns) == 2
    assert "junk" not in tab.data.columns


def test_table_get_svar(tab_full: Table):
    assert isinstance(tab_full.time, Time)


def test_table_get_column(tab_full: Table):
    assert isinstance(tab_full.t, np.ndarray)
    assert isinstance(tab_full.dt, np.ndarray)


def test_table_interpolate(tab_full: Table):
    with pytest.raises(Exception):
        t = tab_full.interpolate(7)

    t = tab_full.interpolate(2.5)
    assert t.t[0] == 2.5
    assert t.dt[0] == 0.5


def test_tab_getitem(tab_full):
    assert tab_full[2].t[0] == 2
    assert tab_full[2.6].t[0] == 2.6


def test_tab_getslice_exact(tab_full):
    assert len(tab_full[2:4]) == 3
    assert tab_full[2:4].t[-1] == 4


def test_tab_getslice_interpolate(tab_full):
    sli = tab_full[2.5:4.5]
    assert len(sli) == 4
    assert sli.t[0] == 2.5
    assert sli.t[-1] == 4.5
    assert sli.dt[0] == 0.5
    assert sli.dt[-2] == 0.5
    assert sli.dt[-1] == 0.5



@fixture
def label_array(tab_full):
    return np.array([f"a{int(i / 2)}" for i in range(len(tab_full))])


@fixture
def tab_lab(tab_full: Table, label_array):
    return tab_full.label(a=label_array)



def test_get_slicer(tab_lab):
    slicer = tab_lab.a
    assert isinstance(slicer, Slicer)


def test_slicer_slice(tab_lab):
    slice = tab_lab.a.a1
    assert slice.t[0] == 2
    assert slice.t[-1] == 4


def test_slice_labels(tab_lab: Table):
    sli = tab_lab[:1]
    assert len(sli) == 2
    assert len(sli.labels["a"]) == 1
    assert sli.labels["a"].labels["a0"].start == 0
    assert sli.labels["a"].labels["a0"].stop == 1



def test_copy(tab_full: Table):
    tab2 = tab_full.copy()
    np.testing.assert_array_equal(tab2.t, tab_full.t)

    tab3 = tab_full.copy(time=Time.from_t(tab_full.t + 10))

    np.testing.assert_array_equal(tab3.t, tab_full.t + 10)


def test_copy_labels_no_path(tab_lab: Table):
#    path=np.array([[0,0], [1,1], [2,2], [3,3], [4,4], [5,5]])
    tfull = Table.from_constructs(Time.from_t(np.arange(2*len(tab_lab))))
    tlab2 = Table.copy_labels(tab_lab, tfull)
    assert "a" in tlab2.labels.lgs

def test_copy_labels_path(tab_lab: Table):
    path=np.array([[0,0], [1,1], [2,2], [3,3], [4,4], [5,5]])
    tlab2 = Table.copy_labels(tab_lab, tab_lab.remove_labels(), path)
    assert "a" in tlab2.labels.lgs


def test_copy_labels_no_substeps(tab_lab: Table):
    path=np.array([[0,0], [1,1], [1,2], [1,3], [4,4], [5,5]])
    tlab2 = Table.copy_labels(tab_lab, tab_lab.remove_labels(), path, None)
    assert "a" in tlab2.labels.lgs
    assert "a1" not in tlab2.labels.a.labels

def test_unsquash_labels(tab_lab: Table):
    #                 0      1      2      3      4      5
    #                A0     A0     A1     A1     A2     A2
    path=np.array([[0,0], [1,1], [1,2], [1,3], [4,4], [5,5]])
    #                A0     A0     A0     A0     A2     A2
    #                A0     A0     A0     A1     A2     A2  
    tlab2 = Table.copy_labels(tab_lab, tab_lab.remove_labels(), path, 1)
    assert tlab2.labels.a.a0.stop==3
    assert tlab2.labels.a.a1.start==3
    assert tlab2.labels.a.a1.stop==4
    assert tlab2.labels.a.a2.start==4
    assert tlab2.labels.a.a2.stop==5
    


def test_shift_time(tab_lab):
    new_lab = tab_lab.shift_time(2)
    assert new_lab.t[0] == 2
    assert new_lab.labels["a"].labels["a0"].start == 2
    assert new_lab.labels["a"].labels["a0"].stop == 4



def test_stack_no_overlap(tab_full: Table):
    tfn = Table.stack(
        [tab_full.label(element="e0"), tab_full.label(element="e1")], overlap=0
    )
    assert tfn.duration == 2 * tab_full.duration + tab_full.dt[-1]
    assert len(tfn) == 2 * len(tab_full)

    assert "element" in tfn.labels.lgs
    assert tfn.element.e0.duration == tab_full.duration
    assert tfn.element.e1.t[0] == tab_full.duration + tab_full.dt[-1]
    assert tfn.element.e1.duration == tab_full.duration


def test_iloc(tab_full: Table):
    t = tab_full.iloc[2:4]
    assert len(t) == 3
    assert t.t[0] == 2
    assert t.t[-1] == 4

def test_iloc_list(tab_full: Table):
    t = tab_full.iloc[[0, -1]]
    assert len(t) == 2
    assert t.t[0] == 0
    assert t.t[-1] == tab_full.t[-1]

def test_stack_overlap(tab_full):
    tfn = Table.stack(
        [tab_full.label(element="e0"), tab_full.label(element="e1")], overlap=1
    )
    assert tfn.duration == 2 * tab_full.duration
    assert len(tfn) == 2 * len(tab_full) - 1

    assert "element" in tfn.labels.lgs
    assert tfn.element.e0.duration == tab_full.duration
    assert tfn.element.e1.t[0] == tab_full.duration
    assert tfn.element.e1.duration == tab_full.duration


def test_over_label(tab_lab: Table):
    tol = tab_lab.over_label("b", "b1")
    assert len(tol.labels) == 1
    assert len(tol.labels.b.b1.sublabels.a) == 3
    assert len(tol.b.b1.a.labels) == 3
    assert len(tol.b.b1.labels) == 1
    assert len(tol.b["b1"].a["a2"]) == 2


def test_sublabels(tab_full: Table):
    tl = Table.stack(
        [
            tab_full.label(b=["b1", "b1", "b1", "b2", "b2", "b2"]),
            tab_full.label(b=["b2", "b2", "b1", "b2", "b2", "b2"]),
        ],
        "a",
        ["a1", "a2"],
        1,
    )

    assert tl.a.a1.b.b1.duration == 3
    assert tl.a.a2.b.b1.duration == 1


def test_set_boundaries(tab_lab: Table):
    boundaries = tab_lab.labels.a.boundaries
    np.testing.assert_array_equal(boundaries, [2, 4, 5])
    newlabs = tab_lab.labels.a.set_boundaries([3, 4, 6])
    assert newlabs.a0.stop == 3
    assert newlabs.a1.start == 3


def test_set_boundary(tab_lab: Table):
    assert tab_lab.labels.a.a0.stop == 2
    assert tab_lab.labels.a.a1.start == 2
    newlabs = tab_lab.labels.a.set_boundary("a0", 3, 1)
    assert newlabs.a0.stop == 3
    assert newlabs.a1.start == 3
    with pytest.raises(ValueError):
        newlabs = tab_lab.labels.a.set_boundary("a0", 4, 1)


def test_nest_labels_single():
    table = Table.from_constructs(Time.from_t(np.arange(10)))
    a=np.concatenate([np.full(5, "a1"), np.full(5, "a2")])
    tlab = table.nest_labels(a=a)
    assert tlab.labels.a.a1 == Label(0, 5)
    assert tlab.labels.a.a2 == Label(5, 9)

def test_nest_labels_multi():
    table = Table.from_constructs(Time.from_t(np.arange(10)))
    a=np.concatenate([np.full(5, "a1"), np.full(5, "a2")])

    b=np.concatenate([np.full(2, "b1"), np.full(3, "b2"), np.full(2, "b1"), np.full(3, "b2")])
    tlab = table.nest_labels(a=a, b=b)
    assert tlab.labels.a.a1 == Label(0, 5)
    assert tlab.labels.a.a2 == Label(5, 9)
    assert tlab.a.a1.labels.b.b1 == Label(0, 2)
    assert tlab.a.a1.labels.b.b2 == Label(2, 5)
    assert tlab.a.a2.labels.b.b1 == Label(5, 7)
    assert tlab.a.a2.labels.b.b2 == Label(7, 9)
    
