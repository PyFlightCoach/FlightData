import numpy as np
import pandas as pd
import pytest
from geometry import Time
from pytest import fixture, mark

from flightdata import Table
from flightdata.base.table import Label, LabelGroup, LabelGroups, Slicer


@fixture
def df():
    df = pd.DataFrame(np.linspace(0, 5, 6), columns=["t"])
    return df.set_index("t", drop=False)

@fixture
def table(df):
    return Table.build(df, fill=True)


def test_table_init(table: Table):
    np.testing.assert_array_equal(table.data.columns, ["t", "dt"])


def test_table_init_junk_cols(df: pd.DataFrame):
    df = df.assign(junk=6)
    tab = Table.build(df)
    assert len(tab.data.columns) == 2
    assert "junk" not in tab.data.columns


def test_table_get_svar(table: Table):
    assert isinstance(table.time, Time)


def test_table_get_column(table: Table):
    assert isinstance(table.t, np.ndarray)
    assert isinstance(table.dt, np.ndarray)


def test_table_interpolate(table: Table):
    with pytest.raises(Exception):
        t = table.interpolate(7)

    t = table.interpolate(2.5)
    assert t.t[0] == 2.5
    assert t.dt[0] == 0.5


def test_tab_getitem(table):
    assert table[2].t[0] == 2
    assert table[2.6].t[0] == 2.6


def test_tab_getslice_exact(table):
    assert len(table[2:4]) == 3
    assert table[2:4].t[-1] == 4


def test_tab_getslice_interpolate(table):
    sli = table[2.5:4.5]
    assert len(sli) == 4
    assert sli.t[0] == 2.5
    assert sli.t[-1] == 4.5
    assert sli.dt[0] == 0.5
    assert sli.dt[-2] == 0.5
    assert sli.dt[-1] == 0.5



@fixture
def label_array(table):
    return np.array([f"a{int(i / 2)}" for i in range(len(table))])


@fixture
def tab_lab(table: Table, label_array):
    return table.label(a=label_array)



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


def test_copy(table: Table):
    tab2 = table.copy()
    np.testing.assert_array_equal(tab2.t, table.t)

    tab3 = table.copy(time=Time.from_t(table.t + 10))

    np.testing.assert_array_equal(tab3.t, table.t + 10)


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



def test_stack_no_overlap(table: Table):
    tfn = Table.stack(
        [table.label(element="e0"), table.label(element="e1")], overlap=0
    )
    assert tfn.duration == 2 * table.duration + table.dt[-1]
    assert len(tfn) == 2 * len(table)

    assert "element" in tfn.labels.lgs
    assert tfn.element.e0.duration == table.duration
    assert tfn.element.e1.t[0] == table.duration + table.dt[-1]
    assert tfn.element.e1.duration == table.duration




def test_stack_three_tables_two_one_long(table: Table):

    stacked = Table.stack([table[0], table, table[-1]])
    assert len(stacked) == len(table) 
    
    


def test_iloc(table: Table):
    t = table.iloc[2:4]
    assert len(t) == 3
    assert t.t[0] == 2
    assert t.t[-1] == 4

def test_iloc_list(table: Table):
    t = table.iloc[[0, -1]]
    assert len(t) == 2
    assert t.t[0] == 0
    assert t.t[-1] == table.t[-1]

def test_stack_overlap(table):
    tfn = Table.stack(
        [table.label(element="e0"), table.label(element="e1")], overlap=1
    )
    assert tfn.duration == 2 * table.duration
    assert len(tfn) == 2 * len(table) - 1

    assert "element" in tfn.labels.lgs
    assert tfn.element.e0.duration == table.duration
    assert tfn.element.e1.t[0] == table.duration
    assert tfn.element.e1.duration == table.duration


def test_over_label(tab_lab: Table):
    tol = tab_lab.over_label("b", "b1")
    assert len(tol.labels) == 1
    assert len(tol.labels.b.b1.sublabels.a) == 3
    assert len(tol.b.b1.a.labels) == 3
    assert len(tol.b.b1.labels) == 1
    assert len(tol.b["b1"].a["a2"]) == 2


def test_sublabels(table: Table):
    tl = Table.stack(
        [
            table.label(b=["b1", "b1", "b1", "b2", "b2", "b2"]),
            table.label(b=["b2", "b2", "b1", "b2", "b2", "b2"]),
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
    

def test_splice_sts_inserts_new_indeces(table: Table):
        table = table.label(element="e0")
        tf = Table.splice([table, table.iloc[[0.5, 1.5, 2.5]]])
        assert len(tf) == 9
        assert tf.t[1] == 0.5
        assert "element" in tf.labels.keys()

        e0 = tf.element.e0
        assert e0.t[-1] == 5.0
        pass