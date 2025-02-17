import pytest
from flightdata import Table
import numpy as np
import pandas as pd
from geometry import Time

from pytest import fixture, mark
from flightdata.base.table import Slicer, Label, LabelGroup
from flightdata.base.table.labels import LabelGroups


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
    with pytest.raises(ValueError):
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


def test_labelgroup_read_array(tab_full, label_array):
    lg = LabelGroup.read_array(tab_full.time, label_array)
    assert len(lg) == 3
    assert lg.a0.start == tab_full.data.index[0]
    assert lg.a0.stop == tab_full.data.index[2]
    assert lg.a1.start == tab_full.data.index[2]
    assert lg.a1.stop == tab_full.data.index[4]
    assert lg.a2.start == tab_full.data.index[4]
    assert lg.a2.stop == tab_full.data.index[-1] + tab_full.data.dt[4]


@fixture
def tab_lab(tab_full: Table, label_array):
    return tab_full.label(a=label_array)


def test_label_array(tab_lab):
    assert len(tab_lab.labels) == 1
    assert isinstance(tab_lab.labels.a, LabelGroup)
    assert len(tab_lab.labels["a"]) == 3


def test_label_string(tab_full: Table):
    tab_lab = tab_full.label(a="a0")
    assert len(tab_lab.labels) == 1
    assert isinstance(tab_lab.labels["a"], LabelGroup)
    assert len(tab_lab.labels["a"]) == 1


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


def test_is_tesselated(tab_lab: Table):
    assert tab_lab.labels["a"].is_tesselated()
    assert tab_lab.labels["a"].is_tesselated(tab_lab.time)


def test_label_intersects(tab_lab: Table):
    assert Label(2, 4).intersects(tab_lab.t[0], tab_lab.t[-1])
    assert not Label(7, 9).intersects(tab_lab.t[0], tab_lab.t[-1])


def test_label_contains():
    assert Label(7, 9).contains(8)
    assert Label(7, 9).contains(9)
    assert not Label(7, 9).contains(9.1)
    assert not Label(7, 9).contains(6.9)
    assert Label(7, 9).contains(7)


def test_interpolate_labelled(tab_lab: Table):
    t = tab_lab.interpolate(2.5)
    assert "a1" in t.labels["a"].labels
    sli = tab_lab[2.5:4.5]
    assert "a1" in sli.labels["a"].labels
    assert "a2" in sli.labels["a"].labels


def test_copy(tab_full: Table):
    tab2 = tab_full.copy()
    np.testing.assert_array_equal(tab2.t, tab_full.t)

    tab3 = tab_full.copy(time=Time.from_t(tab_full.t + 10))

    np.testing.assert_array_equal(tab3.t, tab_full.t + 10)


@fixture
def labst(tab_full):
    return Table.stack(
        [
            tab_full.label(man="m1", el="e1"),
            tab_full.label(man="m1", el="e2"),
            tab_full.label(man="m2", el="e1"),
        ]
    )


@mark.skip("old API")
def test_get_subset_df(tab_full, labst):
    df = labst.get_subset_df(man="m1", el="e1")
    assert len(tab_full) == len(df)


@mark.skip("old API")
def test_shift_labels_ratios(tab_full: Table):
    tf: Table = Table.stack(
        [
            tab_full.label(element="e0", manoeuvre="m0"),
            tab_full.label(element="e1", manoeuvre="m0"),
        ]
    )

    assert sum(tf.shift_labels_ratios([0.5], 2).element == "e1") < sum(
        tf.element == "e1"
    )


def test_labels_dump_array(tab_lab: Table):
    arr = tab_lab.labels.a.to_array(tab_lab.time)
    assert all(arr == ["a0", "a0", "a1", "a1", "a2", "a2"])


def test_label_to_iloc(tab_full: Table):
    lab = Label(2, 4).to_iloc(tab_full.t)
    assert lab.start == 2
    assert lab.stop == 4
    lab = Label(2, 4).to_iloc(tab_full.t * 2)
    assert lab.start == 1
    assert lab.stop == 2
    lab = Label(2.5, 4.5).to_iloc(tab_full.t)
    assert lab.start == 2.5
    assert lab.stop == 4.5


def test_label_to_t(tab_full: Table):
    lab = Label(2, 4).to_t(tab_full.t)
    assert lab.start == 2
    assert lab.stop == 4
    lab = Label(2, 4).to_t(tab_full.t * 2)
    assert lab.start == 4
    assert lab.stop == 8
    lab = Label(2.5, 4.5).to_t(tab_full.t)
    assert lab.start == 2.5
    assert lab.stop == 4.5


def test_label_transfer():
    newlab = Label(2, 4).transfer(
        a=np.arange(5), b=np.arange(5) / 2, path=np.tile(np.arange(5), (2, 1)).T
    )
    assert newlab == Label(1, 2)


def test_shift_time(tab_lab):
    new_lab = tab_lab.shift_time(2)
    assert new_lab.t[0] == 2
    assert new_lab.labels["a"].labels["a0"].start == 2
    assert new_lab.labels["a"].labels["a0"].stop == 4


def test_concat_labelgroup():
    lg1 = LabelGroup({"a0": Label(0, 2), "a1": Label(2, 4)})
    lg2 = LabelGroup({"a1": Label(4, 6), "a2": Label(6, 8)})
    nlg = LabelGroup.concat(lg1, lg2)
    assert len(nlg) == 3
    assert nlg.a0.start == 0
    assert nlg.a0.stop == 2
    assert nlg.a1.start == 2
    assert nlg.a1.stop == 6
    assert nlg.a2.start == 6
    assert nlg.a2.stop == 8


def test_stack_labelgroups():
    lgs1 = LabelGroups({"a": LabelGroup({"a0": Label(0, 2), "a1": Label(2, 4)})})
    lgs2 = LabelGroups({"a": LabelGroup({"a1": Label(4, 6), "a2": Label(6, 8)})})
    nlgs = LabelGroups.concat(lgs1, lgs2)
    assert len(nlgs) == 1
    assert len(nlgs.a) == 3
    assert nlgs.a.a0.start == 0
    assert nlgs.a.a0.stop == 2
    assert nlgs.a.a1.start == 2
    assert nlgs.a.a1.stop == 6
    assert nlgs.a.a2.start == 6
    assert nlgs.a.a2.stop == 8


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
    assert len(t) == 2
    assert t.t[0] == 2
    assert t.t[-1] == 3


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
    assert len(tol.b['b1'].a['a2']) == 2

def test_sublabels(tab_full: Table):
    tl = Table.stack(
        [
            tab_full.label(b=["b1", "b1", "b1", "b2", "b2", "b2"]),
            tab_full.label(b=["b2", "b2", "b1", "b2", "b2", "b2"]),
        ],
        "a", ["a1", "a2"], 1
    )

    assert tl.a.a1.b.b1.duration == 3
    assert tl.a.a2.b.b1.duration == 1



def test_set_boundaries(tab_lab: Table):
    boundaries = tab_lab.labels.a.boundaries
    np.testing.assert_array_equal(boundaries, [2, 4, 6])
    newlabs = tab_lab.labels.a.set_boundaries([3,4,6])
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


@mark.skip("TBC")
def test_shift_multi(tab_full):
    tabs = Table.stack(
        [tab_full.label(element="e0"), tab_full.label(element="e1")], overlap=1
    ).split_labels()
    tb1, tb2 = Table.shift_multi(2, tabs["e0"], tabs["e1"])

    assert len(tb1) == len(tab_full) + 2
    assert len(tb2) == len(tab_full) - 2
    assert tb2.duration == tab_full.data.index[-3]

    tb1, tb2 = Table.shift_multi(-3, tabs["e0"], tabs["e1"])

    assert len(tb1) == len(tab_full) - 3
    assert len(tb2) == len(tab_full) + 3
    assert tb1.duration == tab_full.data.index[-4]


@mark.skip("TBC")
def test_table_cumulative_labels(tab_full):
    tf = (
        tab_full.label(a="a1", b="b1")
        .append(tab_full.label(a="a2", b="b1"))
        .append(tab_full.label(a="a2", b="b2"))
        .append(tab_full.label(a="a1", b="b1"))
    )
    print(tf)
    res = tf.cumulative_labels("a", "b")
    assert isinstance(res, np.ndarray)
    assert len(res) == len(tf)

    indexes = np.unique(res, return_index=True)[1]
    np.testing.assert_array_equal(
        [res[index] for index in sorted(indexes)],
        np.array(["a1_b1_0", "a2_b1_0", "a2_b2_0", "a1_b1_1"]),
    )
    pass


@mark.skip("TBC")
def test_str_replace_label(labst: Table):
    nlabst = labst.str_replace_label(
        el=np.array(
            [
                ["e1", "e1__"],
                ["e2", "e2__"],
            ]
        ),
    )
    assert sum(nlabst.data.el == "e1__") == sum(labst.data.el == "e1")
    assert sum(nlabst.data.el == "e2__") == sum(labst.data.el == "e2")

