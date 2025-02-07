import pytest
from flightdata import Table
import numpy as np
import pandas as pd
from geometry import Time

from pytest import fixture, mark
from flightdata.base.table import Slicer, Label, LabelGroup


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


def test_table_init(tab_full):
    np.testing.assert_array_equal(tab_full.data.columns, ["t", "dt"])


def test_table_get_svar(tab_full):
    assert isinstance(tab_full.time, Time)


def test_table_get_column(tab_full):
    assert isinstance(tab_full.t, np.ndarray)
    assert isinstance(tab_full.dt, np.ndarray)


def test_table_interpolate(tab_full):
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


def test_labelgroup_read_series(tab_full, label_array):
    labels = pd.Series(label_array, index=tab_full.data.index)
    lg = LabelGroup.read_series(labels)
    assert len(lg) == 3
    assert lg.a0.start == tab_full.data.index[0]
    assert lg.a0.stop == tab_full.data.index[2]
    assert lg.a1.start == tab_full.data.index[2]
    assert lg.a1.stop == tab_full.data.index[4]
    assert lg.a2.start == tab_full.data.index[4]
    assert lg.a2.stop is None


@fixture
def tab_lab(tab_full: Table, label_array):
    return tab_full.label(a=label_array)


def test_label_array(tab_lab):
    assert len(tab_lab.labels) == 1
    assert isinstance(tab_lab.labels["a"], LabelGroup)
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


def test_is_tesselated(tab_lab: Table):
    assert tab_lab.labels["a"].is_tesselated()
    assert tab_lab.labels["a"].is_tesselated(tab_lab.time)


def test_label_intersects(tab_lab: Table):
    assert Label(2, 4).intersects(tab_lab)
    assert not Label(7, 9).intersects(tab_lab)


def test_label_contains():
    assert Label(7, 9).contains(8)
    assert not Label(7, 9).contains(9)
    assert not Label(7, 9).contains(9.1)
    assert not Label(7, 9).contains(6.9)
    assert Label(7, 9).contains(7)
    assert Label(7, None).contains(9)


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
    arr = tab_lab.labels["a"].dump_array(tab_lab.time)
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
        a=np.arange(5), 
        b=np.arange(5) / 2, 
        path=np.tile(np.arange(5), (2, 1)).T
    )
    assert newlab == Label(1, 2)



def test_copy_labels(tab_lab: Table, tab_full: Table):
    to = Table.copy_labels(
        labst, tab_full, np.array([[0, 0], [1, ], [2, 1], [3, 2], [4, 4]])
    )
    assert to.labels["a"].length == 3


def test_stack(tab_full):
    tfn = Table.stack(
        [tab_full.label(element="e0"), tab_full.label(element="e1")], overlap=0
    )
    assert tfn.duration == 2 * tab_full.duration + tab_full.dt[-1]
    assert len(tfn) == 2 * len(tab_full)

    tfn = Table.stack(
        [tab_full.label(element="e0"), tab_full.label(element="e1")], overlap=1
    )
    assert tfn.duration == 2 * tab_full.duration
    assert len(tfn) == 2 * len(tab_full) - 1

    tfn = Table.stack([tfn.label(manoeuvre="m1"), tfn.label(manoeuvre="m2")])
    assert tfn.data.index.is_monotonic_increasing
    tfn2 = Table.stack(list(tfn.split_labels().values()))
    assert tfn.duration == tfn2.duration


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
