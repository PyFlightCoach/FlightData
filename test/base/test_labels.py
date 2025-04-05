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
def tab(df):
    return Table(df, False)


@fixture
def tab_full(df):
    return Table.build(df, fill=True)


@fixture
def label_array(tab_full):
    return np.array([f"a{int(i / 2)}" for i in range(len(tab_full))])


@fixture
def tab_lab(tab_full: Table, label_array):
    return tab_full.label(a=label_array)


def test_labelgroup_read_array(tab_full, label_array):
    lg = LabelGroup.read_array(tab_full.t, label_array)
    assert len(lg) == 3
    assert lg.a0.start == tab_full.data.index[0]
    assert lg.a0.stop == tab_full.data.index[2]
    assert lg.a1.start == tab_full.data.index[2]
    assert lg.a1.stop == tab_full.data.index[4]
    assert lg.a2.start == tab_full.data.index[4]
    assert lg.a2.stop == tab_full.data.index[-1]


def test_label_array(tab_lab):
    assert len(tab_lab.labels) == 1
    assert isinstance(tab_lab.labels.a, LabelGroup)
    assert len(tab_lab.labels["a"]) == 3


def test_label_string(tab_full: Table):
    tab_lab = tab_full.label(a="a0")
    assert len(tab_lab.labels) == 1
    assert isinstance(tab_lab.labels["a"], LabelGroup)
    assert len(tab_lab.labels["a"]) == 1


def test_is_tesselated(tab_lab: Table):
    assert tab_lab.labels["a"].is_tesselated()
    assert tab_lab.labels["a"].is_tesselated(tab_lab.t)


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


@fixture
def labst(tab_full):
    return Table.stack(
        [
            tab_full.label(man="m1", el="e1"),
            tab_full.label(man="m1", el="e2"),
            tab_full.label(man="m2", el="e1"),
        ]
    )


def test_labels_dump_array(tab_lab: Table):
    arr = tab_lab.labels.a.to_array(tab_lab.t)
    assert all(arr == ["a0", "a0", "a1", "a1", "a2", "a2"])


def test_labels_dump_array_full(tab_full: Table):
    tlab = tab_full.label(a="a0")
    arr = tlab.labels.a.to_array(tab_full.t)
    assert all(arr == ["a0", "a0", "a0", "a0", "a0", "a0"])


def test_labelgroupss_to_df(tab_lab: Table):
    df = tab_lab.labels.to_df(tab_lab.t)
    assert len(df) == 6


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


def test_label_transfer_shift():
    path = np.array([[0, 0], [1, 1], [1, 2], [1, 3], [4, 4], [5, 5]])
    nlab = Label(0, 1).transfer(np.arange(5), np.arange(5), path)
    assert nlab == Label(0, 3)
    nlab = Label(0, 2.5).transfer(np.arange(5), np.arange(5), path)
    assert nlab == Label(0, 3.5)


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


def test_from_boundaries():
    lg = LabelGroup.from_boundaries(0, dict(a=2, b=5))
    assert lg.a.start == 0
    assert lg.a.stop == 2
    assert lg.b.start==2
    assert lg.b.stop == 5
    assert lg.is_tesselated(np.arange(5))

def test_expand_one():
    lg = LabelGroup.from_boundaries(0, dict(a=2, b=2, c=6))
    lg = lg.expand_one('b', 1)
    np.testing.assert_array_equal(lg.boundaries, [2,3,6])

def test_expand():
    lg = LabelGroup.from_boundaries(0, dict(a=2, b=2, c=6))
    lg = lg.expand(2)
    np.testing.assert_array_equal(lg.boundaries, [2, 4, 6])

def test_expand_difficult():
    lg = LabelGroup.from_boundaries(0, dict(a=0, b=1, c=7))
    
    np.testing.assert_array_equal(lg.expand(1).boundaries, [1, 2, 7])
    np.testing.assert_array_equal(lg.expand(2).boundaries, [2, 4, 7])

def test_expand_infinite_loop():
    lg = LabelGroup.from_boundaries(0, dict(a=1, b=5, c=6, d=8))
    np.testing.assert_array_equal(lg.expand(2).boundaries, [2, 4, 6, 8])