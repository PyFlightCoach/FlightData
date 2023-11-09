import pytest

from flightdata import Constructs, SVar, State
import numpy as np


def test_subset():
    subs = State.constructs.subset(["time", "pos"])

    assert len(subs.data) == 2


def test_existing():
    subs = State.constructs.existing(["x", "y", "z"])
    assert len(subs.data) == 1

def test_missing():
    subs = State.constructs.missing(["x", "y", "z", "rw"])
    assert len(subs.data) == len(State.constructs.data) - 1


def test_contains():
    res = State.constructs.contains(["pos"])
    assert np.all(res==[True])

    res = State.constructs.contains("pos")
    assert res==True

