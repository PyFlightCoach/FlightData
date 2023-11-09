from flightanalysis.base.table import Table, Time
from flightanalysis.base.constructs import Constructs, SVar
import numpy as np
import pandas as pd
from geometry import Point, Quaternion, PZ

from pytest import approx, fixture


@fixture
def df():
    df = pd.DataFrame(np.linspace(0,100, 100), columns=['t'])
    return df.set_index("t", drop=False)


@fixture
def tab(df):
    return Table(df, False)

@fixture
def tab_full(df):
    return Table(df, True)



def test_table_init(tab_full):
    np.testing.assert_array_equal(tab_full.data.columns, ["t", "dt"])
    


def test_table_getattr(tab_full):
    assert isinstance(tab_full.time, Time)


def test_tab_getitem(tab_full):
    t = tab_full[20]
    pass


def test_copy(tab_full):
    tab2 = tab_full.copy()
    np.testing.assert_array_equal(tab2.t, tab_full.t)

    tab3 = tab_full.copy(time=Time.from_t(tab_full.t+10))

    np.testing.assert_array_equal(tab3.t, tab_full.t + 10)