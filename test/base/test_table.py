from flightdata import Table
import numpy as np
import pandas as pd
from geometry import Time

from pytest import fixture


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


def test_shift_labels_ratios(tab_full):
    tf = tab_full.label(element='e0', manoeuvre='m0')
    tf.data.loc[tf.duration/2:,'element'] = 'e1'
    
    assert sum(tf.shift_labels_ratios([0.5],2).element == 'e1') < sum(tf.element == 'e1')
    tf
    
def test_stack(tab_full):
    tfn = Table.stack([tab_full.label(element='e0'), tab_full.label(element='e1')], overlap=0)
    assert tfn.duration == 2 * tab_full.duration + tab_full.dt[-1]
    assert len(tfn) == 2 * len(tab_full)
    
    tfn = Table.stack([tab_full.label(element='e0'), tab_full.label(element='e1')], overlap=1)
    assert tfn.duration == 2 * tab_full.duration 
    assert len(tfn) == 2 * len(tab_full) - 1


def test_shift_multi(tab_full):
    t1, t2 = tab_full.label(element='e1'), tab_full.label(element='e2')
    tb1, tb2 = Table.shift_multi(2, t1, t2)
    
    assert len(tb1) == len(tab_full) + 2
    assert len(tb2) == len(tab_full) - 2
    assert tb2.duration == tab_full.data.index[-3]
    
    tb1, tb2 = Table.shift_multi(-5, t1, t2)
    
    assert len(tb1) == len(tab_full) - 5
    assert len(tb2) == len(tab_full) + 5
    assert tb1.duration == tab_full.data.index[-6]


def test_table_cumulative_labels(tab_full):
    tf = tab_full.label(a='a1', b='b1').append(
        tab_full.label(a='a2', b='b1')
    ).append(
        tab_full.label(a='a2', b='b2')
    ).append(
        tab_full.label(a='a1', b='b1')
    )
    print(tf)
    res = tf.cumulative_labels('a', 'b')
    assert isinstance(res, np.ndarray)
    assert len(res) == len(tf)

    indexes = np.unique(res, return_index=True)[1]
    np.testing.assert_array_equal(
        [res[index] for index in sorted(indexes)],
        np.array(['a1_b1', 'a2_b1', 'a2_b2', 'a1_b1_1'])
    )
    pass