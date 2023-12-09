from flightdata import fields, Field
import numpy as np


def test_get_fields():
    _fields = fields.get_fields(['time', 'position_D'])
    assert len(_fields)==3
    assert all([isinstance(f, Field) for f in _fields])

def test_get_cols():
    cols = fields.get_cols(['time', 'position_D'])
    np.testing.assert_array_equal(cols, ['time_flight', 'time_actual', 'position_D'])
