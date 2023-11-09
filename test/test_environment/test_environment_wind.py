from flightdata import WindModelBuilder, WindModel
import pytest
import numpy as np 
import pandas as pd
from geometry import Point, Quaternion, Transformation



def test_wind_power_law_builder():
    pl = WindModelBuilder.power_law()([0.0, 5.0, 0.2])

    assert pl(300, 0.0).x==5
    assert isinstance(pl(300, 0.0), Point)

    pl_res = pl(np.random.random(100), np.random.random(100))
    assert isinstance(pl_res, Point)
    assert pl_res.count, 100


def test_wind_fit_builder():
    pl = WindModelBuilder.fit()(np.random.random(20))
    assert isinstance(pl(300, 0.0), Point)

    pl_res = pl(np.random.random(100), np.random.random(100))
    assert isinstance(pl_res, Point)
    assert pl_res.count, 100


