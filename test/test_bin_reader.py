from collections.abc import Callable
from json import load

import geometry as g
import numpy as np
import numpy.typing as npt
import pandas as pd
from ardupilot_log_reader import Ardupilot
from pytest import fixture

import flightdata.ardupilot.messages as bf
from flightdata import Flight
from flightdata.ardupilot.state_data import StateData
from flightdata.bindata import BinData


@fixture
def bindata() -> Ardupilot:
    with open("test/data/new_web_bin.json") as f:
        return Ardupilot.from_dict(load(f))


@fixture
def fields(bindata: Ardupilot) -> dict[str, pd.DataFrame]:
    return bindata.dfs


@fixture
def flight(bindata: Ardupilot) -> Flight:
    return Flight.from_log(bindata)


@fixture
def primary_core(
    fields: dict[str, pd.DataFrame],
) -> Callable[[float | npt.NDArray[np.float64]], npt.NDArray[np.int64] | int]:
    return bf.primary_core_at_time(fields)


def test_primary_core_at_time_returns_correct_values():
    df = pd.DataFrame(
        {
            "TimeUS": np.array([4.0, 100.0, 2000.0]) * 1e6,
            "Subsys": [24,24,24],
            "ECode": [1.0, 0.0, 1.0],
        }
    )
    primary_core = bf.primary_core_at_time({"ERR": df})
    t = np.array([0.0, 50.0, 150.0, 2500.0])
    expected = np.array([0, 1, 0, 1]).astype(int)

    assert isinstance(primary_core, Callable)
    assert np.array_equal(primary_core(t), expected, equal_nan=True)


@fixture
def xkf1(fields: dict[str, pd.DataFrame], primary_core: Callable) -> bf.XKF1:
    return bf.XKF1.load(fields, primary_core)


def test_xkf1_returns_dataframe_for_primary_core(xkf1: bf.XKF1):
    assert isinstance(xkf1.t, np.ndarray)
    assert isinstance(xkf1.att, g.Quaternion)
    assert isinstance(xkf1.wvel, g.Point)
    assert len(xkf1.t) == len(xkf1.att) == len(xkf1.wvel)


@fixture
def state_data(fields: dict[str, pd.DataFrame], flight: Flight) -> StateData:
    return StateData.parse_fields(fields, flight.origin)


def test_parse_statedata_returns_state_data(state_data: StateData):
    assert isinstance(state_data.att.data, g.Quaternion)
    assert isinstance(state_data.rvel.data, g.Point)
    assert isinstance(state_data.pos.data, g.Point)
    assert isinstance(state_data.vel.data, g.Point)
    assert isinstance(state_data.acc.data, g.Point)


