from pytest import fixture, mark
from flightdata.bindata import BinData
from json import load
import numpy as np
import pandas as pd
from ardupilot_log_reader import Ardupilot
from flightdata import Flight
from datetime import datetime
from pathlib import Path

@fixture(scope="session")
def bin_parser():
    return Ardupilot.parse(Path(__file__).parent / "data/p23.BIN", types=Flight.ardupilot_types)


@fixture(scope="session")
def web_json():
    with open(Path(__file__).parent / "data/web_bin_parse.json", "r") as f:
        return load(f)


@fixture(scope="session")
def bindata(web_json):
    return BinData.parse_json(web_json)


def test_parse_json(bindata):
    assert "XKF1" in bindata.dfs
    assert isinstance(bindata.XKF1, pd.DataFrame)

    assert isinstance(bindata.PARM, pd.DataFrame)




def test_timestamp_processing(bin_parser, bindata):
    assert datetime.fromtimestamp(bindata.PARM.timestamp.iloc[0]) == datetime.fromtimestamp(
        bin_parser.PARM.timestamp.iloc[0]
    )
    pass

