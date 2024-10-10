from flightdata import Flight, Origin, BinData
from json import load
from pytest import fixture, approx, mark
import numpy as np
import pandas as pd
from ardupilot_log_reader import Ardupilot
from .conftest import fcjson

@fixture(scope='session')
def parser():
    return Ardupilot.parse('test/data/p23.BIN',types=Flight.ardupilot_types)

@fixture(scope='session')
def fl(parser):
    return Flight.from_log(parser)


def test_duration(fl):
    assert fl.duration == approx(687, rel=1e-3)

def test_slice(fl):
    short_flight = fl[100:200]
    assert short_flight.duration == approx(100, 0.01)

def test_to_from_dict(fl):
    data = fl.to_dict()
    fl2 = Flight.from_dict(data)
    assert fl == fl2
    
def test_from_fc_json(fcjson):
    assert isinstance(fcjson, Flight)
    assert fcjson.duration > 200
    assert fcjson.position_D.max() < -10
  
@mark.skip
def test_unique_identifier():
    with open("test/test_inputs/manual_F3A_P21_21_09_24_00000052.json", "r") as f:
        fc_json = load(f)
    flight1 = Flight.from_fc_json(fc_json)  

    flight2 = Flight.from_log('test/test_inputs/test_log_00000052.BIN')
    
    assert flight1.unique_identifier() == flight2.unique_identifier()

@mark.skip
def test_baro(fl):
    press = fl.air_pressure
    assert press.iloc[0,0] <  120000
    assert press.iloc[0,0] >  90000

def test_flying_only(fl: Flight):
    flt = fl.flying_only()
    assert isinstance(flt, Flight)
    assert flt.duration < fl.duration
    assert flt[0].gps_altitude > 5

def test_slice_raw_t(fl: Flight):
    sli = fl.slice_raw_t(slice(100, None, None))
    assert isinstance(sli, Flight)
    assert "time_flight" in sli.data.columns

def test_origin(fl: Flight):
    assert isinstance(fl.origin, Origin)


@fixture(scope='session')
def vtol_hover():
    return Flight.from_json('test/data/vtol_hover.json')

def test_flightmode_split(vtol_hover: Flight):
    smodes = vtol_hover.split_modes()
    assert isinstance(smodes, dict)
    assert isinstance(smodes['QHOVER'], list)
    assert isinstance(smodes['QHOVER'][0], Flight)
    
def _fft(col: pd.Series):
    from scipy.fft import fft, fftfreq
    ts = col.index
    N = len(col)
    T = (ts[-1] - ts[0]) / N

    yf = fft(col.to_numpy())
    xf = fftfreq(N, T)[:N//2]

    return xf, 2.0/N * np.abs(yf[0:N//2])

def test_butter_filter(fl: Flight):
    filtered = fl.butter_filter(1,5)

    x, y = _fft(fl.acceleration_x)
    xf, yf = _fft(filtered.acceleration_x)

    assert np.all(yf[xf>1]<0.025)

def test_remove_time_flutter(fl: Flight):
    flf = fl.remove_time_flutter()
    assert np.gradient(np.gradient(flf.data.index)) == approx(0)


def test_get_parameter_attr(fl: Flight):
    assert fl.AHRS_EKF_TYPE.iloc[0].value == 3

def test_make_param_labels(fl: Flight):
    col = fl.make_param_labels('AHRS_EKF_TYPE')

    assert len(col) == len(fl)  
    assert np.all(col.loc[~np.isnan(col)] == 3)


    col = fl.make_param_labels('AHRS_EKF_TYPE', 'Test')

    assert np.all(col.loc[col!=''] == 'Test3.0')



@fixture(scope='session')
def bindata():
    return BinData.parse_json(load(open('test/data/web_bin_parse.json', 'r')))

def test_from_bindata(bindata: BinData):
    fl = Flight.from_log(bindata)
    assert isinstance(fl, Flight)

    

