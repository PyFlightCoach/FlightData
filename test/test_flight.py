from flightdata import Flight, Origin
from io import open
from json import load
from pytest import fixture, approx, mark
import numpy as np
import pandas as pd
from ardupilot_log_reader import Ardupilot


@fixture(scope='session')
def parser():
    return Ardupilot('test/test_inputs/00000137.BIN',
                     types=Flight.ardupilot_types)

@fixture(scope='session')
def fl():
    return Flight.from_log('test/test_inputs/00000137.BIN')

@fixture(scope='session')
def fcj():
    return Flight.from_fc_json('test/test_inputs/00000137.json')

def test_duration(fl):
    assert fl.duration == approx(685, rel=1e-3)

def test_slice(fl):
    short_flight = fl[100:200]
    assert short_flight.duration == approx(100, 0.01)

def test_to_from_dict(fl):
    data = fl.to_dict()
    fl2 = Flight.from_dict(data)
    assert fl == fl2
    assert fl2.parameters == approx(fl.parameters)

def test_from_fc_json(fcj):
    assert isinstance(fcj, Flight)
    assert fcj.duration > 200
    assert fcj.position_D.max() < -10
  
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
