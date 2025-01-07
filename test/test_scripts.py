from flightdata.scripts import flightline as fl
import geometry as g
from pathlib import Path
import numpy as np
from pytest import approx

def test_get_bin_from_number():
    bin = fl.get_bin_from_number(Path("test/data/script_tests"), 1)
    assert bin.stem == "c6_on_0001"

    bin = fl.get_bin_from_number(Path("test/data/script_tests"), 3)
    assert bin.stem == "center_0003"



def test_flightline_two_bins():
    box, name  = fl.create_flightline(fl.parse_args(["-l", "test/data/script_tests", "-p", "pilot_0004.BIN", "-c", "3"]))
    assert name == "pilot_0004"
    assert isinstance(box.pos, g.GPS)

def test_flightline_c6on():
    box, name  = fl.create_flightline(fl.parse_args(["-l", "test/data/script_tests", "-p", "1"]))
    assert name == "c6_on_0001"

    assert np.degrees(box.heading) == approx(-34.44699010)