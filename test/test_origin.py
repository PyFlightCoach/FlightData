import unittest
from flightdata import Origin
from geometry import GPS
from pytest import approx, mark
from .conftest import flight


def test_to_dict(flight):
    origin = Origin.from_initial(flight)
    di = origin.to_dict()
    assert di["name"] ==  "origin"
    assert di["pos"]['lat'] == origin.pos.lat[0]

@mark.skip('writes to file rather than returns string now')
def test_to_f3azone(origin):
    zone_string = origin.to_f3a_zone()
    lines = zone_string.split("\n")
    
    assert lines[0] == "Emailed box data for F3A Zone Pro - please DON'T modify!"
    assert lines[1] == origin.name

    pilot = GPS(float(lines[2]), float(lines[3]), 0)

    centre = GPS(float(lines[4]), float(lines[5]), 0)

    origin_copy = Origin.from_points("tem", pilot, centre)

    assert origin_copy.heading == approx(origin.heading)
    assert float(lines[6]) == 120





if __name__ == "__main__":
    unittest.main()
