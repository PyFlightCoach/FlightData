import unittest
from flightdata import Flight, Box
from geometry import GPS, Point, Quaternion, PX, PY, PZ, P0, Euler
from geometry.testing import assert_almost_equal
import numpy as np
from .conftest import flight, box
from pytest import approx, fixture


def test_box_from_initial(flight):
    box = Box.from_initial(flight)
    
    assert box.pilot_position.data == approx(GPS(51.6417309, -2.5256134).data)

    assert np.degrees(box.heading) == approx(-67.02, rel=1e-2)


def test_to_dict(flight):
    box = Box.from_initial(flight)
    di = box.to_dict()
    assert di["name"] ==  "origin"
    assert di["pilot_position"]['lat'] == box.pilot_position.lat[0]


def test_from_box(flight, box):
    fl = FlightLine.from_box(box, flight.origin)

    np.testing.assert_array_almost_equal(
        np.sign(fl.transform_to.rotate(Point(1.0, 0.0, 0.0)).data)[0],
        [-1,-1,0]
    )   # My box faces south east ish, so world north should be -x and +y in contest frame

    assert abs(fl.transform_to.rotate(Point(1.0, 0.0, 0.0))) < 3   # Translation should be small, because I turn on close to the pilot position.


@unittest.skip
def test_from_box_true_north():
    home = GPS(39, -105)

    # Box heading specified in radians from North (clockwise)
    box = Box('north', home, 0)

    fl = FlightLine.from_box(box, home)

    oneMeterNorth_NED = Point(1, 0, 0)
    
    np.testing.assert_array_almost_equal(
        fl.transform_from.rotate(oneMeterNorth_NED).to_list(),
        oneMeterNorth_NED.to_list()
    )  

    np.testing.assert_array_almost_equal(
        fl.transform_to.rotate(oneMeterNorth_NED).to_list(),
        oneMeterNorth_NED.to_list()
    )   # Box faces due North, so NED (1,0,0) should be (0,1,0) in ENU world frame

    # lat/lon to x/y is problematic over large distances; need to work with x/y displacements
    # relative to home to avoid issues with accuracy
    # 0.001 degree of latitude at 39N is 111.12 meters: http://www.csgnetwork.com/gpsdistcalc.html
    north_of_home = GPS(39.001, -105)
    deltaPos = home.__sub__(north_of_home)
    np.testing.assert_array_almost_equal(
        deltaPos.to_list(),
        [111.12, 0, 0],
        0
    )


def test_flightline_from_initial(whole_flight):
    flightline = FlightLine.from_initial_position(whole_flight)
    assert flightline.world.origin.x == approx(0)
    assert flightline.world.origin.y == approx(0)


def test_transform_to(flight):
    flightline = FlightLine.from_initial_position(flight)

    npoint = flightline.transform_to.rotate(Point(1, 0, 0))
    assert npoint.x == approx(flightline.contest.x_axis.x)


def test_transform_from(flight, whole_flight):
    flightline = FlightLine.from_initial_position(whole_flight)

    npoint = flightline.transform_to.rotate(PX())
    assert npoint.data == approx(flightline.contest.x_axis.data, 1e-4)

def test_flightline_headings(flight):
    pilotNorth_ENU = Point(0, 1, 1)
    home = flight.origin
    
    ned = Point(flight.read_fields(Fields.POSITION))
    rned = Quaternion.from_euler(Point(flight.read_fields(Fields.ATTITUDE)))

    #North Facing
    enu_flightline =FlightLine.from_box(Box('test',home,0.0),home)
    enu = enu_flightline.transform_to.point(ned)
    renu = enu_flightline.transform_to.rotate(rned)  # TODO think of a test for this
    np.testing.assert_array_almost_equal(ned.x, enu.y)
    np.testing.assert_array_almost_equal(ned.y, enu.x)
    np.testing.assert_array_almost_equal(ned.z, -enu.z)

    pilotNorth_NED = Point(1, 0, -1)
    boxNorth = enu_flightline.transform_to.point(pilotNorth_NED)
    np.testing.assert_array_almost_equal(pilotNorth_ENU.data, boxNorth.data)

    #South Facing
    wsu_flightline =FlightLine.from_box(Box('test',home,np.pi),home)
    wsu = wsu_flightline.transform_to.point(ned)
    rwsu = wsu_flightline.transform_to.rotate(rned)  # TODO think of a test for this
    np.testing.assert_array_almost_equal(ned.x, -wsu.y)
    np.testing.assert_array_almost_equal(ned.y, -wsu.x)
    np.testing.assert_array_almost_equal(ned.z, -wsu.z)

    pilotNorth_NED = Point(-1, 0, -1)
    boxNorth = wsu_flightline.transform_to.point(pilotNorth_NED)
    np.testing.assert_array_almost_equal(pilotNorth_ENU.data, boxNorth.data)

    #West Facing
    nwu_flightline =FlightLine.from_box(Box('test',home,-np.pi/2),home)
    nwu = nwu_flightline.transform_to.point(ned)
    rnwu = nwu_flightline.transform_to.rotate(rned)  # TODO think of a test for this
    np.testing.assert_array_almost_equal(ned.x, nwu.x)
    np.testing.assert_array_almost_equal(ned.y, -nwu.y)
    np.testing.assert_array_almost_equal(ned.z, -nwu.z)

    pilotNorth_NED = Point(0, -1, -1)
    boxNorth = nwu_flightline.transform_to.point(pilotNorth_NED)
    np.testing.assert_array_almost_equal(pilotNorth_ENU.data, boxNorth.data)



def test_transform_from_to(box, flight):
    fl = FlightLine.from_box(box, flight.origin)
    ned = Point(flight.read_fields(Fields.POSITION))
    np.testing.assert_array_almost_equal(
        ned.data,
        fl.transform_to.rotate(fl.transform_from.rotate(ned)).data
    )
    rned = Quaternion.from_euler(Point(flight.read_fields(Fields.ATTITUDE)))
    np.testing.assert_array_almost_equal(
        rned.data,
        fl.transform_from.rotate(fl.transform_to.rotate(rned)).data
    )


def test_to_f3azone(box):
    zone_string = box.to_f3a_zone()
    lines = zone_string.split("\n")
    
    assert lines[0] == "Emailed box data for F3A Zone Pro - please DON'T modify!"
    assert lines[1] == box.name

    pilot = GPS(float(lines[2]), float(lines[3]))

    centre = GPS(float(lines[4]), float(lines[5]))

    box_copy = Box.from_points("tem", pilot, centre)

    assert box_copy.heading == approx(box.heading)
    assert float(lines[6]) == 120




def test_from_f3a_zone():
    box = Box.from_f3a_zone("tests/test_inputs/tl_2_box.f3a")
    assert box.pilot_position == GPS(52.5422769, -1.6318313)
    assert np.degrees(box.heading) == approx(75.281, 1e-3)



def test_box_gps_to_point_north():
    home = GPS(39, -105)
    tests = Point.concatenate([
        PX(10),
        PY(10),
        Point(10, 10, 0)
    ])
    p = home.offset(tests)

    # Box heading specified in radians from North (clockwise)
    box = Box('north', home, 0)

    assert_almost_equal(box.gps_to_point(p), tests)

def test_box_gps_to_point_not_north():
    home = GPS(39, -105)
    tests = Point.concatenate([
        PX(10),
        PY(10),
        Point(10, 10, 0)
    ])
    rot = np.degrees(75)
    p = home.offset(tests)

    # Box heading specified in radians from North (clockwise)
    box = Box('north', home, rot)

    assert_almost_equal(box.gps_to_point(p), Euler(0, 0, rot).transform_point(tests))





if __name__ == "__main__":
    unittest.main()
