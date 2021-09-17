import unittest
from flightdata.fields import Fields
from flightdata.data import Flight
import os
import numpy as np
import pandas as pd
from io import open
from json import load
from geometry import GPSPosition, Points

class TestFlightData(unittest.TestCase):
    def setUp(self):
        self.flight = Flight.from_csv('test/ekfv3_test.csv')

    def test_duration(self):
        self.assertAlmostEqual(self.flight.duration, 601, 0)

    def test_slice(self):
        short_flight = self.flight.subset(100, 200)
        self.assertAlmostEqual(short_flight.duration, 100, 0)

    def test_read_tuples(self):
        vals = self.flight.read_field_tuples(Fields.TIME)
        self.assertAlmostEqual(
            max(vals[0]), 601 + self.flight.zero_time, 0)
        self.assertEqual(len(vals), 2)
        vals1 = self.flight.read_field_tuples(Fields.GPSSATCOUNT)
        self.assertEqual(len(vals1), 1)
    
    def test_to_from_csv(self):
        flight = Flight.from_log('test/ekfv3_test.BIN')
        flight.to_csv('temp.csv')
        flight2 = Flight.from_csv('temp.csv')
        os.remove('temp.csv')
        self.assertEqual(flight2.duration, flight.duration)
        self.assertEqual(flight2.zero_time, flight.zero_time)
   
    def test_missing_arsp(self):
        flight = Flight.from_log('test/00000150.BIN')
        self.assertGreater(flight.duration, 500)

    def test_quaternions(self):
        flight = Flight.from_log('test/00000150.BIN')
        quats = flight.read_fields(Fields.QUATERNION)
        self.assertFalse(quats[pd.isna(quats.quaternion_0)==False].empty)

    def test_from_fc_json(self):
        with open("test/fc_json.json", "r") as f:
            fc_json = load(f)
        flight = Flight.from_fc_json(fc_json)
        self.assertEqual(len(flight.read_fields(Fields.POSITION)), 11205)
        self.assertAlmostEqual(flight.duration, 448.1591)
        self.assertIsInstance(flight.origin(), dict)
        gp = flight.read_fields(Fields.GLOBALPOSITION)

        self.assertFalse(gp[pd.isna(gp)==False].empty)
        pos = Points.from_pandas(flight.read_fields(Fields.POSITION))
        _origin = GPSPosition(fc_json['parameters']['originLat'], fc_json['parameters']['originLng']).offset(pos[0])
        self.assertEqual(_origin, GPSPosition(**flight.origin()))

    def test_unique_identifier(self):
        with open("test/fc_json.json", "r") as f:
            fc_json = load(f)
        flight1 = Flight.from_fc_json(fc_json)
        self.assertIsInstance(flight1.unique_identifier(),str)   
        flight1.to_csv('temp.csv')
        flight2 = Flight.from_csv('temp.csv')
        self.assertIsInstance(flight2.unique_identifier(),str)
        print(flight2.unique_identifier())
        self.assertEqual(flight1.unique_identifier(),flight2.unique_identifier())

