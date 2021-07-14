import unittest
from flightdata.fields import Fields
from flightdata.data import Flight
import os

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

    @unittest.skip("reading log from bin takes a bit longer")
    def test_to_from_csv(self):
        flight = Flight.from_log('test/ekfv3_test.BIN')
        flight.to_csv('temp.csv')
        flight2 = Flight.from_csv('temp.csv')
        os.remove('temp.csv')
        self.assertEqual(flight2.duration, flight.duration)
        self.assertEqual(flight2.zero_time, flight.zero_time)

    @unittest.skip("reading log from bin takes a bit longer")
    def test_missing_arsp(self):
        flight = Flight.from_log('test/00000150.BIN')
        self.assertGreater(flight.duration, 500)
