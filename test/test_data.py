import unittest
from flightdata.fields import Fields
from flightdata.data import Flight
import os
import numpy as np
import pandas as pd
from io import open
from json import load
from geometry import GPS, Point

class TestFlightData(unittest.TestCase):
    def setUp(self):
        self.flight = Flight.from_csv('test/ekfv3_test.csv')

    def test_duration(self):
        self.assertAlmostEqual(self.flight.duration, 601, 0)

    def test_slice(self):
        short_flight = self.flight[100:200]
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
        self.assertAlmostEqual(flight2.duration, flight.duration, 4)
        self.assertAlmostEqual(flight2.zero_time, flight.zero_time, 4)
   
    def test_missing_arsp(self):
        flight = Flight.from_log('test/00000150.BIN')
        self.assertGreater(flight.duration, 500)

    def test_quaternions(self):
        flight = Flight.from_log('test/00000150.BIN')
        quats = flight.read_fields(Fields.QUATERNION)
        self.assertFalse(quats[pd.isna(quats.quaternion_0)==False].empty)

    def test_from_fc_json(self):
        with open("test/test_inputs/manual_F3A_P23_22_08_23_00000055_1.json", "r") as f:
            fc_json = load(f)
        flight = Flight.from_fc_json(fc_json)
        assert isinstance(flight, Flight)
        assert flight.duration > 200
        assert flight.flying_only().duration == flight.duration
        assert flight.read_fields(Fields.POSITION).position_z.max() < -10

        _origin = GPS(fc_json['parameters']['originLat'], fc_json['parameters']['originLng'])
        self.assertEqual(_origin, flight.origin)
        

    def test_unique_identifier(self):
        with open("test/test_inputs/manual_F3A_P21_21_09_24_00000052.json", "r") as f:
            fc_json = load(f)
        flight1 = Flight.from_fc_json(fc_json)
        self.assertIsInstance(flight1.unique_identifier(),str)   

        flight2 = Flight.from_log('test/test_inputs/test_log_00000052.BIN')
        self.assertIsInstance(flight2.unique_identifier(),str)
        print(flight2.unique_identifier())

        self.assertEqual(flight1.unique_identifier(),flight2.unique_identifier())


    def test_frequency(self):
        with open("test/test_inputs/manual_F3A_P21_21_09_24_00000052.json", "r") as f:
            fc_json = load(f)
        flight1 = Flight.from_fc_json(fc_json)

        flight2 = Flight.from_log('test/test_inputs/test_log_00000052.BIN')
        freq1 = flight1.duration / len(flight1.data)
        freq2 = flight2.duration / len(flight2.data)
        self.assertAlmostEqual(freq1, freq2, 5)

    @unittest.skip
    def test_baro(self):
        press = self.flight.read_fields(Fields.PRESSURE)
        temp = self.flight.read_fields(Fields.TEMPERATURE)
        self.assertLess(press.iloc[0,0],  120000)
        self.assertGreater(press.iloc[0,0],  90000)
        #self.assertLess(temp.iloc[0,0], 30)
        #self.assertGreater(temp.iloc[0,0], 0)


    def test_ekfv2(self):
        flight = Flight.from_log("test/xkfv2_log.BIN")
        qs = flight.read_fields(Fields.QUATERNION)
        

        self.assertFalse(qs.isnull().values.all())

    def test_axis_rates(self):
        flight=Flight.from_log("test/test_inputs/test_log_00000052.BIN")

        axis_rates = flight.read_fields(Fields.AXISRATE)

        self.assertFalse(axis_rates.isnull().values.all())


    def test_flying_only(self):
        flt = self.flight.flying_only()
        assert isinstance(flt, Flight)
        assert flt.duration < self.flight.duration
        assert flt[0].position_z < -5


    def test_slice_raw_t(self):
        sli = self.flight.slice_raw_t(slice(100, None, None))
        assert isinstance(sli, Flight)
        assert "time_flight" in sli.data.columns

def test_timestamp():
    fl = Flight.from_log("test/test_inputs/00000129.BIN")
    assert fl.data.time_flight.iloc[0] > 1e6