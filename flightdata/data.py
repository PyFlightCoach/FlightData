"""
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""
from typing import List, Dict, Union
import numpy as np
import pandas as pd
from importlib.util import find_spec
from enum import Enum
from .fields import Fields, CIDTypes
from .field_mapping import get_ardupilot_mapping
from .field_mapping.fc_json_2_1 import fc_json_2_1_io_info
from geometry import GPS, Point, Quaternion, PX
from pathlib import Path


fdict = Fields.to_dict()

class Flight(object):
    def __init__(self, data, parameters: List = None, zero_time_offset: float = 0):
        self.data = data
        self.parameters = parameters
        self.zero_time = self.data.index[0] + zero_time_offset
        self.data.index = self.data.index - self.data.index[0]
        #self.data.index = np.round(self.data.index,3)
        self._origin = None

    def flying_only(self, minalt=5, minv=10):
        vs = abs(Point(self.read_fields(Fields.VELOCITY)))
        above_ground = self.data.loc[(self.data.position_z <= -minalt) & (vs > minv)]
        return self[above_ground.index[0]:above_ground.index[-1]]

    def __getattr__(self, name):
        if name in Fields.all_names:
            return self.data[name]
        if name in fdict.keys():
            return self.data[fdict[name]]

    def __getitem__(self, sli):
        if isinstance(sli, int) or isinstance(sli, float):
            return self.data.iloc[self.data.index.get_loc(sli)]
        else:
            return Flight(self.data.loc[sli], self.parameters, self.zero_time)

    def slice_raw_t(self, sli):
        
        return Flight(
            self.data.set_index("time_flight", drop=False).loc[sli].set_index("time_index"), 
            self.parameters, 
            self.zero_time
        )
        
    
    def to_csv(self, filename):
        self.data.to_csv(filename)
        return filename

    @staticmethod
    def from_csv(filename):
        data = pd.read_csv(filename)
        data.index = data[Fields.TIME.names[0]].copy()
        data.index.name = 'time_index'
        return Flight(data)

    @staticmethod
    def from_log(log_path, *args):
        """Constructor from an ardupilot bin file.
            fields are renamed and units converted to the tool fields defined in ./fields.py
            The input fields, read from the log are specified in ./mapping 

            Args:
                log_path (str): [description]
        """
        from ardupilot_log_reader.reader import Ardupilot

        _field_request = ['POS', 'ATT', 'ACC', 'GYRO', 'IMU', 'ARSP', 'GPS', 'RCIN', 'RCOU', 'BARO', 'MODE', 'RPM', 'MAG', 'BAT', 'BAT2']
        if isinstance(log_path, Path):
            log_path = str(log_path)
        _parser = Ardupilot(log_path, types=_field_request+list(args))#,zero_time_base=True)
        fulldf = _parser.join_logs(_field_request)
        
        return Flight.convert_df(
            fulldf,
            get_ardupilot_mapping(_parser.parms['AHRS_EKF_TYPE']),
            _parser.parms
        )

    @staticmethod
    def convert_df(fulldf, ioinfo, parms):
        # expand the dataframe to include all the columns listed in the io_info instance
        input_data = fulldf.get(
            list(
                set(fulldf.columns.to_list()) & set(ioinfo.io_names)
        ))

        # Generate a reordered io instance to match the columns in the dataframe
        _fewer_io_info = ioinfo.subset(input_data.columns.to_list())

        _data = input_data * _fewer_io_info.factors_to_base  # do the unit conversion
        _data.columns = _fewer_io_info.base_names  # rename the columns

        # add the missing tool columns
        missing_cols = pd.DataFrame(
            columns=list(set(Fields.all_names) - set(_data.columns.to_list())) + [Fields.TIME.names[0]]
        )
        output_data = _data.merge(missing_cols, on=Fields.TIME.names[0], how='left')

        output_data = output_data.set_index(Fields.TIME.names[0], drop=False)
        output_data.index.name = 'time_index'

        return Flight(output_data, parms)

    @staticmethod
    def from_fc_json(fc_json):
        df = pd.DataFrame.from_dict(fc_json['data'])
        df.insert(0, "timestamp", df['time'] * 1E-6)
        
        flight = Flight.convert_df(df, fc_json_2_1_io_info, fc_json['parameters'])
        flight._origin = GPS(fc_json['parameters']['originLat'], fc_json['parameters']['originLng'])
        return flight

    @property
    def duration(self):
        return self.data.tail(1).index.item()

    def read_row_by_id(self, names, index):
        return list(map(self.data.iloc[index].to_dict().get, names))

    def read_closest(self, names, time):
        """Get the row closest to the requested time.

        :param names: list of columns to return
        :param time: desired time in microseconds
        :return: dict[column names, values]
        """
        return self.read_row_by_id(names, self.data.index.get_loc(time, method='nearest'))

    @property
    def column_names(self):
        return self.data.columns.to_list()

    def read_fields(self, fields):
        try:
            return self.data[Fields.some_names(fields)]
        except KeyError:
            return pd.DataFrame()

    def read_numpy(self, fields):
        return self.read_fields(fields).to_numpy().T

    def read_tuples(self, fields):
        return tuple(self.read_numpy(fields))

    def read_field_tuples(self, fields):
        return tuple(self.read_numpy(fields))

    @property
    def origin(self) -> GPS:
        """the latitude and longitude of the origin (first pos in log)

        Returns:
            dict: origin GPS
        """
        if self._origin is None:
            self._origin = GPS(*self.read_fields(Fields.GLOBALPOSITION).loc[self.gps_ready_time()])
        return self._origin

    def gps_ready_time(self):
        gps = self.read_fields(Fields.GLOBALPOSITION)
        gps = gps.loc[~(gps==0).all(axis=1)].dropna()
        return gps.iloc[0].name

    def imu_ready_time(self):
        qs = Quaternion.from_euler(Point(self.read_fields(Fields.ATTITUDE)))
        df = qs.transform_point(PX(1)).to_pandas(index=self.data.index)
        att_ready = df.loc[(df.x!=1.0) | (df.y!=0.0) | (df.z!=0.0)].iloc[0].name

        return max(self.gps_ready_time(), att_ready)

    def unique_identifier(self) -> str:
        """Return a string to identify this flight that is very unlikely to be the same as a different flight

        Returns:
            str: flight identifier
        """
        _ftemp = Flight(self.data.loc[self.data.position_z < -10])
        return "{}_{:.8f}_{:.6f}_{:.6f}".format(len(_ftemp.data), _ftemp.duration, *self.origin.data[0])

    def describe(self):
        info = dict(
            duration = self.duration,
            origin_gps = self.origin.to_dict(),
            last_gps_ = GPS(*self.read_fields(Fields.GLOBALPOSITION).iloc[-1]).to_dict(),
            average_gps = GPS(*self.read_fields(Fields.GLOBALPOSITION).mean()).to_dict(),
            bb_max = Point(self.read_fields(Fields.POSITION)).max().to_dict(),
            bb_min = Point(self.read_fields(Fields.POSITION)).min().to_dict(),
        )

        return pd.json_normalize(info, sep='_')

    def has_pitot(self):
        return not np.all(self.read_fields(Fields.AIRSPEED).iloc[:,0] == 0)