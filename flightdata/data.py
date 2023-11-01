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
from typing import Self, Union, IO
import numpy as np
import pandas as pd
from .fields import fields, Field
from geometry import GPS, Point, Quaternion, PX
from pathlib import Path
from time import time
from json import load


class Flight(object):
    def __init__(self, data: pd.DataFrame, parameters: list = None, origin=None):
        self.data = data
        self.parameters = parameters
        self.data.index = self.data.index - self.data.index[0]
        self.origin = origin

    def __getattr__(self, name):
        cols = getattr(fields, name)
        if isinstance(cols, Field):
            return self.data[cols.column]
        else:
            return self.data.loc[:, [f.column for f in cols]]

    def __getitem__(self, sli):
        if isinstance(sli, int) or isinstance(sli, float):
            return self.data.iloc[self.data.index.get_loc(sli)]
        else:
            return Flight(self.data.loc[sli], self.parameters, self.origin)

    def __len__(self):
        return len(self.data)

    def slice_raw_t(self, sli):
        return Flight(
            self.data.reset_index(drop=True)
                .set_index('time_actual', drop=False)
                    .loc[sli].set_index("time_flight", drop=False), 
            self.parameters, 
            self.origin
        )
    
    def copy(self):
        return Flight(
            self.data.copy(),
            self.parameters.copy() if self.parameters else None,
            self.origin
        )

    def to_csv(self, filename):
        self.data.to_csv(filename)
        return filename

    @staticmethod
    def from_csv(filename):
        data = pd.read_csv(filename)
        data.index = data['time_flight'].copy()
        data.index.name = 'time_index'
        return Flight(data)

    @staticmethod
    def build_cols(**kwargs):
        df = pd.DataFrame(columns=list(fields.data.keys()))
        for k, v in kwargs.items():
            df[k] = v
        return df.dropna(axis=1, how='all')

    def gps_ready_time(self):
        gps = self.gps
        gps = gps.loc[~(gps==0).all(axis=1)].dropna()
        return gps.iloc[0].name
    
    def imu_ready_time(self):
        qs = Quaternion.from_euler(Point(self.attitude))
        df = qs.transform_point(PX(1)).to_pandas(index=self.data.index)
        att_ready = df.loc[(df.x!=1.0) | (df.y!=0.0) | (df.z!=0.0)].iloc[0].name

        return max(self.gps_ready_time(), att_ready)

    @staticmethod
    def synchronise(fls: list[Self]) -> list[Self]:
        """Take a list of overlapping flights and return a list of flights with
        identical time indexes. All Indexes will be equal to the portion of the first
        flights index that overlaps all the other flights.
        """
        start_t = max([fl.time_actual.iloc[0] for fl in fls])
        end_t = min([fl.time_actual.iloc[-1] for fl in fls])
        if end_t < start_t:
            raise Exception('These flights do not overlap')
        otf = fls[0].slice_raw_t(slice(start_t, end_t, None)).time_flight

        flos = []
        for fl in fls:
            flos.append(Flight(
                pd.merge_asof(
                    otf, 
                    fl.data.reset_index(), 
                    on='time_flight'
                ).set_index('time_index'),
                fl.parameters,
                fl.zero_time,
                fl.origin
            ))

        return flos

    @property
    def duration(self):
        return self.data.tail(1).index.item()


    def flying_only(self, minalt=5, minv=10):
        vs = abs(Point(self.velocity))
        above_ground = self.data.loc[(self.gps_altitude >= minalt) & (vs > minv)]

        return self[above_ground.index[0]:above_ground.index[-1]]


    def unique_identifier(self) -> str:
        """Return a string to identify this flight that is very unlikely to be the same as a different flight

        Returns:
            str: flight identifier
        """
        _ftemp = Flight(self.data.loc[self.data.position_z < -10])
        return "{}_{:.8f}_{:.6f}_{:.6f}".format(len(_ftemp.data), _ftemp.duration, *self.origin.data[0])



    @staticmethod
    def from_log(log_path):
        """Constructor from an ardupilot bin file."""
        from ardupilot_log_reader.reader import Ardupilot

        if isinstance(log_path, Path):
            log_path = str(log_path)
        parser = Ardupilot(log_path, types=[
            'XKF1', 'XKF2', 'NKF1', 'NKF2', 
            'POS', 'ATT', 'ACC', 'GYRO', 'IMU', 
            'ARSP', 'GPS', 'RCIN', 'RCOU', 'BARO', 'MODE', 
            'RPM', 'MAG', 'BAT', 'BAT2', 'VEL', 'ORGN'])

        dfs = []

        if parser.parms['AHRS_EKF_TYPE'] == 2:
            ekf1 = 'NKF1'
            ekf2 = 'NKF2'
        else:
            ekf1 = 'XKF1'
            ekf2 = 'XKF2'

        if ekf1 in parser.dfs:
            ekf1 = parser.dfs[ekf1]
            ekf1 = ekf1.loc[ekf1.C==0]
        else:
            ekf1 = None

        if ekf2 in parser.dfs:
            ekf2 = parser.dfs[ekf2]
            ekf2 = ekf2.loc[ekf2.C==0]
        else:
            ekf2 = None


        if 'ATT' in parser.dfs:       
            dfs.append(Flight.build_cols(
                time_actual = parser.ATT.timestamp,
                time_flight = parser.ATT.TimeUS / 1e6,
                attitude_roll = np.radians(parser.ATT.Roll),
                attitude_pitch = np.radians(parser.ATT.Pitch),
                attitude_yaw = np.radians(parser.ATT.Yaw),
            ))
                
        if 'POS' in parser.dfs:
            dfs.append(Flight.build_cols(
                time_actual = parser.POS.timestamp,
                gps_latitude = parser.POS.Lat,
                gps_longitude = parser.POS.Lng,
                gps_altitude = parser.POS.Alt
            ))
        
        if not ekf1 is None: 
            dfs.append(Flight.build_cols(
                time_actual = ekf1.timestamp,
                position_N = ekf1.PN,
                position_E = ekf1.PE,
                position_D = ekf1.PD,
                velocity_N = ekf1.VN,
                velocity_E = ekf1.VE,
                velocity_D = ekf1.VD,
            ))


        if not ekf2 is None:
            dfs.append(Flight.build_cols(
                time_actual = ekf2.timestamp,
                wind_N = ekf2.VWN,
                wind_E = ekf2.VWE,
            ))
        
        if 'IMU' in parser.dfs:

            if not ekf1 is None:  # get gyro biases
                gyro_bias_x = np.radians(ekf1.GX) / 100
                gyro_bias_y = np.radians(ekf1.GY) / 100
                gyro_bias_z = np.radians(ekf1.GZ) / 100
            else:
                gyro_bias_x = 0
                gyro_bias_y = 0
                gyro_bias_z = 0

            if not ekf2 is None:
                acc_bias_x = ekf2.AX / 100
                acc_bias_y = ekf2.AY / 100
                acc_bias_z = ekf2.AZ / 100
            else:
                acc_bias_x = 0
                acc_bias_y = 0
                acc_bias_z = 0

            dfs.append(Flight.build_cols(
                time_actual = parser.IMU.timestamp,
                acceleration_x = parser.IMU.AccX + acc_bias_x,
                acceleration_y = parser.IMU.AccY + acc_bias_y,
                acceleration_z = parser.IMU.AccZ + acc_bias_z,
                axisrate_roll = parser.IMU.GyrX + gyro_bias_x,
                axisrate_pitch = parser.IMU.GyrY + gyro_bias_y,
                axisrate_yaw = parser.IMU.GyrZ + gyro_bias_z,
            ))
        
        if 'MAG' in parser.dfs:
            dfs.append(Flight.build_cols(
                time_actual = parser.MAG.timestamp,
                magnetometer_x = parser.MAG.MagX,
                magnetometer_y = parser.MAG.MagY,
                magnetometer_z = parser.MAG.MagZ
            ))
        
        if 'BARO' in parser.dfs:
            dfs.append(Flight.build_cols(
                time_actual = parser.BARO.timestamp,
                air_pressure = parser.BARO.Press,
                air_temperature = parser.BARO.Temp,
                air_altitude = parser.BARO.Alt,
            ))

        if 'RCIN' in parser.dfs:
            dfs.append(Flight.build_cols(
                time_actual = parser.RCIN.timestamp,
                **{f'rcin_{i}': parser.RCIN[f'C{i}'] for i in range(20) if f'C{i}' in parser.RCIN.columns}
            ))
        
        if 'RCOU' in parser.dfs:
            dfs.append(Flight.build_cols(
                time_actual = parser.RCOU.timestamp,
                **{f'rcout_{i}': parser.RCOU[f'C{i}'] for i in range(20) if f'C{i}' in parser.RCOU.columns}
            ))

        if 'MODE' in parser.dfs:
            dfs.append(Flight.build_cols(
                time_actual = parser.MODE.timestamp,
                flightmode_a = parser.MODE.Mode,
                flightmode_b = parser.MODE.ModeNum,
                flightmode_c = parser.MODE.Rsn,
            ))
        
        if 'BAT' in parser.dfs:
            dfs.append(Flight.build_cols(
                time_actual = parser.BAT.timestamp,
                **{'motor_voltage{i}': parser.BAT[f'Volt{i}'] for i in range(8) if f'Volt{i}' in parser.BAT.columns},
                **{'motor_current{i}': parser.BAT[f'Curr{i}'] for i in range(8) if f'Curr{i}' in parser.BAT.columns},
            ))

        if 'RPM' in parser.dfs:
            dfs.append(Flight.build_cols(
                time_actual = parser.RPM.timestamp,
                **{'motor_rpm{i}': parser.RPM[f'rpm{i}'] for i in range(8) if f'rpm{i}' in parser.RPM.columns},
            ))

        if 'ORGN' in parser.dfs:
            origin = GPS(parser.ORGN.Lat[0], parser.ORGN.Lng[0])
        else:
            origin = None
            #GPS(*self.read_fields(Fields.GLOBALPOSITION).loc[self.gps_ready_time()])

        dfout = dfs[0]

        for df in dfs[1:]:
            dfout = pd.merge_asof(dfout, df, on='time_actual')
        
        return Flight(dfout.set_index('time_flight', drop=False), parser.parms, origin)

    @staticmethod
    def from_fc_json(fc_json: Union[str, dict, IO]):
        
        if isinstance(fc_json, str):
            with open(fc_json, "r") as f:
                fc_json = load(f)
        elif isinstance(fc_json, IO):
            fc_json = load(f)
        
        df = pd.DataFrame.from_dict(fc_json['data'], dtype=float)
                
        df = Flight.build_cols(
            time_actual = df['time']/1e6 + int(time()),
            time_flight = df['time']/1e6,
            attitude_roll = np.radians(df['r']),
            attitude_pitch = np.radians(df['p']),
            attitude_yaw = np.radians(df['yw']),
            gps_latitude = df['N'],
            gps_longitude = df['E'],
            gps_altitude = df['D'],
            velocity_N = df['VN'],
            velocity_E = df['VE'],
            velocity_D = df['VD'],
            wind_N = df['wN'] if 'wN' in df.columns else None,
            wind_E = df['wE'] if 'wE' in df.columns else None,
        )
        
        origin = GPS(fc_json['parameters']['originLat'], fc_json['parameters']['originLng'])
        return Flight(df.set_index('time_flight', drop=False), None, origin)
