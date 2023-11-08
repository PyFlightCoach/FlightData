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
from geometry.testing import assert_almost_equal
from pathlib import Path
from time import time
from json import load, dump
from ardupilot_log_reader.reader import Ardupilot


class Flight(object):
    ardupilot_types = [
        'XKF1', 'XKF2', 'NKF1', 'NKF2', 
        'POS', 'ATT', 'ACC', 'GYRO', 'IMU', 
        'ARSP', 'GPS', 'RCIN', 'RCOU', 'BARO', 'MODE', 
        'RPM', 'MAG', 'BAT', 'BAT2', 'VEL', 'ORGN']
    
    def __init__(self, data: pd.DataFrame, parameters: list = None, origin: GPS = None, primary_pos_source='gps'):
        self.data = data
        self.parameters = parameters
        self.data.index = self.data.index - self.data.index[0]
        self.data.index.name = 'time_index'
        self.origin = origin
        self.primary_pos_source = primary_pos_source
        
    def __getattr__(self, name):
        cols = getattr(fields, name)
        try:
            if isinstance(cols, Field):
                return self.data[cols.column]
            else:
                return self.data.loc[:, [f.column for f in cols]]
        except KeyError:
            if isinstance(cols, Field):
                cols = [cols]
            return pd.DataFrame(data=np.empty((len(self), len(cols))),columns=[f.column for f in cols])
        
    def contains(self, name: Union[str, list[str]]):
        cols = getattr(fields, name)
        if isinstance(cols, Field):
            return name in self.data.columns
        else:
            return [f.column in self.data.columns for f in cols]
                
    def __getitem__(self, sli):
        if isinstance(sli, int) or isinstance(sli, float):
            return self.data.iloc[self.data.index.get_loc(sli)]
        else:
            return Flight(self.data.loc[sli], self.parameters)

    def __len__(self):
        return len(self.data)

    def slice_raw_t(self, sli):
        return Flight(
            self.data.reset_index(drop=True)
                .set_index('time_actual', drop=False)
                    .loc[sli].set_index("time_flight", drop=False), 
            self.parameters
        )
    
    def copy(self):
        return Flight(
            self.data.copy(),
            self.parameters.copy() if self.parameters else None,
            self.origin.copy(),
            self.primary_pos_source
        )

    def to_dict(self):
        return {
            'data': self.data.to_dict('list'),
            'parameters': self.parameters,
            'origin': self.origin.to_dict(),
            'primary_pos_source': self.primary_pos_source
        }
    
    @staticmethod
    def from_dict(data: dict):
        return Flight(
            data=pd.DataFrame.from_dict(data['data']).set_index('time_flight', drop=False),
            parameters=data['parameters'],
            origin=GPS.from_dict(data['origin']),
            primary_pos_source=data['primary_pos_source']
        )
    
    def to_json(self, file: str) -> str:
        with open(file, 'w') as f:
            dump(self.to_dict(), f)
        return file

    @staticmethod
    def from_json(file: str) -> Self:
        with open(file, 'r') as f:
            return Flight.from_dict(load(f))

    @staticmethod
    def build_cols(**kwargs):
        df = pd.DataFrame(columns=list(fields.data.keys()))
        for k, v in kwargs.items():
            df[k] = v
        return df.dropna(axis=1, how='all')
    
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
        otf = fls[0].slice_raw_t(slice(start_t, end_t, None)).time_actual

        flos = []
        for fl in fls:
            flos.append(Flight(
                pd.merge_asof(
                    otf, 
                    fl.data.reset_index(), 
                    on='time_actual'
                ).set_index('time_index', drop=False),
                fl.parameters
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

    def __eq__(self, other):
        try:
            pd.testing.assert_frame_equal(self.data, other.data)
            assert_almost_equal(self.origin, other.origin)
            return True
        except:
            return False
        
    @staticmethod
    def from_log(log:Union[Ardupilot, str]):
        """Constructor from an ardupilot bin file."""

        if isinstance(log, Path):
            log = str(log)
        if isinstance(log, str):
            parser = Ardupilot(log, types=Flight.ardupilot_types)
        else:
            parser = log

        if parser.parms['AHRS_EKF_TYPE'] == 2:
            ekf1 = 'NKF1'
            ekf2 = 'NKF2'
        else:
            ekf1 = 'XKF1'
            ekf2 = 'XKF2'

        if ekf1 in parser.dfs:
            ekf1 = parser.dfs[ekf1]
            if 'C' in ekf1.columns:
                ekf1 = ekf1.loc[ekf1.C==0]
        else:
            ekf1 = None

        if ekf2 in parser.dfs:
            ekf2 = parser.dfs[ekf2]
            if 'C' in ekf2.columns:
                ekf2 = ekf2.loc[ekf2.C==0]
        else:
            ekf2 = None

        dfs = []

        if 'ATT' in parser.dfs:
            att=parser.ATT.iloc[1:, :]
            dfs.append(Flight.build_cols(
                time_actual = att.timestamp,
                time_flight = att.TimeUS / 1e6,
                attitude_roll = np.radians(att.Roll),
                attitude_pitch = np.radians(att.Pitch),
                attitude_yaw = np.radians(att.Yaw),
            ))
        ppsorce = 'gps'
        if 'POS' in parser.dfs:
            dfs.append(Flight.build_cols(
                time_actual = parser.POS.timestamp,
                gps_latitude = parser.POS.Lat,
                gps_longitude = parser.POS.Lng,
                gps_altitude = parser.POS.Alt
            ))
        else:
            ppsorce = 'position'
            
        
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
            imu = parser.IMU
            if 'I' in imu:
                imu = imu.loc[imu.I==0, :]
            
            if not ekf1 is None:
                imu = pd.merge_asof(imu, ekf1, on='timestamp', direction='nearest')
                imu['GyrX'] = imu.GyrX + np.radians(imu.GX) / 100
                imu['GyrY'] = imu.GyrY + np.radians(imu.GY) / 100
                imu['GyrZ'] = imu.GyrZ + np.radians(imu.GZ) / 100

            if not ekf2 is None:
                imu = pd.merge_asof(imu, ekf2, on='timestamp', direction='nearest')
                imu['AccX'] = imu.AccX + imu.AX / 100
                imu['AccY'] = imu.AccY + imu.AY / 100
                imu['AccZ'] = imu.AccZ + imu.AZ / 100

            dfs.append(Flight.build_cols(
                time_actual = imu.timestamp,
                acceleration_x = imu.AccX,
                acceleration_y = imu.AccY,
                acceleration_z = imu.AccZ,
                axisrate_roll = imu.GyrX,
                axisrate_pitch = imu.GyrY,
                axisrate_yaw = imu.GyrZ,
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

        
        origin = GPS(parser.ORGN.iloc[:,-3:]) if 'ORGN' in parser.dfs else None

        dfout = dfs[0]

        for df in dfs[1:]:
            dfout = pd.merge_asof(dfout, df, on='time_actual', direction='nearest')
        
        return Flight(dfout.set_index('time_flight', drop=False), parser.parms, origin, ppsorce)

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
            position_N = df['N'],
            position_E = df['E'],
            position_D = df['D'],
            velocity_N = df['VN'],
            velocity_E = df['VE'],
            velocity_D = df['VD'],
            wind_N = df['wN'] if 'wN' in df.columns else None,
            wind_E = df['wE'] if 'wE' in df.columns else None,
        )
        origin = GPS(fc_json['parameters']['originLat'], fc_json['parameters']['originLng'], fc_json['parameters']['originAlt'])

        return Flight(df.set_index('time_flight', drop=False), None, origin, 'position')
