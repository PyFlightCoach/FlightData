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
from __future__ import annotations
from typing import Self, Union, IO
import numpy as np
import pandas as pd
from .fields import fields, Field
from geometry import GPS, Point
from geometry.testing import assert_almost_equal
from pathlib import Path
from time import time
from json import load, dump
from ardupilot_log_reader.reader import Ardupilot
from flightdata.base.numpy_encoder import NumpyEncoder
from flightdata.flight.fields import Fields
from .ardupilot import flightmodes 
from flightdata import Origin
from scipy.signal import butter, filtfilt


class Flight:
    ardupilot_types = [
        'XKF1', 'XKF2', 'NKF1', 'NKF2', 
        'POS', 'ATT', 'RATE', 'ACC', 'GYRO', 'IMU', 
        'ARSP', 'GPS', 'RCIN', 'RCOU', 'BARO', 'MODE', 
        'RPM', 'MAG', 'BAT', 'BAT2', 'VEL', 'ORGN', 'ESC', 'CURRENT']
    
    def __init__(self, data: pd.DataFrame, parameters: list = None, origin: Origin = None, primary_pos_source='gps'):
        self.data = data
        self.parameters = parameters
        self.data.index = self.data.index - self.data.index[0]
        self.data.index.name = 'time_index'
        self.origin = origin
        self.primary_pos_source = primary_pos_source
        
    def __getattr__(self, name):
        cols = getattr(fields, name)
        if cols is None:
            cols = [f for f in self.data.columns if f.startswith(name)]
            if len(cols) > 0:
                return self.data[cols]
        try:
            if isinstance(cols, Field):
                return self.data[cols.col]
            else:
                return self.data.loc[:, [f.col for f in cols if f.col in self.data.columns]]
        except KeyError:
            if isinstance(cols, Field):
                cols = [cols]
            return pd.DataFrame(data=np.empty((len(self), len(cols))),columns=[f.col for f in cols])
        
    def contains(self, name: Union[str, list[str]]):
        cols = getattr(fields, name)
        if isinstance(cols, Field):
            return name in self.data.columns
        else:
            return [f.column in self.data.columns for f in cols]
                
    def __getitem__(self, sli) -> Self:
        if isinstance(sli, int) or isinstance(sli, float):
            return self.data.iloc[self.data.index.get_loc(sli)]
        else:
            return Flight(self.data.loc[sli], self.parameters, self.origin, self.primary_pos_source)

    def __len__(self):
        return len(self.data)

    def slice_raw_t(self, sli):
        return Flight(
            self.data.reset_index(drop=True)
                .set_index('time_actual', drop=False)
                    .loc[sli].set_index("time_flight", drop=False), 
            self.parameters, self.origin, self.primary_pos_source
        )
        
    def slice_time_flight(self, sli):
        return Flight(
            self.data.reset_index(drop=True)
                .set_index('time_flight', drop=False)
                    .loc[sli], 
            self.parameters, self.origin, self.primary_pos_source
        )
        
    def copy(self, **kwargs):
        return Flight(
            kwargs['data'] if 'data' in kwargs else self.data.copy() ,
            kwargs['parameters'] if 'parameters' in kwargs else self.parameters.copy() if self.parameters else None,
            kwargs['origin'] if 'origin' in kwargs else self.origin.copy(),
            kwargs['primary_pos_source'] if 'primary_pos_source' in kwargs else self.primary_pos_source
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
            origin=Origin.from_dict(data['origin']),
            primary_pos_source=data['primary_pos_source']
        )
    
    def to_json(self, file: str) -> str:
        with open(file, 'w') as f:
            dump(self.to_dict(), f, cls=NumpyEncoder, indent=2)
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
            flos.append(fl.copy(
                data=pd.merge_asof(
                    otf, 
                    fl.data.reset_index(), 
                    on='time_actual'
                ).set_index('time_index', drop=False)
            ))

        return flos

    def split_modes(self):
        """Split the flight into segments of the same flight mode.

        Returns:
            list[Flight]: list of flights
        """
        modechanges = (self.flightmode_a.diff().fillna(value=0) != 0).astype(int).cumsum()

        flights = {flightmodes[m]: [] for m in self.flightmode_a.unique()}

        for mode in modechanges.unique():
            _fl = self.data.loc[modechanges == mode, :]
            flights[flightmodes[_fl.flightmode_a.iloc[0]]].append(Flight(_fl, self.parameters, self.origin, self.primary_pos_source))
        return flights

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
            assert_almost_equal(self.origin.pos, other.origin.pos)
            assert self.origin.heading == other.origin.heading
            return True
        except Exception as ex:
            return False
        
    @staticmethod
    def from_log(log:Union[Ardupilot, str], extra_types: list[str] = None, **kwargs) -> Flight:
        """Constructor from an ardupilot bin file."""
        
        extra_types = [] if extra_types is None else extra_types
        
        if isinstance(log, str) or isinstance(log, Path):
            parser = Ardupilot(str(log), types=list(set(Flight.ardupilot_types + extra_types)))
        else:
            parser = log

        if parser.parms['AHRS_EKF_TYPE'] == 2:
            ekf1 = 'NKF1'
            ekf2 = 'NKF2'
        else:
            ekf1 = 'XKF1'
            ekf2 = 'XKF2'

        ekf1 = parser.dfs[ekf1] if ekf1 in parser.dfs else  None
        ekf2 = parser.dfs[ekf2] if ekf2 in parser.dfs else None

        dfs = []

        if 'ATT' in parser.dfs:
            att=parser.ATT.iloc[1:, :]
            dfs.append(Flight.build_cols(
                time_actual = att.timestamp,
                time_flight = att.TimeUS / 1e6,
                attitude_roll = np.radians(att.Roll),
                attitude_pitch = np.radians(att.Pitch),
                attitude_yaw = np.radians(att.Yaw),
                attdes_roll = np.radians(att.DesRoll),
                attdes_pitch = np.radians(att.DesPitch),
                attdes_yaw = np.radians(att.DesYaw),
            ))

        if 'POS' in parser.dfs:
            ppsorce = 'gps'
            dfs.append(Flight.build_cols(
                time_actual = parser.POS.timestamp,
                gps_latitude = parser.POS.Lat,
                gps_longitude = parser.POS.Lng,
                gps_altitude = parser.POS.Alt
            ))
        else:
            ppsorce = 'position'
            
        if ekf1 is not None: 
            dfs = dfs + Flight.parse_instances(ekf1, dict(
                position_N='PN',
                position_E='PE',
                position_D='PD',
                velocity_N='VN',
                velocity_E='VE',
                velocity_D='VD',
            ), 'C')
            
        if ekf2 is not None:
            dfs = dfs + Flight.parse_instances(ekf2, {
                'wind_N': 'VWN',
                'wind_E': 'VWE',
            }, 'C')
        if 'RATE' in parser.dfs:
            dfs.append(Flight.build_cols(
                time_actual = parser.rate.timestamp,
                axisrate_roll = parser.rate.R,
                axisrate_pitch = parser.rate.P,
                axisrate_yaw = parser.rate.Y,
                desrate_roll = parser.rate.RDes,
                desrate_pitch = parser.rate.PDes,
                desrate_yaw = parser.rate.YDes,
            ))
        
        if 'IMU' in parser.dfs:
            imu = parser.IMU
            if 'I' in imu:
                imu = imu.loc[imu.I==0, :]
            
            if ekf1 is not None:
                if 'C' in ekf1.columns:
                    imu = pd.merge_asof(imu, ekf1.loc[ekf1.C==0], on='timestamp', direction='nearest')
                else:
                    imu = pd.merge_asof(imu, ekf1, on='timestamp', direction='nearest')
                if all([v in imu.columns for v in ['GX', 'GY', 'GZ']]):
                    imu['GyrX'] = imu.GyrX + np.radians(imu.GX) / 100
                    imu['GyrY'] = imu.GyrY + np.radians(imu.GY) / 100
                    imu['GyrZ'] = imu.GyrZ + np.radians(imu.GZ) / 100

            if ekf2 is not None:
                if 'C' in ekf2.columns:
                    imu = pd.merge_asof(imu, ekf2.loc[ekf2.C==0], on='timestamp', direction='nearest')
                else:
                    imu = pd.merge_asof(imu, ekf2, on='timestamp', direction='nearest')
                if all([v in imu.columns for v in ['AX', 'AY', 'AZ']]):
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
            dfs = dfs + Flight.parse_instances(parser.BARO, dict(
                air_pressure= "Press",
                air_temperature= "Temp",
                air_altitude= "Alt"
            ), 'I')

        if 'ARSP' in parser.dfs:
            dfs = dfs + Flight.parse_instances(parser.ARSP,
                dict(air_speed= 'Airspeed'), 'I')

        if 'RCIN' in parser.dfs:
            dfs.append(Flight.build_cols(
                time_actual = parser.RCIN.timestamp,
                **{f'rcin_c{i}': parser.RCIN[f'C{i}'] for i in range(20) if f'C{i}' in parser.RCIN.columns}
            ))
        
        if 'RCOU' in parser.dfs:
            dfs.append(Flight.build_cols(
                time_actual = parser.RCOU.timestamp,
                **{f'rcout_c{i}': parser.RCOU[f'C{i}'] for i in range(20) if f'C{i}' in parser.RCOU.columns}
            ))

        if 'MODE' in parser.dfs:
            
            df = Flight.build_cols(
                time_actual = parser.MODE.timestamp,
                flightmode_a = parser.MODE.Mode,
                flightmode_b = parser.MODE.ModeNum,
                flightmode_c = parser.MODE.Rsn,
            )

            #direction backward sets all the nans to the previous one, fillna backward sets the first bunch of nans to the subsequent
            dfs.append(pd.merge_asof(dfs[0].loc[:,'time_actual'], df, on='time_actual', direction='backward').bfill())
        
        if 'BAT' in parser.dfs:
            dfs = dfs + Flight.parse_instances(parser.BAT, dict(
                battery_voltage = 'Volt',
                battery_current = 'Curr',
                battery_totalcurrent = 'CurrTot',
                battery_totalenergy = 'EnrgTot',
            ))

        if 'ESC' in parser.dfs:
            dfs = dfs + Flight.parse_instances(parser.ESC, dict(
                motor_voltage= 'Volt',
                motor_current= 'Curr',
                motor_rpm= 'RPM'
            ))
        elif 'RPM' in parser.dfs:
            dfs.append(Flight.build_cols(
                time_actual = parser.RPM.timestamp,
                **{f'motor_rpm_{i}': parser.RPM[f'rpm{i}'] for i in range(2) if f'rpm{i}' in parser.RPM.columns},
            ))

        for k, v in kwargs.items():
            if k in parser.dfs:
                dfs = dfs + Flight.parse_instances(parser.dfs[k], v)
        
        dfout = dfs[0]
        dt = dfout.time_actual.diff().max()
        for df in dfs[1:]:
            dfout = pd.merge_asof(
                dfout, df, on='time_actual', direction='nearest', 
                tolerance=min(max(dt, df.time_actual.diff().max()), 0.1)
            )
        
        origin = Origin('ekf_origin', GPS(parser.ORGN.iloc[:,-3:]), 0)
        
        return Flight(dfout.set_index('time_flight', drop=False), parser.parms, origin, ppsorce)

    @staticmethod
    def parse_instances(indf: pd.DataFrame, colmap:dict[str, str], instancecol='Instance'):
            '''Where an instance column exists in an input df split the values into two columns'''
            instances = indf[instancecol].unique() if instancecol in indf.columns else [0]
            dfs = []
            for i in instances:
                _subdf = indf.loc[indf[instancecol]==i, :] if instancecol in indf.columns else indf
                
                dfs.append(Flight.build_cols(
                    time_actual = _subdf.timestamp,
                    **{f'{k}{f"_{i}" if i > 0 else ""}': _subdf[v] for k, v in colmap.items()}
                ))
            return dfs

    @staticmethod
    def from_fc_json(fc_json: Union[str, dict, IO]) -> Self:
        
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
        origin = Origin('fcj_origin',GPS(
            fc_json['parameters']['originLat'], 
            fc_json['parameters']['originLng'], 
            fc_json['parameters']['originAlt']
        ), 0)
        
        return Flight(df.set_index('time_flight', drop=False), None, origin, 'position')#.remove_time_flutter()

    def remove_time_flutter(self):
        #I think the best option is just to take the average of the timestep.
        #FC loop rate seems consistent but there is noise in the time that is recorded
        #to write, probably because different things are being written at different rates
        time_cols = fields.get_cols(['time'])
        avcols = []
        for col in time_cols:
            _col = self.data.loc[:, col]
            avcols.append(np.linspace(_col.iloc[0], _col.iloc[-1], len(_col)))

        return self.copy(
            data=pd.concat([
                pd.DataFrame(np.array(avcols).T, columns=time_cols, index=self.data.index),
                self.data.loc[:,[c for c in self.data.columns if c not in time_cols]],
            ], axis=1).reset_index(drop=True).set_index('time_flight', drop=False)
        )

    def filter(self, b, a):
        dont_filter = [c for c in fields.get_cols(['time', 'flightmode', 'rcin', 'rcout']) if c in self.data.columns]
        unwrap_cols = [c for c in fields.get_cols(['attitude']) if c in self.data.columns]

        filter_cols = [c for c in self.data.columns if c not in dont_filter + unwrap_cols]
        
        filtdf = filtfilt(b, a, self.data.loc[:, filter_cols], axis=0)
        unwfiltdf = filtfilt(b, a, np.unwrap(self.data.loc[:, unwrap_cols], axis=0), axis=0)

        return self.copy(
            data=pd.concat([
                self.data.loc[:,dont_filter],
                pd.DataFrame(unwfiltdf, columns=unwrap_cols, index=self.data.index) % (2*np.pi),
                pd.DataFrame(filtdf, columns=filter_cols, index=self.data.index),
            ], axis=1)
        )
    
    def butter_filter(self, cutoff, order=5):
        ts = self.time_flight.to_numpy()
        N = len(self)
        T = (ts[-1] - ts[0]) / N
        fs = 1/T
        return self.filter(*butter(order, cutoff, fs=fs, btype='low', analog=False))
    