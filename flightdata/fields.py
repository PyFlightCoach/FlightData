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


from pint import UnitRegistry, DimensionalityError
from typing import Union, Self
from itertools import chain

ureg = UnitRegistry()


class Field:
    def __init__(self, column: str, unit: ureg.Unit, description: str = ''):
        self.column = column
        self.unit = unit
        self.description = description
        _sp = column.split('_')
        self.field = _sp[0]
        self.name = _sp[1]


class Fields:
    def __init__(self, data: Union[list[Field], dict[str: Field]]):
        if isinstance(data, list):
            data = {f.column: f for f in data}
        self.data = data
        self.groups = {}
        for k, v in data.items():
            if not v.field in self.groups:
                self.groups[v.field] = {}
            self.groups[v.field][v.name] = v
        
        
    def __getattr__(self, name):
        if name in self.data:
            return self.data[name]
        if '_' in name:
            vs = name.split('_')
            return getattr(getattr(self, vs[0]), vs[1])
        else:
            raise AttributeError(f'Field {name} not found')


    
fields = Fields(list(
        Field('time_flight', ureg.second, 'time since the start of the flight'),
        Field('time_actual', ureg.second, 'time since epoch'),
        *[Field(f'rcin_{i}', ureg.second) for i in range(8)],
        *[Field(f'rcout_{i}', ureg.second) for i in range(14)],
        Field('flightmode', 1),
        Field('position_x', ureg.meter, 'position in the north direction'),
        Field('position_y', ureg.meter, 'position in the east direction'),
        Field('position_z', ureg.meter, 'position in the down direction'),
        Field('gps_latitude', ureg.degree, 'latitude'),
        Field('gps_longitude', ureg.degree, 'longitude'),
        Field('altitude_gps', ureg.meter, 'altitude from gps'),
        Field('altitude_baro', ureg.meter, 'altitude from baro'),
        Field('attitude_roll', ureg.radian, 'roll angle'),
        Field('attitude_pitch', ureg.radian, 'pitch angle'),
        Field('attitude_yaw', ureg.radian, 'yaw angle'),
        Field('quaternion_w', 1),
        Field('quaternion_x', 1),
        Field('quaternion_y', 1),
        Field('quaternion_z', 1),
        Field('axisrate_roll', ureg.radian / ureg.second, 'roll rate'),
        Field('axisrate_pitch', ureg.radian / ureg.second, 'pitch rate'),
        Field('axisrate_yaw', ureg.radian / ureg.second, 'yaw rate'),
        *[Field(f'motor_voltage{i}', ureg.volt) for i in range(8)],
        *[Field(f'motor_current{i}', ureg.amp) for i in range(8)],
        *[Field(f'motor_rpm{i}', 1 / ureg.minute) for i in range(8)],
        Field('air_speed', ureg.meter / ureg.second, 'airspeed'),
        Field('air_pressure', ureg.pascal, 'air pressure'),
        Field('air_temperature', ureg.kelvin, 'air temperature'),
        Field('acceleration_x', ureg.meter / ureg.second / ureg.second, 'Body X Acceleration'),
        Field('acceleration_y', ureg.meter / ureg.second / ureg.second, 'Body Y Acceleration'),
        Field('acceleration_z', ureg.meter / ureg.second / ureg.second, 'Body Z Acceleration'),
        Field('velocity_x', ureg.meter / ureg.second, 'Body X Velocity'),
        Field('velocity_y', ureg.meter / ureg.second, 'Body Y Velocity'),
        Field('velocity_z', ureg.meter / ureg.second, 'Body Z Velocity'),
        Field('wind_x', ureg.meter / ureg.second, 'Wind N'),
        Field('wind_y', ureg.meter / ureg.second, 'Wind E'),
        Field('magnetometer_x', 1, 'Body magnetic field strength X'),
        Field('magnetometer_y', 1, 'Body magnetic field strength Y'),
        Field('magnetometer_z', 1, 'Body magnetic field strength Z'),
))

