
from typing import Union, Self
from itertools import chain


class Field:
    def __init__(self, column: str, description: str = ''):
        self.column = column
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
        for v in data.values():
            if not v.field in self.groups:
                self.groups[v.field] = []
            self.groups[v.field].append(v)
        
    def __getattr__(self, name):
        if name in self.groups:
            return self.groups[name]
        elif name in self.data:
            return self.data[name]
        else:
            raise AttributeError(f'Field {name} not found')


fields = Fields([
        Field('time_flight', 'time since the start of the flight, seconds'),
        Field('time_actual', 'time since epoch, seconds'),
        *[Field(f'rcin_{i}', 'ms') for i in range(8)],
        *[Field(f'rcout_{i}', 'ms') for i in range(14)],
        Field('flightmode_a'),
        Field('flightmode_b'),
        Field('flightmode_c'),
        Field('position_N', 'distance from origin in the north direction, meters'),
        Field('position_E', 'distance from origin in the east direction, meters'),
        Field('position_D', 'distance from origin in the down direction, meters'),
        Field('gps_latitude', 'latitude, degrees'),
        Field('gps_longitude', 'longitude, degrees'),
        Field('gps_altitude', 'altitude, meters'),
        Field('attitude_roll', 'roll angle, radians'),
        Field('attitude_pitch', 'pitch angle, radians'),
        Field('attitude_yaw', 'yaw angle, radians'),
        Field('axisrate_roll', 'roll rate, radians / second'),
        Field('axisrate_pitch', 'pitch rate, radians / second'),
        Field('axisrate_yaw', 'yaw rate, radians / second'),
        *[Field(f'motor_voltage{i}', 'volts') for i in range(8)],
        *[Field(f'motor_current{i}', 'amps') for i in range(8)],
        *[Field(f'motor_rpm{i}', 'rpm') for i in range(8)],
        Field('air_speed', 'airspeed, m/s'),
        Field('air_pressure', 'air pressure, Pa'),
        Field('air_temperature', 'air temperature, k'),
        Field('air_altitude', 'altitude from baro, m'),
        Field('acceleration_x', 'Body x Acceleration, m/s/s'),
        Field('acceleration_y', 'Body y Acceleration, m/s/s'),
        Field('acceleration_z', 'Body z Acceleration, m/s/s'),
        Field('velocity_N', 'World N Velocity, m/s'),
        Field('velocity_E', 'World E Velocity, m/s'),
        Field('velocity_D', 'World D Velocity, m/s'),
        Field('wind_N', 'Wind N, m/s'),
        Field('wind_E', 'Wind E, m/s'),
        Field('magnetometer_x', 'Body magnetic field strength X'),
        Field('magnetometer_y', 'Body magnetic field strength Y'),
        Field('magnetometer_z', 'Body magnetic field strength Z'),
])

