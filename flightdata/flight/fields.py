
from typing import Union, Self
from itertools import chain


class Field:
    def __init__(self, column: str, description: str = '', i: int = 0):
        self.column = column
        self.description = description
        _sp = column.split('_')
        self.field = _sp[0]
        self.name = _sp[1]
        self.i = i

    def instance(self, i) -> Self:
        return Field(self.column, self.description, i)

    @property
    def col(self) -> str:
        return f'{self.column}_{self.i}' if self.i > 0 else self.column

    def __repr__(self):
        return f'{self.column}_{self.i}'
    
class Fields:
    def __init__(self, data: Union[list[Field], dict[str: Field]]):
        if isinstance(data, list):
            data = {f.column: f for f in data}
        self.data = data
        self.groups = {}
        for v in data.values():
            if v.field not in self.groups:
                self.groups[v.field] = []
            self.groups[v.field].append(v)
        
    def __getattr__(self, name):
        _f = name.split('_')
        group = _f[0]
        instance = 0
        col = None
        if len(_f) == 2:
            if _f[1].isnumeric():
                instance = int(_f[1])
            else:
                col = _f[1]
        elif len(_f) == 3:
            col = _f[1]
            instance = int(_f[2])
        
        try:
            if not col:
                return [f.instance(instance) for f in self.groups[group]]
            else:
                return self.data[f'{group}_{col}'].instance(instance)
        except KeyError:
            return None
            #raise AttributeError(f'Field {name} not found')

    def get_fields(self, names: list[str]) -> list[Field]:
        def _l(v):
            return [v] if isinstance(v, Field) else v
        return list(chain(*[_l(getattr(self, n)) for n in names]))
    
    def get_cols(self, names: list[str]) -> list[str]:
        return [f.col for f in self.get_fields(names)]

    def __repr__(self) -> str:
        return f'Fields({','.join(list(self.data.keys()))})'

fields = Fields([
        Field('time_flight', 'time since the start of the flight, seconds'),
        Field('time_actual', 'time since epoch, seconds'),
        *[Field(f'rcin_c{i}', 'ms') for i in range(8)],
        *[Field(f'rcout_c{i}', 'ms') for i in range(14)],
        Field('flightmode_a'),
        Field('flightmode_b'),
        Field('flightmode_c'),
        Field('position_N', 'distance from origin in the north direction, meters'),
        Field('position_E', 'distance from origin in the east direction, meters'),
        Field('position_D', 'distance from origin in the down direction, meters'),
        Field('gps_latitude', 'latitude, degrees'),
        Field('gps_longitude', 'longitude, degrees'),
        Field('gps_altitude', 'altitude, meters'),
        Field('gps_satellites', 'number of satellites'),
        Field('gps_hdop', 'number precision'),
        Field('pos_latitude', 'latitude, degrees'),
        Field('pos_longitude', 'longitude, degrees'),
        Field('pos_altitude', 'altitude, meters'),
        Field('attitude_roll', 'roll angle, radians'),
        Field('attitude_pitch', 'pitch angle, radians'),
        Field('attitude_yaw', 'yaw angle, radians'),
        Field('attdes_roll', 'desired roll angle, radians'),
        Field('attdes_pitch', 'desired pitch angle, radians'),
        Field('attdes_yaw', 'desired yaw angle, radians'),
        Field('axisrate_roll', 'roll rate, radians / second'),
        Field('axisrate_pitch', 'pitch rate, radians / second'),
        Field('axisrate_yaw', 'yaw rate, radians / second'),
        Field('desrate_roll', 'roll rate, radians / second'),
        Field('desrate_pitch', 'pitch rate, radians / second'),
        Field('desrate_yaw', 'yaw rate, radians / second'),
        Field('battery_voltage', 'volts'),
        Field('battery_current', 'amps'),
        Field('battery_totalcurrent', 'Ah'),
        Field('battery_totalenergy', 'Wh'),
        Field('motor_voltage', 'volts'),
        Field('motor_current', 'amps'),
        Field('motor_rpm', 'rpm'),
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

