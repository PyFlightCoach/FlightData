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
from typing import Dict, List
from itertools import chain

ureg = UnitRegistry()


_field_list = []

# TODO rewrite all this, allow multiple units per field. better to look more like /flightdata/mapping/ardupilot_ekfv3.py
class CIDTypes():
    CARTESIAN = 0
    EULER = 1
    BODY = 2
    NA = 3
    GPS = 4
    ZONLY = 5
    XY = 6


class Field(object):
    def __init__(self, name: str, unit: ureg.Unit, length: int, description: str = '', names: List[str] = []):
        self.name = name
        self.unit = unit
        self.length = length
        self.description = description
        self.names = Field._make_names(self.name, names, length)
        _field_list.append(self)

    @staticmethod
    def _make_names(name, names, length):
        _out_names = []
        for i in range(0, length):
            if i < len(names):
                _out_names.append(name + '_' + names[i])
            else:
                _out_names.append(name + '_' + str(i))
        return _out_names



class Fields(object):
    """This class defines the fields. Do not instantiate.
    """
    TIME = Field('time', ureg.second, 2,names=['flight', 'actual'])
    TXCONTROLS = Field('tx_controls', ureg.second, 8, description='PWM Values coming from the TX')
    SERVOS = Field('servos', ureg.second, 14, description='PWN Values going to the Servos')
    FLIGHTMODE = Field('mode', 1, 3, description='The active flight mode ID')
    POSITION = Field('position', ureg.meter, 3, description='position of plane (n, e, d)', names=['x', 'y', 'z'])
    GLOBALPOSITION = Field('global_position', ureg.degree, 2, names=['latitude', 'longitude'])
    GPSSATCOUNT = Field('gps_sat_count', 1, 1, description='number of satellites')
    SENSORALTITUDE = Field('altitude', ureg.meters, 2, names=['gps', 'baro'])
    ATTITUDE = Field('attitude', ureg.radian, 3, description='euler angles, order = yaw, pitch, roll', names=['roll', 'pitch', 'yaw'])
    QUATERNION = Field('quaternion', 1.0, 4)
    AXISRATE = Field('axis_rate', ureg.radian / ureg.second, 3, description='rotational velocities', names=['roll', 'pitch', 'yaw'])
    BATTERY = Field('battery', ureg.volt, 2, description='battery voltages')
    CURRENT = Field('current', ureg.amp, 4, description='motor currents')
    AIRSPEED = Field('airspeed', ureg.meter / ureg.second,2, description='sensor airspeed')
    ACCELERATION = Field('acceleration', ureg.meter / ureg.second / ureg.second, 3, description='accelerations (earth frame)', names=['x', 'y', 'z'])
    VELOCITY = Field('velocity', ureg.meter / ureg.second, 3, description='velocity data (earth frame)', names=['x', 'y', 'z'])
    WIND = Field('wind', ureg.meter / ureg.second, 2, description='wind in earth frame', names=['x', 'y'])
    RPM = Field('rpm', 1 / ureg.minute, 2, description='motor rpm')
    MAGNETOMETER = Field('magnetometer', 1, 3, description='mag field strength n, e, d')
    PRESSURE = Field('pressure', ureg.pascal, 1) 
    TEMPERATURE = Field('temperature', ureg.kelvin, 1) 
    
    all = _field_list
    all_names = list(chain(*[field.names for field in _field_list]))

    @staticmethod
    def some_names(fields):
        return [field.names for field in fields] if isinstance(fields, list) else fields.names

    @staticmethod
    def to_dict():
        return {field.name: field.names for field in _field_list}

    

class MappedField(object):
    def __init__(self, field, position, name, unit):
        super().__init__()
        self.field = field
        self.position = position
        self.name = name
        self.unit = unit
        try:
            self.base_factor = float(self.unit) / float(self.field.unit)
        except DimensionalityError:
            self.base_factor = (1 * self.unit).to(self.field.unit).magnitude


class FieldIOInfo(object):
    def __init__(self, field_maps: Dict[str, MappedField]):
        self._field_maps = field_maps

        self.io_names = []
        self.base_names = []
        self.factors_to_base = []
        self.factors_to_field = []

        for key, value in self._field_maps.items():
            self.io_names.append(key)
            self.base_names.append(value.field.names[value.position])
            self.factors_to_base.append(value.base_factor)
            self.factors_to_field.append(1 / value.base_factor)

    def subset(self, less_base_names):
        return FieldIOInfo({x: self._field_maps[x] for x in less_base_names if x in self._field_maps})
