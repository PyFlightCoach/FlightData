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


from flightdata.fields import Field, Fields, MappedField, FieldIOInfo
from pint import UnitRegistry

ureg = UnitRegistry()

# this maps the inav log variables to the tool variables
# ref https://github.com/ArduPilot/ardupilot/blob/master/ArduPlane/Log.cpp


#TIME = Field('time', ureg.second, 2, names=['flight', 'actual'])
#log_field_map["loopIteration"] = MappedField(Fields.LOOPITERATION, 0, "loopIteration", 1)

mapped_field_args = [
    (Fields.TIME, 0, "timestamp", ureg.second),
    (Fields.TIME, 1, "XKF1TimeUS", ureg.microsecond),
    (Fields.TXCONTROLS, 0, "RCINC1", ureg.second),
    (Fields.TXCONTROLS, 1, "RCINC2", ureg.second),
    (Fields.TXCONTROLS, 2, "RCINC3", ureg.second),
    (Fields.TXCONTROLS, 3, "RCINC4", ureg.second),
    (Fields.TXCONTROLS, 4, "RCINC5", ureg.second),
    (Fields.TXCONTROLS, 5, "RCINC6", ureg.second),
    (Fields.TXCONTROLS, 6, "RCINC7", ureg.second),
    (Fields.TXCONTROLS, 7, "RCINC8", ureg.second),
    (Fields.SERVOS, 0, "RCOUC1", ureg.second),
    (Fields.SERVOS, 1, "RCOUC2", ureg.second),
    (Fields.SERVOS, 2, "RCOUC3", ureg.second),
    (Fields.SERVOS, 3, "RCOUC4", ureg.second),
    (Fields.SERVOS, 4, "RCOUC5", ureg.second),
    (Fields.SERVOS, 5, "RCOUC6", ureg.second),
    (Fields.SERVOS, 6, "RCOUC7", ureg.second),
    (Fields.SERVOS, 7, "RCOUC8", ureg.second),
    (Fields.SERVOS, 8, "RCOUC9", ureg.second),
    (Fields.SERVOS, 9, "RCOUC10", ureg.second),
    (Fields.SERVOS, 10, "RCOUC11", ureg.second),
    (Fields.SERVOS, 11, "RCOUC12", ureg.second),
    (Fields.SERVOS, 12, "RCOUC13", ureg.second),
    (Fields.SERVOS, 13, "RCOUC14", ureg.second),
    (Fields.FLIGHTMODE, 0, "MODEMode", 1),
    (Fields.FLIGHTMODE, 1, "MODEModeNum", 1),
    (Fields.FLIGHTMODE, 2, "MODERsn", 1),
    (Fields.POSITION, 0, "POSx", ureg.meter),
    (Fields.POSITION, 1, "POSy", ureg.meter),
    (Fields.POSITION, 2, "POS", ureg.meter),
    (Fields.GLOBALPOSITION, 0, "GPSLat", ureg.degree),
    (Fields.GLOBALPOSITION, 1, "GPSLng", ureg.degree),
    (Fields.GPSSATCOUNT, 0, "GPSNSats", 1),
    (Fields.ATTITUDE, 0, "ATTRoll", ureg.degree),
    (Fields.ATTITUDE, 1, "ATTPitch", ureg.degree),
    (Fields.ATTITUDE, 2, "ATTYaw", ureg.degree),
    (Fields.AXISRATE, 0, "IMUGyrX", ureg.radian / ureg.second),
    (Fields.AXISRATE, 1, "IMUGyrY", ureg.radian / ureg.second),
    (Fields.AXISRATE, 2, "IMUGyrZ", ureg.radian / ureg.second),
    (Fields.BATTERY, 0, "BATVolt", ureg.V),
    (Fields.BATTERY, 1, "BAT2Volt", ureg.V),
    (Fields.CURRENT, 0, "BATCurr", ureg.A),
    (Fields.CURRENT, 1, "BAT2Curr", ureg.A),
    (Fields.AIRSPEED, 0, "ARSPAirspeed", ureg.meter / ureg.second),
    (Fields.ACCELERATION, 0, "IMUAccX", ureg.meter / ureg.second / ureg.second),
    (Fields.ACCELERATION, 1, "IMUAccY", ureg.meter / ureg.second / ureg.second),
    (Fields.ACCELERATION, 2, "IMUAccZ", ureg.meter / ureg.second / ureg.second),
    (Fields.VELOCITY, 0, "XKF1VN", ureg.meter / ureg.second),
    (Fields.VELOCITY, 1, "XKF1VE", ureg.meter / ureg.second),
    (Fields.VELOCITY, 2, "XKF1VD", ureg.meter / ureg.second),
    (Fields.WIND, 0, "XKF2VWN", ureg.meter / ureg.second),
    (Fields.WIND, 1, "XKF2VWE", ureg.meter / ureg.second),
    (Fields.RPM, 0, "RPMrpm1", 14 / ureg.minute),
    (Fields.RPM, 1, "RPMrpm2", 14 / ureg.minute),
    (Fields.MAGNETOMETER, 0, "MAGMagX", 1),
    (Fields.MAGNETOMETER, 1, "MAGMagY", 1),
    (Fields.MAGNETOMETER, 2, "MAGMagZ", 1),
    (Fields.QUATERNION, 0, "XKQ1Q1", 1),
    (Fields.QUATERNION, 1, "XKQ1Q2", 1),
    (Fields.QUATERNION, 2, "XKQ1Q3", 1),
    (Fields.QUATERNION, 3, "XKQ1Q4", 1),
    (Fields.PRESSURE, 0, "BAROPress", ureg.Pa),
    (Fields.TEMPERATURE, 0, "BAROTemp", ureg.celsius),
]

log_field_map = {fm[2]: MappedField(*fm) for fm in mapped_field_args}

ardupilot_ekfv3_io_info = FieldIOInfo(log_field_map)
