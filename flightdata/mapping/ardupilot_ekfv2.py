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


from flightdata.fields import Fields, MappedField, FieldIOInfo
from pint import UnitRegistry

ureg = UnitRegistry()

# this maps the inav log variables to the tool variables
# ref https://github.com/ArduPilot/ardupilot/blob/master/ArduPlane/Log.cpp
log_field_map = dict()

#TIME = Field('time', ureg.second, 2, names=['flight', 'actual'])
#log_field_map["loopIteration"] = MappedField(Fields.LOOPITERATION, 0, "loopIteration", 1)
log_field_map["timestamp"] = MappedField(
    Fields.TIME, 0, "timestamp", ureg.second)
log_field_map["NKF1TimeUS"] = MappedField(
    Fields.TIME, 1, "NKF1TimeUS", ureg.microsecond)


#TXCONTROLS = Field('tx_controls', ureg.second, 6, description='PWM Values coming from the TX')
#log_field_map["AETR_Ail"] = MappedField(Fields.TXCONTROLS, 0, "AETR_Ail", ureg.second)
#log_field_map["AETR_Elev"] = MappedField(Fields.TXCONTROLS, 1, "AETR_Elev", ureg.second)
#log_field_map["AETR_Thr"] = MappedField(Fields.TXCONTROLS, 2, "AETR_Thr", ureg.second)
#log_field_map["AETR_Rudd"] = MappedField(Fields.TXCONTROLS, 3, "AETR_Rudd", ureg.second)
#log_field_map["AETR_Flap"] = MappedField(Fields.TXCONTROLS, 4, "AETR_Flap", ureg.second)

log_field_map["RCINC1"] = MappedField(
    Fields.TXCONTROLS, 0, "RCINC1", ureg.second)
log_field_map["RCINC2"] = MappedField(
    Fields.TXCONTROLS, 1, "RCINC2", ureg.second)
log_field_map["RCINC3"] = MappedField(
    Fields.TXCONTROLS, 2, "RCINC3", ureg.second)
log_field_map["RCINC4"] = MappedField(
    Fields.TXCONTROLS, 3, "RCINC4", ureg.second)
log_field_map["RCINC5"] = MappedField(
    Fields.TXCONTROLS, 4, "RCINC5", ureg.second)
log_field_map["RCINC6"] = MappedField(
    Fields.TXCONTROLS, 5, "RCINC6", ureg.second)
log_field_map["RCINC7"] = MappedField(
    Fields.TXCONTROLS, 6, "RCINC7", ureg.second)
log_field_map["RCINC8"] = MappedField(
    Fields.TXCONTROLS, 7, "RCINC8", ureg.second)

#SERVOS = Field('servos', ureg.second, 8, description='PWN Values going to the Servos')
log_field_map["RCOUC1"] = MappedField(Fields.SERVOS, 0, "RCOUC1", ureg.second)
log_field_map["RCOUC2"] = MappedField(Fields.SERVOS, 1, "RCOUC2", ureg.second)
log_field_map["RCOUC3"] = MappedField(Fields.SERVOS, 2, "RCOUC3", ureg.second)
log_field_map["RCOUC4"] = MappedField(Fields.SERVOS, 3, "RCOUC4", ureg.second)
log_field_map["RCOUC5"] = MappedField(Fields.SERVOS, 4, "RCOUC5", ureg.second)
log_field_map["RCOUC6"] = MappedField(Fields.SERVOS, 5, "RCOUC6", ureg.second)
log_field_map["RCOUC7"] = MappedField(Fields.SERVOS, 6, "RCOUC7", ureg.second)
log_field_map["RCOUC8"] = MappedField(Fields.SERVOS, 7, "RCOUC8", ureg.second)

#FLIGHTMODE = Field('mode', 1, 1, description='The active flight mode ID')
log_field_map["MODEMode"] = MappedField(Fields.FLIGHTMODE, 0, "MODEMode", 1)
log_field_map["MODEModeNum"] = MappedField(
    Fields.FLIGHTMODE, 1, "MODEModeNum", 1)
log_field_map["MODERsn"] = MappedField(Fields.FLIGHTMODE, 2, "MODERsn", 1)

#EKFPOSITION = Field('ekf_position', ureg.radians, 3, description='position of plane in cartesian coordinates (s, e, u)', names=['x', 'y', 'z'])
log_field_map["NKF1PN"] = MappedField(Fields.POSITION, 0, "NKF1PN", ureg.meter)
log_field_map["NKF1PE"] = MappedField(Fields.POSITION, 1, "NKF1PE", ureg.meter)
log_field_map["NKF1PD"] = MappedField(Fields.POSITION, 2, "NKF1PD", ureg.meter)

#GLOBALPOSITION = Field('global_position', ureg.degrees, 3, names=['latitude', 'longitude', 'altitude'])
log_field_map["GPSLat"] = MappedField(
    Fields.GLOBALPOSITION, 0, "GPSLat", ureg.degree)
log_field_map["GPSLng"] = MappedField(
    Fields.GLOBALPOSITION, 1, "GPSLng", ureg.degree)

#SENSORALTITUDE = Field('altitude', ureg.meters, 2, names=['gps', 'baro'])
log_field_map["GPSAlt"] = MappedField(
    Fields.SENSORALTITUDE, 0, "GPSAlt", ureg.meter)
log_field_map["BAROAlt"] = MappedField(
    Fields.SENSORALTITUDE, 1, "BAROAlt", ureg.meter)

log_field_map["GPSNSats"] = MappedField(Fields.GPSSATCOUNT, 0, "GPSNSats", 1)

#ATTITUDE = Field('attitude', ureg.radian, 3, description='euler angles, order = yaw, pitch, roll', names=['roll', 'pitch', 'yaw'])
log_field_map["NKF1Roll"] = MappedField(
    Fields.ATTITUDE, 0, "NKF1Roll", ureg.degree)
log_field_map["NKF1Pitch"] = MappedField(
    Fields.ATTITUDE, 1, "NKF1Pitch", ureg.degree)
log_field_map["NKF1Yaw"] = MappedField(
    Fields.ATTITUDE, 2, "NKF1Yaw", ureg.degree)

log_field_map["IMUGyrX"] = MappedField(
    Fields.AXISRATE, 0, "IMUGyrX", ureg.degree / ureg.second)
log_field_map["IMUGyrY"] = MappedField(
    Fields.AXISRATE, 1, "IMUGyrY", ureg.degree / ureg.second)
log_field_map["IMUGyrZ"] = MappedField(
    Fields.AXISRATE, 2, "IMUGyrZ", ureg.degree / ureg.second)


#BATTERY = Field('battery', ureg.volt, 2, description='battery voltages')
log_field_map["BATVolt"] = MappedField(Fields.BATTERY, 0, "BATVolt", ureg.V)
log_field_map["BAT2Volt"] = MappedField(Fields.BATTERY, 1, "BAT2Volt", ureg.V)
#log_field_map["sagCompensatedVBat"] = MappedField(Fields.BATTERY, 1, "sagCompensatedVBat", ureg.centiV)

#CURRENT = Field('current', ureg.amp, 4, description='motor currents')
log_field_map["BATCurr"] = MappedField(Fields.CURRENT, 0, "BATCurr", ureg.A)
log_field_map["BAT2Curr"] = MappedField(Fields.CURRENT, 1, "BAT2Curr", ureg.A)

log_field_map["ARSPAirspeed"] = MappedField(
    Fields.AIRSPEED, 0, "ARSPAirspeed", ureg.meter / ureg.second)

log_field_map["IMUAccX"] = MappedField(
    Fields.ACCELERATION, 0, "IMUAccX", ureg.meter / ureg.second / ureg.second)
log_field_map["IMUAccY"] = MappedField(
    Fields.ACCELERATION, 1, "IMUAccY", ureg.meter / ureg.second / ureg.second)
log_field_map["IMUAccZ"] = MappedField(
    Fields.ACCELERATION, 2, "IMUAccZ", ureg.meter / ureg.second / ureg.second)

log_field_map["NKF1VN"] = MappedField(
    Fields.VELOCITY, 0, "NKF1VN", ureg.meter / ureg.second)
log_field_map["NKF1VE"] = MappedField(
    Fields.VELOCITY, 1, "NKF1VE", ureg.meter / ureg.second)
log_field_map["NKF1VD"] = MappedField(
    Fields.VELOCITY, 2, "NKF1VD", ureg.meter / ureg.second)


log_field_map["NKF2VWN"] = MappedField(
    Fields.WIND, 0, "NKF2VWN", ureg.meter / ureg.second)
log_field_map["NKF2VWE"] = MappedField(
    Fields.WIND, 1, "XKF2VWE", ureg.meter / ureg.second)

log_field_map["RPMrpm1"] = MappedField(
    Fields.RPM, 0, "RPMrpm1", 14 / ureg.minute)
log_field_map["RPMrpm2"] = MappedField(
    Fields.RPM, 1, "RPMrpm2", 14 / ureg.minute)


log_field_map['MAGMagX'] = MappedField(Fields.MAGNETOMETER, 0, "MAGMagX", 1)
log_field_map['MAGMagY'] = MappedField(Fields.MAGNETOMETER, 1, "MAGMagY", 1)
log_field_map['MAGMagZ'] = MappedField(Fields.MAGNETOMETER, 2, "MAGMagZ", 1)

ardupilot_ekfv2_io_info = FieldIOInfo(log_field_map)


log_field_map["NKQ1Q1"]  = MappedField(Fields.QUATERNION, 0, "NKQ1Q1", 1)
log_field_map["NKQ1Q2"]  = MappedField(Fields.QUATERNION, 1, "NKQ1Q2", 1)
log_field_map["NKQ1Q3"]  = MappedField(Fields.QUATERNION, 2, "NKQ1Q3", 1)
log_field_map["NKQ1Q4"]  = MappedField(Fields.QUATERNION, 3, "NKQ1Q4", 1)