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


from flightdata.fields import Field, Fields, MappedField
from pint import UnitRegistry

ureg = UnitRegistry()

# this maps the inav log variables to the tool variables
# ref https://github.com/ArduPilot/ardupilot/blob/master/ArduPlane/Log.cpp

ardupilot_ekfv3_maps = [
    MappedField(Fields.TIME, 0, "XKF1", "timestamp", ureg.second),
    MappedField(Fields.TIME, 1, "XKF1", "TimeUS", ureg.microsecond, C=0),
    MappedField(Fields.TXCONTROLS, 0, "RCIN", "C1", ureg.second),
    MappedField(Fields.TXCONTROLS, 1, "RCIN", "C2", ureg.second),
    MappedField(Fields.TXCONTROLS, 2, "RCIN", "C3", ureg.second),
    MappedField(Fields.TXCONTROLS, 3, "RCIN", "C4", ureg.second),
    MappedField(Fields.TXCONTROLS, 4, "RCIN", "C5", ureg.second),
    MappedField(Fields.TXCONTROLS, 5, "RCIN", "C6", ureg.second),
    MappedField(Fields.TXCONTROLS, 6, "RCIN", "C7", ureg.second),
    MappedField(Fields.TXCONTROLS, 7, "RCIN", "C8", ureg.second),
    MappedField(Fields.SERVOS, 0, "RCOU", "C1", ureg.second),
    MappedField(Fields.SERVOS, 1, "RCOU", "C2", ureg.second),
    MappedField(Fields.SERVOS, 2, "RCOU", "C3", ureg.second),
    MappedField(Fields.SERVOS, 3, "RCOU", "C4", ureg.second),
    MappedField(Fields.SERVOS, 4, "RCOU", "C5", ureg.second),
    MappedField(Fields.SERVOS, 5, "RCOU", "C6", ureg.second),
    MappedField(Fields.SERVOS, 6, "RCOU", "C7", ureg.second),
    MappedField(Fields.SERVOS, 7, "RCOU", "C8", ureg.second),
    MappedField(Fields.SERVOS, 8, "RCOU", "C9", ureg.second),
    MappedField(Fields.SERVOS, 9, "RCOU", "C10", ureg.second),
    MappedField(Fields.SERVOS, 10, "RCOU", "C11", ureg.second),
    MappedField(Fields.SERVOS, 11, "RCOU", "C12", ureg.second),
    MappedField(Fields.SERVOS, 12, "RCOU", "C13", ureg.second),
    MappedField(Fields.SERVOS, 13, "RCOU", "C14", ureg.second),
    MappedField(Fields.FLIGHTMODE, 0, "MODE", "Mode", 1),
    MappedField(Fields.FLIGHTMODE, 1, "MODE", "ModeNum", 1),
    MappedField(Fields.FLIGHTMODE, 2, "MODE", "Rsn", 1),
    MappedField(Fields.POSITION, 0, "XKF1", "PN", ureg.meter, C=0),
    MappedField(Fields.POSITION, 1, "XKF1", "PE", ureg.meter, C=0),
    MappedField(Fields.POSITION, 2, "XKF1", "PD", ureg.meter, C=0),
    MappedField(Fields.GLOBALPOSITION, 0, "GPS", "Lat", ureg.degree),
    MappedField(Fields.GLOBALPOSITION, 1, "GPS", "Lng", ureg.degree),
    MappedField(Fields.GPSSATCOUNT, 0, "GPS", "NSats", 1),
    MappedField(Fields.ATTITUDE, 0, "XKF1", "Roll", ureg.degree, C=0),
    MappedField(Fields.ATTITUDE, 1, "XKF1", "Pitch", ureg.degree, C=0),
    MappedField(Fields.ATTITUDE, 2, "XKF1", "Yaw", ureg.degree, C=0),
    MappedField(Fields.AXISRATE, 0, "IMU", "GyrX", ureg.radian / ureg.second),
    MappedField(Fields.AXISRATE, 1, "IMU", "GyrY", ureg.radian / ureg.second),
    MappedField(Fields.AXISRATE, 2, "IMU", "GyrZ", ureg.radian / ureg.second),
    MappedField(Fields.BATTERY, 0, "BAT", "Volt", ureg.V, Instance=0),
    MappedField(Fields.BATTERY, 1, "BAT", "Volt", ureg.V, Instance=1),
    MappedField(Fields.CURRENT, 0, "BAT", "Curr", ureg.A, Instance=0),
    MappedField(Fields.CURRENT, 1, "BAT", "Curr", ureg.A, Instance=1),
    MappedField(Fields.AIRSPEED, 0, "ARSP", "Airspeed", ureg.meter / ureg.second),
    MappedField(Fields.ACCELERATION, 0, "IMU", "AccX", ureg.meter / ureg.second / ureg.second, C=0),
    MappedField(Fields.ACCELERATION, 1, "IMU", "AccY", ureg.meter / ureg.second / ureg.second, C=0),
    MappedField(Fields.ACCELERATION, 2, "IMU", "AccZ", ureg.meter / ureg.second / ureg.second, C=0),
    MappedField(Fields.VELOCITY, 0, "XKF1", "VN", ureg.meter / ureg.second, C=0),
    MappedField(Fields.VELOCITY, 1, "XKF1", "VE", ureg.meter / ureg.second, C=0),
    MappedField(Fields.VELOCITY, 2, "XKF1", "VD", ureg.meter / ureg.second, C=0),
    MappedField(Fields.WIND, 0, "XKF2", "VWN", ureg.meter / ureg.second, C=0),
    MappedField(Fields.WIND, 1, "XKF2", "VWE", ureg.meter / ureg.second, C=0),
    MappedField(Fields.RPM, 0, "RPM", "rpm1", 14 / ureg.minute),
    MappedField(Fields.RPM, 1, "RPM", "rpm2", 14 / ureg.minute),
    MappedField(Fields.MAGNETOMETER, 0, "MAG", "MagX", 1),
    MappedField(Fields.MAGNETOMETER, 1, "MAG", "MagY", 1),
    MappedField(Fields.MAGNETOMETER, 2, "MAG", "MagZ", 1),
    MappedField(Fields.QUATERNION, 0, "XKQ1", "Q1", 1, C=0),
    MappedField(Fields.QUATERNION, 1, "XKQ1", "Q2", 1, C=0),
    MappedField(Fields.QUATERNION, 2, "XKQ1", "Q3", 1, C=0),
    MappedField(Fields.QUATERNION, 3, "XKQ1", "Q4", 1, C=0),
    MappedField(Fields.PRESSURE, 0, "BARO", "Press", ureg.Pa),
    MappedField(Fields.TEMPERATURE, 0, "BARO", "Temp", ureg.celsius),
]


