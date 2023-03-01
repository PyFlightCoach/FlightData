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
log_field_map = dict()

log_field_map["timestamp"] = MappedField(Fields.TIME, 0, "timestamp", ureg.second)
log_field_map["time"] = MappedField(Fields.TIME, 1, "time", ureg.microsecond)

log_field_map["N"] = MappedField(Fields.POSITION, 0, "N", ureg.meter)
log_field_map["E"] = MappedField(Fields.POSITION, 1, "E", ureg.meter)
log_field_map["D"] = MappedField(Fields.POSITION, 2, "D", ureg.meter)

log_field_map["r"] = MappedField(Fields.ATTITUDE, 0, "r", ureg.degree)
log_field_map["p"] = MappedField(Fields.ATTITUDE, 1, "p", ureg.degree)
log_field_map["yw"] = MappedField(Fields.ATTITUDE, 2, "yw", ureg.degree)

log_field_map["VN"] = MappedField(Fields.VELOCITY, 0, "VN", ureg.meter / ureg.second)
log_field_map["VE"] = MappedField(Fields.VELOCITY, 1, "VE", ureg.meter / ureg.second)
log_field_map["VD"] = MappedField(Fields.VELOCITY, 2, "VD", ureg.meter / ureg.second)

log_field_map["wN"] = MappedField(Fields.WIND, 0, "wN", ureg.meter / ureg.second)
log_field_map["wE"] = MappedField(Fields.WIND, 1, "wE", ureg.meter / ureg.second)

fc_json_2_1_io_info = FieldIOInfo(log_field_map)

       