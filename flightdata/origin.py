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

import geometry as g
import numpy as np
from json import load, dump
from typing import Self

class Origin(object):
    '''This class defines an aerobatic box in the world, it uses the pilot position and the direction 
    in which the pilot is facing (normal to the main aerobatic manoeuvering plane)'''

    def __init__(self, name, pos: g.GPS, heading: float):
        self.name = name
        self.pos = pos # position of pilot
        self.heading = heading  # direction pilot faces in radians from North (clockwise)
        self.rotation = g.Euler(np.pi, 0, self.heading + np.pi/2)  # converts NED to x right, y heading direction, z up

    @property
    def pilot_position(self):
        return self.pos

    def to_dict(self) -> dict:
        return dict(
            name=self.name,
            pos=self.pos.to_dict(),
            heading=self.heading
        )

    @staticmethod
    def from_dict(data: dict) -> Self:
        return Origin(
            data['name'], 
            g.GPS(**data['pos']), 
            data['heading']
        )

    @staticmethod
    def from_json(file):
        if hasattr(file, 'read'):
            data = load(file)
        else:
            with open(file, 'r') as f:
                data = load(f)
        return Origin.from_dict(data)

    def to_json(self, file):
        with open(file, 'w') as f:
            dump(self.to_dict(), f)
        return file

    def copy(self):
        return Origin(
            self.name,
            self.pos.copy(),
            self.heading
        )

    def __str__(self):
        return "Origin:{}".format(self.to_dict())

    def __repr__(self):
        return f'Origin(heading={np.degrees(self.heading)},pos={self.pilot_position})'

    @staticmethod
    def from_initial(flight):
        '''Generate a box based on the initial position and heading of the model at the start of the log. 
        This is a convenient, but not very accurate way to setup the box. 
        '''
        
        position = g.GPS(flight.gps_latitude[0], flight.gps_longitude[0], flight.gps_altitude[0])
        heading = g.Euler(flight.attitude)[0].transform_point(g.PX())

        return Origin('origin', position, np.arctan2(heading.y, heading.x)[0])

    @staticmethod
    def from_points(name, pilot: g.GPS, centre: g.GPS):
        direction = centre - pilot
        return Origin(
            name,
            pilot,
            np.arctan2(direction.y[0], direction.x[0])
        )

    def to_f3a_zone(self):
        
        centre = self.pilot_position.offset(
            100 * g.Point(np.cos(self.heading), np.sin(self.heading), 0.0)
        )

        oformat = lambda val: "{}".format(val)

        return "\n".join([
            "Emailed box data for F3A Zone Pro - please DON'T modify!",
            self.name,
            oformat(self.pilot_position.lat[0]),
            oformat(self.pilot_position.long[0]),
            oformat(centre.lat[0]),
            oformat(centre.long[0]),
            "120"
        ])


    @staticmethod
    def from_f3a_zone(file_path: str):
        if hasattr(file_path, "read"):
            lines = file_path.read().splitlines()
        else:
            with open(file_path, "r") as f:
                lines = f.read().splitlines()
        return Origin.from_points(
            lines[1],
            g.GPS(float(lines[2]), float(lines[3]), 0),
            g.GPS(float(lines[4]), float(lines[5]), 0)
        )

    @staticmethod
    def from_fcjson_parmameters(data: dict):
        return Origin.from_points(
            "FCJ_box",
            g.GPS(float(data['pilotLat']), float(data['pilotLng']), 0),
            g.GPS(float(data['centerLat']), float(data['centerLng']), 0)
        )


    def gps_to_point(self, gps: g.GPS) -> g.Point:
        pned = gps - self.pilot_position
        return self.rotation.transform_point(g.Point(pned.y, pned.x, -pned.z ))

    
    