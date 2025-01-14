from __future__ import annotations
import geometry as g
import numpy as np
from json import load, dump
from typing import Self
from dataclasses import dataclass   
from schemas import fcj  


@dataclass
class Origin(object):
    '''This class defines an aerobatic box in the world, it uses the pilot position and the direction 
    in which the pilot is facing (normal to the main aerobatic manoeuvering plane)'''
    name: str
    pos: g.GPS # position of pilot
    heading: float# direction pilot faces in radians from North (clockwise)

    @property
    def lat(self):
        return self.pos.lat[0] 
    
    @property
    def long(self):
        return self.pos.long[0]
    
    @property
    def alt(self):
        return self.pos.alt[0]

    @property
    def rotation(self):
        # converts NED to x right, y heading direction, z up
        return g.Euler(np.pi, 0, self.heading + np.pi/2)  

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
        
        position = g.GPS(flight.pos_latitude.iloc[0], flight.pos_longitude.iloc[0], flight.pos_altitude.iloc[0])
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

    def to_f3a_zone(self, file: str):
        
        centre = self.pilot_position.offset(
            100 * g.Point(np.cos(self.heading), np.sin(self.heading), 0.0)
        )

        with open(file, 'w') as f:
            for line in [
                "Emailed box data for F3A Zone Pro - please DON'T modify!",
                self.name,
                self.pilot_position.lat[0],
                self.pilot_position.long[0],
                centre.lat[0],
                centre.long[0],
                self.pilot_position.alt[0]
            ]:
                print(line, file=f)
                
    @staticmethod
    def from_f3a_zone(file_path: str):
        if hasattr(file_path, "read"):
            lines = file_path.read().splitlines()
        else:
            with open(file_path, "r") as f:
                lines = f.read().splitlines()
        return Origin.from_points(
            lines[1],
            g.GPS(float(lines[2]), float(lines[3]), float(lines[6])),
            g.GPS(float(lines[4]), float(lines[5]), float(lines[6]))
        )

    @staticmethod
    def from_fcjson_parameters(parms: fcj.Parameters):
        return Origin(
            "FCJ_box",
            g.GPS(
                float(parms.pilotLat), 
                float(parms.pilotLng), 
                float(parms.pilotAlt)
            ),
            float(parms.rotation)
        )
    
    def gps_to_point(self, gps: g.GPS) -> g.Point:
        return self.rotation.transform_point(gps - self.pos)

    @staticmethod
    def from_fcj_origin(fcj_origin: fcj.Origin):
        return Origin(
            "fcj",
            g.GPS(fcj_origin.lat, fcj_origin.lng, fcj_origin.alt).offset(
                g.Point(fcj_origin.move_north, fcj_origin.move_east, 0)
            ),
            np.radians(fcj_origin.heading),
        )

    
    @property
    def fcj_origin(self):
        return fcj.Origin(
            lat=self.lat,
            lng=self.long,
            alt=self.alt,
            heading=np.degrees(self.heading),
        )