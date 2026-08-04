from typing import ClassVar

import geometry as g
import numpy as np
import numpy.typing as npt

from flightdata import Flight, Origin, Table
from flightdata.base.table import Construct
from flightdata.base.table.labelgroups import LabelGroups
from flightdata.environment.wind import WindModel


class Environment(Table):
    _constructs: ClassVar[list[Construct]] = Table._constructs + [
        Construct("atm", g.Air, ["P", "T", "rho"], lazy=True),
        Construct("wind", g.Point, ["x", "y", "z"], lazy=True),
    ]

    def __init__(self, time: g.Time, atm: g.Air = None, wind: g.Point = None, labels: LabelGroups = None):
        self._atm = atm
        self._wind = wind
        super().__init__(time=time, labels=labels)
        

    @property
    def atm(self) -> g.Air:
        if self._atm is None:
            self._atm = g.Air.iso_sea_level(len(self))
        return self._atm

    @property 
    def wind(self) -> g.Point:
        if self._wind is None:
            self._wind = g.P0(len(self))
        return self._wind
    

        
    @staticmethod
    def zero(t: npt.NDArray):
        return Environment(
            time=g.Time.from_t(t),
            atm=g.Air.iso_sea_level(len(t)),
            wind=g.P0(len(t))           
        )

    @staticmethod
    def from_flight_wmodel(flight: Flight, origin: Origin, wmodel: WindModel):
        return Environment.from_constructs(
            time=g.Time.from_t(flight.time_flight),
            atm=g.Air(
                flight.air_pressure.to_numpy(),
                flight.air_temperature.to_numpy(),
                g.air.get_rho(flight.air_pressure, flight.air_temperature).to_numpy(),
            ),
            wind=wmodel(flight.gps_altitude - origin.pilot_position.alt),
        )

    @staticmethod
    def from_flight(flight: Flight, origin: Origin = None):
        origin = flight.origin if origin is None else origin
        return Environment.from_constructs(
            g.Time.from_t(np.array(flight.data.time_flight)),
            g.Air.from_pt(
                flight.air_pressure.to_numpy(),
                flight.air_temperature.to_numpy() + 273.15,
            ),
            origin.rotation.transform_point(
                g.Point(
                    flight.wind_N.to_numpy(),
                    flight.wind_E.to_numpy(),
                    np.zeros(len(flight)),
                )
            ),
        )
