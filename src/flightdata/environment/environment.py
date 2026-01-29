from typing import ClassVar, overload, Literal
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import geometry as g
from flightdata import Constructs, Flight, Origin, SVar, Table
from flightdata.environment.wind import WindModel


@dataclass(repr=False)
class Environment(Table):
    constructs: ClassVar[Constructs] = Table.constructs + Constructs(
        [
            SVar(
                "atm", g.Air, ["P", "T", "rho"], lambda tab: g.Air.iso_sea_level(len(tab))
            ),
            SVar("wind", g.Point, ["wvx", "wvy", "wvz"], lambda tab: g.P0(len(tab))),
        ]
    )

    @overload
    def __getattr__(self, key: Literal["atm"]) -> g.Air: ...
    @overload
    def __getattr__(self, key: Literal["wind"]) -> g.Point: ...
    def __getattr__(self, key):
        return super().__getattr__(key)
    
    @staticmethod
    def zero(t: npt.NDArray):
        return Environment.from_constructs(
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
