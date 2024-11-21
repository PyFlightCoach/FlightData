
from flightdata import Flight, Constructs, SVar, Table, Origin
from geometry import Point, Base, P0, Time
import numpy as np
from .wind import WindModel, WindModelBuilder


R = 287.058
GAMMA = 1.4

def get_rho(pressure, temperature):
    return pressure / (R * temperature)

def sl_assumption(sec):
    return np.full((len(sec), 2), [101325, 288.15, get_rho(101325, 288.15)])


class Air(Base):
    cols = ["P", "T", "rho"]
    
    @staticmethod
    def iso_sea_level(length: int):
        return Air(101325, 288.15, get_rho(101325, 288.15)).tile(length)

    @staticmethod
    def from_pt(pressure, temperature):
        return Air(pressure, temperature, get_rho(pressure, temperature))

class Environment(Table):
    constructs = Table.constructs + Constructs([
        SVar("atm", Air, ["P", "T", "rho"], lambda tab: Air.iso_sea_level(len(tab))),
        SVar("wind", Point, ["wvx", "wvy", "wvz"], lambda tab: P0(len(tab)))
    ])

    @staticmethod
    def build(flight: Flight, origin: Origin, wmodel: WindModel):
        return Environment.from_constructs(
            time=flight.time_flight,
            atm=Air(
                flight.air_pressure.to_numpy(), 
                flight.air_temperature.to_numpy(), 
                get_rho(flight.air_pressure, flight.air_temperature).to_numpy()
            ),
            wind=wmodel(flight.gps_altitude - origin.pilot_position.alt)
        )

    @staticmethod
    def from_flight(flight: Flight, origin: Origin=None):
        origin = flight.origin if origin is None else origin
        return Environment.from_constructs(
            Time.from_t(np.array(flight.data.time_flight)),
            Air.from_pt(flight.air_pressure.to_numpy(), flight.air_temperature.to_numpy() + 273.15),
            origin.rotation.transform_point(Point(flight.wind_N.to_numpy(), flight.wind_E.to_numpy(), np.zeros(len(flight))))
        )
        
        