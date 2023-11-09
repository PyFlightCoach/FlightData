
from flightanalysis import Constructs, SVar, Table
from geometry import Point, Base, P0
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



class Environment(Table):
    constructs = Table.constructs + Constructs([
        SVar("atm", Air, ["P", "T", "rho"], lambda tab: Air.iso_sea_level(len(tab))),
        SVar("wind", Point, ["wvx", "wvy", "wvz"], lambda tab: P0(len(tab)))
    ])

    @staticmethod
    def build(flight, sec, wmodel: WindModel):
        from flightdata import Fields

        df = flight.read_fields(Fields.PRESSURE)
        df = df.assign(temperature_0=291.15)
        df = df.assign(rho=get_rho(df["pressure_0"], df["temperature_0"]))

        return Environment.from_constructs(
            time=sec.time,
            atm=Air(df.to_numpy()),
            wind=wmodel(sec.pos.z)
        )




