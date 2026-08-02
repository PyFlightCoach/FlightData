from dataclasses import dataclass

import geometry as g
import numpy.typing as npt

from flightdata import Table
from flightdata.constants import Constants


@dataclass(repr=False)
class Coefficients(Table):
    force: g.Point 
    moment: g.Point

    @staticmethod
    def from_state(sec, q: npt.NDArray, consts: Constants):
        u = sec.vel
        du = sec.acc
        w = sec.rvel
        moment=g.P0(len(sec))#I*(dw + w.cross(w)) / (q * consts.s) 

        return Coefficients.from_constructs(
            sec.time,
            force=(du + w.cross(u)) * consts.M.m[0] / (q * consts.S),
            moment=moment / g.Point(consts.b, consts.c, consts.b).tile(len(moment))
        )
