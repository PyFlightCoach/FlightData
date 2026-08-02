
from dataclasses import dataclass
from typing import ClassVar, Literal, overload

import geometry as g
import numpy as np

from flightdata import Table
from flightdata.coefficients import Coefficients
from flightdata.environment import Environment


class Attack(g.Base):
    cols: ClassVar[list[str]] = ['alpha', 'beta', 'q']


@dataclass(repr=False)
class Flow(Table):
    aspd: g.Point
    flow: Attack


    @staticmethod
    def from_state(body, env: Environment):

        airspeed = body.vel - body.att.inverse().transform_point(env.wind)

        with np.errstate(invalid='ignore'):
            alpha =  np.arctan2(airspeed.z, airspeed.x) 
        alpha[np.isnan(alpha)] = 0.0

        stab_airspeed = g.Euler(
            np.zeros(len(alpha)), 
            alpha, 
            np.zeros(len(alpha))
        ).transform_point(airspeed)
    
        with np.errstate(invalid='ignore'):
            beta = np.arctan2(stab_airspeed.y, stab_airspeed.x)
        beta[np.isnan(beta)] = 0.0

        with np.errstate(invalid='ignore'):
            q = 0.5 * env.rho * abs(airspeed)**2
        q[np.isnan(q)] = 0.0
        
        return Flow.from_constructs(
            body.time, 
            airspeed,
            Attack(alpha, beta, q)
        )
    

    def rotate(self, coefficients: Coefficients, dclda: float, dcydb: float):
        new_flow = Attack(-coefficients.cz / dclda, -coefficients.cy / dcydb, self.flow.q)
        return Flow.from_constructs(coefficients.time, flow=new_flow, aspd=self.aspd)

    
    