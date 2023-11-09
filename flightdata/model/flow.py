
from flightanalysis import Table, Time, SVar, Constructs, SVar
from geometry import Point, Base, PX, Euler
import numpy as np


class Attack(Base):
    cols = ['alpha', 'beta', 'q']


class Flow(Table):
    constructs = Table.constructs + Constructs([
        SVar("aspd", Point, ["asx", "asy", "asz"], None),
        SVar("flow", Attack, ["alpha", "beta", "q"], None)
    ])

    @staticmethod
    def build(body, env):

        airspeed = body.vel - body.att.inverse().transform_point(env.wind)

        with np.errstate(invalid='ignore'):
            alpha =  np.arctan(airspeed.z / airspeed.x) 
        
        stab_airspeed = Euler(
            np.zeros(len(alpha)), 
            alpha, 
            np.zeros(len(alpha))
        ).transform_point(airspeed)
    
        with np.errstate(invalid='ignore'):
            beta = np.arctan(stab_airspeed.y / stab_airspeed.x)

        with np.errstate(invalid='warn'):
            q = 0.5 * env.rho * abs(airspeed)**2

        return Flow.from_constructs(
            body.time, 
            airspeed,
            Attack(alpha, beta, q)
        )
    
    def rotate(self, coefficients, dclda, dcydb):
        new_flow = Attack(-coefficients.cz / dclda, -coefficients.cy / dcydb, self.flow.q)
        return Flow.from_constructs(coefficients.time, flow=new_flow, aspd=self.aspd)

        