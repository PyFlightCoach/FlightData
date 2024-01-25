'''Take a flow and a controls, return the aerodynamic forces and moments'''
from flightdata import Coefficients
from . import Model
import numpy as np
import geometry as g


class SimpleAerodynamic(Model):
    nparms=12
    def __call__(self, time,alpha,beta,a,e,r):
        c = self.parms
        return Coefficients.from_constructs(
            time,
            force=g.Point(
                c[0] + c[1] * alpha**2 ,
                c[2] * beta,
                c[3] + c[4] * alpha
            ),
            moment=g.Point(
                c[5] * a + c[6] * r,
                c[7] + c[8] * alpha + c[9] * e,
                c[10] * beta + c[11] * r
            ),
        )

    initial = [
        0.03, 0.2,
        2,
        0.0, 3.0,
        1.0, 1.0,
        0.0, 1.0, 0.01,
        0.2, 0.01
    ]