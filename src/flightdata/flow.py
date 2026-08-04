from typing import ClassVar

import geometry as g
import numpy as np

from flightdata import Table
from flightdata.base.table import Construct, LabelGroups
from flightdata.coefficients import Coefficients
from flightdata.environment import Environment


class Attack(g.Base):
    cols: ClassVar[list[str]] = ["alpha", "beta", "q"]


class Flow(Table):
    _constructs: ClassVar[list[Construct]] = Table._constructs + [
        Construct("aspd", g.Point, ["x", "y", "z"]),
        Construct("flow", Attack, ["alpha", "beta", "q"]),
    ]

    def __init__(
        self,
        time: g.Time,
        aspd: g.Point,
        flow: Attack,
        labels: LabelGroups = None,
    ):
        self.aspd = aspd
        self.flow = flow
        super().__init__(time=time, labels=labels)

    @staticmethod
    def from_state(body, env: Environment):

        airspeed = body.vel - body.att.inverse().transform_point(env.wind)

        with np.errstate(invalid="ignore"):
            alpha = np.arctan2(airspeed.z, airspeed.x)
        alpha[np.isnan(alpha)] = 0.0

        stab_airspeed = g.Euler(
            np.zeros(len(alpha)), alpha, np.zeros(len(alpha))
        ).transform_point(airspeed)

        with np.errstate(invalid="ignore"):
            beta = np.arctan2(stab_airspeed.y, stab_airspeed.x)
        beta[np.isnan(beta)] = 0.0

        with np.errstate(invalid="ignore"):
            q = 0.5 * env.rho * abs(airspeed) ** 2
        q[np.isnan(q)] = 0.0

        return Flow(body.time, airspeed, Attack(alpha, beta, q))

    def rotate(self, coefficients: Coefficients, dclda: float, dcydb: float):
        new_flow = Attack(
            -coefficients.cz / dclda, -coefficients.cy / dcydb, self.flow.q
        )
        return Flow(coefficients.time, flow=new_flow, aspd=self.aspd)
