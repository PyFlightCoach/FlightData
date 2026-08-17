from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import geometry as g
import numpy as np
import pandas as pd
from ardupilot_log_reader import Ardupilot

import flightdata.ardupilot.messages as bin
from flightdata import Origin
from flightdata.bindata import BinData

from .base import Field


@dataclass
class StateData:
    att: Field[g.Quaternion]
    rvel: Field[g.Point]
    pos: Field[g.Point]
    vel: Field[g.Point]
    acc: Field[g.Point]
    
    @cached_property
    def t0(self) -> float:
        return np.max([_g.t[0] for _g in [self.att, self.rvel, self.pos, self.vel, self.acc]])

    @cached_property
    def t1(self) -> float:
        return np.min([_g.t[-1] for _g in [self.att, self.rvel, self.pos, self.vel, self.acc]])

    @staticmethod
    def parse_fields(
        fields: dict[str, pd.DataFrame], origin: Origin
    ) -> dict[str, g.Time | Field[g.Point | g.Quaternion]]:
        """
        Create a dictionary of state entities from the bin file fields and origin.
        """
        active_core = bin.primary_core_at_time(fields)

        imu = bin.IMU.load(fields, active_core)
        xkf1 = bin.XKF1.load(fields, active_core, origin)
        xkf2 = bin.XKF2.load(fields, active_core, origin)
        pos = bin.Pos.load(fields, origin)
        att = bin.Att.load_att(fields, origin)

        xkf2_att = att.att.rotation_spline(att.t)(xkf2.t)

        return StateData(
            Field(att.t, att.att),
            Field(xkf1.t, imu.gyro.interp_spline(imu.t)(xkf1.t) - xkf1.gyro_bias),
            Field(pos.t, pos.pos),
            Field(
                xkf1.t,
                origin.rotation.transform_point(xkf1.wvel)
            ),
            Field(xkf2.t, xkf2_att.transform_point(imu.acc.interp_spline(imu.t)(xkf2.t) - xkf2.acc_bias)),
        )

    @staticmethod
    def parse_bin(
        bin_file: Path | str | BinData | Ardupilot, origin: Origin | None = None
    ) -> StateData:
        if not isinstance(bin_file, (BinData, Ardupilot)):
            bin_file = Ardupilot.parse(
                Path(bin_file), ["ATT", "POS", "IMU", "XKF1", "XKF2", "ERR", "GPS", "ORGN"]
            )

        if origin is None:
            origin = Origin("bin_orgn", g.GPS(bin_file.ORGN.iloc[0].Lat, bin_file.ORGN.iloc[0].Lng, bin_file.ORGN.iloc[0].Alt), 0)

        return StateData.parse_fields(bin_file.dfs, origin)

    def slice(self, start: float, end: float) -> StateData:
        return StateData(
            self.att.slice(start, end),
            self.rvel.slice(start, end),
            self.pos.slice(start, end),
            self.vel.slice(start, end),
            self.acc.slice(start, end),
        )
