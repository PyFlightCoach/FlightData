from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import geometry as g
import numpy as np
import numpy.typing as npt
import pandas as pd
from ardupilot_log_reader import Ardupilot

import flightdata.ardupilot.messages as msgs
from flightdata import Origin
from flightdata.ardupilot.base import BinFunc

from .base import Field


@dataclass
class StateData:
    att: Field[g.Quaternion]
    rvel: Field[g.Point] | None
    pos: Field[g.Point]
    vel: Field[g.Point]
    acc: Field[g.Point] | None
    
    @cached_property
    def t0(self) -> float:
        return np.max([_g.t[0] for _g in [self.att, self.rvel, self.pos, self.vel, self.acc] if _g is not None])

    @cached_property
    def t1(self) -> float:
        return np.min([_g.t[-1] for _g in [self.att, self.rvel, self.pos, self.vel, self.acc] if _g is not None])

    @staticmethod
    def parse_bin(
        bin_file: Path | str | Ardupilot, origin: Origin | None = None
    ) -> StateData:
        if not isinstance(bin_file, Ardupilot):
            bin_file = Ardupilot.parse(
                Path(bin_file), ["ATT", "POS", "IMU", "XKF1", "XKF2", "ERR", "GPS", "ORGN"]
            )

        if origin is None:
            origin = Origin("bin_orgn", g.GPS(bin_file.ORGN.iloc[0].Lat, bin_file.ORGN.iloc[0].Lng, bin_file.ORGN.iloc[0].Alt), 0)

        return StateData.parse_fields(bin_file.dfs, origin)

    @staticmethod
    def parse_fields(
        fields: dict[str, pd.DataFrame], origin: Origin
    ) -> dict[str, g.Time | Field[g.Point | g.Quaternion]]:
        """
        Create a dictionary of state entities from the bin file fields and origin.
        """
        active_core = msgs.primary_core_at_time(fields)
        imu_msg = msgs.IMU.load(fields, active_core) if "IMU" in fields else None
        xkf1_msg = msgs.XKF1.load(fields, active_core, origin) if "XKF1" in fields else None
        xkf2_msg = msgs.XKF2.load(fields, active_core, origin) if "XKF2" in fields else None
        pos_msg = msgs.Pos.load(fields, origin)
        att_msg = msgs.Att.load_att(fields, origin)
        return StateData.parse_messages(imu_msg, xkf1_msg, xkf2_msg, pos_msg, att_msg)

    @staticmethod
    def parse_messages(
        imu_msg: msgs.IMU | None, 
        xkf1_msg: msgs.XKF1 | None, 
        xkf2_msg: msgs.XKF2 | None, 
        pos_msg: msgs.Pos, 
        att_msg: msgs.Att,
    ):
        att = Field(att_msg.t, att_msg.att)
        pos = Field(pos_msg.t, pos_msg.pos)

        vel = Field(xkf1_msg.t, xkf1_msg.vel) if xkf1_msg is not None else None

        if imu_msg is not None and xkf1_msg is not None and xkf2_msg is not None:
            gyro = Field(imu_msg.t, imu_msg.gyro)
            gyro_bias = Field(xkf1_msg.t, xkf1_msg.gyro_bias)
            rvel = Field(
                gyro.t, gyro.data - gyro_bias.data.linterp(gyro_bias.t, "nearest")(gyro.t)
            )

            
            
            accelerometer = Field(imu_msg.t, imu_msg.acc)
            accelerometer_bias = Field(xkf2_msg.t, xkf2_msg.acc_bias)
            acc = Field(
                accelerometer.t,
                accelerometer.data
                - accelerometer_bias.data.linterp(accelerometer_bias.t, "nearest")(
                    accelerometer.t
                ),
            )
        else:
            rvel = None
            acc = None
        return StateData( att, rvel, pos, vel, acc)

    def slice(self, start: float, end: float) -> StateData:
        return StateData(
            self.att.slice(start, end),
            self.rvel.slice(start, end) if self.rvel is not None else None,
            self.pos.slice(start, end),
            self.vel.slice(start, end) if self.vel is not None else None,
            self.acc.slice(start, end) if self.acc is not None else None,
        )
