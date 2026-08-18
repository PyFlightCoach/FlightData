from __future__ import annotations

from dataclasses import dataclass

import geometry as g
import numpy as np
import numpy.typing as npt
import pandas as pd

from flightdata.origin import Origin

from .base import BinFunc


@dataclass
class XKF1:
    t: npt.NDArray[np.float64]
    att: g.Quaternion
    vel: g.Point
    gyro_bias: g.Point

    @staticmethod
    def load(
        fields: dict[str, pd.DataFrame],
        primary_core: BinFunc[npt.NDArray[np.int64] | int],
        origin: Origin | None = None,
    ) -> XKF1:
        """Get the xkf1 data for the primary core only"""
        xkf1 = filter_core(fields["XKF1"], "C", primary_core).dropna()
        _rotation = origin.rotation if origin is not None else g.Q0()
        att = _rotation * g.Euldeg(xkf1.Roll, xkf1.Pitch, xkf1.Yaw)
        wvel = _rotation.transform_point(g.Point(xkf1.VN, xkf1.VE, xkf1.VD))
        return XKF1(
            (xkf1.TimeUS / 1e6).to_numpy(),
            att,
            att.inverse().transform_point(wvel),
            g.Point(xkf1.GX, xkf1.GY, xkf1.GZ).radians(),
        )


@dataclass
class XKF2:
    t: npt.NDArray[np.float64]
    acc_bias: g.Point
    wind: g.Point

    @staticmethod
    def load(
        fields: dict[str, pd.DataFrame],
        primary_core: BinFunc[npt.NDArray[np.int64] | int],
        origin: Origin | None = None,
    ) -> XKF2:
        """Get the xkf2 data for the primary core only"""
        xkf2 = filter_core(fields["XKF2"], "C", primary_core).dropna()
        _rotation = origin.rotation if origin is not None else g.Q0()
        return XKF2(
            (xkf2.TimeUS / 1e6).to_numpy(),
            g.Point(xkf2.AX, xkf2.AY, xkf2.AZ),
            _rotation.transform_point(
                g.Point(xkf2.VWN.to_numpy(), xkf2.VWE.to_numpy(), np.zeros(len(xkf2)))
            ),
        )


@dataclass
class Pos:
    t: npt.NDArray[np.float64]
    pos: g.Point

    @staticmethod
    def load(fields: dict[str, pd.DataFrame], origin: Origin) -> Pos:
        """
        Get the position data from a dict of bin file field dataframes.
        """
        pos = fields["POS"].dropna()

        return Pos(
            (pos.TimeUS / 1e6).to_numpy(),
            origin.gps_to_point(g.GPS(pos.Lat, pos.Lng, pos.Alt)),
        )


@dataclass
class Att:
    t: npt.NDArray[np.float64]
    att: g.Quaternion

    @staticmethod
    def load_att(fields: dict[str, pd.DataFrame], origin: Origin) -> Att:
        """
        Get the attitude data from a dict of bin file field dataframes.
        """
        att = fields["ATT"].dropna()

        return Att(
            (att.TimeUS / 1e6).to_numpy(),
            origin.rotation * g.Euldeg(att.Roll, att.Pitch, att.Yaw),
        )


@dataclass
class IMU:
    t: npt.NDArray[np.float64]
    gyro: g.Point
    acc: g.Point

    @staticmethod
    def load(
        fields: dict[str, pd.DataFrame],
        primary_core: BinFunc[npt.NDArray[np.int64] | int],
    ) -> IMU:
        """Get the imu dataframe"""
        imu = filter_core(fields["IMU"], "I", primary_core).dropna()

        return IMU(
            (imu.TimeUS / 1e6).to_numpy(),
            g.Point(imu.GyrX, imu.GyrY, imu.GyrZ),
            g.Point(imu.AccX, imu.AccY, imu.AccZ),
        )


def primary_core_at_time(
    fields: dict[str, pd.DataFrame],
) -> BinFunc[npt.NDArray[np.int64] | int]:
    """
    Creates a function that takes a time and returns the primary core from the ERR field Subsys 24.

    """
    err = fields.get("ERR", None)
    if err is None:
        cores = np.array([[0,0]])
    else:
        cores = (
            fields["ERR"]
            .loc[fields["ERR"].Subsys == 24, ["TimeUS", "ECode"]]
            .multiply((1e-6, 1))
            .astype({"TimeUS": np.float64, "ECode": np.int64})
            .to_numpy()
        )
        cores = np.pad(cores, ((1,0), (0,0)))

    def get_core(
            t: float | npt.NDArray[np.float64],
        ) -> npt.NDArray[np.int64] | float:
            if len(cores) == 0:
                return np.zeros_like(t, dtype=np.int64)
            else:
                _t = np.atleast_1d(t)
                diff = np.subtract.outer(_t, cores[:, 0])
                rows = np.argmin(np.where(diff > 0, diff, np.inf), axis=1)
                pcores = cores[rows, 1].astype(np.int64)
            return pcores if pd.api.types.is_list_like(t) else pcores[0]

    return get_core


def parameter_at_time(
    fields: dict[str, pd.DataFrame],
    param_name: str,
    type: type = np.int64,
) -> BinFunc[npt.NDArray[np.int64] | int]:
    """
    Creates a function that takes a time and returns the value of the requested parameter at that time
    """
    param_values = (
        fields["PARM"]
        .loc[fields["PARM"].Name == param_name, ["TimeUS", "Value"]]
        .multiply((1e-6, 1))
        .astype({"TimeUS": np.float64})
        .to_numpy()
    )

    def get_parameter_value(
        t: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.int64] | float:
        _t = np.atleast_1d(t)
        diff = np.subtract.outer(_t, param_values[:, 0])
        rows = np.argmin(np.where(diff > 0, diff, np.inf), axis=1)
        pcores = param_values[rows, 1].astype(type)
        return pcores if pd.api.types.is_list_like(t) else pcores[0]

    return get_parameter_value


def filter_core(
    df: pd.DataFrame,
    core_col: str,
    primary_core: BinFunc[npt.NDArray[np.int64] | int],
) -> pd.DataFrame:
    """Get the xkfi data for the primary core only"""

    pcore = primary_core(df.TimeUS / 1e6)
    return df.loc[df[core_col] == pcore] if core_col in df.columns else df
