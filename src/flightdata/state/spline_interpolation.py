from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import geometry as g
import numpy as np
import numpy.typing as npt
from scipy.interpolate import BSpline

from flightdata.ardupilot import Field, StateData


@dataclass
class SplineState:
    rotation: RSpline
    translation: TSpline

    def pos(self, t: npt.NDArray[np.float64] | float) -> g.Point:
        return self.translation.pos(t)

    def att(self, t: npt.NDArray[np.float64] | float) -> g.Quaternion:
        return self.rotation.att(t)

    def rvel(self, t: npt.NDArray[np.float64] | float) -> g.Point:
        return self.rotation.rvel(t)

    def wvel(self, t: npt.NDArray[np.float64] | float) -> g.Point:
        return self.translation.vel(t)

    def vel(self, t: npt.NDArray[np.float64] | float) -> g.Point:
        return self.att(t).inverse().transform_point(self.wvel(t))

    def wacc(self, t: npt.NDArray[np.float64] | float) -> g.Point:
        return self.translation.acc(t)

    def acc(self, t: npt.NDArray[np.float64] | float) -> g.Point:
        return self.att(t).inverse().transform_point(self.wacc(t))

    @staticmethod
    def build(
        data: StateData,
        s: float = 10,
    ):
        rspline = RSpline.rotation_spline(data.att, data.rvel)
        tspline = TSpline.independent(data.pos, data.vel, data.acc, s=s)
        return SplineState(rspline, tspline)

    def to_dict(self) -> dict[str, dict[str, str | npt.NDArray[np.float64]]]:
        return {
            "rotation": self.rotation.to_dict(),
            "translation": self.translation.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict[str, dict[str, str | npt.NDArray[np.float64]]]) -> SplineState:
        return SplineState(
            rotation=RSpline.from_dict(data["rotation"]),
            translation=TSpline.from_dict(data["translation"]),
        )

@dataclass
class RSpline:
    """rvel is body frame rotational velocity"""

    att: Callable[[npt.NDArray[np.float64]], g.Quaternion]
    rvel: Callable[[npt.NDArray[np.float64]], g.Point]
    mode: Literal["slerp", "squad", "rotation_spline"]
    obj: g.splines.RotationSplineFunction | g.splines.SquadFunction | g.splines.SlerpFunction

    @staticmethod
    def slerp(att: Field[g.Quaternion], rvel: Field[g.Point] | None = None) -> RSpline:
        _slerp = att.data.slerp(att.t)
        return RSpline(
            att=lambda t: _slerp(t, axis_rates=False, mode="body"),
            rvel=lambda t: _slerp(t, mode="body", axis_rates=True)[1],
            mode="slerp",
            obj=_slerp
        )

    @staticmethod
    def squad(att: Field[g.Quaternion], rvel: Field[g.Point]) -> RSpline:
        assert all(att.t == rvel.t), (
            "Squad interpolation requires matching time indices for att and rvel"
        )
        _squad = att.data.squad(rvel.data, att.t)
        return RSpline(
            att=lambda t: _squad(t)[0],
            rvel=lambda t: _squad(t, mode="body")[1],
            mode="squad",
            obj=_squad
        )

    @staticmethod
    def rotation_spline(
        att: Field[g.Quaternion], rvel: Field[g.Point] | None = None
    ) -> RSpline:
        _spline = att.data.rotation_spline(att.t)
        return RSpline(
            att=lambda t: _spline(t, 0),
            rvel=lambda t: _spline(t, 1),
            mode="rotation_spline",
            obj=_spline
        )

    def to_dict(self) -> dict[str, str | npt.NDArray[np.float64]]:
        return {
            "mode": self.mode,
            "spline": self.obj.to_dict()
        }

    @staticmethod
    def from_dict(data: dict[str, str | npt.NDArray[np.float64]]) -> RSpline:
        mode = data["mode"]
        match mode:
            case "slerp":
                _slerp = g.splines.SlerpFunction.from_dict(data["spline"])
                return RSpline(
                    att=lambda t: _slerp(t, axis_rates=False, mode="body"),
                    rvel=lambda t: _slerp(t, mode="body", axis_rates=True)[1],
                    mode="slerp",
                    obj=_slerp
                )

            case "squad":
                _squad = g.splines.SquadFunction.from_dict(data["spline"])
                return RSpline(
                    att=lambda t: _squad(t)[0],
                    rvel=lambda t: _squad(t, mode="body")[1],
                    mode="squad",
                    obj=_squad
                )
            case "rotation_spline":
                _spline = g.splines.RotationSplineFunction.from_dict(data["spline"])
                return RSpline(
                    att=lambda t: _spline(t, 0),
                    rvel=lambda t: _spline(t, 1),
                    mode="rotation_spline",
                    obj=_spline
                )    

@dataclass
class TSpline:
    """This is all World frame"""

    pos: Callable[[npt.NDArray[np.float64]], g.Point]
    vel: Callable[[npt.NDArray[np.float64]], g.Point]
    acc: Callable[[npt.NDArray[np.float64]], g.Point]
    mode: Literal["independent", "interpolating", "smoothing", "quintic_hermite"]
    obj: g.quintic_hermite_spline.QuinticHermiteSpline | None = None

    @staticmethod
    def independent(
        pos: Field[g.Point],
        vel: Field[g.Point],
        acc: Field[g.Point],
        pos_kwargs: dict[str, object] | None = None,
        vel_kwargs: dict[str, object] | None = None,
        acc_kwargs: dict[str, object] | None = None,
        **kwargs,
    ) -> TSpline:

        return TSpline(
            pos=pos.data.univariate_spline(pos.t, **(kwargs | (pos_kwargs or {}))),
            vel=vel.data.univariate_spline(vel.t, **(kwargs | (vel_kwargs or {}))),
            acc=acc.data.univariate_spline(acc.t, **(kwargs | (acc_kwargs or {}))),
            mode="independent",
        )

    @staticmethod
    def interpolating(
        pos: Field[g.Point],
        vel: Field[g.Point] | None = None,
        acc: Field[g.Point] | None = None,
    ) -> TSpline:
        pos_spline = pos.data.interp_spline(pos.t)
        return TSpline(
            pos=lambda t: pos_spline(t, 0),
            vel=lambda t: pos_spline(t, 1),
            acc=lambda t: pos_spline(t, 2),
            mode="interpolating",
        )

    @staticmethod
    def smoothing(
        pos: Field[g.Point],
        vel: Field[g.Point] | None = None,
        acc: Field[g.Point] | None = None,
        **kwargs,
    ) -> TSpline:
        pos_spline = pos.data.univariate_spline(pos.t, **kwargs)
        return TSpline(
            pos=lambda t: pos_spline(t, 0),
            vel=lambda t: pos_spline(t, 1),
            acc=lambda t: pos_spline(t, 2),
            mode="smoothing",
        )

    @staticmethod
    def quintic_hermite(
        pos: Field[g.Point],
        vel: Field[g.Point],
        acc: Field[g.Point],
    ) -> TSpline:

        spline = g.quintic_hermite_spline.QuinticHermiteSpline(
            pos.t, pos.data, vel.data, acc.data
        )
        return TSpline(
            pos=spline.pos_spline,
            vel=spline.vel_spline,
            acc=spline.acc_spline,
            mode="quintic_hermite",
            obj=spline,
        )

    def to_dict(self) -> dict[str, str | npt.NDArray[np.float64]]:
        odata = {"mode": self.mode}
        match self.mode:
            case "independent":
                return odata | {
                    "pos": self.pos.to_dict(),
                    "vel": self.vel.to_dict(),
                    "acc": self.acc.to_dict(),
                }
            case "smoothing" | "interpolating":
                return odata | {
                    "pos": self.pos.to_dict(),
                }
            case "quintic_hermite":
                return odata | {
                    "time": self.obj.time.to_list(),
                    "pos": self.obj.pos.to_dict(),
                    "vel": self.obj.vel.to_dict(),
                    "acc": self.obj.acc.to_dict(),
                }

    @staticmethod
    def from_dict(data: dict[str, str | npt.NDArray[np.float64]]) -> TSpline:
        mode = data["mode"]
        match mode:
            case "independent":
                return TSpline(
                    pos=BSpline(data["pos"].values())
                )
            case "smoothing" | "interpolating":
                return TSpline.interpolating(
                    pos=Field.from_dict(data["pos"]),
                )
            case "quintic_hermite":
                return TSpline.quintic_hermite(
                    pos=Field.from_dict(data["pos"]),
                    vel=Field.from_dict(data["vel"]),
                    acc=Field.from_dict(data["acc"]),
                )