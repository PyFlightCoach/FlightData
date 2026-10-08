from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import geometry as g
import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.spatial.transform import Rotation, RotationSpline


@dataclass
class SplineInterpolator3D:
    time: g.Time
    pos: g.Point
    vel: g.Point
    acc: g.Point | None = None
    mode: Literal["Univariate", "Cubic", "CubicHermite", "QuinticHermite"] = None
    smoothing: float = 0.01

    _pos_spline: tuple[UnivariateSpline | CubicSpline, ...] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        n = len(self.time)
        assert len(self.pos) == n

        assert len(self.pos) == n
        if self.vel is not None:
            assert len(self.vel) == n, (
                "Velocity array length must match time and position arrays"
            )
        if self.acc is not None:
            assert len(self.acc) == n, (
                "Acceleration array length must match time and position arrays"
            )

        if self.mode is None:
            if self.vel is not None:
                if self.acc is not None:
                    self.mode = "QuinticHermite"
                else:
                    self.mode = "CubicHermite"
            else:
                self.mode = "Univariate"

        match self.mode:
            case "Univariate":
                self._pos_spline = tuple(
                    UnivariateSpline(
                        self.time.t, self.pos.data[:, i], k=3, s=self.smoothing
                    )
                    for i in range(3)
                )
            case "Cubic":
                self._pos_spline = tuple(
                    CubicSpline(self.time.t, self.pos.data[:, i], bc_type="natural")
                    for i in range(3)
                )
            case "CubicHermite":
                assert self.vel is not None, (
                    "Velocity must be provided for Cubic Hermite interpolation"
                )
            case "QuinticHermite":
                assert self.vel is not None, (
                    "Velocity must be provided for Quintic Hermite interpolation"
                )
                assert self.acc is not None, (
                    "Acceleration must be provided for Quintic Hermite interpolation"
                )
            case None:
                if self.acc is not None and self.vel is not None:
                    self.mode = "QuinticHermite"
                elif self.vel is not None:
                    self.mode = "CubicHermite"

    def __call__(
        self, t_query: npt.NDArray[np.float64] | float
    ) -> tuple[g.Point, g.Point, g.Point]:
        """Evaluate position, velocity, and acceleration at query times."""
        t_q = np.atleast_1d(t_query).astype(float)
        t_clipped = np.clip(t_q, self.time.t[0], self.time.t[-1])

        if self.mode in ["Cubic", "Univariate"]:
            return self._evaluate_position_spline(t_clipped)

        # 1. Locate indices and compute normalized time metrics
        idx, u, dt = self._locate_segments(t_clipped)

        # 2. Gather boundary constraints for the active segments
        p0, p1 = self.pos[idx - 1], self.pos[idx]
        v0, v1 = self.vel[idx - 1], self.vel[idx]

        if self.acc is None:
            pos = self._cubic_position(u, dt, p0, p1, v0, v1)
            vel = self._cubic_velocity(u, dt, p0, p1, v0, v1)
            acc = self._cubic_acceleration(u, dt, p0, p1, v0, v1)
        else:
            a0, a1 = self.acc[idx - 1], self.acc[idx]

            # 3. Compute all interpolated values
            pos = self._quintic_position(u, dt, p0, p1, v0, v1, a0, a1)
            vel = self._quintic_velocity(u, dt, p0, p1, v0, v1, a0, a1)
            acc = self._quintic_acceleration(u, dt, p0, p1, v0, v1, a0, a1)

        return pos, vel, acc

    def _evaluate_position_spline(
        self,
        t: npt.NDArray[np.float64],
    ):
        assert self._pos_spline is not None

        pos = g.Point(np.column_stack([spline(t) for spline in self._pos_spline]))

        vel = g.Point(
            np.column_stack(
                [
                    (
                        spline.derivative(1)(t)
                        if self.mode == "UnivariateSpline"
                        else spline(t, 1)
                    )
                    for spline in self._pos_spline
                ]
            )
        )

        acc = g.Point(
            np.column_stack(
                [
                    (
                        spline.derivative(2)(t)
                        if self.mode == "UnivariateSpline"
                        else spline(t, 2)
                    )
                    for spline in self._pos_spline
                ]
            )
        )

        return pos, vel, acc

    def _locate_segments(self, t_clipped):
        """Finds active segments and returns indexes, local time 'u', and durations."""
        idx = np.searchsorted(self.time.t, t_clipped)
        idx = np.clip(idx, 1, len(self.time.t) - 1)

        t0 = self.time.t[idx - 1]
        t1 = self.time.t[idx]
        dt = np.where(t1 - t0 == 0.0, 1e-9, t1 - t0)
        u = (t_clipped - t0) / dt

        return idx, u, dt

    def _cubic_position(self, u, dt, p0, p1, v0, v1):
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2

        return h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1

    def _quintic_position(self, u, dt, p0, p1, v0, v1, a0, a1):
        """Evaluates 3D position using base quintic polynomials."""
        h0 = 1 - 10 * u**3 + 15 * u**4 - 6 * u**5
        h1 = u - 6 * u**3 + 8 * u**4 - 3 * u**5
        h2 = 0.5 * u**2 - 1.5 * u**3 + 1.5 * u**4 - 0.5 * u**5
        h3 = 0.5 * u**3 - u**4 + 0.5 * u**5
        h4 = -4 * u**3 + 7 * u**4 - 3 * u**5
        h5 = 10 * u**3 - 15 * u**4 + 6 * u**5

        return (
            h0 * p0
            + h1 * (v0 * dt)
            + h2 * (a0 * dt**2)
            + h3 * (a1 * dt**2)
            + h4 * (v1 * dt)
            + h5 * p1
        )

    def _cubic_velocity(self, u, dt, p0, p1, v0, v1):
        dh00 = 6 * u**2 - 6 * u
        dh10 = 3 * u**2 - 4 * u + 1
        dh01 = -6 * u**2 + 6 * u
        dh11 = 3 * u**2 - 2 * u

        return (dh00 * p0 + dh10 * dt * v0 + dh01 * p1 + dh11 * dt * v1) / dt

    def _quintic_velocity(self, u, dt, p0, p1, v0, v1, a0, a1):
        """Evaluates 3D velocity using first-derivative polynomials."""
        dh0 = -30 * u**2 + 60 * u**3 - 30 * u**4
        dh1 = 1 - 18 * u**2 + 32 * u**3 - 15 * u**4
        dh2 = u - 4.5 * u**2 + 6 * u**3 - 2.5 * u**4
        dh3 = 1.5 * u**2 - 4 * u**3 + 2.5 * u**4
        dh4 = -12 * u**2 + 28 * u**3 - 15 * u**4
        dh5 = 30 * u**2 - 60 * u**3 + 30 * u**4

        return (
            dh0 * p0
            + dh1 * (v0 * dt)
            + dh2 * (a0 * dt**2)
            + dh3 * (a1 * dt**2)
            + dh4 * (v1 * dt)
            + dh5 * p1
        ) / dt

    def _cubic_acceleration(self, u, dt, p0, p1, v0, v1):
        d2h00 = 12 * u - 6
        d2h10 = 6 * u - 4
        d2h01 = -12 * u + 6
        d2h11 = 6 * u - 2

        return (d2h00 * p0 + d2h10 * dt * v0 + d2h01 * p1 + d2h11 * dt * v1) / dt**2

    def _quintic_acceleration(self, u, dt, p0, p1, v0, v1, a0, a1):
        """Evaluates 3D acceleration using second-derivative polynomials."""
        d2h0 = -60 * u + 180 * u**2 - 120 * u**3
        d2h1 = -36 * u + 96 * u**2 - 60 * u**3
        d2h2 = 1 - 9 * u + 18 * u**2 - 10 * u**3
        d2h3 = 3 * u - 12 * u**2 + 10 * u**3
        d2h4 = -24 * u + 84 * u**2 - 60 * u**3
        d2h5 = 60 * u - 180 * u**2 + 120 * u**3

        return (
            d2h0 * p0
            + d2h1 * (v0 * dt)
            + d2h2 * (a0 * dt**2)
            + d2h3 * (a1 * dt**2)
            + d2h4 * (v1 * dt)
            + d2h5 * p1
        ) / (dt**2)



@dataclass
class RotationInterpolator:
    time: g.Time
    att: g.Quaternion
    rvel: g.Point | None = None
    mode: Literal["Slerp", "RotationSpline", "Squad"] = None
    _interpolator: (
        RotationSpline | Callable[[npt.NDArray[np.float64]], g.Quaternion] | None
    ) = field(init=False, repr=False, default=None)
 
    def __post_init__(self):
        n = len(self.time)
        assert len(self.att) == n, "Attitude array length must match time array"
        if self.rvel is not None:
            assert len(self.rvel) == n, (
                "Rotational velocity array length must match time array"
            )
 
        if self.mode is None:
            if self.rvel is not None:
                self.mode = "Squad"
            else:
                self.mode = "Slerp"
 
        if self.mode == "RotationSpline":
            self._interpolator = RotationSpline(self.time.t, Rotation(self.att.xyzw))
        elif self.mode == "Slerp":
            self._interpolator = self.att.slerp(self.time.t, "nearest")
        elif self.mode == "Squad":
            assert self.rvel is not None, (
                "Rotational velocity must be provided for Squad interpolation"
            )
            self._interpolator = self.att.squad(self.rvel, self.time)
 
    def __call__(
        self,
        t_query: npt.NDArray[np.float64] | float,
        mode: Literal["body", "world"] = "body",
    ) -> tuple[g.Quaternion, g.Point]:
        """ 
        TODO check rotationspline mode
        """
        t_q = np.atleast_1d(t_query).astype(float)
        t_clipped = np.clip(t_q, self.time.t[0], self.time.t[-1])
 
        if self.mode == "RotationSpline":
            att = g.Quaternion(self._interpolator(t_clipped).as_quat()[:, [3, 0, 1, 2]])
            rvel = g.Point(self._interpolator(t_clipped, 1))
            if mode == "world":
                rvel = att.transform_point(rvel)
            return att, rvel
        elif self.mode in ["Slerp", "Squad"]:
            return self._interpolator(t_clipped, mode)
