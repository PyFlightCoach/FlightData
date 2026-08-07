from dataclasses import dataclass

import geometry as g
import numpy as np
import numpy.typing as npt


@dataclass
class QuinticHermiteSpline3D:
    time: g.Time
    pos: g.Point
    vel: g.Point
    acc: g.Point

    def __post_init__(self):
        assert len(self.time) == len(self.pos) == len(self.vel) == len(self.acc), (
            "All input arrays must have the same length."
        )

    def __call__(
        self, t_query: npt.NDArray[np.float64] | float
    ) -> tuple[g.Point, g.Point, g.Point]:
        """Evaluate position, velocity, and acceleration at query times."""
        t_q = np.atleast_1d(t_query).astype(float)
        t_clipped = np.clip(t_q, self.time.t[0], self.time.t[-1])

        # 1. Locate indices and compute normalized time metrics
        idx, u, dt = self._locate_segments(t_clipped)

        # 2. Gather boundary constraints for the active segments
        p0, p1 = self.pos[idx - 1], self.pos[idx]
        v0, v1 = self.vel[idx - 1], self.vel[idx]
        a0, a1 = self.acc[idx - 1], self.acc[idx]

        # 3. Compute all interpolated values
        pos = self._compute_position(u, dt, p0, p1, v0, v1, a0, a1)
        vel = self._compute_velocity(u, dt, p0, p1, v0, v1, a0, a1)
        acc = self._compute_acceleration(u, dt, p0, p1, v0, v1, a0, a1)

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

    def _compute_position(self, u, dt, p0, p1, v0, v1, a0, a1):
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

    def _compute_velocity(self, u, dt, p0, p1, v0, v1, a0, a1):
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

    def _compute_acceleration(self, u, dt, p0, p1, v0, v1, a0, a1):
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
