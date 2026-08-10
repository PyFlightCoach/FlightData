"""
Tests for RotationInterpolator.

ASSUMPTIONS TO VERIFY / ADJUST:
- Import of RotationInterpolator below assumes it lives in
  `rotation_interpolator.py` at the project root / importable path. Adjust
  to wherever you actually put the class.
- geometry.Time / geometry.Point / geometry.Quaternion import paths mirror
  the earlier squad tests (geometry.time / geometry.point / geometry.quaternion).
  Fix `make_time()` if Time's constructor differs.
- Assumes scipy is available (RotationSpline mode depends on it).
- Quaternion.almost_equal() handles the q / -q double-cover ambiguity.
"""
import numpy as np
import pytest
from geometry.point import Point
from geometry.quaternion import Quaternion
from geometry.time import Time

from flightdata.state.interpolation import RotationInterpolator


def make_time(ts: np.ndarray) -> Time:
    return Time.from_t(ts)


def z_axis_keyframes(n: int, w: float, t_end: float):
    """Single-axis (z) constant-rate keyframes -- world and body rate are
    identical here, useful for isolating mode-independent behavior."""
    ts = np.linspace(0, t_end, n)
    angles = w * ts
    r = Quaternion.from_axis_angle(Point(np.zeros(n), np.zeros(n), angles))
    rvel = Point(np.zeros(n), np.zeros(n), np.full(n, w))
    return r, rvel, ts


def multi_axis_keyframes():
    """A handful of keyframes spanning multiple rotation axes, with
    explicit (body-frame, per squad_control_points' own convention) target
    rates -- world and body rate genuinely differ here."""
    ts = np.array([0.0, 1.0, 2.0, 3.5, 5.0])
    axang = 0.2 * np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, np.pi / 2],
        [np.pi / 3, np.pi / 4, -np.pi / 6],
        [-np.pi / 4, np.pi / 2, np.pi / 3],
        [np.pi / 2, -np.pi / 3, np.pi / 4],
    ])
    r = Quaternion.from_axis_angle(Point(axang[:, 0], axang[:, 1], axang[:, 2]))
    rate = 0.2 * np.array([
        [0.0, 0.0, 1.2],
        [0.8, -0.5, 0.4],
        [-0.6, 0.9, 0.3],
        [0.5, 0.5, -0.8],
        [0.2, 0.4, 0.6],
    ])
    rvel = Point(rate[:, 0], rate[:, 1], rate[:, 2])
    return r, rvel, ts


class TestValidation:
    def test_mismatched_attitude_length_raises(self):
        r, rvel, ts = z_axis_keyframes(5, 1.0, 4.0)
        t = make_time(ts)
        with pytest.raises(AssertionError):
            RotationInterpolator(time=t, att=Quaternion(r.data[:-1]), rvel=rvel)

    def test_mismatched_rvel_length_raises(self):
        r, rvel, ts = z_axis_keyframes(5, 1.0, 4.0)
        t = make_time(ts)
        with pytest.raises(AssertionError):
            RotationInterpolator(time=t, att=r, rvel=Point(rvel.data[:-1]))

    def test_default_mode_is_squad_when_rvel_given(self):
        r, rvel, ts = z_axis_keyframes(5, 1.0, 4.0)
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r, rvel=rvel)
        assert interp.mode == "Squad"

    def test_default_mode_is_slerp_when_rvel_absent(self):
        r, _, ts = z_axis_keyframes(5, 1.0, 4.0)
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r)
        assert interp.mode == "Slerp"

    def test_squad_mode_without_rvel_raises(self):
        r, _, ts = z_axis_keyframes(5, 1.0, 4.0)
        t = make_time(ts)
        with pytest.raises(AssertionError):
            RotationInterpolator(time=t, att=r, mode="Squad")

    def test_explicit_mode_is_respected(self):
        """Explicitly requesting Slerp even though rvel is provided should
        not silently switch to Squad."""
        r, rvel, ts = z_axis_keyframes(5, 1.0, 4.0)
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r, rvel=rvel, mode="Slerp")
        assert interp.mode == "Slerp"


class TestCallReturnShape:
    @pytest.mark.parametrize("mode_kw", ["Slerp", "Squad", "RotationSpline"])
    def test_returns_quaternion_and_point_tuple(self, mode_kw):
        r, rvel, ts = z_axis_keyframes(5, np.pi / 4, 4.0)
        t = make_time(ts)
        interp = RotationInterpolator(
            time=t, att=r, rvel=rvel if mode_kw != "Slerp" else None, mode=mode_kw
        )
        query = np.linspace(ts[0], ts[-1], 11)
        att, rate = interp(query)
        assert isinstance(att, Quaternion)
        assert isinstance(rate, Point)
        assert len(att) == len(query)
        assert len(rate) == len(query)

    def test_scalar_query_works(self):
        r, rvel, ts = z_axis_keyframes(5, np.pi / 4, 4.0)
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r, rvel=rvel)
        att, rate = interp(2.3)
        assert len(att) == 1
        assert len(rate) == 1


class TestClipping:
    """__call__ clips query times into range instead of raising, unlike
    the raw doslerp/dosquad closures which raise ExtrapolationError."""

    @pytest.mark.parametrize("mode_kw", ["Slerp", "Squad"])
    def test_query_before_range_clips_to_first_sample(self, mode_kw):
        r, rvel, ts = z_axis_keyframes(5, np.pi / 5, 4.0)
        t = make_time(ts)
        interp = RotationInterpolator(
            time=t, att=r, rvel=rvel if mode_kw == "Squad" else None, mode=mode_kw
        )
        att, _ = interp(np.array([ts[0] - 10.0]))
        assert att.almost_equal(Quaternion(r.data[[0]]), tol=1e-6)

    @pytest.mark.parametrize("mode_kw", ["Slerp", "Squad"])
    def test_query_after_range_clips_to_last_sample(self, mode_kw):
        r, rvel, ts = z_axis_keyframes(5, np.pi / 5, 4.0)
        t = make_time(ts)
        interp = RotationInterpolator(
            time=t, att=r, rvel=rvel if mode_kw == "Squad" else None, mode=mode_kw
        )
        att, _ = interp(np.array([ts[-1] + 10.0]))
        assert att.almost_equal(Quaternion(r.data[[-1]]), tol=1e-6)

    def test_no_exception_for_mixed_in_and_out_of_range_query(self):
        r, rvel, ts = z_axis_keyframes(5, np.pi / 5, 4.0)
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r, rvel=rvel)
        query = np.array([ts[0] - 1, ts[1], ts[-1] + 1])
        att, rate = interp(query)  # should not raise
        assert len(att) == 3


class TestSlerpInterpolation:
    def test_matches_samples_at_knot_times(self):
        r, _, ts = z_axis_keyframes(6, np.pi / 6, 5.0)
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r)
        att, _ = interp(ts)
        assert att.almost_equal(r, tol=1e-6)

    def test_output_is_unit_quaternion(self):
        r, _, ts = z_axis_keyframes(5, np.pi / 3, 4.0)
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r)
        query = np.linspace(ts[0], ts[-1], 15)
        att, _ = interp(query)
        np.testing.assert_allclose(abs(att), np.ones(len(query)), atol=1e-8)


class TestSquadInterpolation:
    def test_matches_samples_at_knot_times(self):
        r, rvel, ts = multi_axis_keyframes()
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r, rvel=rvel)
        att, _ = interp(ts)
        assert att.almost_equal(r, tol=1e-6)

    def test_default_mode_is_body_and_matches_explicit_body(self):
        r, rvel, ts = multi_axis_keyframes()
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r, rvel=rvel)
        query = np.linspace(ts[0], ts[-1], 9)
        att_default, rate_default = interp(query)
        att_body, rate_body = interp(query, "body")
        assert att_default.almost_equal(att_body, tol=1e-12)
        np.testing.assert_allclose(rate_default.data, rate_body.data, atol=1e-12)

    def test_body_mode_rate_matches_supplied_rvel_at_knots(self):
        """The whole point of the squad_control_points fix: in body frame
        (the frame the control points are actually built in), the
        interpolated rate should match the supplied rvel almost exactly
        at each keyframe -- for genuinely multi-axis motion, not just a
        single-axis special case."""
        r, rvel, ts = multi_axis_keyframes()
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r, rvel=rvel)
        _, rate = interp(ts, "body")
        np.testing.assert_allclose(rate.data, rvel.data, atol=1e-3)

    def test_world_mode_rate_differs_from_body_for_multiaxis_motion(self):
        """Sanity check that `mode` actually does something for genuinely
        3D motion -- world and body rate should NOT coincide here."""
        r, rvel, ts = multi_axis_keyframes()
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r, rvel=rvel)
        query = np.linspace(ts[0], ts[-1], 9)
        _, rate_body = interp(query, "body")
        _, rate_world = interp(query, "world")
        assert not np.allclose(rate_body.data, rate_world.data, atol=1e-3)

    def test_world_and_body_rate_agree_for_single_axis_motion(self):
        """For pure single-axis rotation, world and body rate coincide --
        contrasts with the multi-axis case above."""
        r, rvel, ts = z_axis_keyframes(6, np.pi / 4, 5.0)
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r, rvel=rvel)
        query = np.linspace(ts[0], ts[-1], 11)
        _, rate_body = interp(query, "body")
        _, rate_world = interp(query, "world")
        np.testing.assert_allclose(rate_body.data, rate_world.data, atol=1e-6)

    def test_mode_does_not_affect_orientation(self):
        """mode should only change which frame the *rate* is reported in,
        never the interpolated orientation itself."""
        r, rvel, ts = multi_axis_keyframes()
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r, rvel=rvel)
        query = np.linspace(ts[0], ts[-1], 9)
        att_body, _ = interp(query, "body")
        att_world, _ = interp(query, "world")
        assert att_body.almost_equal(att_world, tol=1e-12)


class TestRotationSplineInterpolation:
    def test_returns_finite_att_and_rate(self):
        r, rvel, ts = multi_axis_keyframes()
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r, rvel=rvel, mode="RotationSpline")
        query = np.linspace(ts[0], ts[-1], 15)
        att, rate = interp(query)
        assert np.all(np.isfinite(att.data))
        assert np.all(np.isfinite(rate.data))

    def test_output_is_unit_quaternion(self):
        r, rvel, ts = multi_axis_keyframes()
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r, rvel=rvel, mode="RotationSpline")
        query = np.linspace(ts[0], ts[-1], 15)
        att, _ = interp(query)
        np.testing.assert_allclose(abs(att), np.ones(len(query)), atol=1e-6)

    def test_matches_samples_approximately_at_knot_times(self):
        r, rvel, ts = multi_axis_keyframes()
        t = make_time(ts)
        interp = RotationInterpolator(time=t, att=r, rvel=rvel, mode="RotationSpline")
        att, _ = interp(ts)
        assert att.almost_equal(r, tol=1e-4)

