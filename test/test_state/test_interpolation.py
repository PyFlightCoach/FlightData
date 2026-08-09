import geometry as g
import numpy as np
import pytest

from flightdata.state.interpolation import SplineInterpolator3D, RotationInterpolator


@pytest.fixture
def sample_spline_data():
    """Generates a simple 2-point valid trajectory segment."""
    time = g.Time.from_t(np.array([0.0, 2.0]))
    pos = g.Point([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
    vel = g.Point([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    acc = g.Point([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return time, pos, vel, acc



def test_boundary_conditions_exact_match(sample_spline_data):
    """Verifies the spline exactly returns the starting and ending node states."""
    time, pos, vel, acc = sample_spline_data
    spline = SplineInterpolator3D(time, pos, vel, acc)
    
    # Evaluate at exactly t = 0.0 (Start node)
    p_start, v_start, a_start = spline(0.0)
    g.checks.assert_almost_equal(p_start, pos[0])
    g.checks.assert_almost_equal(v_start, vel[0])
    g.checks.assert_almost_equal(a_start, acc[0])
    
    # Evaluate at exactly t = 2.0 (End node)
    p_end, v_end, a_end = spline(2.0)
    g.checks.assert_almost_equal(p_end, pos[1])
    g.checks.assert_almost_equal(v_end, vel[1])
    g.checks.assert_almost_equal(a_end, acc[1])


def test_scalar_vs_array_output_shapes(sample_spline_data):
    """Checks that scalar inputs yield tuples of vectors, and array inputs yield matrices."""
    time, pos, vel, acc = sample_spline_data
    spline = SplineInterpolator3D(time, pos, vel, acc)
    
    # Scalar check
    res_scalar = spline(1.0)
    assert isinstance(res_scalar, tuple)
    assert len(res_scalar[0]) == 1
    
    # Array check
    t_queries = np.array([0.5, 1.0, 1.25, 1.5])
    res_array = spline(t_queries)
    assert isinstance(res_array, tuple)
    assert len(res_array[0]) == 4


def test_time_clipping_out_of_bounds(sample_spline_data):
    """Verifies that queries outside the time bounds are safely clipped."""
    time, pos, vel, acc = sample_spline_data
    spline = SplineInterpolator3D(time, pos, vel, acc)
    
    # Below lower bound (-1.0 -> should clip to 0.0)
    p_low, v_low, a_low = spline(-1.0)
    p_exact, v_exact, a_exact = spline(0.0)
    g.checks.assert_equal(p_low, p_exact)
    
    # Above upper bound (5.0 -> should clip to 2.0)
    p_high, _, _ = spline(5.0)
    p_end, _, _ = spline(2.0)
    g.checks.assert_equal(p_high, p_end)


def test_multi_segment_spline():
    """Tests if multi-interval configurations map indices correctly."""
    time = g.Time.from_t([0.0, 1.0, 2.0])
    pos = g.Point([[0, 0, 0], [1, 1, 1], [2, 4, 8]])
    vel = g.Point([[0, 0, 0], [1, 2, 3], [2, 4, 8]])
    acc = g.Point([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    
    spline = SplineInterpolator3D(time, pos, vel, acc)
    
    # Check a value distinctly inside the second segment (t=1.5)
    p, v, a = spline(1.5)
    assert len(p) == 1



