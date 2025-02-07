from fastdtw.fastdtw import fastdtw
import numpy as np
import geometry as g


def test_dtw_all_indices_returned():
    arr1 = np.concatenate([np.full(5, 0), np.full(5, 1)])
    arr2 = np.concatenate([np.full(2, 0), np.full(8, 1)])
    arr2[9] = 0
    distance, path = fastdtw(arr1, arr2)
    np.testing.assert_array_equal(np.unique(np.array(path)[:,0]), np.arange(10)) 
    np.testing.assert_array_equal(np.unique(np.array(path)[:,1]), np.arange(10)) 
    assert distance == 1


def test_dtw_points():
    p1 = g.PX(1) * np.concatenate([np.full(5, 0), np.full(5, 1)])
    p2 = g.PX(1) * np.concatenate([np.full(2, 0), np.full(8, 1)])
    distance, path = fastdtw(p1.data, p2.data)
    pass