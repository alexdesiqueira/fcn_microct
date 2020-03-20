from fullconvnets import utils
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_array_equal

import numpy as np


def test_montage_3d_simple_gray():
    n_images, n_planes, n_rows, n_cols = 4, 2, 2, 2
    array_input = np.arange(n_images * n_planes * n_rows * n_cols,
                            dtype=np.float)
    array_input = array_input.reshape(n_images, n_planes, n_rows, n_cols)

    array_output = utils.montage_3d(array_input)
    print(array_output)
    array_reference = np.array([[[0, 1, 8, 9],
                                 [2, 3, 10, 11],
                                 [15.5, 15.5, 15.5, 15.5],
                                 [15.5, 15.5, 15.5, 15.5]],
                                [[4, 5, 12, 13],
                                 [6, 7, 14, 15],
                                 [15.5, 15.5, 15.5, 15.5],
                                 [15.5, 15.5, 15.5, 15.5]],
                                [[15.5, 15.5, 15.5, 15.5],
                                 [15.5, 15.5, 15.5, 15.5],
                                 [16, 17, 24, 25],
                                 [18, 19, 26, 27]],
                                [[15.5, 15.5, 15.5, 15.5],
                                 [15.5, 15.5, 15.5, 15.5],
                                 [20, 21, 28, 29],
                                 [22, 23, 30, 31]]])
    assert_array_equal(array_output, array_reference)
