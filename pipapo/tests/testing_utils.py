import numpy as np


def assert_equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_array_equal(a, b)
        return True
    else:
        return a == b


def assert_close(a, b, tol=1e-8):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_allclose(a, b, atol=tol, rtol=0)
        return True
    else:
        return np.abs(a - b) < tol
