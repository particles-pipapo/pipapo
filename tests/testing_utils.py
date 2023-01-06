from collections.abc import Iterable

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


def indexer(field, idx, reshape=True):
    # Lists can not be indexed by lists
    if isinstance(idx, Iterable):
        result = [field[i] for i in idx]

    else:
        result = field[idx]

    if isinstance(field, np.ndarray):
        result = np.array(result)
        if reshape:
            result = np.array(result).reshape(-1, field.shape[1])
        return result
    else:
        return result


def get_object_dict(obj):
    return obj.__dict__
