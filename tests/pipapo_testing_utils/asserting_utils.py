"""Utils for tests."""

from collections.abc import Iterable

import numpy as np


def assert_equal(a, b):
    """Assert if a and b are equal."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


def assert_close(a, b, tol=1e-8):
    """Assert if a and b are equal within tolerance."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_allclose(a, b, atol=tol, rtol=0)
    else:
        assert np.abs(a - b) < tol


def indexer(field, idx):
    """Index iterables based on list."""
    # Lists can not be indexed by lists

    if isinstance(field, list):
        if isinstance(idx, Iterable):
            return [field[i] for i in idx]

        if isinstance(idx, int):
            return [field[idx]]

        if isinstance(idx, slice):
            return field[idx]

        raise IndexError("Index type not supported for lists.")

    if isinstance(field, np.ndarray):
        if isinstance(idx, (list, np.ndarray, slice)):
            return field[idx]
        if isinstance(idx, int):
            return field[[idx]]

        raise IndexError("Index type not supported for np.ndarray.")

    raise TypeError(f"Field type {type(field)} not supported for indexing.")


def get_object_dict(obj):
    """Get dictionary of object."""
    return obj.__dict__
