"""Type hinting utilities for pipapo package."""

from typing import TypeAlias

import numpy as np

float_np_array: TypeAlias = np.typing.NDArray[  # pylint: disable=invalid-name
    np.float64
]
int_np_array: TypeAlias = np.typing.NDArray[np.integer]  # pylint: disable=invalid-name
