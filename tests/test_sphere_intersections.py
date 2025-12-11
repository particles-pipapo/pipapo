"""Test sphere intersections."""

import numpy as np

from pipapo.utils import sphere_intersection


def test_center_to_center():
    """Test center to center computation."""
    center_i = np.array([0.0, 0.0, 0.0])
    center_j = np.array([1.0, 1.0, 1.0])
    expected = np.array([1.0, 1.0, 1.0])
    result = sphere_intersection.get_center_to_center(center_i, center_j)
    np.testing.assert_equal(result, expected)


def test_get_interface_area():
    """Test interface area computation."""
    distance = 3.0
    radius_i = 2.5
    radius_j = 1
    expected_area = 1.91440802328128
    result = sphere_intersection.get_interface_area(distance, radius_i, radius_j)
    np.testing.assert_almost_equal(result, expected_area)


def test_get_interface_position_and_size():
    """Test interface position and size computation."""
    distance = 3.0
    radius_i = 2.5
    center_to_center = np.ones(3) / np.sqrt(3) * distance
    excepted_radius = 0.7806247497997997
    expected_normal = np.ones(3) / np.sqrt(3)
    expected_center = np.sqrt(radius_i**2 - excepted_radius**2) * expected_normal
    interface_radius, interface_normal, interface_center = (
        sphere_intersection.get_interface_position_and_size(
            center_to_center, 1.91440802328128, np.array([0.0, 0.0, 0.0]), radius_i
        )
    )
    np.testing.assert_almost_equal(interface_radius, excepted_radius)
    np.testing.assert_almost_equal(interface_normal, expected_normal)
    np.testing.assert_almost_equal(interface_center, expected_center)


def test_get_gap():
    """Test gap computation."""
    distance = 3.0
    radius_i = 2.5
    radius_j = 1
    expected_gap = -0.5
    result = sphere_intersection.get_gap(distance, radius_i, radius_j)
    np.testing.assert_almost_equal(result, expected_gap)
