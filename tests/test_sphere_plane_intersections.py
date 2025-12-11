"""Sphere plane intersection."""

import numpy as np
import pytest

from pipapo.utils import sphere_plane_intersection


@pytest.fixture(name="particle_data")
def fixture_particle_data():
    """Particle data."""
    radius = 5
    center = np.zeros(3)
    return center, radius


@pytest.fixture(name="wall_point")
def fixture_wall_point():
    """Wall point."""
    return np.ones(3)


@pytest.fixture(name="wall_normal", params=(np.ones(3), -np.ones(3)))
def fixture_wall_normal(request):
    """Wall normal."""
    return request.param / np.linalg.norm(request.param)


def test_signed_distance_to_plane(wall_normal, particle_data):
    """Test signed distance."""
    center, _ = particle_data

    plane_point = np.ones(3)

    center_to_plane_signed_distance = (
        sphere_plane_intersection.get_center_to_plane_signed_distance(
            center, plane_point, wall_normal
        )
    )

    np.testing.assert_allclose(
        center_to_plane_signed_distance, np.sign(wall_normal[0]) * np.sqrt(3)
    )


def test_get_gap(particle_data, wall_normal, wall_point):
    """Test gap function."""
    center, radius = particle_data
    signed_distance = sphere_plane_intersection.get_center_to_plane_signed_distance(
        center, wall_point, wall_normal
    )

    expected_gap = -5 + np.abs(signed_distance)
    gap = sphere_plane_intersection.get_gap(signed_distance, radius)
    assert gap == expected_gap


def test_get_sphere_plane(particle_data, wall_normal, wall_point):
    """Test sphere plane instersection data."""
    center, radius = particle_data

    signed_distance = sphere_plane_intersection.get_center_to_plane_signed_distance(
        center, wall_point, wall_normal
    )

    interface_radius, interface_normal, interface_center = (
        sphere_plane_intersection.get_sphere_plane_intersection_data(
            center, radius, signed_distance, wall_normal
        )
    )

    excepted_interface_radius = np.sqrt(5**2 - 3)
    excepted_interface_normal = np.ones(3) / np.sqrt(3)
    excepted_interface_center = np.ones(3)

    assert np.isclose(interface_radius, excepted_interface_radius)
    assert np.allclose(interface_normal, excepted_interface_normal)
    assert np.allclose(interface_center, excepted_interface_center)
