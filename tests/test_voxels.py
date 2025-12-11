"""Test voxel container stuff."""

import numpy as np
import pytest
from pipapo_testing_utils.asserting_utils import assert_equal

from pipapo.particles import ParticleContainer
from pipapo.utils.container import Container
from pipapo.utils.voxels import (
    VoxelContainer,
    reverse_running_index,
    round_up_division,
    running_index,
)

# pylint: disable=protected-access


@pytest.fixture(
    name="voxels",
    params=["single_particle", "two_particles", "n_particles"],
)
def fixture_voxels(request):
    """Voxel container fixture."""
    particles = None
    n_voxels = None
    if request.param == "single_particle":
        particles = create_container(1)
        n_voxels = 280
    elif request.param == "two_particles":
        particles = create_container(2)
        n_voxels = 528
    else:
        particles = create_container(10)
        n_voxels = 2592

    return (
        VoxelContainer.from_particles(particles, particles.get_bounding_box()),
        n_voxels,
    )


@pytest.mark.parametrize(
    "test_values",
    [
        ((20, 3), 7),
        ((20, 4), 5),
        ((np.array([10, 5, 20]), 4), np.array([3, 2, 5])),
    ],
)
def test_round_up_division(test_values):
    """Test the round up utilility."""
    (a, b), expected_result = test_values
    assert_equal(round_up_division(a, b), expected_result)


@pytest.mark.parametrize(
    "test_values",
    [
        ((1, 2, 3, np.array([5, 5, 5])), 86),
        ((0, 2, 3, np.array([5, 4, 8])), 70),
    ],
)
def test_running_index(test_values):
    """Test running index."""
    (i, j, k, n_dim), expected_result = test_values
    assert running_index(i, j, k, n_dim) == expected_result


@pytest.mark.parametrize(
    "test_values",
    [
        ((86, np.array([5, 5, 5])), (1, 2, 3)),
        ((70, np.array([5, 4, 8])), (0, 2, 3)),
        ((60, np.array([5, 4, 8])), (0, 0, 3)),
    ],
)
def test_reverse_running_index(test_values):
    """Test running index."""
    (c, n_dim), expected_result = test_values
    assert reverse_running_index(c, n_dim) == expected_result


def test_warning_plt(voxels):
    """Test if warning is raised when trying to plot large sets."""
    voxels, n_voxels_expected = voxels
    if n_voxels_expected > 2000:
        with pytest.raises(ValueError):
            voxels.plot(show=False)


def test_voxelize_particlecontainers(voxels):
    """Test if number of expected particles is correct."""
    voxels, n_voxels_expected = voxels
    assert len(voxels) == n_voxels_expected


def create_container(n_elements):
    """Particle container fixture."""
    radius = np.ones(n_elements)
    position = np.column_stack([np.arange(n_elements) * 1.6, np.ones((n_elements, 2))])
    container = ParticleContainer(radius=radius, position=position)
    return container


def test_new_instance(voxels):
    """Test new voxels."""
    voxels, _ = voxels
    new_voxels: VoxelContainer = voxels.create_new_instance(
        voxels._properties,
        voxels._field_data,
        voxels._additional_properties(),
    )

    assert new_voxels._properties == voxels._properties
    assert new_voxels._field_data == voxels._field_data
    assert new_voxels._additional_properties() == voxels._additional_properties()
    assert new_voxels.box == voxels.box
    assert new_voxels.voxel_size == voxels.voxel_size
    np.testing.assert_array_equal(new_voxels.n_voxels_dim, voxels.n_voxels_dim)


def test_properties(voxels):
    """Test properties."""
    for property_name, value in voxels[0]._properties.items():
        assert_equal(getattr(voxels[0], property_name), value)


def test_failure_wrong_size(voxels):
    """Test failure."""
    voxels[0].data["some_data"] = np.ones(42)
    with pytest.raises(ValueError, match="Data and Properties"):
        voxels[0].data  # pylint: disable=pointless-statement


def test_wrong_size_container(voxels):
    """Test no failure."""
    Container._safe = False
    voxels[0].data["some_data"] = np.ones(42)
    assert voxels[0].data
    Container._safe = True


# pylint: enable=protected-access
