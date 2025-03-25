"""Test voxel container stuff."""
import numpy as np
import pytest
from pipapo.particles import ParticleContainer
from pipapo.utils.voxels import (
    VoxelContainer,
    chunkify,
    reverse_running_index,
    round_up_division,
    running_index,
)

from .testing_utils import assert_equal, assert_equal_list_with_array


@pytest.fixture(
    name="voxelcontainer",
    params=["single_particle", "two_particles", "n_particles"],
)
def fixture_particlecontainer(request):
    """Voxel container fixture."""
    if request.param == "single_particle":
        return VoxelContainer.from_particles(create_container(1)), 280
    elif request.param == "two_particles":
        return VoxelContainer.from_particles(create_container(2)), 528
    else:
        return VoxelContainer.from_particles(create_container(10)), 2592


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
    assert assert_equal(round_up_division(a, b), expected_result)


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


def test_chunkify():
    """Test chunkify."""
    data = np.arange(10)
    assert assert_equal_list_with_array(chunkify(data, 1)[0], data)
    assert assert_equal_list_with_array(chunkify(data, 2), [data[:5], data[5:]])
    assert assert_equal_list_with_array(
        chunkify(data, 3), [data[:3], data[3:6], data[6:]]
    )


def test_warning_reset_id(voxelcontainer):
    """Test if warning is raised when trying to reset the ids."""
    voxels, _ = voxelcontainer
    with pytest.warns():
        voxels.reset_ids()


def test_warning_plt(voxelcontainer):
    """Test if warning is raised when trying to plot large sets."""
    voxels, n_voxels_expected = voxelcontainer
    if n_voxels_expected > 2000:
        with pytest.warns():
            voxels.plot(show=False)


def test_voxelize_particlecontainers(voxelcontainer):
    """Test if number of expected particles is correct."""
    voxels, n_voxels_expected = voxelcontainer
    assert len(voxels) == n_voxels_expected


def create_container(n_elements):
    """Particle container fixture."""
    radius = np.ones(n_elements).reshape(-1, 1)
    position = np.column_stack([np.arange(n_elements) * 1.6, np.ones((n_elements, 2))])
    container = ParticleContainer(radius=radius, position=position)
    return container
