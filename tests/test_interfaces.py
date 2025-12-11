"""Tests for the InterfaceContainer class."""

import numpy as np
import pytest
from pipapo_testing_utils.asserting_utils import assert_equal

from pipapo.particles import ParticleContainer
from pipapo.utils.container import Container
from pipapo.utils.interfaces import InterfaceContainer

# pylint: disable=protected-access


@pytest.fixture(name="interfaces")
def test_fixture_interfaces():
    """Fixture providing sample data for InterfaceContainer tests."""

    position = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.8, 1.0, 0.0],
            [0.8, 1.0, 0.0],
            [1, 2.5, 0.0],
            [5.0, 0.0, 0.0],
            [5 - 0.8, 1.0, 0.0],
            [5 + 0.8, 1.0, 0.0],
            [-5, 1, 0],
        ]
    )

    particles = ParticleContainer(
        position,
        radius=np.ones(len(position)),
        data={"ids": np.arange(len(position))},
    )

    return particles.get_interfaces()


def test_particle_indices(interfaces):
    """Test particle indices property."""
    expected_particle_indices = [(0, 1), (0, 2), (1, 2), (2, 3), (4, 5), (4, 6), (5, 6)]

    assert_equal(interfaces.particle_indices, expected_particle_indices)


def test_contacts(interfaces):
    """Test contacts method."""
    expected_contacts = {
        0: {1, 2},
        1: {0, 2},
        2: {0, 1, 3},
        3: {2},
        4: {5, 6},
        5: {4, 6},
        6: {4, 5},
        7: set(),
    }
    contacts = interfaces.get_contacts(n_particles=8)
    assert len(contacts) == len(expected_contacts)
    for i, v in expected_contacts.items():
        assert contacts[i] == v


def test_get_connected_clusters(interfaces):
    """Test get_connected_clusters method."""
    expected_clusters = [[0, 1, 2, 3], [4, 5, 6], [7]]
    clusters = interfaces.get_connected_clusters(n_particles=8)
    assert len(clusters) == len(expected_clusters)
    for cluster in expected_clusters:
        assert cluster in clusters


def test_get_isolated_particle_indices(interfaces):
    """Test get_isolated_particle_indices method."""
    isolated_indices = interfaces.get_isolated_particle_indices(n_particles=8)
    expected_isolated_indices = [7]
    assert_equal(isolated_indices, expected_isolated_indices)


def test_new_instance(interfaces):
    """Test new interfaces."""
    new_interfaces: InterfaceContainer = interfaces.create_new_instance(
        interfaces._properties,
        interfaces._field_data,
        interfaces._additional_properties(),
    )

    assert new_interfaces._properties == interfaces._properties
    assert new_interfaces._field_data == interfaces._field_data
    assert (
        new_interfaces._additional_properties() == interfaces._additional_properties()
    )


def test_failure_wrong_size(interfaces):
    """Test failure."""
    interfaces.data["some_data"] = np.ones(42)
    with pytest.raises(ValueError, match="Data and Properties"):
        interfaces.data  # pylint: disable=pointless-statement


def test_wrong_size_container(interfaces):
    """Test no failure."""
    Container._safe = False
    interfaces.data["some_data"] = np.ones(42)
    assert interfaces.data
    Container._safe = True


# pylint: enable=protected-access
