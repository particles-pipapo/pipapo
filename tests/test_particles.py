"""Tests for the ParticleContainer class."""

import pathlib

import numpy as np
import pytest
from pipapo_testing_utils.asserting_utils import assert_close, assert_equal

from pipapo.particles import ParticleContainer
from pipapo.utils.container import Container
from pipapo.utils.domain import Box

np.random.seed(42)


# pylint: disable=protected-access


@pytest.fixture(name="reference_positions")
def fixture_reference_positions():
    """Fixture for particles positions."""
    position = np.random.rand(10, 3) * 10
    return position


@pytest.fixture(name="reference_radii")
def fixture_reference_radii():
    """Fixture for particles radii."""
    radius = 1 + np.random.rand(10) * 2
    return radius


@pytest.fixture(name="reference_data")
def fixture_refernce_data():
    """Fixture for reference particles data."""
    data_entry_1 = np.random.rand(10)
    data_entry_2 = list(range(10))
    return {"data_entry_1": data_entry_1, "data_entry_2": data_entry_2}


@pytest.fixture(name="particles")
def fixture_particles(reference_positions, reference_radii, reference_data):
    """Fixture for ParticleContainer."""
    particles = ParticleContainer(
        position=reference_positions,
        radius=reference_radii,
        data=reference_data,
    )
    return particles


def test_initialization(
    particles, reference_positions, reference_radii, reference_data
):
    """Test initialization of ParticleContainer."""
    assert len(particles) == 10
    np.testing.assert_equal(particles.position, reference_positions)
    np.testing.assert_equal(particles.radius, reference_radii)
    for key in reference_data:
        assert_equal(particles.data[key], reference_data[key])


def test_volume(particles, reference_radii):
    """Test computation of the sum of volumes."""
    reference_volumes = 4 * np.pi / 3 * np.sum(reference_radii**3)
    np.testing.assert_allclose(particles.get_volume_sum(), reference_volumes)


@pytest.mark.parametrize("export_path", ["test.vtk", "test.vtp", "test.csv"])
def test_export(tmp_path, particles, export_path):
    """Test if file is exported."""
    path = pathlib.Path(tmp_path) / export_path
    particles.export(path)
    assert path.is_file()


def test_export_failure(tmp_path, particles):
    """Test if file is not exported and an error is raised."""
    path = pathlib.Path(tmp_path) / "export.not_existing"
    with pytest.raises(
        IOError,
        match="Filetype .not_existing unknown. Supported export file types are vtk, vtp or csv.",
    ):
        particles.export(path)


def test_bounding_box_position(particles):
    """Test bounding box by position."""
    box = particles.get_bounding_box(False)
    lower_bounds_obtained = box.center - box.lengths * 0.5
    upper_bounds_obtained = box.center + box.lengths * 0.5
    lower_bounds_reference = np.min(particles.position, axis=0)
    upper_bounds_reference = np.max(particles.position, axis=0)
    np.testing.assert_allclose(lower_bounds_reference, lower_bounds_obtained)
    np.testing.assert_allclose(upper_bounds_reference, upper_bounds_obtained)


def test_bounding_box(particles):
    """Test bounding box for the full particles."""
    box = particles.get_bounding_box()
    lower_bounds_obtained = box.center - box.lengths * 0.5
    upper_bounds_obtained = box.center + box.lengths * 0.5
    lower_bounds_reference = np.min(
        particles.position - np.atleast_2d(particles.radius).T, axis=0
    )
    upper_bounds_reference = np.max(
        particles.position + np.atleast_2d(particles.radius).T, axis=0
    )
    np.testing.assert_allclose(lower_bounds_reference, lower_bounds_obtained)
    np.testing.assert_allclose(upper_bounds_reference, upper_bounds_obtained)


def test_particle_center_in_box():
    """Test particles in box."""
    xx, yy, zz = np.meshgrid(a := np.linspace(-5, 5, 10), a, a)
    position = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
    particles = ParticleContainer(position=position, radius=np.ones(len(position)))
    box = Box(center=np.array([5, 5, 5]) / 2, lengths=np.array([5, 5, 5]))
    in_box = box.get_particles_in_domain(particles)

    # The in box function is set for one octant -> number of particles divided by 8
    assert len(in_box) == len(particles) // 8


def test_from_vtk(tmp_path, particles):
    """Test if particles are reloaded correctly from vtk."""
    path = pathlib.Path(tmp_path) / "export.vtk"
    particles.export(path)

    particles_loaded = ParticleContainer.from_vtk(path)
    for field_name in particles.data:
        assert_close(
            particles.data[field_name], particles_loaded.data[field_name], tol=1e-8
        )
    for property_name in particles._properties:
        assert_close(
            particles._properties[property_name],
            particles_loaded._properties[property_name],
            tol=1e-8,
        )


def test_from_csv(tmp_path, particles):
    """Test if particles are reloaded correctly from csv."""
    path = pathlib.Path(tmp_path) / "export.csv"
    particles.export(path)

    particles_loaded = ParticleContainer.from_csv(
        path, position_keywords=[f"position_{i}" for i in range(3)]
    )

    assert set(particles.data.keys()) == set(particles_loaded.data.keys())
    for field_name in particles.data:
        assert_close(
            particles.data[field_name], particles_loaded.data[field_name], tol=1e-8
        )
    for property_name in particles._properties:
        assert_close(
            particles._properties[property_name],
            particles_loaded._properties[property_name],
            tol=1e-8,
        )


def test_from_csv_failure(tmp_path, particles):
    """Check if error is raised."""
    path = pathlib.Path(tmp_path) / "export.csv"
    particles.export(path)

    with pytest.raises(KeyError, match="x"):
        ParticleContainer.from_csv(path)


def test_export_warning():
    """Test if warning is raised when trying to export empty set."""
    particles = ParticleContainer(position=np.empty((0, 3)), radius=np.empty((0,)))
    with pytest.warns():
        particles.export("test.vtk")


def test_new_instance(particles):
    """Test new particles."""
    new_particles: ParticleContainer = particles.create_new_instance(
        particles._properties,
        particles._field_data,
        particles._additional_properties(),
    )

    assert new_particles._properties == particles._properties
    assert new_particles._field_data == particles._field_data
    assert new_particles._additional_properties() == particles._additional_properties()


def test_properties(particles):
    """Test properties."""
    for property_name, value in particles._properties.items():
        assert_equal(getattr(particles, property_name), value)


def test_failure_wrong_size(particles):
    """Test failure."""
    particles.data["some_data"] = np.ones(42)
    with pytest.raises(ValueError, match="Data and Properties"):
        particles.data  # pylint: disable=pointless-statement


def test_wrong_size_container(particles):
    """Test no failure."""
    Container._safe = False
    particles.data["some_data"] = np.ones(42)
    assert particles.data
    Container._safe = True


# pylint: enable=protected-access
