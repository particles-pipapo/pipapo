from pathlib import Path

import numpy as np
import pytest
from pipapo import ParticleContainer

from .testing_utils import assert_close, assert_equal


def test_initialization(particlecontainer_fixture):
    """Test initialization."""
    particles, (
        position,
        radius,
        radius_squared,
        n_particles,
    ) = particlecontainer_fixture
    assert assert_equal(particles.position, position.reshape(-1, 3))
    assert assert_equal(particles.radius, radius.reshape(-1, 1))
    assert assert_equal(particles.radius_squared, radius_squared.reshape(-1, 1))
    assert assert_equal(particles.id, np.arange(n_particles).reshape(-1, 1))
    assert assert_equal(
        set(particles.field_names), {"position", "radius", "radius_squared", "id"}
    )


def test_initialization_mismatching_arrays():
    """Test array dimension mismatch."""
    with pytest.raises(
        ValueError, match="Dimension mismatch between fields! radius: 10, position: 5"
    ):
        ParticleContainer(radius=np.arange(10), position=np.ones((5, 3)))


def test_initialization_invalid_mandatory_field_not_provided():
    """Test if exception is raised in case mandatory fields are provided."""
    with pytest.raises(
        Exception,
        match="The mandatory fields position, radius were not provided.",
    ):
        ParticleContainer(field_2=np.ones(2))


def test_initialization_invalid_field_and_field_names():
    """Test if exception is raised in case field and field_names are provided."""
    with pytest.raises(
        Exception,
        match="You provided field_names and fields. You can only provide the one or the other!",
    ):
        ParticleContainer(
            "field_1", "field_2", radius=np.ones(2), position=np.ones((3, 2))
        )


def test_volume(particlecontainer_fixture):
    """Test computation of the sum of volumes."""
    particles, (_, radius, _, _) = particlecontainer_fixture
    reference_volumes = 4 * np.pi / 3 * radius**3
    assert np.abs(particles.volume_sum() - np.sum(reference_volumes)) < 1e-8


def test_surface(particlecontainer_fixture):
    """Test computation of the sum of volumes."""
    particles, (_, radius, _, _) = particlecontainer_fixture
    reference_surface = 4 * np.pi * radius**2
    assert np.abs(particles.surface_sum() - np.sum(reference_surface)) < 1e-8


def test_update_particle_by_id(particlecontainer_fixture, particle_1):
    """Test update by id."""
    particles, _ = particlecontainer_fixture
    particle, (position, radius, radius_squared) = particle_1
    particles.update_particle(particle, 5)
    assert assert_equal(particles.position[5], position)
    assert assert_equal(particles.radius[5], radius)
    assert assert_equal(particles.radius_squared[5], radius_squared)


def test_update_particle_by_id_in_particle(particlecontainer_fixture, particle_1):
    """Test update by id in particle."""
    particles, _ = particlecontainer_fixture
    particle, (position, radius, radius_squared) = particle_1
    particle.id = 5
    particles.update_particle(particle)
    assert assert_equal(particles.position[5], position)
    assert assert_equal(particles.radius[5], radius)
    assert assert_equal(particles.radius_squared[5], radius_squared)


def test_export_csv(tmp_path, particlecontainer_fixture):
    """Test if file is exported."""
    particles, _ = particlecontainer_fixture
    path = Path(tmp_path) / "export.csv"
    particles.export(path)
    assert path.is_file()


def test_export_vtk(tmp_path, particlecontainer_fixture):
    """Test if file is exported"""
    particles, _ = particlecontainer_fixture
    path = Path(tmp_path) / "export.vtk"
    particles.export(path)
    assert path.is_file()


def test_export_failure(tmp_path, particlecontainer_fixture):
    """Test if file is not exported and an error is raised."""
    particles, _ = particlecontainer_fixture
    path = Path(tmp_path) / "export.not_existing"
    with pytest.raises(
        IOError,
        match="Filetype .not_existing unknown. Supported export file types are vtk, vtp or csv.",
    ):
        particles.export(path)


def test_bounding_box_position(particlecontainer_fixture):
    """Test bounding box by position."""
    particles, _ = particlecontainer_fixture
    center, lengths = particles.bounding_box(by_position=True)
    lower_bounds_obtained = center - lengths * 0.5
    upper_bounds_obtained = center + lengths * 0.5
    lower_bounds_reference = np.min(particles.position, axis=0)
    upper_bounds_reference = np.max(particles.position, axis=0)
    np.testing.assert_allclose(lower_bounds_reference, lower_bounds_obtained)
    np.testing.assert_allclose(upper_bounds_reference, upper_bounds_obtained)


def test_bounding_box(particlecontainer_fixture):
    """Test bounding box for the full particles."""
    particles, _ = particlecontainer_fixture
    center, lengths = particles.bounding_box()
    lower_bounds_obtained = center - lengths * 0.5
    upper_bounds_obtained = center + lengths * 0.5
    lower_bounds_reference = np.min(particles.position - particles.radius, axis=0)
    upper_bounds_reference = np.max(particles.position + particles.radius, axis=0)
    np.testing.assert_allclose(lower_bounds_reference, lower_bounds_obtained)
    np.testing.assert_allclose(upper_bounds_reference, upper_bounds_obtained)


def test_particle_center_in_box():
    """Test particles in box."""
    xx, yy, zz = np.meshgrid(a := np.linspace(-5, 5, 10), a, a)
    position = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
    particles = ParticleContainer(position=position, radius=np.ones(len(position)))
    in_box = particles.particle_center_in_box(
        np.array([5, 5, 5]), box_center=np.array([5, 5, 5]) / 2
    )

    # The in box function is set for one octant -> number of particles divided by 8
    assert len(in_box) == len(particles) // 8


def test_from_vtk(tmp_path, particlecontainer_fixture):
    """Test if particles are reloaded correctly from vtk."""
    particles, _ = particlecontainer_fixture
    path = Path(tmp_path) / "export.vtk"
    particles.export(path)

    particles_loaded = ParticleContainer.from_vtk(path)
    assert set(particles.field_names) == set(particles_loaded.field_names)
    for field_name in particles.field_names:
        reference = getattr(particles, field_name)
        loaded = getattr(particles_loaded, field_name)
        assert assert_close(reference, loaded, tol=1e-8)


def test_from_csv(tmp_path, particlecontainer_fixture):
    """Test if particles are reloaded correctly from csv."""
    particles, _ = particlecontainer_fixture
    path = Path(tmp_path) / "export.csv"
    particles.export(path)

    particles_loaded = ParticleContainer.from_csv(
        path, position_keywords=[f"position_{i}" for i in range(3)]
    )

    assert set(particles.field_names) == set(particles_loaded.field_names)
    for field_name in particles.field_names:
        reference = getattr(particles, field_name)
        loaded = getattr(particles_loaded, field_name)
        assert assert_close(reference, loaded, tol=1e-8)


def test_from_csv_failure(tmp_path, particlecontainer_fixture):
    """Check if error is raised."""
    particles, _ = particlecontainer_fixture
    path = Path(tmp_path) / "export.csv"
    particles.export(path)

    with pytest.raises(KeyError, match="x"):
        ParticleContainer.from_csv(path)


def test_export_warning():
    """Test if warning is raised when trying to export empty set."""
    particles = ParticleContainer("radius", "position")
    with pytest.warns():
        particles.export("test.vtk")


def test_remove_mandatory_field(particlecontainer_fixture):
    """Test remove of mandatory field."""
    particles, _ = particlecontainer_fixture
    field_name_to_be_removed = "radius"
    with pytest.raises(
        Exception, match="Mandatory fields: position, radius, id can not be deleted!"
    ):
        particles.remove_field(field_name_to_be_removed)
