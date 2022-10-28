import numpy as np
import pytest
from pipapo import Particle, ParticleContainer


@pytest.fixture(name="particle_1", scope="session")
def fixture_particle_1():
    """Particle fixture."""
    position = np.array([1, 2, 3])
    radius = 5
    radius_squared = radius**2
    particle = Particle(position, radius, radius_squared=radius_squared)
    return particle, (position, radius, radius_squared)


@pytest.fixture(name="particles_1", scope="session")
def fixture_particles_1():
    np.random.seed(42)
    n_particles = 10
    position = np.random.randn(n_particles, 3)
    radius = np.arange(n_particles)
    radius_squared = radius**2
    particles = ParticleContainer(
        position=position, radius=radius, radius_squared=radius_squared
    )
    return particles, (position, radius, radius_squared, n_particles)
