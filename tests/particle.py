import numpy as np
import pytest
from pipapo import Particle

from .testing_utils import assert_equal


def eval_function_example(particle):
    """Nonlinear example function with particle arguments."""
    return eval_function_example_args(particle.radius, particle.position)


def eval_function_example_args(radius, position):
    """Nonlinear example function."""
    return radius + np.sum(position**2)


def test_particle_init(particle_1):
    """Test particle initialization."""
    particle, (position, radius, radius_squared) = particle_1
    assert isinstance(particle, Particle)
    assert assert_equal(particle.position, position)
    assert particle.radius == radius
    assert particle.radius_squared == radius_squared


def test_evaluate_function(particle_1):
    """Test evaluate function method."""
    particle, (position, radius, _) = particle_1
    result = eval_function_example(particle)
    result_particle_call = particle.evaluate_function(eval_function_example)
    result_reference = eval_function_example_args(radius, position)
    assert result == result_particle_call == result_reference


def test_items(particle_1):
    """Test items method."""
    particle, data = particle_1
    reference_dictionary = dict(zip(["position", "radius", "radius_squared"], data))
    for key, value in particle.items():
        assert assert_equal(reference_dictionary[key], value)


def test_volume(particle_1):
    """Test volume computation."""
    particle, (_, radius, _) = particle_1
    assert particle.volume() == 4 * np.pi / 3 * radius**3


def test_surface(particle_1):
    """Test surface computation."""
    particle, (_, radius, _) = particle_1
    assert particle.surface() == 4 * np.pi * radius**2
