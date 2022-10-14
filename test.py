import numpy as np

from pipapo import Particle, ParticleContainer
from pipapo.utils.bining import *

particles = ParticleContainer.from_vtk("input.vtk")

particles = particles.where(particles.phase_ID == 2, index_only=False)
# particles = particles.in_box(b := np.array([40, 40, 25]), -0.5 * b)

print(particles.mean_of_field("diameter"))


def test_function(p):
    return p.radius**2


x_bins = np.linspace(-40, 40, 40)
y_bins = np.linspace(-40, 40, 40)
z_bins = np.linspace(-25, 25, 25)
xx, yy, zz = np.meshgrid(x_bins, y_bins, z_bins)
positions = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
radius = np.ones(len(positions))
ids = np.arange(len(radius))
# particles = ParticleContainer(position=positions, radius=radius, id=ids)

# particles.evaluate_function(test_function, add_as_field=True)


# bin_id = bin_particles(particles, np.array([40, 40, 25]) * 2 + 12, [2, 5, 3])
bin_id = bin_bounding_box_by_n_bins(particles, [2, 5, 3])
particles.add_field("bin_id", bin_id)
bin_id = bin_bounding_box_by_max_radius(particles)
particles.add_field("bin_id_max", bin_id)
particles.export("test.vtk")
print(particles.bounding_box(True))
print(particles.bounding_box())
breakpoint()
