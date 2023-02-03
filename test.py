from pathlib import Path

import numpy as np

from pipapo import Particle, ParticleContainer
from pipapo.utils.bining import *

particles = ParticleContainer.from_vtk("input.vtk", diameter_keyword="diameter")

particles = particles.where(particles.phase_ID == 2, index_only=False)
# particles = particles.in_box(b := np.array([40, 40, 25]), -0.5 * b)

print(particles.mean_of_field("diameter"))


def test_function(p):
    return p.radius**2


x_bins = np.linspace(-10, 10, 8)
y_bins = np.linspace(-10, 10, 8)
z_bins = np.linspace(-10, 10, 8)
xx, yy, zz = np.meshgrid(x_bins, y_bins, z_bins)
positions = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
radius = np.ones(len(positions)) * 2
ids = np.arange(len(radius))
# particles = ParticleContainer(position=positions, radius=radius, id=ids)

#
## bin_id = bin_particles(particles, np.array([40, 40, 25]) * 2 + 12, [2, 5, 3])
# bin_id = bin_bounding_box_by_n_bins(particles, [2, 5, 3])
# particles.add_field("bin_id", bin_id.reshape(-1, 1))
# bin_id = bin_bounding_box_by_max_radius(particles)
# particles.add_field("bin_id_max", bin_id.reshape(-1, 1))
# particles.export("test.vtk")
# print(particles.bounding_box(True))
# print(particles.bounding_box())
#
# a = particles.position
# particles.export("test.csv")
#
# particles = ParticleContainer.from_csv(
#    "test.csv", position_keywords=[f"position_{i}" for i in range(3)]
# )
# print(a - particles.position)
# a = particles[-1]
# print(a)
# a.phase_ID = 10
# a.bin_id = a.bin_id * 2
# print(a)
# print(particles.id)
# print(particles)
particles.reset_ids()
contacts = particles.get_contacts()
particles.add_field(
    "coordination_number", np.array(contacts.coordination_number).reshape(-1, 1)
)

outs = Path("outs")
for i, cp in enumerate(contacts):
    parcon = particles[cp.contact_partners_ids]
    parcon.add_field("gaps", cp.gaps)
    parcon.export(outs / f"con_{i}.vtk")


particles.export("test_contacts.vtk")
print(contacts.coordination_number)


from collections import defaultdict


def connected_components(lists):
    neighbors = defaultdict(set)
    seen = set()
    for each in lists:
        for item in each:
            neighbors[item].update(each)

    def component(node, neighbors=neighbors, seen=seen, see=seen.add):
        nodes = set([node])
        next_node = nodes.pop
        while nodes:
            node = next_node()
            see(node)
            nodes |= neighbors[node] - seen
            yield node

    for node in neighbors:
        if node not in seen:
            yield sorted(component(node))


cluster = -np.ones(len(particles))
total = 0
for c, ids in enumerate(connected_components(contacts.contact_partners_ids)):
    print(len(ids))
    for i in ids:
        cluster[i] = c
        total = c
        particles[ids].export(outs / f"clusters_{total}.vtk")
total += 1
# for i, id in enumerate(particles.get_isolated_particles_id()):
#    cluster[id] = total + i
# particles.add_field("cluster_id", cluster)

mean = contacts.mean_of_field("gaps", concatenate=True)
print(mean)
import matplotlib.pyplot as plt

# hist, bins = contacts.histogram_of_field("gaps", concatenate=True, density=True)
# plt.plot(bins[1:], hist, "b-x")
hist, bins = contacts.histogram_of_field(
    "coordination_number", concatenate=True, density=True
)
plt.plot(bins[:-1], hist, "r-x")
plt.show()
