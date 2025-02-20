"""Classes related to contacts and contact pairs."""

import abc

import numpy as np

from pipapo.utils import sphere_intersection, sphere_plane_intersection
from pipapo.utils.dataclass import Container, NumpyContainer, has_len
from pipapo.utils.interfaces import InterfaceContainer


class Pair:
    """Pair dataclass."""

    def __init__(self, particle_ids, gap, interface_area, **fields):
        """Initialise particle object.

        Args:
            particle_ids (np.integer): Ids of particle for this pair
            gap (float): Gap of the pair
            interface_area (float): Interface area of the pair.
        """
        self.particle_ids = particle_ids
        self.gap = gap
        self.interface_area = interface_area
        for key, value in fields.items():
            if has_len(value) and len(value) == 1:
                value = value[0]
            setattr(self, key, value)


class PairContainer(NumpyContainer):
    """Container for contact data per contact pair."""

    # Id is handled by the dataclass
    _MANDATORY_FIELDS = ["particle_ids", "gap", "interface_area"]

    def __init__(self, *field_names, **fields):
        """Initialise particles container.

        Args:
            field_names (list): Field names list
            fields (dict): Dictionary with field names and values.
        """
        self.particle_ids = None
        self.gap = None
        self.interface_area = None

        if fields:
            if not set(ParticlePairContainer._MANDATORY_FIELDS).intersection(
                list(fields.keys())
            ) == set(ParticlePairContainer._MANDATORY_FIELDS):
                raise TypeError(
                    f"The mandatory fields {', '.join(ParticlePairContainer._MANDATORY_FIELDS)} "
                    "were not provided."
                )
        else:
            field_names = list(
                set(field_names).union(ParticlePairContainer._MANDATORY_FIELDS)
            )
        super().__init__(Pair, *field_names, **fields)

    def get_total_interface_area(self):
        """Get the total interface area.

        Returns:
            float: Sum of all interface areas
        """
        return np.sum(self.interface_area)

    @abc.abstractmethod
    def to_interfaces(self, particles):
        """Generate interfaces from particle pairs."""

    @abc.abstractmethod
    def export(self, particles, file_path):
        """Export the pairs."""


class ParticlePairContainer(PairContainer):
    """Container for contact data per particle pair."""

    def __str__(self):
        """Particle particle pair container description."""
        string = "\npipapo particle-particle pairs\n"
        string += f"  with {len(self)} pairs\n"
        string += f"  with fields: {', '.join(list(self.field_names))}"
        return string

    @classmethod
    def from_particles(cls, particles):
        """Construct contacts from particles.

        Args:
            particles (pipapo.ParticleContainer): Particle set to look for contacts

        Returns:
            ParticlePairContainer: particle pairs
        """
        particle_ids = []
        gap = []
        interface_area = []

        for i, (radius_i, position_i) in enumerate(
            zip(particles.radius, particles.position)
        ):
            for j, (radius_j, position_j) in enumerate(
                zip(particles.radius, particles.position)
            ):
                if j <= i:
                    continue
                center_to_center = sphere_intersection.get_center_to_center(
                    position_i, position_j
                )
                interface_gap = sphere_intersection.get_gap(
                    np.sqrt(np.sum(center_to_center**2)), radius_i, radius_j
                )

                if interface_gap < 0:
                    particle_ids.append([i, j])
                    gap.append(interface_gap)
                    interface_area.append(
                        sphere_intersection.get_interface_area(
                            np.sqrt(np.sum(center_to_center**2)), radius_i, radius_j
                        )
                    )
        return cls(particle_ids=particle_ids, gap=gap, interface_area=interface_area)

    def to_interfaces(self, particles):
        """Convert particle pairs and particles to interface objects.

        Args:
            particles (pipapo.ParticleContainer): Particles associated to the interfaces.

        Returns:
            pipapo.InterfaceContainer: Interface container
        """
        positions = []
        radii = []
        normals = []

        for interface in self:
            particle_i = particles[interface.particle_ids[0]]
            particle_j = particles[interface.particle_ids[1]]
            interface_radius, interface_normal, interface_center = (
                sphere_intersection.get_sphere_intersection_data(
                    particle_i.position,
                    particle_i.radius,
                    particle_j.position,
                    particle_j.radius,
                )
            )

            positions.append(interface_center)
            radii.append(interface_radius)
            normals.append(interface_normal)

        return InterfaceContainer(
            position=positions, radius=radii, normal=normals, **self.to_dict()
        )

    def to_contacts(self, particles):
        """Create contacts for each particle.

        Args:
            particles (pipapo.Particles): Particle container

        Returns:
            ContactContainer: contact data for each particle
        """
        contacts = dict((particle_id, []) for particle_id in range(len(particles)))

        for particle_particle_ids in self.particle_ids:
            i = particle_particle_ids[0]
            j = particle_particle_ids[1]
            contacts[i].append(j)
            contacts[j].append(i)

        return ContactContainer([kv[1] for kv in sorted(contacts.items())])

    def export(self, particles, file_path):
        """Export particle pairs.

        Args:
            particles (pipapo.Particles): Particle container
            file_path (pathlib.Path): Path to export the pairs
        """
        self.to_interfaces(particles).export(file_path)


class ParticleWallPairContainer(PairContainer):
    """Container for contact data per contact between wall and particle."""

    def __str__(self):
        """Particle wall pair container description."""
        string = "\npipapo particle-wall pairs\n"
        string += f"  with {len(self)} pairs\n"
        string += f"  with fields: {', '.join(list(self.field_names))}"
        return string

    def to_interfaces(self, particles, wall_point, wall_normal):
        """Create interface.

        Args:
            particles (pipapo.ParticleContainer): Particles looking for contacts
            wall_point (np.ndarray): Point on the wall to define the plane
            wall_normal (np.ndarray): Wall normal POINTING INWARDS

        Returns:
            pipapo.InterfaceContainer: Interface container
        """
        positions = []
        radii = []
        normals = []

        for particle_id in self.particle_ids.flatten():
            particle_i = particles[particle_id]

            interface_radius, interface_normal, interface_center = (
                sphere_plane_intersection.get_sphere_plane_intersection_data(
                    particle_i.position, particle_i.radius, wall_point, wall_normal
                )
            )

            positions.append(interface_center)
            radii.append(interface_radius)
            normals.append(interface_normal)

        return InterfaceContainer(
            position=positions, radius=radii, normal=normals, **self.to_dict()
        )

    @classmethod
    def from_particles_and_wall(cls, particles, wall_point, wall_normal):
        """Get particle wall pairs form particles and wall data.

        Args:
            particles (pipapo.ParticleContainer): Particles looking for contacts
            wall_point (np.ndarray): Point on the wall to define the plane
            wall_normal (np.ndarray): Wall normal POINTING INWARDS

        Returns:
            ParticleWallPairContainer: the pairs
        """
        particle_ids = []
        gap = []
        interface_area = []

        for i, (radius_i, position_i) in enumerate(
            zip(particles.radius, particles.position)
        ):
            center_to_wall_point = sphere_plane_intersection.get_center_to_plane_point(
                position_i, wall_point
            )
            center_to_wall_distance = (
                sphere_plane_intersection.get_center_to_plane_distance(
                    center_to_wall_point, wall_normal
                )
            )
            interface_gap = sphere_plane_intersection.get_gap(
                center_to_wall_distance, radius_i
            )

            if interface_gap < 0:
                particle_ids.append(i)
                gap.append(interface_gap)
                interface_area.append(
                    sphere_plane_intersection.get_interface_area(
                        center_to_wall_distance, radius_i
                    )
                )
        return cls(particle_ids=particle_ids, gap=gap, interface_area=interface_area)

    def export(self, particles, wall_point, wall_normal, file_path):
        """Export particle wall pairs.

        Args:
            particles (pipapo.ParticleContainer): Particles looking for contacts
            wall_point (np.ndarray): Point on the wall to define the plane
            wall_normal (np.ndarray): Wall normal POINTING INWARDS
            file_path (pathlib.Path): Path to export the pairs
        """
        self.to_interfaces(particles, wall_point, wall_normal).export(file_path)


class ContactContainer(Container):
    """Container for contact data per particle."""

    def __init__(self, contact_partners_ids):
        """Initialise contact container.

        Args:
            contact_partners_ids (list, optional): List of contact partners. Defaults to None.
        """
        self.contact_partners_ids = None

        super().__init__(self, contact_partners_ids=contact_partners_ids)

    def __str__(self):
        """Contact container description."""
        string = "\npipapo contacts\n"
        string += f"  with fields: {', '.join(list(self.field_names))}"
        return string

    def get_isolated_particles_id(self):
        """Array with isolated particles."""
        return [
            particle_id
            for particle_id, contact_partners in enumerate(self.contact_partners_ids)
            if len(contact_partners) == 0
        ]

    def get_connected_clusters(self):
        """Get the connected clusters, i.e. particle sets which a in contact with each other.

        This solution is inspired from https://stackoverflow.com/a/13837045
        """

        def connected_components(neighbors):
            seen = set()

            def component(node):
                nodes = set([node])
                while nodes:
                    node = nodes.pop()
                    seen.add(node)
                    nodes |= neighbors[node] - seen
                    yield node

            for node in neighbors:
                if node not in seen:
                    yield component(node)

        graph = dict(((i, set(c)) for i, c in enumerate(self.contact_partners_ids)))
        return [list(c) for c in connected_components(graph)]

    def to_particle_pairs(self, particles):
        """Get particle pairs from contacts.

        Args:
            particles (pipapo.ParticleContainer): Particles looking for contacts

        Returns:
            pipapo.ParticlePairContainer: The particle pairs
        """
        particle_particle_ids = []
        gap = []
        interface_area = []
        other_fields = {}

        for i, partners in enumerate(self.contact_partners_ids):
            particle_i = particles[i]
            contact_i = self[i]

            for ij, j in enumerate(partners):
                if j <= i:
                    continue
                particle_j = particles[j]
                center_to_center = sphere_intersection.get_center_to_center(
                    particle_i.position, particle_j.position
                )
                gap_interaction = sphere_intersection.get_gap(
                    center_to_center, particle_i.radius, particle_j.radius
                )

                particle_particle_ids.append([i, j])
                gap.append(gap_interaction)
                interface_area.append(
                    sphere_intersection.get_interface_area(
                        np.sqrt(np.sum(center_to_center**2)),
                        particle_i.radius,
                        particle_j.radius,
                    )
                )

                for field_name in set(self.field_names).difference(
                    ["contact_partners_ids", "id"]
                ):
                    other_fields[field_name].append(getattr(contact_i, field_name)[ij])

        return ParticlePairContainer(
            particle_particle_ids=particle_particle_ids,
            gap=gap,
            interface_area=interface_area,
        )

    def export(self, particles, file_path):
        """Export the contacts as particle pairs.

        Args:
            particles (pipapo.Particles): Particle container
            file_path (pathlib.Path): Path to export the pairs
        """
        self.to_particle_pairs(particles).export(particles, file_path)
