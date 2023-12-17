"""Voxel container."""
import warnings
from collections.abc import Iterable
from functools import partial
from multiprocessing import Pool

import meshio
import numpy as np
import pyvista as pv

from pipapo.utils.dataclass import NumpyContainer
from pipapo.utils.io import export, pathify

pv.set_plot_theme("document")


class VoxelContainer(NumpyContainer):
    """Voxel container."""

    def __init__(
        self, center, lengths, voxel_size, n_voxels_dim, *field_names, **fields
    ):
        """Initialise voxel container.

        Args:
            center (np.ndarray): Center of the box
            lengths (np.ndarray): Lengths of the box
            voxel_size (float): Voxel size
            n_voxel_dim (np.ndarray): Number of voxels per dimension
            add_centers (bool, optional): Compute the cell centers. Defaults to True.
            field_names (list): Field names list
            fields (dict): Dictionary with field names and values.
        """
        self.center = center
        self.lengths = lengths
        self.voxel_size = voxel_size
        self.n_voxels_dim = n_voxels_dim
        self.voxel_volume = self.voxel_size * self.voxel_size * self.voxel_size
        super().__init__(*field_names, **fields)

    @classmethod
    def from_box_and_voxel_size(cls, center, lengths, voxel_size, add_centers=True):
        """Create empty voxel container from a box and voxel_size.

        Note that be domain might be slightly bigger as all the voxels are equal sized.

        Args:
            center (np.ndarray): Center of the box
            lengths (np.ndarray): Lengths of the box
            voxel_size (float): Voxel size
            add_centers (bool, optional): Compute the cell centers. Defaults to True.

        Returns:
            VoxelContainer: initialized voxel container
        """
        n_voxels_dim = round_up_division(lengths, voxel_size)
        voxel_container = cls(
            center=center,
            lengths=lengths,
            voxel_size=voxel_size,
            n_voxels_dim=n_voxels_dim,
        )

        # Add the voxel centers if desired
        if add_centers:
            voxel_container.id = np.arange(
                voxel_container.total_number_of_voxels_in_outer_domain()
            ).reshape(-1, 1)
            voxel_container.add_voxel_centers()

        return voxel_container

    def __str__(self):
        """Voxel container descriptions."""
        string = "\npipapo voxel set\n"
        string += f"  with {len(self)} voxels\n"
        string += f"  with center {self.center}\n"
        string += f"  with lengths {self.lengths}\n"
        string += f"  with voxel size {self.voxel_size}\n"
        string += f"  with number of voxel per dim {self.n_voxels_dim}\n"
        string += f"  with fields: {', '.join(list(self.field_names))}"
        return string

    def volume_of_voxels(self):
        """Sum of all voxels with the discretized domain.

        Returns:
            float: Volume
        """
        return self.voxel_volume * len(self)

    def total_number_of_voxels_in_outer_domain(self):
        """Get total number of voxels within the outer domain.

        Returns:
            int: Number of voxels
        """
        return int(self.n_voxels_dim[0] * self.n_voxels_dim[1] * self.n_voxels_dim[2])

    def volume_of_outer_domain(self):
        """Return volume of total domain.

        Returns:
            float: Volume of domain
        """
        return self.voxel_volume * self.total_number_of_voxels_in_outer_domain()

    def to_dict(self):
        """Create dictionary from voxels.

        Returns:
            dict: dictionary
        """
        dictionary = super().to_dict()
        dictionary["voxel_size"] = np.ones(self.id.shape) * self.voxel_size
        return dictionary

    def add_voxel_centers(self):
        """Add voxel centers to fields."""
        voxel_positions = []
        for c in self.id:
            x, y, z = reverse_running_index(c, self.n_voxels_dim)
            center_voxel = (
                self.center
                - 0.5 * self.lengths
                + (np.array([x, y, z]) + 0.5) * self.voxel_size
            )
            voxel_positions.append(center_voxel)
        self.add_field("position", np.array(voxel_positions))

    def export(self, file_path):
        """Export voxels.

        Args:
            file_path (pathlib.Path): Path to be exported
        """
        if not hasattr(self, "position"):
            self.add_voxel_centers()

        file_path = pathify(file_path)
        if file_path.suffix == ".vtu":
            self._export_vtu(file_path)
        else:
            export(
                self.to_dict(),
                file_path,
            )

    def _export_vtu(self, file_path):
        """Create hexahedrons and export as vtu.

        Note that notes will appear multiple times.

        Args:
            file_path (pathlib.Path): Path to file.
        """

        # meshio hexahedron definition
        ref_coordinates = np.array(
            [
                [-1, 1, -1],
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, 1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
            ]
        )

        nodes = []
        for position in self.position:
            nodes_element = ref_coordinates * 0.5 * self.voxel_size + position
            nodes.extend(nodes_element)

        cell_data = self.to_dict()

        # remove position
        cell_data.pop("position")

        # meshio wants the cell data to be wrapped in a list
        for k in cell_data:
            cell_data[k] = [cell_data[k]]

        # create mesh
        mesh = meshio.Mesh(
            np.array(nodes),
            [("hexahedron", np.arange(len(nodes)).reshape(-1, 8))],
            cell_data=cell_data,
        )

        # export the mesh
        mesh.write(file_path)

    def get_pv_cubes(self, field_name=None):
        """Get pyvista cubes.

        Returns:
            list: List of all the pyvista cubes.
        """
        voxels = []
        for i in range(len(self)):
            cube = pv.Cube(
                center=self.position[i],
                x_length=self.voxel_size,
                y_length=self.voxel_size,
                z_length=self.voxel_size,
            )
            if field_name:
                cube["Data"] = np.ones(len(cube.points)) * getattr(self, field_name)[i]
            voxels.append(cube)
        return voxels

    def reset_ids(self):
        """Reset ids."""
        warnings.warn("You can not reset ids of voxel containers!")

    def plot(self, pv_plotter=None, show=True, field_name=None, force=False, **kwargs):
        """Plot voxels.

        Args:
            pv_plotter (pv.Plotter, optional): Plotter object to plot. Defaults to None.
            show (bool, optional): Open the plotting window. Defaults to True.
            field_name (str,optional): Field to plot. Defaults to None
            force (bool,optional): Force plotting also for large sets. Defaults to None
            kwargs (dict): additional keyword arguments for add_mesh

        Returns:
            pv.Plotter: Plotter object
        """
        if not force:
            if len(self) > 2000:
                warnings.warn(
                    "You are trying to plot a large voxel set. If you really want to do this add"
                    " the kwarg force=True."
                )
                return

        if not hasattr(self, "position"):
            self.add_voxel_centers()

        if not pv_plotter:
            pv_plotter = pv.Plotter()

        if not "color" in kwargs and not field_name:
            kwargs["color"] = kwargs.get("color", "purple")

        kwargs["show_edges"] = kwargs.get("show_edges", True)

        for voxel in self.get_pv_cubes(field_name):
            pv_plotter.add_mesh(voxel, scalar_bar_args={"title": field_name}, **kwargs)

        if show:
            pv_plotter.show()

        return pv_plotter

    def __getitem__(self, i):
        """Getitem method.

        Args:
            i (int, slice, iterable): Indexes to get the items

        Returns:
            self.datatype: If index i is an int
            type(self): If index i is iterable or slice
        """
        if isinstance(i, int):
            if i >= len(self) or i < -len(self):
                raise IndexError(f"Index {i} out of range for size {self.__len__()}")
            item = self.datatype(**{key: value[i] for key, value in self._items()})
        elif isinstance(i, Iterable):
            item = type(self)(
                self.center,
                self.lengths,
                self.voxel_size,
                self.n_voxels_dim,
                **{key: [value[j] for j in i] for key, value in self._items()},
            )
        else:
            item = type(self)(
                self.center,
                self.lengths,
                self.voxel_size,
                self.n_voxels_dim,
                **{key: value[i] for key, value in self._items()},
            )
        return item

    @classmethod
    def from_particles(
        cls,
        particles,
        center=None,
        lengths=None,
        voxel_size=None,
        n_voxels_dim=None,
        n_threads=1,
    ):
        """Voxelize particles.

        Args:
            particles (pipapo.ParticleContainer): particles to be voxelized
            center (np.ndarray): center of outer domain. Defaults to the center of the bounding box
                                 of the particle container
            lengths (np.ndarray): lengths of outer domain. Defaults to the lengths of the bounding
                                  box of the particle container
            voxel_size (float): voxel size. defaults to a quarter of the smallest radius of the
                                particles
            n_voxels_dim (np.ndarray): voxels per dimension. Defaults to the number of voxels that
                                       fit in a length of the bounding box of the container
            n_threads (float): threads to parallelize the calculation of the voxels
            return_all (bool): Return all the information of the voxels. Defaults to False
        Returns:
            VoxelContainer: voxel container from particles
        """

        if center is None and lengths is None:
            center, lengths = particles.bounding_box()

        if voxel_size is None:
            voxel_size = float(min(particles.radius) / 4)

        if n_voxels_dim is None:
            n_voxels_dim = round_up_division(lengths, voxel_size)

        if n_threads > 1:
            voxel_ids = _parallel_voxelize_particlecontainer(
                particles, center, lengths, voxel_size, n_voxels_dim, n_threads
            )
        else:
            voxel_ids = _voxelize_particlecontainer(
                particles, center, lengths, voxel_size, n_voxels_dim
            )
        voxel_ids = np.array(list(voxel_ids)).reshape(-1, 1)
        return cls(center, lengths, voxel_size, n_voxels_dim, id=voxel_ids)


def _voxelize_particlecontainer(particles, center, lengths, voxel_size, n_voxels_dim):
    """Voxelize particles.
    Args:
        particles (pipapo.ParticleContainer): particles to be voxelized
        center (np.ndarray): center of outer domain
        lengths (np.ndarray): lengths of outer domain
        voxel_size (float): voxel size
        n_voxels_dim (np.ndarray): voxels per dimension
    Returns:
        set: set of indices of the voxels
    """
    outer_left_boundary = center - lengths * 0.5
    voxel_ids = set()
    for particle in particles:
        voxels_in_particle_ids = voxelize_particle(
            particle.position,
            particle.radius,
            outer_left_boundary,
            voxel_size,
            n_voxels_dim,
        )
        voxel_ids.update(voxels_in_particle_ids)
    return voxel_ids


def _parallel_voxelize_particlecontainer(
    particles, center, lengths, voxel_size, n_voxels_dim, n_threads
):
    """Voxelize particles.
    Args:
        particles (pipapo.ParticleContainer): particles to be voxelized
        center (np.ndarray): center of outer domain
        lengths (np.ndarray): lengths of outer domain
        voxel_size (float): voxel size
        n_voxels_dim (np.ndarray): voxels per dimension
        n_threads (float): threads to parallelize the calculation of the voxels
    Returns:
        set: set of indices of the voxels
    """
    voxel_ids = set()
    with Pool(n_threads) as pool:
        voxel_ids = pool.starmap(
            partial(
                _voxelize_particlecontainer,
                center=center,
                lengths=lengths,
                voxel_size=voxel_size,
                n_voxels_dim=n_voxels_dim,
            ),
            chunkify(particles, n_threads),
        )
    voxel_ids = set().union(*voxel_ids)
    return voxel_ids


def chunkify(indexable_object, number_of_chunks):
    """Create particles chunks

    Args:
        indexable_object (obj): Object that can be indexed
        number_of_chunks (int): number of chunks to be split up

    Returns:
        list: chunks of indexable_object
    """
    chunks = []
    n_particles = len(indexable_object)
    rounded_particles_per_chunk = n_particles // number_of_chunks

    # divide rounded_particles_per_chunk sized sets
    for i in range(0, number_of_chunks - 1):
        chunks.append(
            indexable_object[
                i * rounded_particles_per_chunk : (i + 1) * rounded_particles_per_chunk
            ]
        )

    # like in a bar, the last has to pay extra
    chunks.append(
        indexable_object[
            (number_of_chunks - 1) * rounded_particles_per_chunk : n_particles
        ]
    )
    return chunks


def round_up_division(a, b):
    """Divide and round up.

    Args:
        a (int,np.ndarray): numerator
        b (int,np.ndarray): denominator
    Returns:
        np.ndarray: rounded up division
    """
    return np.ceil(a / b).astype(int)


def running_index(i, j, k, n_dim):
    """Generate running index for 3d matrix.

    Args:
        i (int): first index
        j (int): second index
        k (int): third index
        n_dim (np.ndarray): length per dimension

    Returns:
        int: running index
    """
    return int(i + j * n_dim[0] + k * n_dim[0] * n_dim[1])


def reverse_running_index(c, n_dim):
    """Reverse running index c to ijk.

    Args:
        c (int): running index
        n_dim (int): length per dimension
    Returns:
        (int,int,int): indices i,j,k
    """
    c = int(c)
    k = c // (n_dim[0] * n_dim[1])
    c1 = c - k * n_dim[0] * n_dim[1]
    j = c1 // n_dim[0]
    i = c1 - j * n_dim[0]
    return i, j, k


def voxelize_particle(
    particle_center,
    particle_radius,
    outer_left_boundary,
    voxel_size,
    n_voxels_dim,
):
    """Get voxels for a single particle.

    The domain is given by `outer_left_boundary` which is the vertex of the domain with the
    smallest coordinate in every direction. The indices are based on a background mesh defined by
    `n_voxels_dim`.

    Idea:
      1. Raster bounding box of the particle to the background mesh
      2. Loop through the voxels of the rastered bounding box

    Args:
        particle_center (np.ndarray): particle center
        particle_radius (float): particle radius
        outer_left_boundary (np.ndarray): vertex of outer box with smallest coordinates
        voxel_size (float): voxel size
        n_voxels_dim (np.ndarray): voxels per dimension

    Returns:
        list: list of indices for voxels within the particle
    """
    bounding_box_left_boundary = particle_center - particle_radius
    ijk = (bounding_box_left_boundary - outer_left_boundary) // voxel_size
    rastered = outer_left_boundary + ijk * voxel_size

    dijk = round_up_division(
        bounding_box_left_boundary + particle_radius * 2 - rastered, voxel_size
    )
    radius_sq = particle_radius * particle_radius
    voxel_indices = []

    # Catch in case voxels would be outside of the domain
    upper_bound = np.minimum(ijk + dijk, n_voxels_dim).astype(int)
    lower_bound = np.maximum(ijk, 0).astype(int)

    # offset in the radius computation in order to only doing in once
    # half the voxel size is added to address the voxel centers
    offset = outer_left_boundary - particle_center + 0.5 * voxel_size

    for k in range(lower_bound[2], upper_bound[2]):
        dxyz = np.zeros(3)
        dxyz[2] = k * voxel_size
        for j in range(lower_bound[1], upper_bound[1]):
            dxyz[1] = j * voxel_size
            for i in range(lower_bound[0], upper_bound[0]):
                dxyz[0] = i * voxel_size
                # distance to particle center
                dist_voxel_particle_center = offset + dxyz
                if (
                    dist_voxel_particle_center[0] * dist_voxel_particle_center[0]
                    + dist_voxel_particle_center[1] * dist_voxel_particle_center[1]
                    + dist_voxel_particle_center[2] * dist_voxel_particle_center[2]
                    - radius_sq
                ) <= 0:  # check if the voxel center is inside particle
                    voxel_indices.append(running_index(i, j, k, n_voxels_dim))
    return voxel_indices
