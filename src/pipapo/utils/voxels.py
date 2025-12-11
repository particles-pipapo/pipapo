"""Voxel container."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, Generic, NotRequired, Self, TypedDict

import meshio
import numpy as np
import pyvista as pv

from pipapo.utils.container import Container, D
from pipapo.utils.domain import Box
from pipapo.utils.io import export
from pipapo.utils.type_hinting import float_np_array, int_np_array

if TYPE_CHECKING:
    from pipapo.particles import ParticleContainer

pv.set_plot_theme("document")  # type: ignore[no-untyped-call]


class VoxelProperties(TypedDict):
    """Voxel properties.

    Attributes:
        index: Index of the voxel in the box grid
        position: Voxel position
    """

    index: list[int]
    position: NotRequired[float_np_array]


class VoxelContainer(Container[VoxelProperties, D], Generic[D]):
    """Voxel container."""

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        box: Box,
        voxel_size: float,
        n_voxels_dim: int_np_array,
        index: list[int],
        data: D | None = None,
    ):
        """Initialise voxel container.

        Assumes an axis-aligned box domain discretized into voxels of equal size.

        Args:
            box: Box of the voxels
            voxel_size Voxel size
            n_voxel_dim: Number of voxels per dimension
            index: Indices of the voxels
            data: voxel data
        """
        self.box = box
        self.voxel_size = voxel_size
        self.n_voxels_dim = n_voxels_dim
        self.voxel_volume = self.voxel_size * self.voxel_size * self.voxel_size

        super().__init__(VoxelProperties(index=index), data)

    @property
    def index(self) -> list[int]:
        """Index."""
        self._check_lens()
        return self._properties["index"]

    @index.setter
    def index(self, new_index: list[int]) -> None:
        """Index."""
        self._properties["index"] = new_index
        self._check_lens()

    @property
    def position(self) -> float_np_array:
        """Position."""
        self._check_lens()
        if "position" not in self._properties:
            self._create_voxel_centers()
        return self._properties["position"]

    @position.setter
    def position(self, new_position: float_np_array) -> None:
        """Position."""
        self._properties["position"] = new_position
        self._check_lens()

    def get_porosity(self) -> float:
        """Get porosity.

        Returns:
            porosity
        """
        porosity = 1 - self.volume_of_voxels() / self.volume_of_outer_domain()
        return porosity

    def __str__(self) -> str:
        """Voxel container descriptions."""
        string = super().__str__()
        string += "\n Domain:"
        string += f"\n   with center {self.box.center}\n"
        string += f"\n   with lengths {self.box.lengths}\n"
        string += f"\n   with voxel size {self.voxel_size}\n"
        string += f"\n   with number of voxel per dim {self.n_voxels_dim}\n"
        return string

    def volume_of_voxels(self) -> float:
        """Sum of all voxels with the discretized domain.

        Returns:
            float: Volume
        """
        return self.voxel_volume * len(self)

    def total_number_of_voxels_in_outer_domain(self) -> int:
        """Get total number of voxels within the outer domain.

        Returns:
            int: Number of voxels
        """
        return int(self.n_voxels_dim[0] * self.n_voxels_dim[1] * self.n_voxels_dim[2])

    def volume_of_outer_domain(self) -> float:
        """Return volume of total domain.

        Returns:
            float: Volume of domain
        """
        return self.voxel_volume * self.total_number_of_voxels_in_outer_domain()

    def to_dict(self) -> dict:
        """Create dictionary from voxels.

        Returns:
            dict: dictionary
        """
        dictionary = super().to_dict()
        dictionary["voxel_size"] = np.ones(len(self.index)) * self.voxel_size
        return dictionary

    def _create_voxel_centers(self) -> None:
        """Add voxel centers to fields."""
        voxel_positions = []
        for c in self.index:
            x, y, z = reverse_running_index(c, self.n_voxels_dim)
            center_voxel = (
                self.box.center
                - 0.5 * self.box.lengths
                + (np.array([x, y, z]) + 0.5) * self.voxel_size
            )
            voxel_positions.append(center_voxel)
        self._properties["position"] = np.array(voxel_positions)

    def export(self, file_path: pathlib.Path | str) -> None:
        """Export voxels.

        Args:
            file_path: to be exported
        """
        file_path = pathlib.Path(file_path)
        if file_path.suffix == ".vtu":
            self._export_vtu(file_path)
        else:
            export(
                self.to_dict(),
                file_path,
            )

    def _export_vtu(self, file_path: pathlib.Path | str) -> None:
        """Create hexahedrons and export as vtu.

        Note that notes will appear multiple times.

        Args:
            file_path: to file.
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

    def plot(
        self,
        pv_plotter: pv.Plotter | None = None,
        show: bool = True,
        field_name: str | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> pv.Plotter:
        """Plot voxels.

        Args:
            pv_plotter:Plotter object to plot. Defaults to None.
            show:Open the plotting window. Defaults to True.
            field_name:Field to plot. Defaults to None
            force:Force plotting also for large sets. Defaults to None
            kwargs :additional keyword arguments for add_mesh

        Returns:
            pv.Plotter: Plotter object
        """
        if not force:
            if len(self) > 2000:
                raise ValueError(
                    "You are trying to plot a large voxel set. If you really want to do this add"
                    " the kwarg force=True."
                )

        if not pv_plotter:
            pv_plotter = pv.Plotter()

        if "color" not in kwargs and not field_name:
            kwargs["color"] = kwargs.get("color", "purple")

        kwargs["show_edges"] = kwargs.get("show_edges", True)

        cube = pv.Cube()

        voxels = pv.PolyData(self.position)
        voxels.point_data["diameter"] = np.ones(len(self.position)) * self.voxel_size

        voxels_glyph = voxels.glyph(scale="diameter", geom=cube)

        pv_plotter.add_mesh(voxels_glyph, scalars=field_name)

        if show:
            pv_plotter.show()

        return pv_plotter

    @classmethod
    def from_particles(
        cls,
        particles: ParticleContainer,
        box: Box,
        voxel_size: float | None = None,
        n_voxels_dim: int_np_array | None = None,
    ) -> VoxelContainer:
        """Voxelize particles.

        Args:
            particles (pipapo.ParticleContainer): particles to be voxelized
            box: Box of the voxels
            voxel_size Voxel size
            n_voxel_dim: Number of voxels per dimension
        Returns:
            VoxelContainer: voxel container from particles
        """
        if voxel_size is None:
            voxel_size = float(min(particles.radius) / 4)

        if n_voxels_dim is None:
            n_voxels_dim = round_up_division(box.lengths, voxel_size)

        voxel_ids = _voxelize_particlecontainer(
            particles.position,
            particles.radius,
            box.center,
            box.lengths,
            voxel_size,
            n_voxels_dim,
        )
        return cls(
            box,
            voxel_size,
            n_voxels_dim,
            index=voxel_ids,
            data={},  # type: ignore[arg-type]
        )

    @classmethod
    def create_new_instance(
        cls, properties: VoxelProperties, data: D, additional_properties: dict
    ) -> Self:
        """Create new voxel properties instance.

        Args:
            properties: Voxel properties
            data: Data
            additional_properties: Additional properties which might be constant

        Returns:
            new instance
        """
        properties = properties.copy()
        properties.pop("position", None)
        return cls(**properties, data=data, **additional_properties)  # type: ignore[misc]

    def _additional_properties(self) -> dict:
        """Return additional properties.

        Returns:
            Additional data
        """
        return {
            "box": self.box,
            "voxel_size": self.voxel_size,
            "n_voxels_dim": self.n_voxels_dim,
        }


def _voxelize_particlecontainer(  # pylint: disable=too-many-arguments
    position: float_np_array,
    radius: float_np_array,
    center: float_np_array,
    lengths: float_np_array,
    voxel_size: float,
    n_voxels_dim: int_np_array,
) -> list[int]:
    """Voxelize particles.

    Args:
        particles (pipapo.ParticleContainer): particles to be voxelized
        center: center of outer domain
        lengths: lengths of outer domain
        voxel_size voxel size
        n_voxels_dim: voxels per dimension
    Returns:
        set: set of indices of the voxels
    """
    outer_left_boundary = center - lengths * 0.5
    voxel_ids = set()
    for p, r in zip(position, radius):
        voxels_in_particle_ids = voxelize_particle(
            p,
            r,
            outer_left_boundary,
            voxel_size,
            n_voxels_dim,
        )
        voxel_ids.update(voxels_in_particle_ids)
    return list(voxel_ids)


def round_up_division(a: float_np_array, b: float) -> int_np_array:
    """Divide and round up.

    Args:
        a (int,np.ndarray): numerator
        b (int,np.ndarray): denominator
    Returns:
        rounded up division
    """
    return np.ceil(a / b).astype(int)


def running_index(i: int, j: int, k: int, n_dim: int_np_array) -> int:
    """Generate running index for 3d matrix.

    Args:
        i first index
        j second index
        k third index
        n_dim: length per dimension

    Returns:
        int: running index
    """
    return int(i + j * n_dim[0] + k * n_dim[0] * n_dim[1])


def reverse_running_index(c: int, n_dim: int_np_array) -> tuple[int, int, int]:
    """Reverse running index c to ijk.

    Args:
        c running index
        n_dim length per dimension
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
    particle_center: float_np_array,
    particle_radius: float_np_array,
    outer_left_boundary: float_np_array,
    voxel_size: float,
    n_voxels_dim: int_np_array,
) -> list[int]:
    """Get voxels for a single particle.

    The domain is given by `outer_left_boundary` which is the vertex of the domain with the
    smallest coordinate in every direction. The indices are based on a background mesh defined by
    `n_voxels_dim`.

    Idea:
      1. Raster bounding box of the particle to the background mesh
      2. Loop through the voxels of the rastered bounding box

    Args:
        particle_center: particle center
        particle_radius particle radius
        outer_left_boundary: vertex of outer box with smallest coordinates
        voxel_size voxel size
        n_voxels_dim: voxels per dimension

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
