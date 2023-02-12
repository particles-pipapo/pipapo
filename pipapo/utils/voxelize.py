"""Voxelise particles."""
import numpy as np
from pipapo.utils.io import export


def round_up_division(a, b):
    """Divide and round up.

    Args:
        a (int): numerator
        b (int): denominator
    Returns:
        int: rounded up division
    """
    return -(-a // b).astype(int)


def running_index(i, j, k, n_dim):
    """Generate running index for 3d matrix.

    Args:
        i (int): first index
        j (int): second index
        k (int): third index
        n_dim (int): length per dimension

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
    if c < n_dim[0]:
        return c, 0, 0
    elif c < n_dim[0] * n_dim[1]:
        j = c // n_dim[0]
        i, _, _ = reverse_running_index(c - j * n_dim[0], n_dim)
        return i, j, 0
    else:
        k = c // (n_dim[0] * n_dim[1])
        i, j, _ = reverse_running_index(c - k * n_dim[0] * n_dim[1], n_dim)
        return i, j, k


def voxelize_particle(
    particle_center,
    particle_radius,
    outer_left_boundary,
    voxel_size,
    n_voxels_dim,
):
    """Get voxels for a single partilce.

    The domain is given by `outer_left_boundary` which is the vertex of the domain with the
    smallest coordinate in every direction. The indices are based on a background mesh defined by
    `n_voxels_dim`.

    Idea:
      1. Raster bounding box of the particle to the background mesh
      2. Loop through the voxels of the rasted bounding box

    Args:
        particle_center (np.ndarray): particle center
        particle_radius (float): particle radius
        outer_left_boundary (np.ndarray): vertex of outerbox with smallest coordinates
        voxel_size (float): voxel size
        n_voxels_dim (np.ndarray): voxels per dimension

    Returns:
        list: list of indices for voxels within the particle
    """
    bounding_box_left_boundary = particle_center - particle_radius
    ijk = (bounding_box_left_boundary - outer_left_boundary) // voxel_size
    rasted = outer_left_boundary + ijk * voxel_size

    dijk = round_up_division(
        bounding_box_left_boundary + particle_radius * 2 - rasted, voxel_size
    )
    radius_sq = particle_radius * particle_radius
    voxel_indices = []

    # Catch in case voxels would be outside of the domain
    upper_bound = np.minimum(ijk + dijk, n_voxels_dim).astype(int)
    lower_bound = np.maximum(ijk, 0).astype(int)

    # offset in the radius compuation
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
    for p in particles:
        voxels_in_particle_ids = voxelize_particle(
            p.position, p.radius, outer_left_boundary, voxel_size, n_voxels_dim
        )
        voxel_ids.update(voxels_in_particle_ids)
    return voxel_ids


def chunkify(particles, number_of_chunks):
    """Create particles chunks

    Args:
        particles (pipapo.ParticleContainer): particles to be chunked
        number_of_chunks (int): number of chunks to be split up

    Returns:
        list: chunks of particles
    """
    chunks = []
    n_particles = len(particles)
    rounded_particles_per_chunk = n_particles // number_of_chunks

    # divide rounded_particles_per_chunk sized sets
    for i in range(0, number_of_chunks - 1):
        chunks.append(
            particles[
                i * rounded_particles_per_chunk : (i + 1) * rounded_particles_per_chunk
            ]
        )

    # like in a bar, the last has to pay extra
    chunks.append(
        particles[(number_of_chunks - 1) * rounded_particles_per_chunk : n_particles]
    )
    return chunks


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
        n_threads (float): voxels number of threads

    Returns:
        set: set of indices of the voxels
    """
    from multiprocessing import Pool
    from functools import partial

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


def export_voxels(voxel_ids, center, lengths, voxel_size, n_voxels_dim, export_name):
    """Export voxels.

    Args:
        voxel_ids (set): voxel ids to be exported
        center (np.ndarray): center of outer domain. Defaults to the center of the bounding box of
                             the particle container
        lengths (np.ndarray): lengths of outer domain. Defaults to the lengths of the bounding box
                              of the particle container
        voxel_size (float): voxel size. defaults to a quarter of the smallest radius of the
                            particles
        n_voxels_dim (np.ndarray): voxels per dimension. Defaults to the number of voxels that fit
                                   in a length of the bounding box of the particle container
        export_name (pathlib.Path): Path to export data
    """
    voxel_positions = []
    for c in voxel_ids:
        x, y, z = reverse_running_index(c, n_voxels_dim)
        center_voxel = center - 0.5 * lengths + (np.array([x, y, z]) + 0.5) * voxel_size
        voxel_positions.append(center_voxel)

    voxel_positions = np.array(voxel_positions)
    voxel_size = np.ones((len(voxel_positions), 1)) * voxel_size

    export(
        {
            "voxel_id": np.array(list(voxel_ids)).reshape(-1, 1),
            "position": voxel_positions,
            "voxel_size": voxel_size,
        },
        export_name,
    )


def voxelize_particlecontainer(
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
        center (np.ndarray): center of outer domain. Defaults to the center of the bounding box of
                             the particle container
        lengths (np.ndarray): lengths of outer domain. Defaults to the lengths of the bounding box
                              of the particle container
        voxel_size (float): voxel size. defaults to a quarter of the smallest radius of the
                            particles
        n_voxels_dim (np.ndarray): voxels per dimension. Defaults to the number of voxels that fit
                                   in a length of the bounding box of the particle container
        n_threads (float): voxels number of threads

    Returns:
        set: set of indices of the voxels
    """
    if center is None and lengths is None:
        center, lengths = particles.bounding_box()

    if voxel_size is None:
        voxel_size = min(particles.radius) / 4

    if n_voxels_dim is None:
        n_voxels_dim = round_up_division(lengths, voxel_size)

    breakpoint()
    if n_threads > 1:
        return _parallel_voxelize_particlecontainer(
            particles, center, lengths, voxel_size, n_voxels_dim, n_threads
        )
    else:
        return _voxelize_particlecontainer(
            particles, center, lengths, voxel_size, n_voxels_dim
        )
