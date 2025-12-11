"""Utils for binning the paricles."""

import numpy as np

from pipapo.particles import ParticleContainer
from pipapo.utils.type_hinting import float_np_array, int_np_array


def bin_particles(
    particles: ParticleContainer,
    box_dimensions: float_np_array,
    n_bins: int_np_array,
    box_center: float_np_array = np.zeros(3),
) -> None:
    """Bin the particles.

    Args:
        particles: Particles to be binned
        box_dimensions: Dimension of binning box of the particles
        n_bins: Number of bins per dimension
        box_center: Binning box center. Defaults to np.zeros(3).
    """
    # Precompute bin edges for all dimensions
    half_box = 0.5 * box_dimensions
    bin_edges = [
        np.linspace(
            -half_box[i] + box_center[i], half_box[i] + box_center[i], n_bins[i] + 1
        )
        for i in range(3)
    ]

    # Compute bin indices for all dimensions
    bin_indices = [
        np.digitize(particles.position[:, i], bin_edges[i]) - 1 for i in range(3)
    ]

    # Compute the bin ID using broadcasting
    bin_id = (
        bin_indices[0] * n_bins[1] * n_bins[2]
        + bin_indices[1] * n_bins[2]
        + bin_indices[2]
    )

    # Add the bin_id field to particles
    particles.data["bin_id"] = bin_id


def bin_bounding_box_by_n_bins(
    particles: ParticleContainer, n_bins: int_np_array
) -> None:
    """Bin the particles.

    Args:
        particles: Particles to be binned
        n_bins: Number of bins per dimension
    """
    box = particles.get_bounding_box()
    bin_particles(particles, box.lengths, n_bins, box.center)


def bin_bounding_box_by_max_radius(particles: ParticleContainer) -> None:
    """Bin the particles based on maximal radius.

    Args:
        particles: Particles to be binned
    """
    box = particles.get_bounding_box()
    max_radius = np.max(particles.radius)
    n_bins = (box.lengths // max_radius + 1).astype(int)
    box.lengths = n_bins * max_radius
    return bin_particles(particles, box.lengths, n_bins, box.center)
