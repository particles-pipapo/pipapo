"""Utils for binning the paricles."""
import numpy as np


def bin_particles(particles, box_dimensions, n_bins, box_center=np.zeros(3)):
    """Bin the particles.

    Args:
        particles (ParticleContainer): Particles to be binned
        box_dimensions (np.array): Dimension of binning box of the particles
        n_bins (np.array): Number of bins per dimension
        box_center (np.array, optional): Binning box center. Defaults to np.zeros(3).
    """
    x_bins = np.linspace(
        -0.5 * box_dimensions[0] + box_center[0],
        0.5 * box_dimensions[0] + box_center[0],
        n_bins[0] + 1,
    )
    y_bins = np.linspace(
        -0.5 * box_dimensions[1] + box_center[1],
        0.5 * box_dimensions[1] + box_center[1],
        n_bins[1] + 1,
    )
    z_bins = np.linspace(
        -0.5 * box_dimensions[2] + box_center[2],
        0.5 * box_dimensions[2] + box_center[2],
        n_bins[2] + 1,
    )

    # -1 is needed as indexing within the box starts at 1
    x_bin = np.digitize(particles.position[:, 0], x_bins) - 1
    y_bin = np.digitize(particles.position[:, 1], y_bins) - 1
    z_bin = np.digitize(particles.position[:, 2], z_bins) - 1

    # Create a running index
    bin_id = x_bin * n_bins[2] * n_bins[1] + y_bin * n_bins[2] + z_bin

    particles.add_field("bin_id", bin_id)


def get_bounding_box(position, tol=1e-8):
    """Get the bounding box from position.

    Args:
        position (np.array): Positions
        tol (float, optional): Tolerance. Defaults to 1e-8.
    """
    mins = np.min(position, axis=0)
    maxs = np.max(position, axis=0)
    center = 0.5 * (mins + maxs)
    lengths = maxs - mins + tol
    return center, lengths


def bin_bounding_box_by_n_bins(particles, n_bins):
    """Bin the particles.

    Args:
        particles (ParticleContainer): Particles to be binned
        n_bins (np.array): Number of bins per dimension
    """
    center, lengths = get_bounding_box(particles.position)
    bin_particles(particles, lengths, n_bins, center)


def bin_bounding_box_by_max_radius(particles):
    """Bin the particles based on maximal radius.

    Args:
        particles (ParticleContainer): Particles to be binned
    """
    center, lengths = get_bounding_box(particles.position)
    max_radius = np.max(particles.radius)
    n_bins = (lengths // max_radius + 1).astype(int)
    lengths = n_bins * max_radius
    return bin_particles(particles, lengths, n_bins, center)
