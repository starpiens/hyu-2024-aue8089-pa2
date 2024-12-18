import numpy as np

from distort_points import distort_points


def project_points(points_3d: np.ndarray,
                   K: np.ndarray,
                   D: np.ndarray) -> np.ndarray:
    """
    Projects 3d points to the image plane, given the camera matrix,
    and distortion coefficients.

    Args:
        points_3d: 3d points (Nx3)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        projected_points: 2d points (Nx2)
    """

    # [TODO] get image coordinates
    projected_points = points_3d @ K.T  # (Nx3)
    projected_points = projected_points[:, :2] / projected_points[:, 2:]  # (Nx2)

    # [TODO] apply distortion
    projected_points = distort_points(projected_points, D, K)

    return projected_points
