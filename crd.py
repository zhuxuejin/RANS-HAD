"""Distance-weighted collaborative representation detector of Li and Du.

W. Li and Q. Du, IEEE TGRS 53 (2015), 1463–1474.
https://doi.org/10.1109/TGRS.2014.2343955
"""

import numpy as np
from scipy import linalg


def weighted_residual(target, atoms, ridge):
    """L2 residual of distance-weighted Tikhonov reconstruction; one atom per row."""
    if not len(atoms):
        return np.linalg.norm(target)
    delta = atoms - target
    distances_sq = np.einsum("ij,ij->i", delta, delta)
    if np.any(distances_sq == 0):
        # Selecting a matching atom gives both zero residual and zero penalty.
        return 0.0
    penalty = ridge * distances_sq
    try:
        if len(atoms) <= target.size:
            gram = atoms @ atoms.T
            gram.flat[::len(atoms) + 1] += penalty
            chol = linalg.cho_factor(gram, check_finite=False)
            coefficients = linalg.cho_solve(chol, atoms @ target, check_finite=False)
            return np.linalg.norm(target - coefficients @ atoms)

        # With D=diag(penalty), the residual is (I + X D^-1 X.T)^-1 y.
        weighted = atoms / np.sqrt(penalty)[:, None]
        gram = weighted.T @ weighted
        gram.flat[::target.size + 1] += 1.0
        chol = linalg.cho_factor(gram, check_finite=False)
        return np.linalg.norm(linalg.cho_solve(chol, target, check_finite=False))
    except linalg.LinAlgError:
        # Avoid squaring the condition number when the normal system loses
        # numerical positive definiteness for very similar spectra.
        design = np.vstack((atoms.T, np.diag(np.sqrt(penalty))))
        response = np.r_[target, np.zeros(len(atoms))]
        coefficients = linalg.lstsq(design, response, check_finite=False)[0]
        return np.linalg.norm(target - coefficients @ atoms)


def detect(cube, *, inner=5, outer=11, ridge=1e-6):
    """Return raw residual norms for a finite (rows, columns, retained bands) cube.

    inner and outer are odd square side lengths, and ridge is the positive
    multiplier on the squared distance-weighted coefficient norm. These are
    configurable experiment settings. Cropped rings include each real pixel
    at most once; an empty dictionary predicts the zero vector.
    """
    cube = np.asarray(cube)
    if (cube.ndim != 3 or min(cube.shape) < 1 or cube.dtype.kind not in "biuf"
            or not np.isfinite(cube).all()):
        raise ValueError("cube must be a nonempty, finite real H x W x B array")
    if any(not isinstance(v, (int, np.integer)) or v < 1 or v % 2 == 0
           for v in (inner, outer)) or inner >= outer:
        raise ValueError("inner and outer must be positive odd side lengths with inner < outer")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    cube = cube.astype(np.float64, copy=False)
    height, width, _ = cube.shape
    guard, radius = inner // 2, outer // 2
    max_y, max_x = min(radius, height - 1), min(radius, width - 1)
    dy, dx = np.mgrid[-max_y:max_y + 1, -max_x:max_x + 1]
    ring = np.maximum(np.abs(dy), np.abs(dx)) > guard
    dy, dx = dy[ring], dx[ring]
    scores = np.empty((height, width))
    for row in range(height):
        y = row + dy
        on_image = (y >= 0) & (y < height)
        for col in range(width):
            x = col + dx
            inside = on_image & (x >= 0) & (x < width)
            atoms = cube[y[inside], x[inside]]
            scores[row, col] = weighted_residual(cube[row, col], atoms, ridge)
    return scores
