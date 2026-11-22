"""Adaptive-window collaborative representation (AWCR).

M. Imani (2023), https://doi.org/10.1016/j.ejrs.2023.05.002, Eqs. 1–15.
The defaults use the San Diego column of Table 1 and the regularization in
Section 3.2. Settings for other scenes can be supplied explicitly. The
original band count determines dictionary sizes even after PCA reduction.
"""

import numpy as np
from crd import weighted_residual
from skimage.segmentation import slic


def _window_radii(image, count, compactness):
    labels = slic(image, n_segments=min(count, image.size),
                  compactness=compactness, channel_axis=None, start_label=0)
    areas = np.bincount(labels.ravel())[labels]
    sides = np.floor(np.sqrt(areas) + 0.5).astype(int)
    # Round the side up to odd, then convert it to a radius (Eqs. 1 and 2).
    return sides // 2


def detect(cube, *, n_components=5, fine_segments=150, coarse_segments=100,
           background_factor=0.2, anomaly_factor=0.005, local_weight=0.15,
           ridge=1e-6, scale_global=True, compactness=1.0):
    """Return the raw H by W AWCR score for a finite H by W by B array.

    SLIC compactness, clipping windows at the image edge, rounding dictionary
    sizes half-up with a one-atom minimum, and resolving tied RX scores by
    row-major pixel order are implementation choices. An empty clipped ring
    gives a zero local prediction. The global PCs are individually scaled
    to [0, 255] by default (disable for the paper's Texas Coast setting).
    The local branch uses the complete, unscaled, centered PCA scores.
    """
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim != 3 or min(cube.shape) < 1 or not np.isfinite(cube).all():
        raise ValueError("cube must be a finite, nonempty H by W by B array")
    for count in (n_components, fine_segments, coarse_segments):
        if not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("PCA and segment counts must be positive integers")
    if coarse_segments >= fine_segments:
        raise ValueError("fine_segments must exceed coarse_segments")
    for value in (background_factor, anomaly_factor, compactness, ridge):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("dictionary factors, compactness and ridge must be positive")
    if not 0 <= local_weight <= 1:
        raise ValueError("local_weight must lie in [0, 1]")

    height, width, bands = cube.shape
    spectra = cube.reshape(-1, bands)
    centered = spectra - spectra.mean(axis=0)
    if not np.any(centered):
        return np.zeros((height, width))
    eigenvalues, vectors = np.linalg.eigh(centered.T @ centered)
    # A fixed sign convention also makes per-component min-max scaling reproducible.
    pivots = vectors[np.argmax(np.abs(vectors), axis=0), np.arange(bands)]
    vectors *= np.sign(pivots)
    pcs = centered @ vectors[:, ::-1]

    if local_weight > 0:
        first = pcs[:, 0].reshape(height, width)
        first = (first - first.min()) / np.ptp(first)
        inner = _window_radii(first, fine_segments, compactness)
        outer = np.maximum(inner + 1, _window_radii(first, coarse_segments, compactness))
        local_pcs = pcs.reshape(height, width, bands)

    if local_weight < 1:
        # Use the pseudoinverse in a rank-deficient PCA space. In particular,
        # min-max scaling must not magnify numerical noise in null components.
        m = min(n_components, np.count_nonzero(
            eigenvalues > eigenvalues[-1] * np.finfo(float).eps * max(centered.shape)))
        global_pcs = pcs[:, :m]
        rx = np.sum(global_pcs**2 / eigenvalues[::-1][:m], axis=1)
        if scale_global:
            global_pcs = 255 * ((global_pcs - global_pcs.min(axis=0)) /
                                np.ptp(global_pcs, axis=0))
        order = np.argsort(rx, kind="stable")
        nb = max(1, int(np.floor(background_factor * bands + 0.5)))
        na = max(1, int(np.floor(anomaly_factor * bands + 0.5)))
        background = global_pcs[order[:nb]]
        anomalies = global_pcs[order[-na:]]

    score = np.zeros((height, width))
    for row, col in np.ndindex(height, width):
        index = row * width + col
        if local_weight > 0:
            radius = outer[row, col]
            top, bottom = max(0, row - radius), min(height, row + radius + 1)
            left, right = max(0, col - radius), min(width, col + radius + 1)
            rr, cc = np.ogrid[top:bottom, left:right]
            ring = np.maximum(abs(rr - row), abs(cc - col)) > inner[row, col]
            atoms = local_pcs[top:bottom, left:right][ring]
            score[row, col] += local_weight * weighted_residual(pcs[index], atoms, ridge)
        if local_weight < 1:
            pixel = global_pcs[index]
            difference = weighted_residual(pixel, background, ridge) - weighted_residual(pixel, anomalies, ridge)
            score[row, col] += (1 - local_weight) * difference
    return score
