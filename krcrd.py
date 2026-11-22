"""KRCRD hyperspectral anomaly scores (Zhao et al., 2024).

https://doi.org/10.1109/JSTARS.2024.3476319
Scaled-MST construction follows the traversal described in the cited MS2OD
paper, https://doi.org/10.1088/2632-2153/ad2492 . No labels are used.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _scaled_mst(pixels):
    """Prim traversal from a shortest-edge endpoint, storing O(N) edge data."""
    count = len(pixels)
    nearest = cKDTree(pixels).query(pixels, k=2)[0][:, 1]
    current = int(np.argmin(nearest))
    previous_length = nearest[current]
    unused = np.ones(count, dtype=bool)
    parents = np.empty(count, dtype=np.intp)
    weights = np.full(count, np.inf)
    edges = np.empty((count - 1, 2), dtype=np.intp)
    scaled = np.empty(count - 1)
    for step in range(count - 1):
        unused[current] = False
        weights[current] = np.inf
        # All candidate edges at this step share the same scaling divisor.
        distances = cdist(pixels[current : current + 1], pixels)[0]
        better = unused & (distances < weights)
        parents[better] = current
        weights[better] = distances[better]
        current = int(np.argmin(weights))
        length = weights[current]
        edges[step] = parents[current], current
        if previous_length > 0:
            scaled[step] = length / previous_length
        else:
            scaled[step] = np.inf if length > 0 else 0.0
        previous_length = length
    return edges, scaled


def _dictionary(pixels, cluster_count, minimum, maximum):
    edges, scaled = _scaled_mst(pixels)
    # Dropping the C-1 largest scaled edges leaves C connected components.
    kept = np.argsort(scaled, kind="stable")[: len(pixels) - cluster_count]
    src, dst = edges[kept].T
    graph = coo_matrix(
        (np.ones(len(kept), dtype=np.uint8), (src, dst)),
        shape=(len(pixels), len(pixels)),
    )
    _, labels = connected_components(graph, directed=False)
    groups = np.argsort(labels, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(labels[groups])) + 1, len(labels)]
    medoids = []
    for start, end in zip(starts[:-1], starts[1:]):
        if minimum <= end - start <= maximum:
            members = pixels[groups[start:end]]
            # The minimum total-distance point in a cluster is its medoid.
            totals = np.empty(len(members))
            for i in range(0, len(members), 128):
                totals[i : i + 128] = cdist(members[i : i + 128], members).sum(axis=1)
            medoids.append(members[np.argmin(totals)])
    if not medoids:
        raise ValueError("No cluster meets the configured dictionary size bounds")
    return np.asarray(medoids)


def _kernel(distance, bandwidth):
    # Eq. (14), log((distance + e*h)/(distance + h)), evaluated with log1p.
    return np.log1p(np.expm1(1.0) / (1.0 + distance / bandwidth))


def _scores(pixels, dictionary, neighbor_indices, bandwidth, ridge, coupling):
    count = neighbor_indices.shape[1]
    identity = np.eye(count)
    scores = np.empty(len(pixels))
    # Identical unordered k-NN supports share their per-band Gram matrices.
    sets, assignment = np.unique(np.sort(neighbor_indices, axis=1), axis=0, return_inverse=True)
    assignment = assignment.reshape(-1)
    order = np.argsort(assignment, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(assignment))]
    for group, indices in enumerate(sets):
        atoms = dictionary[indices].T
        gram = _kernel(np.abs(atoms[:, :, None] - atoms[:, None, :]), bandwidth)
        # P_i in Eq. (17) and the mean-coefficient system in Eq. (20).
        matrices = gram + (ridge + coupling) * identity
        inverse = np.linalg.solve(matrices, np.broadcast_to(identity, matrices.shape))
        mean_system = identity - coupling * inverse.mean(axis=0)
        rows = order[bounds[group] : bounds[group + 1]]
        for start in range(0, len(rows), 32):
            batch = rows[start : start + 32]
            target = pixels[batch]
            cross = _kernel(np.abs(target[:, :, None] - atoms[None, :, :]), bandwidth)
            base = np.einsum("bij,nbj->nbi", inverse, cross)
            mean = np.linalg.solve(mean_system, base.mean(axis=1).T).T
            coefficients = base + coupling * np.einsum("bij,nj->nbi", inverse, mean)
            # k(y_i, y_i) = 1; Eq. (22) sums squared feature-space residuals.
            fitted_norm = np.einsum("nbi,bij,nbj->nb", coefficients, gram, coefficients)
            alignment = np.einsum("nbi,nbi->nb", coefficients, cross)
            scores[batch] = np.maximum(1.0 - 2 * alignment + fitted_norm, 0.0).sum(axis=1)
    return scores


def detect(cube, *, k=15, h=0.5, ridge=0.01, coupling=0.01, cluster_count=None,
           min_cluster_size=2, max_cluster_size=None):
    """Return the H-by-W reconstruction error of a real H-by-W-by-B cube.

    Input is rescaled by the scene-wide minimum and range. Zhao et al. use
    k=15 and h=0.5. The ridge (lambda) and coupling (tau) defaults are fixed
    choices within their search range. The paper specifies user-defined
    cluster cuts and size bounds without universal values. Here the default
    cut leaves ceil(sqrt(N)) components (at least 4*k, capped at N/minimum), followed
    by a minimum size of two and an optional upper bound. Specify these
    arguments to match a particular experiment. Each surviving cluster
    contributes its medoid. If fewer than k medoids survive, all are used.
    """
    raw = np.asarray(cube)
    if raw.ndim != 3 or not all(raw.shape) or raw.dtype.kind not in "iuf":
        raise ValueError("cube must be a nonempty real numeric H-by-W-by-B array")
    pixels = np.asarray(raw, dtype=np.float64).reshape(-1, raw.shape[2])
    if not np.isfinite(pixels).all():
        raise ValueError("cube contains non-finite values")
    k = _positive_integer(k, "k")
    minimum = _positive_integer(min_cluster_size, "min_cluster_size")
    maximum = len(pixels) if max_cluster_size is None else _positive_integer(
        max_cluster_size, "max_cluster_size"
    )
    if minimum > maximum:
        raise ValueError("min_cluster_size exceeds max_cluster_size")
    if not np.isfinite([h, ridge, coupling]).all() or h <= 0 or ridge <= 0 or coupling < 0:
        raise ValueError("h and ridge must be positive and coupling nonnegative")
    if np.all(pixels == pixels[0]):
        return np.zeros(raw.shape[:2], dtype=float)
    if cluster_count is None:
        cluster_count = min(len(pixels) // minimum, max(4 * k, int(np.ceil(np.sqrt(len(pixels))))))
    cluster_count = _positive_integer(cluster_count, "cluster_count")
    if cluster_count > len(pixels):
        raise ValueError("cluster_count cannot exceed the number of pixels")
    pixels = (pixels - pixels.min()) / np.ptp(pixels)
    dictionary = _dictionary(pixels, cluster_count, minimum, maximum)
    k = min(k, len(dictionary))
    neighbors = cKDTree(dictionary).query(pixels, k=list(range(1, k + 1)))[1]
    return _scores(pixels, dictionary, neighbors, h, ridge, coupling).reshape(raw.shape[:2])
