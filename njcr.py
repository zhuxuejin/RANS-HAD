"""Linear NJCR for hyperspectral anomaly detection.

S. Chang and P. Ghamisi (2022), https://doi.org/10.1109/TGRS.2022.3195339.
Reference MATLAB package: https://github.com/ShizhenChang/NJCR/tree/32138a3815ea5e13b247ca6e15e1c8a94eb27a2e

The San Diego demo supplies the default dictionary sizes and regularization.
The model minimizes ||X-DA||_F^2 + (ridge/2)||A||_F^2 subject to A >= 0 and
each column summing to one. Anomaly scores use only background coefficients.
"""

from heapq import heappush, heappop

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix, triu
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform


def _ncut_segments(pc, count, min_size, bandwidth):
    """Recursive normalized cut on an eight-neighbour first-PC affinity graph."""
    height, width = pc.shape
    index = np.arange(pc.size).reshape(pc.shape)
    edges = [(index[:, :-1], index[:, 1:]), (index[:-1], index[1:]),
             (index[:-1, :-1], index[1:, 1:]), (index[1:, :-1], index[:-1, 1:])]
    i = np.concatenate([a.ravel() for a, _ in edges])
    j = np.concatenate([b.ravel() for _, b in edges])
    scaled = pc.ravel() / max(np.ptp(pc), np.finfo(float).tiny)
    affinity = np.exp(np.maximum(-0.5 * ((scaled[i] - scaled[j]) / bandwidth)**2, -30))
    graph = coo_matrix((affinity, (i, j)), shape=(pc.size, pc.size)).tocsr()
    graph = graph + graph.T

    pending = [(-pc.size, 0, index.ravel())]
    finished = []
    serial = 0
    while len(pending) + len(finished) < count and pending:
        _, _, members = heappop(pending)
        n = len(members)
        if n < 2 * min_size:
            finished.append(members)
            continue
        sub = graph[members][:, members]
        degree = np.asarray(sub.sum(axis=1)).ravel()
        normalized = laplacian(sub, normed=True)
        if n <= 16:
            _, vectors = np.linalg.eigh(normalized.toarray())
            fiedler = vectors[:, 1]
        else:
            values, vectors = eigsh(normalized.tocsc(), k=2, sigma=-1e-7,
                                    which="LM", v0=np.linspace(0.1, 1, n))
            fiedler = vectors[:, np.argsort(values)[1]]
        order = np.argsort(fiedler / np.sqrt(np.maximum(degree, 1e-30)), kind="stable")

        # Each edge crosses the sweep cut from its first to its last endpoint.
        upper = triu(sub, k=1, format="coo")
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(n)
        low = np.minimum(rank[upper.row], rank[upper.col])
        high = np.maximum(rank[upper.row], rank[upper.col])
        changes = np.bincount(low, weights=upper.data, minlength=n)
        changes -= np.bincount(high, weights=upper.data, minlength=n)
        cuts = np.maximum(0, changes.cumsum()[:-1])
        volume = np.cumsum(degree[order])
        denom = np.maximum(volume[:-1] * (volume[-1] - volume[:-1]), 1e-30)
        costs = cuts * volume[-1] / denom
        split = min_size + np.argmin(costs[min_size - 1:n - min_size])
        for child in (members[order[:split]], members[order[split:]]):
            serial += 1
            heappush(pending, (-len(child), serial, child))
    return finished + [entry[2] for entry in pending]


def _density_centres(samples, count):
    """Gaussian density followed by distance to the nearest denser observation."""
    if len(samples) == count:
        return samples
    condensed = pdist(samples)
    nonzero = condensed[condensed > 0]
    if not nonzero.size:
        return samples[:count]
    # cluster_dp.m indexes the sorted, positive upper triangle by 0.02*n*(n-1).
    k = int(np.floor(0.02 * len(samples) * (len(samples) - 1) + 0.5))
    k = np.clip(k, 1, len(nonzero)) - 1
    cutoff = np.partition(nonzero, k)[k]
    distances = squareform(condensed)
    rho = np.exp(-(distances / cutoff)**2).sum(axis=1) - 1
    order = np.argsort(-rho, kind="stable")
    delta = np.zeros(len(samples))
    for rank, idx in enumerate(order[1:], 1):
        delta[idx] = distances[idx, order[:rank]].min()
    delta[order[0]] = delta.max()
    selected = np.argsort(-delta, kind="stable")[:count]
    return samples[selected]


def detect(cube, *, n_segments=100, atoms_per_segment=5, anomaly_atoms=50,
           ridge=100.0, graph_bandwidth=0.10, labels=None, rho=0.01,
           max_iter=30, tolerance=0.01, block_size=1024):
    """Return an H by W continuous anomaly map for a finite H by W by B cube.

    The MATLAB package uses Ncut_9 on PC1. Here the graph is defined by a
    Gaussian of relative PC1 differences, with graph_bandwidth in units of
    the PC1 range. Supply an H by W integer labels array to use a saved
    partition (including Ncut_9 output). Each child of the built-in cut has
    at least atoms_per_segment pixels. RX uses the covariance pseudoinverse.
    Spectral ties in the density-peak step use stable input order. ADMM uses
    the MATLAB solver's signed maximum primal residual and full-scene dual
    Frobenius norm. All blocks complete the same number of iterations.
    """
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim != 3 or min(cube.shape) < 1 or not np.isfinite(cube).all():
        raise ValueError("cube must be a finite, nonempty H by W by B array")
    for value in (n_segments, atoms_per_segment, anomaly_atoms, max_iter, block_size):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("counts, block_size and max_iter must be positive integers")
    for value in (ridge, rho, graph_bandwidth, tolerance):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("ridge, rho, graph_bandwidth and tolerance must be positive")
    height, width, bands = cube.shape
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (height, width) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be an H by W integer array")
    spectra = cube.reshape(-1, bands)
    scale = np.ptp(spectra)
    if scale == 0:
        return np.zeros((height, width))
    x = (spectra - spectra.min()) / scale
    centered = x - x.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    valid = values > values[-1] * np.finfo(float).eps * max(centered.shape)
    whitened = (centered @ vectors[:, valid]) / np.sqrt(values[valid])
    rx = np.einsum("ij,ij->i", whitened, whitened)
    anomalies = x[np.argsort(-rx, kind="stable")[:anomaly_atoms]]

    atom_count = min(atoms_per_segment, len(x))
    if labels is None:
        pc = (centered @ vectors[:, -1]).reshape(height, width)
        groups = _ncut_segments(pc, n_segments, atom_count, graph_bandwidth)
    else:
        groups = [np.flatnonzero(labels.ravel() == label) for label in np.unique(labels)]
    atoms = [_density_centres(x[group], atom_count) for group in groups if len(group) >= atom_count]
    if not atoms:
        raise ValueError("all segments are smaller than atoms_per_segment")
    background = np.concatenate(atoms)
    background = background[background.any(axis=1)]
    dictionary = np.concatenate([background, anomalies])
    n_atoms = len(dictionary)

    # Equality and nonnegativity each have a scaled multiplier. The shared
    # system includes rho*11^T from the per-pixel sum-to-one constraints.
    system = 2 * (dictionary @ dictionary.T) + rho
    system.flat[::n_atoms + 1] += ridge + rho
    factor = cho_factor(system, lower=True)
    data_term = 2 * (x @ dictionary.T)
    # For the projection onto the nonnegative orthant, omega and its updated
    # scaled multiplier are the positive and negative parts of the same array.
    projection_input = np.zeros_like(data_term)
    sum_dual = np.zeros(len(x))
    dual_change_squared = np.empty(len(x))
    background_coeff = np.empty((len(x), len(background)))
    for _ in range(max_iter):
        primal_max = -np.inf
        for start in range(0, len(x), block_size):
            block = slice(start, start + block_size)
            previous = projection_input[block]
            rhs = data_term[block] + rho * (np.abs(previous) + 1 - sum_dual[block, None])
            coeff = cho_solve(factor, rhs.T, check_finite=False).T
            updated = coeff + np.minimum(previous, 0)
            positive = np.maximum(updated, 0)
            change = positive - np.maximum(previous, 0)
            dual_change_squared[block] = np.einsum("ij,ij->i", change, change)
            sum_error = coeff.sum(axis=1) - 1
            primal_max = max(primal_max, np.max(coeff - positive), np.max(sum_error))
            projection_input[block] = updated
            sum_dual[block] += sum_error
            background_coeff[block] = coeff[:, :len(background)]
        if primal_max < tolerance and rho * np.sqrt(dual_change_squared.sum()) < tolerance:
            break

    result = np.empty(len(x))
    for start in range(0, len(x), block_size):
        block = slice(start, start + block_size)
        estimate = background_coeff[block] @ background
        result[block] = np.linalg.norm(x[block] - estimate, axis=1)
    return result.reshape(height, width)
