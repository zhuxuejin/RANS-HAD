"""SP-ProCRD for hyperspectral anomaly detection (Zhang et al., 2024).

Detector and adaptive weights: https://doi.org/10.1109/TGRS.2024.3393303
Optimal neighborhood reconstruction: https://doi.org/10.1109/TGRS.2020.2987955
Dual mean-shift dictionary: https://doi.org/10.1016/j.isprsjprs.2020.09.008
"""

from numbers import Integral

import numpy as np
from scipy.spatial import cKDTree


def _interval_losses(pixels):
    # Dummy zero bands bracket the first and last real bands (ONR Eq. 2).
    gram = np.pad(pixels.T @ pixels, ((1, 1), (1, 1)))
    size = len(gram)
    left, right = np.triu_indices(size, k=2)
    lengths = right - left - 1
    starts = np.r_[0, np.cumsum(lengths)]
    values = np.empty(starts[-1])
    minimum_loss = np.full(size, np.inf)
    diagonals = np.diag(gram)
    inverse = np.linalg.pinv(
        np.stack([gram[left, left], gram[left, right], gram[right, left],
                  gram[right, right]], axis=-1).reshape(-1, 2, 2), hermitian=True
    )
    for index, (a, b) in enumerate(zip(left, right)):
        interior = slice(a + 1, b)
        cross = gram[[a, b], interior]
        fitted = np.einsum("ij,ij->j", cross, inverse[index] @ cross)
        loss = np.sqrt(np.maximum(diagonals[interior] - fitted, 0.0))
        values[starts[index]:starts[index + 1]] = loss
        minimum_loss[interior] = np.minimum(minimum_loss[interior], loss)
    return left, right, starts, values, minimum_loss[1:-1]


def _choose_bands(gram_size, count, left, right, starts, values, threshold):
    # Eq. 8: an interval costs the sum of its clipped reconstruction errors.
    cost = np.full((gram_size, gram_size), np.inf)
    cost[np.arange(gram_size - 1), np.arange(1, gram_size)] = 0.0
    cost[left, right] = np.add.reduceat(np.minimum(values, threshold), starts[:-1])
    # Eq. 13: choose count interior knots and finally the right sentinel.
    previous = cost[0].copy()
    parent = np.zeros((count + 1, gram_size), dtype=np.intp)
    for step in range(1, count + 1):
        candidates = previous[:, None] + cost
        parent[step] = np.argmin(candidates, axis=0)
        previous = candidates[parent[step], np.arange(gram_size)]
    chosen = np.empty(count, dtype=np.intp)
    endpoint = gram_size - 1
    for step in range(count, 0, -1):
        endpoint = parent[step, endpoint]
        chosen[step - 1] = endpoint
    return chosen


def _onr_bands(pixels, count):
    band_count = pixels.shape[1]
    if count >= band_count:
        return np.arange(band_count)
    size = band_count + 2
    left, right, starts, values, minimum = _interval_losses(pixels)
    offsets = np.zeros((size, size), dtype=np.intp)
    offsets[left, right] = starts[:-1]

    def errors(selected):
        result = np.zeros(size)
        knots = np.r_[0, selected, size - 1]
        for a, b in zip(knots[:-1], knots[1:]):
            start = offsets[a, b]
            result[a + 1:b] = values[start:start + b - a - 1]
        return result[1:-1]

    # Eq. 16 locates the drop in a 2w+1-bin moving count, w=5.
    bins = max(1, int(np.floor(0.6 * band_count + 0.5)))
    hist, limits = np.histogram(minimum, bins=bins)
    clean = np.ones(band_count, dtype=bool)
    if bins >= 11:
        counts = np.convolve(hist, np.ones(11), mode="valid")
        drops = np.flatnonzero(counts < 0.4 * counts[0])
        if len(drops):
            clean = minimum < limits[drops[0] + 5]

    initial = _choose_bands(size, count, left, right, starts, values, np.inf)
    upper = errors(initial).max()
    if upper == 0 or not clean.any():
        return initial - 1
    # Algorithm 2 tests 100 equally spaced caps and requires >95% coverage.
    for cap in upper * np.arange(1, 101) / 100:
        selected = _choose_bands(size, count, left, right, starts, values, cap)
        if np.mean(errors(selected)[clean] < cap) > 0.95:
            return selected - 1
    return initial - 1


def _self_paced_scores(pixels, dictionary, alpha, beta, initial_age):
    # Rows of dictionary are atoms D_i. Precompute the shared Gram factors.
    gram = dictionary @ dictionary.T
    diag = np.diag(gram)
    c = len(dictionary)
    eye = np.eye(c)
    t = 2 - (c - 10) / 8 if c <= 17 else 1.1
    power = 1.0 / (t - 1)
    scores = np.empty(len(pixels))
    first_normal = (c + alpha * (c - 2)) * gram + beta * eye
    first_normal.flat[::c + 1] += alpha * diag
    first_projection = np.linalg.solve(first_normal, c * dictionary)
    atom_numbers = np.arange(c)
    # Bound the memory consumed by the per-pixel c-by-c systems.
    batch_size = max(1, min(64, (64 * 1024**2) // (8 * c * c)))
    # For every y start v at one, and execute at least one iteration. This
    # resolves Algorithm 2's guard, which would skip its first update at v=1.
    for start in range(0, len(pixels), batch_size):
        targets = pixels[start:start + batch_size]
        active = np.arange(len(targets))
        steps = np.ones(len(targets), dtype=np.int64)
        cross = targets @ dictionary.T
        coeff = targets @ first_projection.T
        while len(active):
            fitted = coeff @ dictionary
            residual = np.sum((targets[active] - fitted) ** 2, axis=1)
            # Eq. 6, with ||Du - D_i u_i||^2 expanded for all atoms.
            excluded = (np.sum(fitted**2, axis=1)[:, None]
                        - 2 * coeff * (fitted @ dictionary.T) + coeff**2 * diag)
            loss = residual[:, None] + alpha * np.maximum(excluded, 0)
            age = steps[:, None] * initial_age
            weights = np.maximum(1.0 - loss / age, 0.0) ** power
            done = np.all(weights > 0, axis=1)
            scores[start + active[done]] = residual[done]
            active = active[~done]
            if not len(active):
                break
            weights = weights[~done]
            steps = steps[~done] + 1
            # With every v_i=0, Eq. 8 gives u=0 and all losses equal
            # ||y||^2. Advance past repeated zero-weight steps analytically.
            empty = ~np.any(weights > 0, axis=1)
            if empty.any():
                norms = np.sum(targets[active[empty]] ** 2, axis=1)
                steps[empty] = np.maximum(steps[empty], np.floor(norms / initial_age) + 1)
            sums = weights.sum(axis=1)
            # Expand sum_i v_i D_{-i}.T D_{-i}, where atom i is zeroed;
            # the diagonal correction restores the singly excluded term.
            multipliers = sums[:, None, None] + alpha * (
                sums[:, None, None] - weights[:, :, None] - weights[:, None, :]
            )
            normal = multipliers * gram + beta * eye
            normal[:, atom_numbers, atom_numbers] += alpha * weights * diag
            rhs = sums[:, None] * cross[active]
            coeff = np.linalg.solve(normal, rhs[:, :, None])[:, :, 0]
    return scores


def _modes(points, bandwidth):
    """Flat-kernel mean shift, initialized from occupied spectral grid cells."""
    tree = cKDTree(points)
    cells = np.floor(points / bandwidth).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    candidates = []
    for index in np.sort(indices):
        center = points[index]
        for _ in range(300):
            members = tree.query_ball_point(center, bandwidth)
            new_center = points[members].mean(axis=0)
            delta = np.linalg.norm(new_center - center)
            center = new_center
            if delta <= bandwidth * 1e-3:
                break
        else:
            raise RuntimeError("Mean shift did not converge in 300 updates")
        candidates.append((len(members), center))
    # Collapse nearby converged modes in descending order of local density.
    modes = []
    for _, center in sorted(candidates, key=lambda item: -item[0]):
        if not modes or np.all(np.linalg.norm(np.asarray(modes) - center, axis=1) > bandwidth):
            modes.append(center)
    modes = np.asarray(modes)
    labels = cKDTree(modes).query(points)[1]
    return modes, labels


def _kdist_bandwidth(pixels):
    # The b-th nearest *other* pixel (Su et al., Section 2.2); query rank
    # b+1 includes the zero-distance self match. The source plots this curve
    # and chooses its elbow visually. Use maximum chord separation here.
    rank = min(pixels.shape[1] + 1, len(pixels))
    distances = np.sort(cKDTree(pixels).query(pixels, k=[rank])[0][:, 0])[::-1]
    if distances[0] == distances[-1]:
        estimate = distances[0]
    else:
        baseline = np.linspace(distances[0], distances[-1], len(pixels))
        estimate = distances[np.argmax(baseline - distances)]
    if estimate <= 0:
        positive = distances[distances > 0]
        if not len(positive):
            raise ValueError("The k-distance plot has no positive radius; supply bandwidth1")
        estimate = positive[-1]
    return estimate


def _dual_dictionary(pixels, bandwidth1, bandwidth2):
    if bandwidth1 is None:
        bandwidth1 = _kdist_bandwidth(pixels)
    _, first = _modes(pixels, bandwidth1)
    counts = np.bincount(first)
    # Algorithm 2 in Su et al.: apply the 0.1% cutoff first, then drop one
    # smallest surviving coarse class. Preserve a sole class on a degenerate
    # scene so that the dictionary is defined; ties use the first mode index.
    parents = np.flatnonzero(counts * 1000 > len(pixels))
    if not len(parents):
        raise ValueError("Primary mean shift produced no cluster exceeding 0.1% of the scene")
    if len(parents) > 1:
        parents = np.delete(parents, np.argmin(counts[parents]))
    atoms = []
    for parent in parents:
        background = pixels[first == parent]
        modes, labels = _modes(background, bandwidth2)
        sizes = np.bincount(labels, minlength=len(modes))
        for group in np.flatnonzero(sizes * 1000 > len(background)):
            members = background[labels == group]
            nearest = np.argmin(np.sum((members - modes[group]) ** 2, axis=1))
            atoms.append(members[nearest])
    if not atoms:
        raise ValueError("No fine cluster exceeds 0.1% of its parent cluster")
    return np.asarray(atoms)


def detect(cube, *, bands=10, alpha=0.01, beta=0.01, initial_age=1.0,
           bandwidth1=0.5, bandwidth2=0.28, dictionary_pixel_indices=None):
    """Produce an H-by-W map of squared SP-ProCRD reconstruction residuals.

    Default alpha, beta, initial age and bandwidths are the paper's San Diego
    settings, held fixed when applied to other scenes. ONR uses up to 10 bands.
    The original detector uses a global dictionary, with no spatial windows.
    Input spectra are scaled using a common scene-wide minimum and range.
    Both dictionary construction and scoring use the selected bands. Setting
    bandwidth1=None estimates the first radius at the k-distance curve elbow.
    Flat-kernel mean shift uses occupied-cell seeds and merges modes within
    one bandwidth. The 0.1% cut and per-parent second stage follow Su et al.,
    Algorithm 2. A sole surviving coarse class is retained. If the exact
    source dictionary pixel indices are known, dictionary_pixel_indices
    selects those observed spectra (zero-based row-major flattened indices).
    """
    raw = np.asarray(cube)
    if raw.ndim != 3 or not all(raw.shape) or raw.dtype.kind not in "fiu":
        raise ValueError("cube must be a nonempty real numeric H-by-W-by-B array")
    pixels = np.ascontiguousarray(raw, dtype=np.float64).reshape(-1, raw.shape[-1])
    if not np.isfinite(pixels).all():
        raise ValueError("cube contains non-finite values")
    if not isinstance(bands, Integral) or isinstance(bands, bool) or bands < 1:
        raise ValueError("bands must be a positive integer")
    if (not np.all(np.isfinite([alpha, beta, initial_age, bandwidth2]))
            or alpha < 0 or min(beta, initial_age, bandwidth2) <= 0
            or (bandwidth1 is not None and (not np.isfinite(bandwidth1) or bandwidth1 <= 0))):
        raise ValueError("alpha must be nonnegative; beta, age and bandwidths must be positive")
    span = np.ptp(pixels)
    if span == 0:
        return np.zeros(raw.shape[:2])
    pixels = (pixels - pixels.min()) / span
    selected = _onr_bands(pixels, min(bands, raw.shape[-1]))
    data = np.ascontiguousarray(pixels[:, selected])
    if dictionary_pixel_indices is None:
        atoms = _dual_dictionary(data, bandwidth1, bandwidth2)
    else:
        indices = np.asarray(dictionary_pixel_indices)
        if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
                or indices.min() < 0 or indices.max() >= len(data)
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("dictionary_pixel_indices must list distinct valid pixel indices")
        atoms = data[indices]
    scores = _self_paced_scores(data, atoms, alpha, beta, initial_age)
    return scores.reshape(raw.shape[:2])
