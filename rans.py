"""Reliability-Guided Adaptive Neighborhood Selection for hyperspectral images.

Input axes are (rows, columns, retained bands). Higher output scores indicate
stronger anomalies. Computation is scene-specific and uses no reference labels.
"""

import numpy as np


VARIANTS = (
    "rans", "fixed_5_2", "fixed_7_3", "grx_pca8", "no_joint_screening",
    "no_b_gate", "no_h_gate", "no_ess_gate", "no_typicality", "uniform_weights",
    "no_log_penalty", "no_compatibility", "unbounded_compatibility",
    "residual_selection", "no_context",
)
DIRECTIONS = ("full", "north", "south", "west", "east")


def _whiten(spectra, d, epsilon_p):
    keep = np.ptp(spectra, axis=0) > 0
    if not keep.any():
        raise ValueError("PCA requires at least one nonconstant retained band")
    x = spectra[:, keep]
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    eigenvalues, axes = np.linalg.eigh((x.T @ x) / len(x))
    d = min(d, x.shape[1], len(x) - 1)
    order = np.arange(len(eigenvalues) - 1, len(eigenvalues) - 1 - d, -1)
    scales = np.sqrt(np.maximum(eigenvalues[order], epsilon_p * eigenvalues[-1]))
    return (x @ axes[:, order]) / scales


def _log_probabilities(z, centers, variances, priors):
    # Keeping one component's residuals in memory also permits full-scene scoring.
    out = np.empty((len(z), len(centers)))
    for k, (center, variance, prior) in enumerate(zip(centers, variances, priors)):
        delta = z - center
        distance = np.einsum("nd,d,nd->n", delta, 1.0 / variance, delta)
        out[:, k] = np.log(prior) - 0.5 * (np.log(2 * np.pi * variance).sum() + distance)
    return out


def _typicality(z, k, seed, variance_floor, max_iter, tol):
    indices = np.linspace(0, len(z) - 1, min(4096, len(z)), dtype=int)
    train = z[indices]
    k = min(k, len(np.unique(train, axis=0)))
    rng = np.random.default_rng(seed)
    centers = np.empty((k, z.shape[1]))
    centers[0] = train[rng.integers(len(train))]
    nearest = np.full(len(train), np.inf)
    for j in range(1, k):
        nearest = np.minimum(nearest, np.sum((train - centers[j - 1]) ** 2, axis=1))
        centers[j] = train[rng.choice(len(train), p=nearest / nearest.sum())]

    # Seeded k-means++ centers, equal initial priors and pooled diagonal variance.
    priors = np.full(k, 1 / k)
    variances = np.tile(np.maximum(train.var(axis=0), variance_floor), (k, 1))
    previous = None
    for _ in range(max_iter):
        joint = _log_probabilities(train, centers, variances, priors)
        loglik = np.logaddexp.reduce(joint, axis=1)
        current = loglik.mean()
        if previous is not None and abs(current - previous) <= tol * max(abs(previous), 1e-12):
            break
        previous = current
        responsibilities = np.exp(joint - loglik[:, None])
        mass = responsibilities.sum(axis=0)
        # A numerically empty component contributes zero density and can be dropped.
        live = mass > np.finfo(float).tiny
        responsibilities = responsibilities[:, live]
        mass = mass[live]
        priors = mass / mass.sum()
        centers = (responsibilities.T @ train) / mass[:, None]
        variances = np.empty_like(centers)
        for j, center in enumerate(centers):
            delta = train - center
            variances[j] = np.einsum("n,nd,nd->d", responsibilities[:, j], delta, delta) / mass[j]
        variances = np.maximum(variances, variance_floor)

    joint = _log_probabilities(z, centers, variances, priors)
    nll = -np.logaddexp.reduce(joint, axis=1)
    q25, q75, q80 = np.quantile(nll, (0.25, 0.75, 0.80))
    logistic_arg = np.clip((q80 - nll) / max(q75 - q25, 1e-6), -40, 40)
    return 1.0 / (1.0 + np.exp(-logistic_arg))


def _offsets(radius, guard, direction):
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    ring = np.maximum(np.abs(dy), np.abs(dx)) > guard
    half = (np.ones_like(ring), dy < 0, dy > 0, dx < 0, dx > 0)[direction]
    dy, dx = dy[ring & half], dx[ring & half]
    taper = np.exp(-(dy * dy + dx * dx) / (2.0 * radius**2))
    return dy, dx, taper


def _neighbors(rows, cols, dy, dx, valid):
    # Out-of-bounds entries receive zero weight. Clipped indices are gather-only.
    h, w = valid.shape
    y = rows[:, None] + dy
    x = cols[:, None] + dx
    inside = (y >= 0) & (y < h) & (x >= 0) & (x < w)
    idx = np.clip(y, 0, h - 1) * w + np.clip(x, 0, w - 1)
    return idx, inside & valid.ravel()[idx]


def _candidate(rows, cols, offsets, valid, z, b, weights):
    dy, dx, taper = offsets
    idx, members = _neighbors(rows, cols, dy, dx, valid)
    count = members.sum(axis=1)
    credibility = (b[idx] * members).sum(axis=1) / np.maximum(count, 1)
    u = weights[idx] * taper * members
    total = u.sum(axis=1)
    w = u / np.where(total > 0, total, 1.0)[:, None]
    ess = 1.0 / np.where(count > 0, np.sum(w * w, axis=1), np.inf)
    samples = z[idx]
    center = np.einsum("nm,nmd->nd", w, samples)
    centered = samples - center[:, None, :]
    scatter = centered.swapaxes(1, 2) @ (w[:, :, None] * centered)
    return count, credibility, ess, center, scatter


def _residual_energy(centered, scatter, lambda0, nu):
    d = centered.shape[-1]
    scatter = 0.5 * (scatter + scatter.swapaxes(-1, -2))
    trace = np.maximum(scatter.trace(axis1=-2, axis2=-1), 0.0)
    ridge = lambda0 * trace / d + nu
    diag = np.arange(d)
    scatter[:, diag, diag] += ridge[:, None]
    q = np.linalg.solve(scatter, centered[:, :, None])[:, :, 0]
    energy = np.einsum("nd,nd->n", q, q) * ridge**2
    return energy, trace + d * nu


def _global_cache(z, weights, eligible):
    a = np.where(eligible, weights, 0.0)
    return a, eligible.sum(), a.sum(), a @ z, z.T @ (a[:, None] * z)


def _global_moments(rows, cols, z, valid, guard, caches):
    """Subtract each clipped guard from scene totals, with a centered direct fallback."""
    n, d = len(rows), z.shape[1]
    centers = np.zeros((n, d))
    scatter = np.zeros((n, d, d))
    routes = np.full(n, "unscorable", dtype="U16")
    dy, dx = np.mgrid[-guard:guard + 1, -guard:guard + 1]
    idx, members = _neighbors(rows, cols, dy.ravel(), dx.ravel(), valid)
    samples = z[idx]
    for route, (weights, count, total, first, second) in caches:
        local = weights[idx] * members
        n_out = count - (local > 0).sum(axis=1)
        use = (routes == "unscorable") & (n_out > 0)
        if not use.any():
            continue
        mass = total - local.sum(axis=1)
        stable = use & (mass > 1e-7 * total) & (n_out > 1)
        if stable.any():
            w, spectra = local[stable], samples[stable]
            mu = (first - np.einsum("nm,nmd->nd", w, spectra)) / mass[stable, None]
            raw = second - spectra.swapaxes(1, 2) @ (w[:, :, None] * spectra)
            centers[stable] = mu
            scatter[stable] = raw / mass[stable, None, None] - mu[:, :, None] * mu[:, None, :]
        for j in np.flatnonzero(use & ~stable):
            # Subtraction is ill-conditioned when almost all mass lies in the guard.
            outside = weights > 0
            outside[idx[j, members[j]]] = False
            w = weights[outside]
            w = w / w.sum()
            spectra = z[outside]
            centers[j] = w @ spectra
            delta = spectra - centers[j]
            scatter[j] = delta.T @ (w[:, None] * delta)
        routes[use] = route
    return centers, scatter, routes


def detect(cube, variant="rans", *, d=8, k=8, seed=0, epsilon_p=1e-6,
           variance_floor=1e-3, max_iter=200, tol=1e-5, tau_b=0.5,
           tau_h=1.5, n_min=None, c=4.0, lambda0=1e-2, nu=1e-3,
           chunk_size=256, valid_mask=None, return_diagnostics=False):
    """Score a scene with the manuscript defaults and one of the named VARIANTS.

    Supply the cube after sensor-specific band removal. Nonfinite spectra and
    pixels excluded by valid_mask do not enter scene statistics or neighborhoods.
    The default output is an H x W float64 array (NaN for unscorable pixels).
    With return_diagnostics=True, also return a dictionary of diagnostic maps.
    Selected directions are coded 0..4 in DIRECTIONS order, with -1 for none.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from {VARIANTS}")
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3 or min(cube.shape) < 1:
        raise ValueError("cube must have three nonempty axes: rows, columns, bands")
    integers = (d, k, max_iter, chunk_size)
    if any(not isinstance(x, (int, np.integer)) or x < 1 for x in integers):
        raise ValueError("d, k, max_iter, and chunk_size must be positive integers")
    positive = (epsilon_p, variance_floor, lambda0, nu)
    if not all(np.isfinite(x) and x > 0 for x in positive):
        raise ValueError("epsilon_p, variance_floor, lambda0, and nu must be positive")
    if not (0 <= tau_b <= 1 and np.isfinite(tau_h) and tau_h >= 0
            and np.isfinite(c) and c >= 0 and np.isfinite(tol) and tol >= 0):
        raise ValueError("Invalid screening, compatibility, or convergence parameter")
    if n_min is not None and (not np.isfinite(n_min) or n_min < 0):
        raise ValueError("n_min must be finite and nonnegative")
    h, width, bands = cube.shape
    valid = np.isfinite(cube).all(axis=2)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != valid.shape:
            raise ValueError("valid_mask dimensions must match the scene")
        valid &= mask
    pixels = np.flatnonzero(valid)
    if pixels.size < 2:
        raise ValueError("Scene PCA requires at least two valid spectra")
    features = _whiten(cube.reshape(-1, bands)[pixels], d, epsilon_p)
    d = features.shape[1]
    score = np.full(h * width, np.nan)
    route = np.full(h * width, "invalid", dtype="U16")
    diagnostics = {"route": route}
    if variant == "grx_pca8":
        centered = features - features.mean(axis=0)
        cov = centered.T @ centered / len(features)
        cov.flat[::d + 1] += 1e-6 * cov.trace() / d
        q = np.linalg.solve(cov, centered.T).T
        score[pixels] = np.einsum("nd,nd->n", centered, q)
        route[pixels] = "global_rx"
    else:
        z = np.zeros((h * width, d))
        z[pixels] = features
        b = np.zeros(h * width)
        if variant == "no_typicality":
            b[pixels] = 1.0
        else:
            b[pixels] = _typicality(features, k, seed, variance_floor, max_iter, tol)
        weights = valid.ravel().astype(float) if variant == "uniform_weights" else b
        n_min = 2 * d + 1 if n_min is None else n_min
        fixed = {"fixed_5_2": (5, 2), "fixed_7_3": (7, 3)}.get(variant)
        pairs = [fixed] if fixed else [(3, 1), (5, 2), (7, 3)]
        bank = [(r, direction, _offsets(r, g, direction))
                for r, g in pairs for direction in range(1 if fixed else 5)]
        guard = pairs[-1][1]
        caches = None
        if return_diagnostics:
            diagnostics.update(typicality=np.where(valid.ravel(), b, np.nan),
                               radius=np.full(h * width, -1, dtype=np.int16),
                               direction=np.full(h * width, -1, dtype=np.int8))
            for key in ("B", "H", "ess", "alpha"):
                diagnostics[key] = np.full(h * width, np.nan)

        for begin in range(0, len(pixels), chunk_size):
            idx = pixels[begin:begin + chunk_size]
            rows, cols = np.divmod(idx, width)
            n = len(idx)
            best = np.full(n, np.inf)
            mu = np.zeros((n, d))
            cov = np.zeros((n, d, d))
            local_ess = np.zeros(n)
            for r, direction, offsets in bank:
                count, B, ess, m, C = _candidate(rows, cols, offsets, valid, z, b, weights)
                if r == pairs[-1][0] and direction == 0:
                    context_count, context_ess, context_mu, context_cov = count, ess, m, C
                if fixed:
                    continue
                H = C.trace(axis1=1, axis2=2) / d
                accepted = count >= 2
                if variant != "no_joint_screening":
                    if variant not in ("no_b_gate", "no_typicality"):
                        accepted &= B >= tau_b
                    if variant != "no_h_gate":
                        accepted &= H <= tau_h
                    if variant != "no_ess_gate":
                        accepted &= ess >= n_min
                delta = z[idx] - m
                if variant == "residual_selection":
                    objective, _ = _residual_energy(delta, C, lambda0, nu)
                else:
                    objective = H.copy()
                    if variant != "no_compatibility":
                        distance = np.einsum("nd,nd->n", delta, delta) / d
                        objective += distance if variant == "unbounded_compatibility" else np.minimum(distance, c)
                    if variant not in ("no_log_penalty", "no_typicality"):
                        objective -= np.log(np.maximum(B, np.finfo(float).tiny))
                # Strict improvement preserves radius and direction order on exact ties.
                better = accepted & (objective < best)
                best[better] = objective[better]
                mu[better], cov[better], local_ess[better] = m[better], C[better], ess[better]
                if return_diagnostics:
                    selected = idx[better]
                    diagnostics["radius"][selected] = r
                    diagnostics["direction"][selected] = direction
                    diagnostics["B"][selected] = B[better]
                    diagnostics["H"][selected] = H[better]
                    diagnostics["ess"][selected] = ess[better]

            selected = np.isfinite(best)
            if variant == "no_context":
                fallback = ~selected
                alpha = selected.astype(float)
                route[idx] = "selected"
            else:
                fallback = context_count < 2
                mix = selected & ~fallback
                alpha = np.zeros(n)
                alpha[mix] = local_ess[mix] / (local_ess[mix] + context_ess[mix])
                a = alpha[:, None]
                delta = mu - context_mu
                mu = context_mu + a * delta
                between = delta[:, :, None] * delta[:, None, :]
                cov = (a[:, :, None] * cov + (1 - a[:, :, None]) * context_cov
                       + (a * (1 - a))[:, :, None] * between)
                route[idx] = "fixed" if fixed else "context"
                route[idx[mix]] = "mixed"
            if fallback.any():
                if caches is None:
                    eligible = valid.ravel() & (b >= tau_b)
                    caches = [(name, _global_cache(z, weights, keep))
                              for name, keep in (("global_screened", eligible),
                                                 ("global_relaxed", valid.ravel()))]
                gm, gc, gr = _global_moments(rows[fallback], cols[fallback], z, valid, guard, caches)
                mu[fallback], cov[fallback], route[idx[fallback]] = gm, gc, gr
                alpha[fallback] = np.nan
                if return_diagnostics:
                    for key in ("radius", "direction"):
                        diagnostics[key][idx[fallback]] = -1
                    for key in ("B", "H", "ess"):
                        diagnostics[key][idx[fallback]] = np.nan
            if return_diagnostics:
                diagnostics["alpha"][idx] = alpha
            ok = route[idx] != "unscorable"
            if ok.any():
                energy, normalizer = _residual_energy(z[idx[ok]] - mu[ok], cov[ok], lambda0, nu)
                score[idx[ok]] = energy / normalizer

    score = score.reshape(h, width)
    if return_diagnostics:
        return score, {key: value.reshape(h, width) for key, value in diagnostics.items()}
    return score
