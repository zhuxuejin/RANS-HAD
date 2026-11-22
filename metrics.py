"""Scene-level metrics for continuous anomaly scores (larger is more anomalous)."""

import numpy as np


def scene_metrics(scores, truth):
    """Calculate tie-aware AP, ROC area, and discrete operating points on all pixels.

    Supply matching 2-D arrays. Labels must be zero (background) or one (anomaly)
    and both classes must occur. Nonfinite scores are rejected, never omitted.
    """
    scores, truth = np.asarray(scores), np.asarray(truth)
    if scores.ndim != 2 or truth.shape != scores.shape or not scores.size:
        raise ValueError("scores and truth must be nonempty arrays of the same H x W shape")
    if scores.dtype.kind not in "biuf" or not np.isfinite(scores).all():
        raise ValueError("scores must be finite real numbers at every pixel")
    if truth.dtype.kind not in "biuf" or not np.all((truth == 0) | (truth == 1)):
        raise ValueError("truth must contain only binary labels (0 and 1)")

    # Index sorting preserves distinctions even when the input is a large integer.
    flat = scores.ravel()
    order = np.argsort(flat, kind="stable")[::-1]
    ranked = flat[order]
    # A threshold can only include or exclude an entire tied group.
    ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]) + 1, len(ranked)]
    tp = np.r_[0, np.cumsum(truth.ravel()[order], dtype=np.int64)[ends - 1]]
    fp = np.r_[0, ends] - tp
    positive, negative = int(tp[-1]), int(fp[-1])
    if not positive or not negative:
        raise ValueError("both background and anomaly pixels are required")

    recall = tp / positive
    precision = tp[1:] / ends
    ap = np.dot(np.diff(recall), precision)
    auroc = np.dot(np.diff(fp) / negative, (recall[:-1] + recall[1:]) / 2)
    # Compare integer counts to avoid floating-point threshold errors.
    first95 = np.searchsorted(tp, (95 * positive + 99) // 100)
    pd001 = tp[np.searchsorted(fp, negative // 1000, side="right") - 1] / positive
    pd01 = tp[np.searchsorted(fp, negative // 100, side="right") - 1] / positive
    return {
        "AP": float(ap),
        "AUROC": float(auroc),
        "FPR95": float(fp[first95] / negative),
        "PD_1e-3": float(pd001),
        "PD_1e-2": float(pd01),
    }
