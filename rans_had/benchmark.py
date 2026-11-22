"""Score and evaluate the 13 Airport–Beach–Urban (ABU) scenes.

Requirements: Python, NumPy and SciPy. AWCR also requires scikit-image.
From this directory, with the public abu-airport-1.mat, etc. under DATA:

    python benchmark.py score --data DATA --out RUN
    python benchmark.py evaluate --data DATA --out RUN
    python benchmark.py score --data DATA --out PILOT --methods crd awcr --scenes airport-1
    python benchmark.py evaluate --data DATA --out RUN --scenes airport-1 --report-dir REPORT

Both commands default to RANS and the five external baselines on all scenes.
Use --methods all for all baselines and internal variants. Pass the same
selection to both commands. For score, --settings accepts a UTF-8 JSON file
such as {"njcr": {"max_iter": 50}, "fixed_5_2": {"chunk_size": 128}}.
Each method key affects only its named detector. Score reads only MAT variable
'data'. Evaluate accepts previously saved score maps and reads 'map' for the
labels. Put an unthresholded H by W NumPy array (higher = more anomalous) at
RUN/scores/METHOD/SCENE.npy, e.g. RUN/scores/rans/airport-1.npy.

Score also records effective settings, runtime versions, and input, detector,
and output hashes beside each .npy. Evaluate writes scene_metrics.csv and
macro_metrics.csv under RUN/metrics by default. Use --report-dir to keep a
subset report alongside a full run. Reusing a report directory replaces its
two CSV files. Macro rows are equal-
weight means over the selected scenes. AP and AP_4dp retain the full value
and its four-decimal display. Macro rows additionally give AP_mean_scene_4dp,
the mean after rounding each scene's AP (used for some manuscript tables).
"""

import argparse
import csv
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.io import loadmat


SCENES = tuple(f"{kind}-{i}" for kind, count in (("airport", 4), ("beach", 4), ("urban", 5))
               for i in range(1, count + 1))
BASELINES = ("rans", "crd", "njcr", "awcr", "sp_procrd", "krcrd")
INTERNAL = ("fixed_5_2", "fixed_7_3", "grx_pca8", "no_joint_screening",
            "no_b_gate", "no_h_gate", "no_ess_gate", "no_typicality",
            "uniform_weights", "no_log_penalty", "no_compatibility",
            "unbounded_compatibility", "residual_selection", "no_context")
METRICS = ("AP", "AUROC", "FPR95", "PD_1e-3", "PD_1e-2")


def _scene_files(directory, scenes):
    files = {}
    for scene in scenes:
        name = f"abu-{scene}.mat"
        matches = list(directory.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Expected one {name} under {directory}, found {len(matches)}")
        files[scene] = matches[0]
    return files


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, callback, binary=False):
    """Replace a single output only after writing it completely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {"mode": "wb"} if binary else {"mode": "w", "encoding": "utf-8", "newline": ""}
    with NamedTemporaryFile(dir=path.parent, delete=False, **options) as stream:
        temporary = Path(stream.name)
        try:
            callback(stream)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_versions():
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "scipy": version("scipy")}
    try:
        runtime["scikit-image"] = version("scikit-image")
    except PackageNotFoundError:
        pass
    return runtime


def _detector(method, overrides):
    internal = method == "rans" or method in INTERNAL
    module = import_module(".rans" if internal else f".baselines.{method}", __package__)
    supplied = dict(overrides)
    reserved = {"cube", "variant", "return_diagnostics", "valid_mask"} & supplied.keys()
    if reserved:
        raise ValueError(f"{method} settings contain reserved fields: {sorted(reserved)}")
    if internal:
        supplied["variant"] = method
    signature = inspect.signature(module.detect)
    bound = signature.bind(None, **supplied)
    bound.apply_defaults()
    effective = dict(bound.arguments)
    del effective[next(iter(signature.parameters))]
    record = {"method": method, "settings": effective,
              "module_sha256": _sha256(Path(module.__file__))}
    if method == "awcr":
        record["helper_sha256"] = {"crd.py": _sha256(Path(inspect.getfile(module.weighted_residual)))}
    return module.detect, supplied, record


def _score(methods, files, output, settings_path):
    settings = json.loads(settings_path.read_text(encoding="utf-8-sig")) if settings_path else {}
    if not isinstance(settings, dict) or not all(isinstance(v, dict) for v in settings.values()):
        raise ValueError("settings must be a JSON object mapping method names to objects")
    unknown = settings.keys() - set(BASELINES + INTERNAL)
    if unknown:
        raise ValueError(f"Unknown method(s) in settings: {sorted(unknown)}")
    detectors = {m: _detector(m, settings.get(m, {})) for m in methods}
    runtime = _runtime_versions()
    for scene, mat_path in files.items():
        arrays = loadmat(mat_path, variable_names=["data"])
        if "data" not in arrays or arrays["data"].ndim != 3:
            raise ValueError(f"{mat_path} must supply the H by W by B array 'data'")
        cube = arrays["data"]
        input_digest = _sha256(mat_path)
        for method, (detect, kwargs, record) in detectors.items():
            path = output / "scores" / method / f"{scene}.npy"
            scores = np.asarray(detect(cube, **kwargs))
            if scores.shape != cube.shape[:2] or scores.dtype.kind not in "fiu":
                raise ValueError(f"{method} returned an invalid score map for {scene}")
            _write(path, lambda stream: np.save(stream, scores, allow_pickle=False), binary=True)
            metadata = dict(record, scene=scene, runtime=runtime, mat_sha256=input_digest,
                            score_sha256=_sha256(path))
            _write(path.with_suffix(".json"), lambda f: json.dump(metadata, f, indent=2, allow_nan=False))
            print(f"{method} {scene} -> {path}", flush=True)


def _csv(path, rows):
    def write_rows(stream):
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write(path, write_rows)


def _evaluate(methods, files, output, report_dir):
    from .metrics import scene_metrics

    rows = []
    for scene, mat_path in files.items():
        arrays = loadmat(mat_path, variable_names=["data", "map"])
        if "data" not in arrays or "map" not in arrays or arrays["data"].ndim != 3:
            raise ValueError(f"{mat_path} must supply 'data' and 'map'")
        cube, truth = arrays["data"], arrays["map"]
        if truth.shape != cube.shape[:2]:
            raise ValueError(f"Reference mask shape differs from cube dimensions for {scene}")
        # A detector's nonfinite prediction must not silently reduce its test set.
        valid = np.isfinite(cube).all(axis=2) & np.isfinite(truth)
        if not np.isin(truth[valid], [0, 1]).all():
            raise ValueError(f"{scene} has nonbinary ground truth at a valid pixel")
        labels = truth[valid]
        input_digest = _sha256(mat_path)
        for method in methods:
            path = output / "scores" / method / f"{scene}.npy"
            scores = np.load(path, allow_pickle=False)
            if scores.shape != truth.shape or scores.dtype.kind not in "fiu":
                raise ValueError(f"Expected a real-valued {truth.shape} score array at {path}")
            if not np.isfinite(scores[valid]).all():
                raise ValueError(f"{path} has nonfinite predictions within the evaluation mask")
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                record = json.loads(sidecar.read_text(encoding="utf-8"))
                if (record.get("method") != method or record.get("scene") != scene or
                        record.get("score_sha256") != _sha256(path) or
                        record.get("mat_sha256") != input_digest):
                    raise ValueError(f"Score or input conflicts with its settings record: {sidecar}")
            metrics = scene_metrics(scores[valid].reshape(1, -1), labels.reshape(1, -1))
            row = {"method": method, "scene": scene, "pixels": labels.size,
                   "anomaly_pixels": int(np.count_nonzero(labels))}
            row.update({name: float(metrics[name]) for name in METRICS})
            row["AP_4dp"] = format(row["AP"], ".4f")
            rows.append(row)
    macro = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        sums = {name: float(np.mean([row[name] for row in subset])) for name in METRICS}
        macro.append({"method": method, "scenes": len(subset), **sums,
                      "AP_4dp": format(sums["AP"], ".4f"),
                      "AP_mean_scene_4dp": float(np.mean([float(r["AP_4dp"]) for r in subset]))})
    _csv(report_dir / "scene_metrics.csv", rows)
    _csv(report_dir / "macro_metrics.csv", macro)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("score", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--data", type=Path, required=True, help="directory containing ABU MAT files")
        sub.add_argument("--out", type=Path, required=True, help="run directory for score maps and metrics")
        sub.add_argument("--methods", nargs="+", choices=(*BASELINES, *INTERNAL, "all"), default=BASELINES)
        sub.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES)
        if command == "score":
            sub.add_argument("--settings", type=Path, help="method-to-parameters JSON file")
        else:
            sub.add_argument("--report-dir", type=Path, help="CSV output directory (default: OUT/metrics)")
    args = parser.parse_args()
    if "all" in args.methods and len(args.methods) != 1:
        parser.error("use --methods all by itself")
    methods = BASELINES + INTERNAL if "all" in args.methods else tuple(dict.fromkeys(args.methods))
    files = _scene_files(args.data, dict.fromkeys(args.scenes))
    if args.command == "score":
        _score(methods, files, args.out, args.settings)
    else:
        _evaluate(methods, files, args.out, args.report_dir or args.out / "metrics")


if __name__ == "__main__":
    main()
