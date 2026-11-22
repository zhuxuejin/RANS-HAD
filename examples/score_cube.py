"""Score one NumPy hyperspectral cube with the default RANS detector.

After installing the package, supply a real-valued array with axes
(rows, columns, retained spectral bands). The output uses the same two
spatial axes, with larger values indicating stronger anomaly responses.

    python examples/score_cube.py inputs/scene.npy results/scene_scores.npy
"""

import argparse
from pathlib import Path

import numpy as np
from rans_had import detect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cube", type=Path, help="input hyperspectral cube (.npy)")
    parser.add_argument("output", type=Path, help="output raw score map (.npy)")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npy":
        parser.error("use a .npy filename for the output score map")
    if args.cube.resolve() == args.output.resolve():
        parser.error("choose a separate output file for the scores")

    cube = np.load(args.cube, allow_pickle=False)
    if cube.ndim != 3 or cube.dtype.kind not in "fiu":
        parser.error("expected a real numeric cube with axes (rows, columns, bands)")
    scores = detect(cube)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        np.save(stream, scores, allow_pickle=False)
    print(f"Saved {scores.shape[0]} by {scores.shape[1]} scores to {args.output}")


if __name__ == "__main__":
    main()
