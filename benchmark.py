"""Compatibility entry point for rans_had.benchmark."""

from rans_had.benchmark import BASELINES, INTERNAL, METRICS, SCENES, main

__all__ = ["BASELINES", "INTERNAL", "METRICS", "SCENES", "main"]

if __name__ == "__main__":
    main()
