# Evaluating detectors on ABU

The benchmark compares RANS and five collaborative-representation detectors on 13 Airport–Beach–Urban scenes. Each detector returns one continuous anomaly score per pixel. Scoring uses the spectral cube. Evaluation ranks the scores within each scene and compares them with the released binary reference map.

## Input and commands

The dataset submodule provides all 13 MAT files under `data/ABU/`. Follow the [repository setup instructions](../README.md#installation) to fetch them. To evaluate a separate copy, pass its directory to `--data`. Each selected filename must occur once. Names follow `abu-airport-1.mat` through `abu-airport-4.mat`, `abu-beach-1.mat` through `abu-beach-4.mat`, and `abu-urban-1.mat` through `abu-urban-5.mat`. The `data` variable is an H × W × B cube in the released band order. The `map` variable is an H × W array with 0 for background and 1 for anomaly.

With the package installed, these commands score all 13 scenes and produce the six-method comparison. Replace the example data and output paths with your directories.

```sh
rans-had score --data data/ABU --out runs/comparison
rans-had evaluate --data data/ABU --out runs/comparison
```

For a smaller comparison, pass the same methods and scenes to both commands. Select a separate output directory when trying different settings. The parent directories are created automatically.

```sh
rans-had score --data data/ABU --out runs/pilot --methods rans crd --scenes airport-1 beach-1 --settings examples/settings.json
rans-had evaluate --data data/ABU --out runs/pilot --methods rans crd --scenes airport-1 beach-1
```

Both commands accept `--methods all` to include the internal RANS variants. Individual variants such as `fixed_5_2`, `fixed_7_3`, `grx_pca8`, and `no_context` are also valid method names. Use `rans-had score --help` for the full list. The same arguments work with `python -m rans_had`.

## Detector settings

The UTF-8 settings file maps each method name to its keyword arguments. Unspecified parameters take their implementation defaults. The example makes selected defaults explicit for all six methods. A variant takes its own settings entry, so use a `fixed_5_2` entry to configure that control. [Example settings](../examples/settings.json)

| Method | Score and principal defaults |
| --- | --- |
| `rans` | Reliability-guided neighborhood selection and context mixing. PCA dimension `d=8`, mixture components `k=8`, `seed=0`. Outer/guard radius pairs are (3, 1), (5, 2), and (7, 3). The gates use `tau_b=0.5`, `tau_h=1.5`, and minimum effective size `2*d+1` at the retained PCA dimension. |
| `crd` | Local collaborative-reconstruction residual. `inner=5` and `outer=11` are square side lengths. Distance-weighted regularization uses `ridge=1e-6`. |
| `njcr` | Background-only residual after fitting a union dictionary. Defaults request 100 normalized-cut regions, 5 background atoms per region, 50 RX anomaly atoms, and `ridge=100`. ADMM uses `rho=0.01`, `tolerance=0.01`, and up to 30 iterations. The Python graph partition can differ from the authors' MATLAB partition. |
| `awcr` | Local residual fused with the difference between global background and anomaly residuals. Defaults use 5 global PCs, 150 fine and 100 coarse SLIC segments, `background_factor=0.2`, `anomaly_factor=0.005`, `local_weight=0.15`, and `ridge=1e-6`. Dictionary sizes scale with the original band count. |
| `sp_procrd` | Squared residual with self-paced probabilistic atom weights. Uses up to 10 selected bands, `alpha=0.01`, `beta=0.01`, `initial_age=1`, and mean-shift bandwidths 0.5 and 0.28. |
| `krcrd` | Per-band kernel residuals using a scaled minimum-spanning-tree dictionary. Uses up to 15 nearest surviving medoids, `h=0.5`, `ridge=0.01`, `coupling=0.01`, and a minimum cluster size of 2. The default cut requests the larger of `ceil(sqrt(N))` and `4*k`, capped at `floor(N/min_cluster_size)` for N scene pixels. |

The baseline modules document the reference papers, implementation choices, and optional settings. [CRD](../rans_had/baselines/crd.py) · [NJCR](../rans_had/baselines/njcr.py) · [AWCR](../rans_had/baselines/awcr.py) · [SP-ProCRD](../rans_had/baselines/sp_procrd.py) · [KRCRD](../rans_had/baselines/krcrd.py)

## Score maps and external results

`score` writes `RUN/scores/METHOD/SCENE.npy` and a JSON file with the same stem. For example, the Airport-1 RANS map is `runs/comparison/scores/rans/airport-1.npy`. The JSON records effective parameters, runtime versions, and SHA-256 hashes of the input MAT file, detector module, and score map. AWCR also records its shared CRD helper's hash.

To evaluate maps from another implementation of a supported method, place its raw H × W real-valued NumPy arrays in that layout in a separate run directory and invoke `evaluate` with the matching method and scene names. Keep the original pixel alignment and use scores for which larger means more anomalous. External maps can be supplied without a JSON sidecar. When a sidecar is present, the evaluator checks the method, scene, MAT hash, and score hash.

## Metrics and aggregation

Each evaluation uses pixels with finite spectra and finite reference labels. Both classes must occur. Every selected detector must supply a finite score for every evaluated pixel. Pixels with identical scores enter the ranked list together.

| CSV field | Definition |
| --- | --- |
| `AP` | Sum of each recall increment multiplied by precision after including that entire score group. |
| `AUROC` | Trapezoidal area under the empirical ROC curve, using false-positive rate on its original linear scale. |
| `FPR95` | False-positive rate at the first threshold reaching at least 95% recall as the score threshold decreases. Lower values indicate fewer false alarms. |
| `PD_1e-3`, `PD_1e-2` | Maximum observed recall with false-positive rate at most 0.001 or 0.01, respectively. With 900 background pixels, the 0.001 budget allows zero false positives. |

`RUN/metrics/scene_metrics.csv` includes valid-pixel counts, anomaly counts, and one row per method and scene. `macro_metrics.csv` gives each selected scene equal weight. Its `AP` field is the mean of the full-precision scene AP values. `AP_4dp` formats that row's AP to four decimal places in either CSV. The macro-only `AP_mean_scene_4dp` averages the scene AP values **after** each has been rounded to four decimal places.

Use `evaluate --report-dir PATH` to keep a separate report for a subset of existing maps. Evaluating again in the same report directory replaces its two CSV files.
