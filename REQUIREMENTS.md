# REQUIREMENTS — Reproducibility Specification

## Overview

This document is the authoritative companion to [`README.md`](README.md). It defines the exact datasets, software versions, evaluation protocol, hardware, and acceptance criteria needed to **reproduce and validate** the results published in this repository. A reviewer should be able to follow this file end-to-end without further clarification.

---

## Benchmarked Practices

Three classical 3D spatial-indexing structures are compared. All three implement KNN and radius queries with iterative stack-based traversal and Numba-accelerated kernels.

| # | Practice | One-line definition | Tested by |
|---|---|---|---|
| 1 | **k-d Tree** ([`src/kdtree.py`](src/kdtree.py)) | Binary space-partitioning tree built by recursive median splits along the longest axis. | All four rounds. |
| 2 | **Octree** ([`src/octree.py`](src/octree.py)) | Pointer-based 8-way hierarchical subdivision of an axis-aligned 3D bounding cube. | All four rounds. |
| 3 | **Sparse Voxel Octree (SVO)** ([`src/svo.py`](src/svo.py)) | Octree variant that allocates child slots only for occupied octants using bitmask + popcount addressing. | All four rounds; compression ratio additionally reported in Round 3. |

---

## Datasets

### Synthetic ([`data/synthetic/`](data/synthetic/))

Generated deterministically by [`src/synthetic_gen.py`](src/synthetic_gen.py) with `seed=42`. 21 files total = 3 distributions × 7 scales.

| Distribution | Description | Sizes (N) |
|---|---|---|
| `uniform` | i.i.d. uniform in `[0, 1]³` | 10k, 50k, 100k, 500k, 1M, 5M, 10M |
| `clustered` | 8 isotropic Gaussian blobs, σ = 0.05 | same scales |
| `surface` | Sphere ∪ cylinder ∪ plane with Gaussian noise | same scales |

Each cloud is stored as a `float32` `.npy` array of shape `(N, 3)` with a JSON sidecar in `data/configs/` recording `{shape, seed, dtype, distribution, N}`.

To regenerate from scratch:

```bash
python run_all.py --generate
```

### Real-world ([`realworld_data/`](realworld_data/))

| Dataset | Source | Local path | Size |
|---|---|---|---|
| **SemanticKITTI seq 00** | http://semantic-kitti.org/dataset.html (Velodyne data) | `realworld_data/00/velodyne/*.bin` | ~8.2 GB, 4,543 frames, ~120k pts/frame |
| **Stanford Bunny** | https://graphics.stanford.edu/data/3Dscanrep/ | `realworld_data/stanford/bunny/reconstruction/bun_zipper.ply` | ~3 MB |
| **Stanford Dragon** | https://graphics.stanford.edu/data/3Dscanrep/ | `realworld_data/stanford/dragon_recon/dragon_vrip.ply` | ~50 MB |

Acquisition:

```bash
# SemanticKITTI: download "Velodyne point clouds" (80 GB total; only seq 00 is needed)
#   → unzip into realworld_data/ such that realworld_data/00/velodyne/000000.bin exists
# Stanford Bunny / Dragon:
mkdir -p realworld_data/stanford
curl -L https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz   | tar xz -C realworld_data/stanford
curl -L https://graphics.stanford.edu/pub/3Dscanrep/dragon_recon.tar.gz | tar xz -C realworld_data/stanford
```

All real-world clouds are normalized to the unit cube `[0, 1]³` by [`src/data_loader.py`](src/data_loader.py) before benchmarking, so absolute coordinate scales do not affect query timings.

### Expected directory layout

```
spatial_benchmark/
├── data/
│   ├── synthetic/         # 21 .npy files
│   └── configs/           # 21 .json sidecars
└── realworld_data/
    ├── 00/velodyne/       # 4,543 .bin frames
    └── stanford/
        ├── bunny/reconstruction/bun_zipper.ply
        └── dragon_recon/dragon_vrip.ply
```

---

## Evaluation Protocol

### Splits and seeds

The benchmark is read-only on point clouds — there is no train/val/test split in the ML sense. Instead:

- **Index points**: the full point cloud `P` (size `N`) is used to build the structure.
- **Query points**: `N_QUERIES = 1000` (default; `200` in `--quick` mode) are drawn uniformly at random from `P` with `numpy.random.default_rng(seed=42)`.
- **Per-rep seed**: rep `i ∈ [0, N_REPS)` uses derived seed `42 + i` for query sampling, ensuring run-to-run determinism.

### Metrics

Definitions (all timings measured with `time.perf_counter_ns`, reported in microseconds unless stated):

| Metric | Definition |
|---|---|
| `build_time_s` | Wall-clock seconds to construct the structure from `P`. |
| `knn_mean_us` / `knn_median_us` / `knn_std_us` / `knn_p99_us` | Per-query latency statistics over `N_QUERIES` KNN queries with neighbor count `k`. |
| `radius_mean_us` / `radius_median_us` / `radius_std_us` / `radius_p99_us` | Same, for radius queries with radius `r`. |
| `peak_memory_bytes` | Peak resident set increment during build, via `tracemalloc`. |
| `struct_memory_bytes` | Steady-state in-memory size of the constructed structure. |
| `node_count` / `leaf_count` / `internal_count` / `max_depth` | Structural counts. |
| `compression_ratio` (SVO only) | `dense_octree_node_count / svo_node_count`. |
| `avg_radius_result_size` | Mean number of points returned per radius query. |

### Runs and statistical reporting

- **Reps**: `N_REPS = 5` (default) per configuration; `--quick` uses `2`.
- **Reporting**: per-rep rows in `results/benchmark_results.csv`; downstream aggregation (mean ± std, 95% CI via 1.96 σ / √N_REPS) is performed in [`src/plotting.py`](src/plotting.py) and the summary printer of `run_combined_test.py`.
- **Warm-up**: the first 10 queries of each rep are discarded to absorb Numba JIT warm-up cost.

### Comparison rules

- **Baselines**: `scipy.spatial.cKDTree` is used for correctness validation only ([`src/validate.py`](src/validate.py)) — not as a timing baseline, because its C implementation is not directly comparable to the pure-Python+Numba structures under test.
- **Hyperparameter tuning budget**: none. Each structure uses the leaf-size / max-depth defaults committed in `src/`. No per-dataset tuning is permitted.
- **Allowed adaptations**: bug fixes only; no algorithmic substitutions for a given practice.

### Experimental rounds (defined in [`src/benchmark.py`](src/benchmark.py))

| Round | Sweep | Fixed |
|---|---|---|
| 1 — Scalability | `N ∈ {10k, 50k, 100k, 500k, 1M, 5M, 10M}` | `k=10`, `r=0.5`, distribution=`uniform` |
| 2A — KNN sensitivity | `k ∈ {1, 5, 10, 25, 50, 100}` | `N=1M`, `r=0.5`, `uniform` |
| 2B — Radius sensitivity | `r ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 5.0}` | `N=1M`, `k=10`, `uniform` |
| 3 — Distribution sensitivity | `{uniform, clustered, surface, semantickitti, stanford}` | `N=1M`, `k=10`, `r=0.5` |

---

## Software and Dependencies

- **OS**: Linux (tested on Ubuntu 22.04) or macOS 13+. Windows is untested.
- **Python**: 3.10 or 3.11.
- **Pinned package versions** (suggested `requirements.txt`):

```
numpy==1.26.4
numba==0.59.1
scipy==1.13.0
pandas==2.2.2
matplotlib==3.8.4
open3d==0.18.0
```

Installation:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # if you add the file above
# or, equivalently:
pip install numpy==1.26.4 numba==0.59.1 scipy==1.13.0 \
            pandas==2.2.2 matplotlib==3.8.4 open3d==0.18.0
```

### Containerization (suggested)

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "run_combined_test.py"]
```

Build and run:

```bash
docker build -t spatial-benchmark:latest .
docker run --rm -v "$PWD/results:/app/results" -v "$PWD/plots:/app/plots" \
           spatial-benchmark:latest
```

---

## Hardware and Resource Requirements

| Tier | CPU | RAM | Disk | Wall-clock (full run) |
|---|---|---|---|---|
| **Minimum** (smoke / `--quick`) | 4-core x86_64 @ 2.5 GHz | 8 GB | 1 GB free (synthetic only) | ~10 min |
| **Recommended** | 8-core x86_64 @ 3.5 GHz (e.g., Ryzen 7 / i7-12700) | 32 GB | 15 GB free (incl. SemanticKITTI) | 2–4 hours |
| **Large-N (10M points)** | 16-core, 64 GB RAM | 64 GB | 20 GB | additional 1–2 hours |

GPU is **not** required. The 10M-point uniform configuration is the most memory-intensive single run (~8 GB peak resident).

---

## Reproducibility Steps

End-to-end reproduction from a clean checkout:

```bash
# 0. Clone
git clone <repo-url> spatial_benchmark && cd spatial_benchmark

# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Acquire datasets (see Datasets section above)
python run_all.py --generate                            # synthetic clouds
# manually place SemanticKITTI seq 00 + Stanford .ply files

# 3. Validate correctness against scipy.cKDTree
python run_all.py --validate                            # → console PASS/FAIL

# 4. Run benchmarks
python run_all.py --quick                               # smoke run (~10 min)
# OR
python run_combined_test.py                             # full pipeline

# 5. Generate plots
python run_all.py --plot                                # → plots/*.png

# 6. Inspect outputs
column -s, -t < results/benchmark_results.csv | less    # raw numbers
open plots/                                             # figures (macOS)
```

- **Seeds**: synthetic generation uses `seed=42`; per-rep query sampling uses `42 + rep_index`.
- **Logs**: stdout from the runners is the canonical log; redirect with `python run_combined_test.py 2>&1 | tee run.log` if archiving is needed.
- **Outputs**: every run appends to (or overwrites, per the runner's flag) `results/benchmark_results.csv`. Prior runs are preserved under `results/archive/`.

---

## CI and Automation

Suggested CI (e.g., GitHub Actions) jobs to run on every push:

| Job | Command | Purpose |
|---|---|---|
| `lint` | `python -m pyflakes src/ run_*.py` | Catch unused imports / undefined names. |
| `unit` | `python -m pytest tests/ -q` (when added) | Module-level tests. |
| `validate` | `python run_all.py --validate` | Cross-check vs. `scipy.cKDTree`. |
| `smoke-bench` | `python run_all.py --quick` | End-to-end smoke benchmark (~10 min). |

Sketch (`.github/workflows/ci.yml`):

```yaml
name: CI
on: [push, pull_request]
jobs:
  smoke:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: python run_all.py --generate
      - run: python run_all.py --validate
      - run: python run_all.py --quick
      - uses: actions/upload-artifact@v4
        with: { name: results, path: results/benchmark_results.csv }
```

The full real-world benchmark is **not** suitable for CI (multi-hour runtime, 8+ GB dataset); it must be run manually on the recommended hardware.

---

## Acceptance Criteria and Validation Checklist

A reproduction is considered **successful** if and only if every item below holds.

### Per-experiment pass/fail

| Round | Pass criterion |
|---|---|
| Validation | `python run_all.py --validate` prints `PASS` for KNN and radius on all three structures across all sizes; zero mismatches vs. `scipy.cKDTree`. |
| Round 1 (scalability) | All three structures build and query successfully for `N ∈ {10k … 10M}`; build time grows sub-quadratically (slope ≤ 1.3 in log-log fit). |
| Round 2A (KNN-k) | KNN latency monotonically non-decreasing in `k` for each structure. |
| Round 2B (radius) | Radius latency monotonically non-decreasing in `r`; `avg_radius_result_size` increases with `r`. |
| Round 3 (distribution) | All five distributions complete; SVO `compression_ratio > 1` on `clustered` and `surface`. |

### Reviewer checklist

- [ ] Repository cloned and dependencies installed without error.
- [ ] `python run_all.py --generate` produced 21 files in `data/synthetic/`.
- [ ] SemanticKITTI seq 00 and Stanford bunny + dragon present at the documented paths.
- [ ] `python run_all.py --validate` reports PASS on all structures.
- [ ] `python run_all.py --quick` completes in < 30 min on recommended hardware.
- [ ] `results/benchmark_results.csv` contains rows tagged with all four rounds.
- [ ] All 21 plots present in `plots/` after `--plot`.
- [ ] No `ERROR`/`FAIL` lines in stdout logs.
- [ ] Headline metrics (k-d Tree fastest KNN on uniform; SVO `compression_ratio > 1` on clustered/surface) match the README's Key Findings within ±20 %.

---

## Troubleshooting and Known Limitations

| Symptom | Cause / fix |
|---|---|
| `numba` warm-up dominates first rep | Expected; `N_REPS ≥ 2` and the 10-query warm-up handle this. Use `--quick` cautiously when reading first-rep numbers. |
| `MemoryError` on `N=10M` | Insufficient RAM. Skip 10M scale or use the Large-N hardware tier. |
| `open3d` import fails on macOS arm64 | Install `open3d>=0.18.0` (earlier versions lack arm64 wheels). |
| `FileNotFoundError: realworld_data/00/velodyne/...` | SemanticKITTI not downloaded; Round 3's `semantickitti` slice will be skipped. |
| Plots blank or fonts missing in headless Docker | The plotting module forces the `Agg` backend; ensure `libgl1` and a default font package are installed in the container. |

### Known limitations

- **No GPU implementations.** Pure-Python + Numba only; CUDA-accelerated baselines (e.g., FAISS, RAPIDS cuML) are out of scope.
- **No deletion/insertion benchmarks.** Structures are built once and queried; dynamic updates are not measured.
- **`scipy.cKDTree` is used for correctness, not as a timing baseline** — its C implementation is not directly comparable to the structures under test.
- **Real-world coverage is limited** to one LiDAR sequence and two mesh models; results may not generalize to e.g. indoor RGB-D or aerial photogrammetry data.

---

## Changelog

| Date | Author | Notes |
|---|---|---|
| 2026-04-28 | Spatial Benchmark Maintainers | Initial REQUIREMENTS specification. |
