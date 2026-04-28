# Spatial Benchmark — k-d Tree vs. Octree vs. Sparse Voxel Octree

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-pending-lightgrey.svg)](#ci-and-automation)
[![Reproducible](https://img.shields.io/badge/reproducible-yes-brightgreen.svg)](REQUIREMENTS.md)

> A reproducible benchmark of three classical 3D spatial-indexing structures — **k-d Tree**, **Octree**, and **Sparse Voxel Octree (SVO)** — measuring build time, query latency, and memory across synthetic and real-world point clouds.

---

## Motivation and Scope

Nearest-neighbor and radius queries on 3D point clouds are foundational to robotics, graphics, and geospatial systems, yet practitioners rarely have an apples-to-apples comparison of the most common indexing structures across realistic data scales and distributions. This repository benchmarks **three algorithmic practices** for spatial indexing:

1. **k-d Tree** — binary space partitioning with median splits.
2. **Octree** — pointer-based 8-way hierarchical cubic subdivision.
3. **Sparse Voxel Octree (SVO)** — memory-optimized octree using bitmask child encoding.

Each is implemented with Numba-accelerated kernels and exercised over uniform, clustered, and surface-like synthetic distributions, plus real-world LiDAR (SemanticKITTI) and Stanford 3D mesh data. The intended audience is researchers, engineers, and reviewers who need a defensible, end-to-end performance comparison they can rerun on their own hardware.

---

## Key Findings

> Headline takeaways from the included `results/benchmark_results.csv` (full numbers and plots in [`plots/`](plots/) and [`results/`](results/)):

- **k-d Tree** delivers the fastest KNN latency on uniform data across all scales tested (10k → 10M points).
- **SVO** achieves substantial memory compression on clustered and surface-like distributions where many octants are empty, at the cost of modest query overhead.
- **Octree** sits between the two — competitive radius-query performance, but heavier memory than SVO on sparse data.
- All three structures match `scipy.spatial.cKDTree` results exactly on the validation suite.

---

## What's in this Repo

| Path | Description |
|---|---|
| [`run_combined_test.py`](run_combined_test.py) | One-shot pipeline: generate → validate → benchmark → plot → summarize. |
| [`run_all.py`](run_all.py) | Modular runner with `--validate`, `--benchmark`, `--plot`, `--quick`, `--generate` flags. |
| [`src/kdtree.py`](src/kdtree.py) | k-d Tree implementation (Numba JIT, quickselect median split). |
| [`src/octree.py`](src/octree.py) | Pointer-based Octree with 8-way subdivision. |
| [`src/svo.py`](src/svo.py) | Sparse Voxel Octree with bitmask child encoding. |
| [`src/benchmark.py`](src/benchmark.py) | Harness defining the four experimental rounds and CSV logging. |
| [`src/validate.py`](src/validate.py) | Correctness check against `scipy.spatial.cKDTree`. |
| [`src/data_loader.py`](src/data_loader.py) | Loaders for `.npy`, SemanticKITTI `.bin`, and Stanford `.ply`. |
| [`src/synthetic_gen.py`](src/synthetic_gen.py) | Uniform / clustered / surface point-cloud generators. |
| [`src/plotting.py`](src/plotting.py) | Matplotlib figures (log-log scaling, sensitivity, structural stats). |
| [`data/synthetic/`](data/synthetic/) | 21 `.npy` synthetic clouds (10k–10M × 3 distributions). |
| [`data/configs/`](data/configs/) | JSON metadata sidecars for synthetic clouds. |
| [`realworld_data/`](realworld_data/) | SemanticKITTI seq 00 + Stanford bunny / dragon meshes. |
| [`results/`](results/) | `benchmark_results.csv` and archived runs. |
| [`plots/`](plots/) | 21 publication-ready PNG figures. |
| [`REQUIREMENTS.md`](REQUIREMENTS.md) | Exact datasets, versions, seeds, and reviewer checklist. |

---

## Quickstart

Reproduce a baseline run in under 30 minutes on a modern laptop (8 GB RAM, no GPU required).

```bash
# 1. Clone and enter
git clone <repo-url> spatial_benchmark
cd spatial_benchmark

# 2. Install dependencies (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install numpy numba scipy pandas matplotlib open3d

# 3. Run the quick smoke benchmark (~5–15 min, 2 reps × 200 queries)
python run_all.py --quick

# 4. Inspect results
ls results/benchmark_results.csv
ls plots/
```

Expected output:

- `results/benchmark_results.csv` with one row per `(round, structure, N, k, radius, distribution, rep)` tuple.
- A console summary table of mean build / KNN / radius latencies per structure.
- PNG plots written to `plots/`.

For a full run (5 reps × 1000 queries × all rounds, several hours on the recommended hardware):

```bash
python run_combined_test.py
```

---

## How the Benchmark Works (High Level)

The harness in [`src/benchmark.py`](src/benchmark.py) sweeps **four experimental rounds**:

| Round | What varies | Holds fixed |
|---|---|---|
| **1. Scalability** | `N ∈ {10k, 50k, 100k, 500k, 1M, 5M, 10M}` | `k=10`, `r=0.5`, uniform |
| **2A. KNN sensitivity** | `k ∈ {1, 5, 10, 25, 50, 100}` | `N=1M`, uniform |
| **2B. Radius sensitivity** | `r ∈ {0.01, 0.05, 0.1, 0.5, 1.0, 5.0}` | `N=1M`, uniform |
| **3. Distribution sensitivity** | uniform / clustered / surface / SemanticKITTI / Stanford | `N=1M`, `k=10`, `r=0.5` |

For each configuration, every structure is built and queried 5 times (default). Reported metrics include build time, mean / median / std / p99 query latencies (µs), peak and structural memory (bytes), node / leaf / depth counts, and SVO compression ratio. Correctness is cross-checked against `scipy.spatial.cKDTree` in [`src/validate.py`](src/validate.py).

See [`REQUIREMENTS.md`](REQUIREMENTS.md) for exact definitions, seeds, and acceptance criteria.


---

## License and Contact

Released under the MIT License (see `LICENSE`). For questions, bug reports, or contributions, please open an issue on the GitHub tracker or contact the maintainers via the repository's issue template.

---

## Changelog

| Date | Author | Notes |
|---|---|---|
| 2026-04-28 | Spatial Benchmark Maintainers | Initial README. |
