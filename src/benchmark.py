"""
Benchmarking harness for spatial indexing structures.
Runs all experimental rounds and logs results to CSV.
"""

import os
import gc
import csv
import time
import tracemalloc
import numpy as np

from src.kdtree import KDTree
from src.octree import Octree
from src.svo import SVO
from src.data_loader import (load_and_normalize_semantickitti,
                             load_and_normalize_stanford,
                             VELODYNE_DIR, STANFORD_DRAGON)
from src.synthetic_gen import generate_uniform, generate_clustered, generate_surface


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'synthetic')

N_QUERIES = 1000
N_REPS = 5

CSV_HEADER = [
    'round', 'structure', 'N', 'k', 'radius', 'distribution', 'rep',
    'build_time_s', 'knn_mean_us', 'knn_median_us', 'knn_std_us', 'knn_p99_us',
    'radius_mean_us', 'radius_median_us', 'radius_std_us', 'radius_p99_us',
    'peak_memory_bytes', 'struct_memory_bytes', 'node_count',
    'leaf_count', 'internal_count', 'max_depth',
    'compression_ratio', 'avg_radius_result_size'
]


# =============================================================================
# Data loading helpers
# =============================================================================

def get_or_generate_data(distribution, N):
    """Get dataset for the given distribution and size.

    For synthetic data: loads from saved .npy files in data/synthetic/.
    Falls back to on-the-fly generation if .npy not found.
    For real-world data: loads from realworld_data/ directory.
    """
    # Try loading from saved .npy file first (synthetic distributions)
    if distribution in ('uniform', 'clustered', 'surface'):
        npy_path = os.path.join(DATA_DIR, f"{distribution}_{N}.npy")
        if os.path.exists(npy_path):
            print(f"    [loading {npy_path}]", flush=True)
            return np.load(npy_path).astype(np.float32)
        else:
            # Fall back to on-the-fly generation
            print(f"    [generating {distribution} N={N} on-the-fly]", flush=True)
            if distribution == 'uniform':
                return generate_uniform(N, seed=42)
            elif distribution == 'clustered':
                return generate_clustered(N, n_clusters=8, sigma=0.05, seed=42)
            elif distribution == 'surface':
                return generate_surface(N, seed=42)
    elif distribution == 'semantickitti':
        max_frames = max(N // 120000 + 2, 10)
        print(f"    [loading SemanticKITTI, target N={N}]", flush=True)
        return load_and_normalize_semantickitti(VELODYNE_DIR, target_n=N,
                                                max_frames=max_frames)
    elif distribution == 'stanford':
        print(f"    [loading Stanford Dragon, target N={N}]", flush=True)
        return load_and_normalize_stanford(STANFORD_DRAGON, target_n=N)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def generate_query_points(points, n_queries, seed):
    """Generate random query points within the bounding box of the data."""
    rng = np.random.default_rng(seed)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    queries = rng.uniform(bbox_min, bbox_max, (n_queries, 3)).astype(np.float32)
    return queries


# =============================================================================
# Warm-up: trigger JIT compilation
# =============================================================================

_warmed_up = set()

def warmup(structure_name):
    """Trigger Numba JIT compilation on tiny data."""
    if structure_name in _warmed_up:
        return
    dummy = np.random.rand(100, 3).astype(np.float32)
    dq = np.random.rand(2, 3).astype(np.float32)

    if structure_name == 'kdtree':
        t = KDTree(dummy, leaf_capacity=16)
        t.knn_batch(dq, 3)
        t.radius_batch(dq, 0.5)
    elif structure_name == 'octree':
        t = Octree(dummy, max_depth=5, max_points_per_leaf=16)
        t.knn_batch(dq, 3)
        t.radius_batch(dq, 0.5)
    elif structure_name == 'svo':
        t = SVO(dummy, max_depth=5, max_points_per_leaf=16)
        t.knn_batch(dq, 3)
        t.radius_batch(dq, 0.5)

    _warmed_up.add(structure_name)


# =============================================================================
# Measurement functions
# =============================================================================

def build_structure(structure_name, points):
    """Build a structure and return (structure, build_time, peak_memory, node_count, compression_ratio)."""
    warmup(structure_name)
    gc.collect()

    tracemalloc.start()

    t0 = time.perf_counter()
    if structure_name == 'kdtree':
        struct = KDTree(points, leaf_capacity=32)
    elif structure_name == 'octree':
        struct = Octree(points, max_depth=10, max_points_per_leaf=32)
    elif structure_name == 'svo':
        struct = SVO(points, max_depth=10, max_points_per_leaf=32)
    else:
        raise ValueError(f"Unknown structure: {structure_name}")
    build_time = time.perf_counter() - t0

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    node_count = struct.node_count
    compression_ratio = getattr(struct, 'compression_ratio', 0.0)
    struct_mem = struct.memory_bytes()
    stats = struct.structural_stats()

    return struct, build_time, peak_mem, struct_mem, stats, compression_ratio


def measure_knn(struct, query_points, k):
    """Measure KNN query latency. Returns dict of stats in microseconds."""
    M = len(query_points)
    latencies = np.empty(M, dtype=np.float64)

    for i in range(M):
        t0 = time.perf_counter()
        struct.knn(query_points[i], k)
        latencies[i] = (time.perf_counter() - t0) * 1e6  # microseconds

    return {
        'mean': float(np.mean(latencies)),
        'median': float(np.median(latencies)),
        'std': float(np.std(latencies)),
        'p99': float(np.percentile(latencies, 99)),
    }


def measure_radius(struct, query_points, radius):
    """Measure radius query latency. Returns dict of stats in microseconds + avg result size."""
    M = len(query_points)
    latencies = np.empty(M, dtype=np.float64)
    result_sizes = np.empty(M, dtype=np.int32)

    for i in range(M):
        t0 = time.perf_counter()
        result = struct.radius_query(query_points[i], radius)
        latencies[i] = (time.perf_counter() - t0) * 1e6
        result_sizes[i] = len(result)

    return {
        'mean': float(np.mean(latencies)),
        'median': float(np.median(latencies)),
        'std': float(np.std(latencies)),
        'p99': float(np.percentile(latencies, 99)),
        'avg_result_size': float(np.mean(result_sizes)),
    }


# =============================================================================
# CSV helpers
# =============================================================================

def init_csv(csv_path):
    """Create CSV file with header if it doesn't exist."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def row_exists(csv_path, round_name, structure, N, k, radius, distribution, rep):
    """Check if a result row already exists (for crash resilience)."""
    if not os.path.exists(csv_path):
        return False
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row['round'] == round_name and row['structure'] == structure and
                int(row['N']) == N and int(row['k']) == k and
                float(row['radius']) == radius and
                row['distribution'] == distribution and int(row['rep']) == rep):
                return True
    return False


def append_row(csv_path, row_dict):
    """Append a single result row to CSV."""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writerow(row_dict)


# =============================================================================
# Experiment runners
# =============================================================================

def run_single_config(csv_path, round_name, structure_name, N, k, radius,
                      distribution, rep, points=None):
    """Run a single benchmark configuration and append to CSV."""
    if row_exists(csv_path, round_name, structure_name, N, k, radius, distribution, rep):
        return

    if points is None:
        points = get_or_generate_data(distribution, N)

    query_points = generate_query_points(points, N_QUERIES, seed=rep * 1000 + 7)

    # Build
    struct, build_time, peak_mem, struct_mem, stats, comp_ratio = build_structure(
        structure_name, points)

    # KNN
    knn_stats = measure_knn(struct, query_points, k)

    # Radius
    radius_stats = measure_radius(struct, query_points, radius)

    row = {
        'round': round_name,
        'structure': structure_name,
        'N': N,
        'k': k,
        'radius': radius,
        'distribution': distribution,
        'rep': rep,
        'build_time_s': f"{build_time:.6f}",
        'knn_mean_us': f"{knn_stats['mean']:.2f}",
        'knn_median_us': f"{knn_stats['median']:.2f}",
        'knn_std_us': f"{knn_stats['std']:.2f}",
        'knn_p99_us': f"{knn_stats['p99']:.2f}",
        'radius_mean_us': f"{radius_stats['mean']:.2f}",
        'radius_median_us': f"{radius_stats['median']:.2f}",
        'radius_std_us': f"{radius_stats['std']:.2f}",
        'radius_p99_us': f"{radius_stats['p99']:.2f}",
        'peak_memory_bytes': peak_mem,
        'struct_memory_bytes': struct_mem,
        'node_count': stats['node_count'],
        'leaf_count': stats['leaf_count'],
        'internal_count': stats['internal_count'],
        'max_depth': stats['max_depth'],
        'compression_ratio': f"{comp_ratio:.8f}",
        'avg_radius_result_size': f"{radius_stats['avg_result_size']:.1f}",
    }
    append_row(csv_path, row)

    # Clean up
    del struct
    gc.collect()


def run_round1(csv_path):
    """Round 1: Scalability — vary N with fixed k=10, r=0.5, uniform."""
    print("\n" + "=" * 60)
    print("ROUND 1: Scalability (vary N)")
    print("=" * 60)
    N_values = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    structures = ['kdtree', 'octree', 'svo']
    k = 10
    r = 0.5

    for N in N_values:
        print(f"\n--- N = {N:,} ---")
        points = get_or_generate_data('uniform', N)
        for sname in structures:
            for rep in range(N_REPS):
                print(f"  {sname} rep {rep+1}/{N_REPS}...", end='', flush=True)
                t0 = time.time()
                run_single_config(csv_path, 'round1', sname, N, k, r,
                                  'uniform', rep, points=points)
                print(f" {time.time()-t0:.1f}s")
        del points
        gc.collect()


def run_round2a(csv_path):
    """Round 2A: KNN sensitivity — vary k with fixed N=1M, uniform."""
    print("\n" + "=" * 60)
    print("ROUND 2A: KNN Sensitivity (vary k)")
    print("=" * 60)
    N = 1_000_000
    k_values = [1, 5, 10, 25, 50, 100]
    r = 0.5
    structures = ['kdtree', 'octree', 'svo']
    points = get_or_generate_data('uniform', N)

    for k in k_values:
        print(f"\n--- k = {k} ---")
        for sname in structures:
            for rep in range(N_REPS):
                print(f"  {sname} rep {rep+1}/{N_REPS}...", end='', flush=True)
                t0 = time.time()
                run_single_config(csv_path, 'round2a', sname, N, k, r,
                                  'uniform', rep, points=points)
                print(f" {time.time()-t0:.1f}s")
    del points
    gc.collect()


def run_round2b(csv_path):
    """Round 2B: Radius sensitivity — vary r with fixed N=1M, uniform."""
    print("\n" + "=" * 60)
    print("ROUND 2B: Radius Sensitivity (vary r)")
    print("=" * 60)
    N = 1_000_000
    k = 10
    r_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    structures = ['kdtree', 'octree', 'svo']
    points = get_or_generate_data('uniform', N)

    for r in r_values:
        print(f"\n--- r = {r} ---")
        for sname in structures:
            for rep in range(N_REPS):
                print(f"  {sname} rep {rep+1}/{N_REPS}...", end='', flush=True)
                t0 = time.time()
                run_single_config(csv_path, 'round2b', sname, N, k, r,
                                  'uniform', rep, points=points)
                print(f" {time.time()-t0:.1f}s")
    del points
    gc.collect()


def run_round3(csv_path):
    """Round 3: Distribution sensitivity — vary distribution with fixed N=1M, k=10, r=0.5."""
    print("\n" + "=" * 60)
    print("ROUND 3: Distribution Sensitivity")
    print("=" * 60)
    N = 1_000_000
    k = 10
    r = 0.5
    distributions = ['uniform', 'clustered', 'surface', 'semantickitti', 'stanford']
    structures = ['kdtree', 'octree', 'svo']

    for dist in distributions:
        print(f"\n--- distribution = {dist} ---")
        points = get_or_generate_data(dist, N)
        for sname in structures:
            for rep in range(N_REPS):
                print(f"  {sname} rep {rep+1}/{N_REPS}...", end='', flush=True)
                t0 = time.time()
                run_single_config(csv_path, 'round3', sname, N, k, r,
                                  dist, rep, points=points)
                print(f" {time.time()-t0:.1f}s")
        del points
        gc.collect()


def run_all_benchmarks(csv_path=None):
    """Run all experimental rounds."""
    if csv_path is None:
        csv_path = os.path.join(RESULTS_DIR, 'benchmark_results.csv')
    init_csv(csv_path)

    print(f"Results will be saved to: {csv_path}")

    run_round1(csv_path)
    run_round2a(csv_path)
    run_round2b(csv_path)
    run_round3(csv_path)

    print("\n" + "=" * 60)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 60)
    return csv_path


if __name__ == '__main__':
    run_all_benchmarks()
