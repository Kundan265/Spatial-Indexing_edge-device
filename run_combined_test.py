#!/usr/bin/env python3
"""
Full pipeline: generate synthetic data, validate, benchmark, plot, summarize.

This is the recommended way to run the complete project:
    python3 run_combined_test.py

Steps:
  1. Generate synthetic datasets (.npy) if not already on disk
  2. Inspect real-world data (SemanticKITTI, Stanford)
  3. Validate correctness of all 3 structures vs scipy
  4. Run benchmarks (4 rounds, 360 configs, 5 reps each)
  5. Generate all plots
  6. Print summary table
"""

import os
import sys
import time

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
SYNTH_DIR = os.path.join(BASE_DIR, 'data', 'synthetic')
CONFIG_DIR = os.path.join(BASE_DIR, 'data', 'configs')
CSV_PATH = os.path.join(RESULTS_DIR, 'benchmark_results.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def step1_generate_synthetic():
    """Generate and save all synthetic datasets to disk."""
    print("\n" + "=" * 70)
    print("STEP 1: GENERATE SYNTHETIC DATASETS")
    print("=" * 70)
    import numpy as np
    from src.synthetic_gen import generate_all_datasets
    generate_all_datasets(SYNTH_DIR, CONFIG_DIR)

    files = [f for f in os.listdir(SYNTH_DIR) if f.endswith('.npy')]
    print(f"\n  Total synthetic datasets on disk: {len(files)}")
    total_mb = sum(os.path.getsize(os.path.join(SYNTH_DIR, f))
                   for f in files) / 1e6
    print(f"  Total size: {total_mb:.0f} MB")

    print(f"\n  {'Distribution':<12} {'N':>12} {'File Size':>12} {'Shape':>16}")
    print("  " + "-" * 56)
    for f in sorted(files):
        path = os.path.join(SYNTH_DIR, f)
        size_mb = os.path.getsize(path) / 1e6
        data = np.load(path)
        dist, n_str = f.replace('.npy', '').rsplit('_', 1)
        print(f"  {dist:<12} {int(n_str):>12,} {size_mb:>10.1f} MB {str(data.shape):>16}")


def step2_verify_real_world():
    """Load and inspect real-world datasets."""
    print("\n" + "=" * 70)
    print("STEP 2: INSPECT REAL-WORLD DATA")
    print("=" * 70)
    from src.data_loader import (load_semantickitti_bin, load_stanford_ply,
                                 normalize_points, VELODYNE_DIR,
                                 STANFORD_BUNNY, STANFORD_DRAGON)

    bin_files = sorted(f for f in os.listdir(VELODYNE_DIR) if f.endswith('.bin'))
    print(f"\n  SemanticKITTI:")
    print(f"    Directory: {VELODYNE_DIR}")
    print(f"    Total frames: {len(bin_files)}")

    sample = load_semantickitti_bin(os.path.join(VELODYNE_DIR, bin_files[0]))
    print(f"    Frame 0 shape: {sample.shape} ({sample.dtype})")
    print(f"    Range: x=[{sample[:,0].min():.1f}, {sample[:,0].max():.1f}] "
          f"y=[{sample[:,1].min():.1f}, {sample[:,1].max():.1f}] "
          f"z=[{sample[:,2].min():.1f}, {sample[:,2].max():.1f}]")

    norm, _, _ = normalize_points(sample)
    print(f"    After normalization: [{norm.min():.3f}, {norm.max():.3f}]")

    for name, path in [("Bunny", STANFORD_BUNNY), ("Dragon", STANFORD_DRAGON)]:
        pts = load_stanford_ply(path)
        print(f"\n  Stanford {name}:")
        print(f"    Path: {path}")
        print(f"    Points: {len(pts):,} ({pts.dtype})")
        norm, _, _ = normalize_points(pts)
        print(f"    After normalization: [{norm.min():.3f}, {norm.max():.3f}]")


def step3_validation():
    """Validate correctness of all 3 structures."""
    print("\n" + "=" * 70)
    print("STEP 3: CORRECTNESS VALIDATION")
    print("=" * 70)
    from src.validate import run_validation
    success = run_validation()
    if not success:
        print("\nFATAL: Validation failed. Cannot proceed.")
        sys.exit(1)


def step4_benchmark():
    """Run all 4 benchmark rounds via benchmark.py."""
    print("\n" + "=" * 70)
    print("STEP 4: BENCHMARKING")
    print("=" * 70)
    from src.benchmark import run_all_benchmarks
    run_all_benchmarks(CSV_PATH)


def step5_plots():
    """Generate all plots from results."""
    print("\n" + "=" * 70)
    print("STEP 5: GENERATE PLOTS")
    print("=" * 70)
    from src.plotting import generate_all_plots
    generate_all_plots(CSV_PATH, PLOTS_DIR)


def step6_summary():
    """Print summary of key results."""
    print("\n" + "=" * 70)
    print("STEP 6: RESULTS SUMMARY")
    print("=" * 70)
    import pandas as pd

    df = pd.read_csv(CSV_PATH)

    # Round 1 summary
    r1 = df[df['round'] == 'round1']
    print("\n  Round 1 — Scalability (median across reps):")
    print(f"  {'N':>12} | {'Metric':<14} | {'k-d Tree':>10} | {'Octree':>10} | {'SVO':>10}")
    print("  " + "-" * 68)
    for N in [10_000, 100_000, 1_000_000, 10_000_000]:
        sub = r1[r1['N'] == N]
        if sub.empty:
            continue
        for metric, label, unit in [('build_time_s', 'Build time', 's'),
                                     ('knn_median_us', 'KNN latency', 'us'),
                                     ('peak_memory_bytes', 'Memory', 'MB')]:
            vals = {}
            for s in ['kdtree', 'octree', 'svo']:
                v = sub[sub['structure'] == s][metric].median()
                if metric == 'peak_memory_bytes':
                    v = v / (1024 * 1024)
                vals[s] = v
            if metric == 'peak_memory_bytes':
                print(f"  {N:>12,} | {label:<14} | {vals['kdtree']:>8.1f} MB | "
                      f"{vals['octree']:>8.1f} MB | {vals['svo']:>8.1f} MB")
            else:
                print(f"  {N:>12,} | {label:<14} | {vals['kdtree']:>9.2f}{unit} | "
                      f"{vals['octree']:>9.2f}{unit} | {vals['svo']:>9.2f}{unit}")

    # Round 3 summary
    r3 = df[df['round'] == 'round3']
    if not r3.empty:
        print("\n  Round 3 — Distribution Sensitivity (KNN median latency, us):")
        print(f"  {'Distribution':<16} | {'k-d Tree':>10} | {'Octree':>10} | {'SVO':>10}")
        print("  " + "-" * 56)
        for dist in r3['distribution'].unique():
            vals = {}
            for s in ['kdtree', 'octree', 'svo']:
                v = r3[(r3['distribution'] == dist) &
                       (r3['structure'] == s)]['knn_median_us'].median()
                vals[s] = v
            print(f"  {dist:<16} | {vals['kdtree']:>10.1f} | "
                  f"{vals['octree']:>10.1f} | {vals['svo']:>10.1f}")

    # Structural stats if available
    if 'leaf_count' in df.columns:
        print("\n  Structural Stats (N=1M, uniform, median):")
        print(f"  {'Structure':<10} | {'Nodes':>8} | {'Leaves':>8} | {'Internal':>8} | {'Depth':>5} | {'Struct MB':>9}")
        print("  " + "-" * 60)
        r1_1m = r1[r1['N'] == 1_000_000]
        for s in ['kdtree', 'octree', 'svo']:
            ss = r1_1m[r1_1m['structure'] == s]
            if ss.empty:
                continue
            print(f"  {s:<10} | {ss['node_count'].median():>8.0f} | "
                  f"{ss['leaf_count'].median():>8.0f} | "
                  f"{ss['internal_count'].median():>8.0f} | "
                  f"{ss['max_depth'].median():>5.0f} | "
                  f"{ss['struct_memory_bytes'].median()/1e6:>8.1f}")

    # SVO compression
    svo_r3 = r3[r3['structure'] == 'svo'] if not r3.empty else pd.DataFrame()
    if not svo_r3.empty:
        print("\n  SVO Compression Ratio by Distribution:")
        for dist in svo_r3['distribution'].unique():
            cr = svo_r3[svo_r3['distribution'] == dist]['compression_ratio'].median()
            print(f"    {dist:<16}: {cr:.6f}")

    print(f"\n  Plots directory: {PLOTS_DIR}/")
    print(f"  Results CSV: {CSV_PATH}")


def main():
    print("=" * 70)
    print("  SPATIAL INDEXING BENCHMARK")
    print("  Structures: k-d Tree | Octree | Sparse Voxel Octree")
    print(f"  Platform: {os.uname().sysname} {os.uname().machine}")
    print("=" * 70)

    total_start = time.time()

    step1_generate_synthetic()
    step2_verify_real_world()
    step3_validation()
    step4_benchmark()
    step5_plots()
    step6_summary()

    total_elapsed = time.time() - total_start
    print(f"\n  Total time: {total_elapsed/60:.1f} minutes")
    print("\nDone!")


if __name__ == '__main__':
    main()
