#!/usr/bin/env python3
"""
Master runner for the Spatial Indexing Benchmark project.
Validates correctness, runs benchmarks, and generates plots.

Usage:
    python run_all.py              # Run everything
    python run_all.py --validate   # Validation only
    python run_all.py --benchmark  # Benchmarks only
    python run_all.py --plot       # Plots only
    python run_all.py --quick      # Quick run (smaller N, fewer reps)
"""

import os
import sys
import time
import argparse


BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
CSV_PATH = os.path.join(RESULTS_DIR, 'benchmark_results.csv')
SYNTH_DIR = os.path.join(BASE_DIR, 'data', 'synthetic')
CONFIG_DIR = os.path.join(BASE_DIR, 'data', 'configs')


def run_generate():
    print("\n" + "=" * 60)
    print("STEP 0: GENERATE SYNTHETIC DATA")
    print("=" * 60)
    os.makedirs(SYNTH_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    from src.synthetic_gen import generate_all_datasets
    generate_all_datasets(SYNTH_DIR, CONFIG_DIR)
    files = [f for f in os.listdir(SYNTH_DIR) if f.endswith('.npy')]
    print(f"  {len(files)} synthetic datasets on disk.")


def run_validation():
    print("\n" + "=" * 60)
    print("STEP 1: VALIDATION")
    print("=" * 60)
    from src.validate import run_validation as validate
    success = validate()
    if not success:
        print("\nValidation FAILED. Aborting.")
        sys.exit(1)
    print("\nValidation PASSED.\n")


def run_benchmarks(quick=False):
    print("\n" + "=" * 60)
    print("STEP 2: BENCHMARKS")
    print("=" * 60)
    import src.benchmark as bm

    if quick:
        # Override for quick mode: smaller sizes, fewer reps
        print("QUICK MODE: reduced sizes and reps")
        bm.N_REPS = 2
        bm.N_QUERIES = 200

    t0 = time.time()
    bm.run_all_benchmarks(CSV_PATH)
    elapsed = time.time() - t0
    print(f"\nBenchmarks completed in {elapsed/60:.1f} minutes.")


def run_plots():
    print("\n" + "=" * 60)
    print("STEP 3: PLOTTING")
    print("=" * 60)
    from src.plotting import generate_all_plots
    generate_all_plots(CSV_PATH, PLOTS_DIR)


def main():
    parser = argparse.ArgumentParser(description='Spatial Indexing Benchmark Runner')
    parser.add_argument('--validate', action='store_true', help='Run validation only')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks only')
    parser.add_argument('--plot', action='store_true', help='Generate plots only')
    parser.add_argument('--generate', action='store_true',
                        help='Generate synthetic datasets')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: smaller N values and fewer repetitions')
    args = parser.parse_args()

    # If no specific flag, run everything
    run_all = not (args.validate or args.benchmark or args.plot or args.generate)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Efficient 3D Spatial Indexing Benchmark")
    print("Structures: k-d Tree, Octree, Sparse Voxel Octree")
    print(f"Platform: {os.uname().sysname} {os.uname().machine}")
    print("=" * 60)

    total_start = time.time()

    if run_all or args.generate:
        run_generate()

    if run_all or args.validate:
        run_validation()

    if run_all or args.benchmark:
        run_benchmarks(quick=args.quick)

    if run_all or args.plot:
        if os.path.exists(CSV_PATH):
            run_plots()
        else:
            print(f"\nNo results CSV found at {CSV_PATH}. Run benchmarks first.")

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print("Done!")


if __name__ == '__main__':
    main()
