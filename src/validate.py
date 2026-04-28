"""
Correctness validation: compare all three structures against scipy.spatial.KDTree.
Must pass before any benchmarking.
"""

import numpy as np
from scipy.spatial import cKDTree
from src.kdtree import KDTree
from src.octree import Octree
from src.svo import SVO


def validate_knn(structure, name, points, query_points, k_values, scipy_tree):
    """Validate KNN queries against scipy. Returns (pass_count, fail_count)."""
    passes = 0
    fails = 0
    for k in k_values:
        scipy_dists, scipy_idxs = scipy_tree.query(query_points, k=k)
        if k == 1:
            scipy_dists = scipy_dists.reshape(-1, 1)
            scipy_idxs = scipy_idxs.reshape(-1, 1)

        for i in range(len(query_points)):
            our_dists, our_idxs = structure.knn(query_points[i], k)
            # Compare distances (handle ties: check that k-th distance matches)
            scipy_d_sorted = np.sort(scipy_dists[i])
            our_d_sorted = np.sort(our_dists)
            if np.allclose(scipy_d_sorted, our_d_sorted, rtol=1e-4, atol=1e-6):
                passes += 1
            else:
                fails += 1
                if fails <= 3:
                    print(f"  FAIL {name} KNN k={k} query {i}:")
                    print(f"    scipy dists: {scipy_d_sorted}")
                    print(f"    our dists:   {our_d_sorted}")
    return passes, fails


def validate_radius(structure, name, points, query_points, r_values, scipy_tree):
    """Validate radius queries against scipy. Returns (pass_count, fail_count)."""
    passes = 0
    fails = 0
    for r in r_values:
        scipy_results = scipy_tree.query_ball_point(query_points, r)
        for i in range(len(query_points)):
            our_result = structure.radius_query(query_points[i], r)
            scipy_set = set(scipy_results[i])
            our_set = set(our_result.tolist())
            if scipy_set == our_set:
                passes += 1
            else:
                fails += 1
                if fails <= 3:
                    missing = scipy_set - our_set
                    extra = our_set - scipy_set
                    print(f"  FAIL {name} radius r={r} query {i}: "
                          f"missing={len(missing)} extra={len(extra)} "
                          f"(scipy={len(scipy_set)} ours={len(our_set)})")
    return passes, fails


def run_validation(N=10000, n_queries=100, seed=42):
    """Run full validation suite. Returns True if all pass."""
    print(f"=== Validation: N={N}, queries={n_queries}, seed={seed} ===")
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 1, (N, 3)).astype(np.float32)
    query_points = rng.uniform(0, 1, (n_queries, 3)).astype(np.float32)

    k_values = [1, 5, 10, 25]
    r_values = [0.01, 0.05, 0.1, 0.5]

    # Build scipy reference
    print("Building scipy.spatial.cKDTree (reference)...")
    scipy_tree = cKDTree(points)

    # Build our structures
    structures = {}
    print("Building k-d Tree...", end='', flush=True)
    structures['kdtree'] = KDTree(points, leaf_capacity=32)
    print(f" done ({structures['kdtree'].node_count} nodes)")

    print("Building Octree...", end='', flush=True)
    structures['octree'] = Octree(points, max_depth=10, max_points_per_leaf=32)
    print(f" done ({structures['octree'].node_count} nodes)")

    print("Building SVO...", end='', flush=True)
    structures['svo'] = SVO(points, max_depth=10, max_points_per_leaf=32)
    print(f" done ({structures['svo'].node_count} nodes)")

    all_passed = True
    total_pass = 0
    total_fail = 0

    for name, struct in structures.items():
        print(f"\nValidating {name}...")
        kp, kf = validate_knn(struct, name, points, query_points, k_values, scipy_tree)
        rp, rf = validate_radius(struct, name, points, query_points, r_values, scipy_tree)
        print(f"  KNN:    {kp} pass, {kf} fail")
        print(f"  Radius: {rp} pass, {rf} fail")
        total_pass += kp + rp
        total_fail += kf + rf
        if kf > 0 or rf > 0:
            all_passed = False

    print(f"\n=== Total: {total_pass} pass, {total_fail} fail ===")
    if all_passed:
        print("ALL VALIDATIONS PASSED")
    else:
        print("VALIDATION FAILED")
    return all_passed


if __name__ == '__main__':
    success = run_validation()
    exit(0 if success else 1)
