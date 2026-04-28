"""
Synthetic 3D point cloud generator.
Generates uniform, clustered, and surface-like distributions at various scales.
"""

import os
import json
import numpy as np


def generate_uniform(N, seed=42):
    """Uniform random points in [0, 1]^3."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (N, 3)).astype(np.float32)


def generate_clustered(N, n_clusters=8, sigma=0.05, seed=42):
    """Gaussian blobs: n_clusters centers with std=sigma."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.15, 0.85, (n_clusters, 3))
    points_per_cluster = N // n_clusters
    remainder = N % n_clusters
    clouds = []
    for i in range(n_clusters):
        n_pts = points_per_cluster + (1 if i < remainder else 0)
        pts = rng.normal(centers[i], sigma, (n_pts, 3))
        clouds.append(pts)
    points = np.vstack(clouds).astype(np.float32)
    np.clip(points, 0, 1, out=points)
    return points


def generate_surface(N, seed=42):
    """Points on parametric surfaces (sphere, cylinder, plane) with noise."""
    rng = np.random.default_rng(seed)
    n_sphere = N // 3
    n_cylinder = N // 3
    n_plane = N - n_sphere - n_cylinder

    # Sphere: radius 0.3, center (0.5, 0.5, 0.5)
    theta = rng.uniform(0, 2 * np.pi, n_sphere)
    phi = rng.uniform(0, np.pi, n_sphere)
    r = 0.3
    sx = 0.5 + r * np.sin(phi) * np.cos(theta)
    sy = 0.5 + r * np.sin(phi) * np.sin(theta)
    sz = 0.5 + r * np.cos(phi)
    sphere = np.column_stack([sx, sy, sz])
    sphere += rng.normal(0, 0.005, sphere.shape)

    # Cylinder: radius 0.15, axis along z, center (0.5, 0.5, *)
    ctheta = rng.uniform(0, 2 * np.pi, n_cylinder)
    cr = 0.15
    cx = 0.5 + cr * np.cos(ctheta)
    cy = 0.5 + cr * np.sin(ctheta)
    cz = rng.uniform(0.1, 0.9, n_cylinder)
    cylinder = np.column_stack([cx, cy, cz])
    cylinder += rng.normal(0, 0.005, cylinder.shape)

    # Plane: z = 0.5 with noise
    px = rng.uniform(0.1, 0.9, n_plane)
    py = rng.uniform(0.1, 0.9, n_plane)
    pz = np.full(n_plane, 0.5)
    plane = np.column_stack([px, py, pz])
    plane += rng.normal(0, 0.01, plane.shape)

    points = np.vstack([sphere, cylinder, plane]).astype(np.float32)
    np.clip(points, 0, 1, out=points)
    # Shuffle so distributions are mixed
    rng.shuffle(points)
    return points


DISTRIBUTIONS = {
    'uniform': generate_uniform,
    'clustered': generate_clustered,
    'surface': generate_surface,
}

SCALE_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]


def generate_all_datasets(output_dir, config_dir=None):
    """Generate all synthetic datasets and save as .npy files."""
    os.makedirs(output_dir, exist_ok=True)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)

    for dist_name, gen_fn in DISTRIBUTIONS.items():
        for N in SCALE_SIZES:
            fname = f"{dist_name}_{N}.npy"
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                print(f"  [skip] {fname} already exists")
                continue
            print(f"  Generating {dist_name} N={N:,}...", end='', flush=True)
            points = gen_fn(N, seed=42)
            np.save(fpath, points)
            print(f" saved ({points.nbytes / 1e6:.1f} MB)")

            if config_dir:
                config = {
                    'distribution': dist_name,
                    'N': N,
                    'seed': 42,
                    'dtype': 'float32',
                    'shape': list(points.shape),
                }
                cfg_path = os.path.join(config_dir, fname.replace('.npy', '.json'))
                with open(cfg_path, 'w') as f:
                    json.dump(config, f, indent=2)


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(__file__))
    generate_all_datasets(
        os.path.join(base, 'data', 'synthetic'),
        os.path.join(base, 'data', 'configs'),
    )
