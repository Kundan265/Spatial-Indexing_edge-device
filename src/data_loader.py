"""
Data loading and normalization utilities.
Handles SemanticKITTI .bin files, Stanford .ply files, and .npy datasets.
"""

import os
import json
import numpy as np


def load_semantickitti_bin(bin_path):
    """Load a single SemanticKITTI Velodyne .bin file -> (N, 3) float32."""
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return scan[:, :3].copy()


def load_semantickitti_sequence(velodyne_dir, max_frames=None):
    """Load multiple frames from a SemanticKITTI velodyne directory."""
    files = sorted(f for f in os.listdir(velodyne_dir) if f.endswith('.bin'))
    if max_frames is not None:
        files = files[:max_frames]
    clouds = []
    for f in files:
        pts = load_semantickitti_bin(os.path.join(velodyne_dir, f))
        clouds.append(pts)
    return np.vstack(clouds).astype(np.float32)


def load_stanford_ply(ply_path):
    """Load a Stanford .ply file -> (N, 3) float32 using Open3D."""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points, dtype=np.float32)


def normalize_points(points):
    """Normalize points to [0, 1]^3. Returns (normalized_points, bbox_min, bbox_max)."""
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = bbox_max - bbox_min
    extent[extent == 0] = 1.0  # avoid division by zero for flat dimensions
    normalized = (points - bbox_min) / extent
    return normalized.astype(np.float32), bbox_min, bbox_max


def load_and_normalize_semantickitti(velodyne_dir, target_n=None, max_frames=None):
    """Load SemanticKITTI data, optionally subsample to target_n, normalize."""
    points = load_semantickitti_sequence(velodyne_dir, max_frames=max_frames)
    if target_n is not None and len(points) > target_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), target_n, replace=False)
        points = points[idx]
    normalized, bbox_min, bbox_max = normalize_points(points)
    return normalized


def load_and_normalize_stanford(ply_path, target_n=None):
    """Load Stanford .ply, optionally subsample, normalize."""
    points = load_stanford_ply(ply_path)
    if target_n is not None and len(points) > target_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), target_n, replace=False)
        points = points[idx]
    elif target_n is not None and len(points) < target_n:
        # Upsample by adding jittered duplicates
        rng = np.random.default_rng(42)
        n_extra = target_n - len(points)
        idx = rng.choice(len(points), n_extra, replace=True)
        noise = rng.normal(0, 1e-4, (n_extra, 3)).astype(np.float32)
        extra = points[idx] + noise
        points = np.vstack([points, extra])
    normalized, bbox_min, bbox_max = normalize_points(points)
    return normalized


def save_dataset(points, npy_path, metadata=None):
    """Save dataset as .npy with optional JSON metadata sidecar."""
    np.save(npy_path, points)
    if metadata is not None:
        json_path = npy_path.replace('.npy', '_meta.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def load_dataset(npy_path):
    """Load a .npy dataset."""
    return np.load(npy_path).astype(np.float32)


# Paths for the real-world data
VELODYNE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'realworld_data', '00', 'velodyne')
STANFORD_BUNNY = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'realworld_data', 'stanford', 'bunny',
                              'reconstruction', 'bun_zipper.ply')
STANFORD_DRAGON = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'realworld_data', 'stanford', 'dragon_recon', 'dragon_vrip.ply')
