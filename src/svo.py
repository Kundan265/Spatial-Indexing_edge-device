"""
Sparse Voxel Octree (SVO) implementation using Numba @njit.
Only allocates children for occupied octants using bitmask + compact child pool.
Tracks compression ratio as a key metric.
"""

import numpy as np
from numba import njit


# =============================================================================
# Heap helpers (same as kdtree/octree)
# =============================================================================

@njit(cache=True)
def _heap_sift_down(heap_dists, heap_idxs, size, i):
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < size and heap_dists[left] > heap_dists[largest]:
            largest = left
        if right < size and heap_dists[right] > heap_dists[largest]:
            largest = right
        if largest == i:
            break
        heap_dists[i], heap_dists[largest] = heap_dists[largest], heap_dists[i]
        heap_idxs[i], heap_idxs[largest] = heap_idxs[largest], heap_idxs[i]
        i = largest


@njit(cache=True)
def _heap_replace_max(heap_dists, heap_idxs, k, new_dist, new_idx):
    if new_dist < heap_dists[0]:
        heap_dists[0] = new_dist
        heap_idxs[0] = new_idx
        _heap_sift_down(heap_dists, heap_idxs, k, 0)


@njit(cache=True)
def _heap_sort(heap_dists, heap_idxs, k):
    size = k
    for i in range(k - 1, 0, -1):
        heap_dists[0], heap_dists[i] = heap_dists[i], heap_dists[0]
        heap_idxs[0], heap_idxs[i] = heap_idxs[i], heap_idxs[0]
        size -= 1
        _heap_sift_down(heap_dists, heap_idxs, size, 0)


# =============================================================================
# SVO-specific helpers
# =============================================================================

@njit(cache=True)
def _popcount(x):
    """Count set bits in a byte."""
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


@njit(cache=True)
def _get_octant(px, py, pz, cx, cy, cz):
    octant = 0
    if px >= cx:
        octant |= 1
    if py >= cy:
        octant |= 2
    if pz >= cz:
        octant |= 4
    return octant


@njit(cache=True)
def _svo_get_child(child_mask, child_offset, child_pool, node_idx, octant):
    """Get child node index for a given octant. Returns -1 if not occupied."""
    mask = child_mask[node_idx]
    bit = np.uint8(1 << octant)
    if (mask & bit) == 0:
        return -1
    # Count bits below this octant to find offset in child_pool
    below_mask = mask & np.uint8(bit - 1)
    local_idx = _popcount(below_mask)
    return child_pool[child_offset[node_idx] + local_idx]


# =============================================================================
# Build
# =============================================================================

@njit(cache=True)
def _svo_build(points, indices, max_depth, max_points_per_leaf,
               child_mask, child_offset, child_pool,
               center_x, center_y, center_z, half_size,
               point_start, point_end, is_leaf):
    """Build SVO iteratively. Returns (node_count, pool_count)."""
    N = len(points)

    temp_points = np.empty_like(points)
    temp_indices = np.empty_like(indices)

    max_stack = 256
    s_node = np.empty(max_stack, dtype=np.int32)
    s_start = np.empty(max_stack, dtype=np.int32)
    s_end = np.empty(max_stack, dtype=np.int32)
    s_depth = np.empty(max_stack, dtype=np.int32)
    s_cx = np.empty(max_stack, dtype=np.float32)
    s_cy = np.empty(max_stack, dtype=np.float32)
    s_cz = np.empty(max_stack, dtype=np.float32)
    s_hs = np.empty(max_stack, dtype=np.float32)

    # Bounding box
    min_x = points[0, 0]
    max_x = points[0, 0]
    min_y = points[0, 1]
    max_y = points[0, 1]
    min_z = points[0, 2]
    max_z = points[0, 2]
    for i in range(1, N):
        if points[i, 0] < min_x: min_x = points[i, 0]
        if points[i, 0] > max_x: max_x = points[i, 0]
        if points[i, 1] < min_y: min_y = points[i, 1]
        if points[i, 1] > max_y: max_y = points[i, 1]
        if points[i, 2] < min_z: min_z = points[i, 2]
        if points[i, 2] > max_z: max_z = points[i, 2]

    root_cx = (min_x + max_x) * 0.5
    root_cy = (min_y + max_y) * 0.5
    root_cz = (min_z + max_z) * 0.5
    root_hs = max(max_x - min_x, max(max_y - min_y, max_z - min_z)) * 0.5 + 1e-6

    node_count = 1
    pool_count = 0

    stack_top = 0
    s_node[0] = 0
    s_start[0] = 0
    s_end[0] = N
    s_depth[0] = 0
    s_cx[0] = root_cx
    s_cy[0] = root_cy
    s_cz[0] = root_cz
    s_hs[0] = root_hs

    while stack_top >= 0:
        ni = s_node[stack_top]
        start = s_start[stack_top]
        end = s_end[stack_top]
        depth = s_depth[stack_top]
        cx = s_cx[stack_top]
        cy = s_cy[stack_top]
        cz = s_cz[stack_top]
        hs = s_hs[stack_top]
        stack_top -= 1

        center_x[ni] = cx
        center_y[ni] = cy
        center_z[ni] = cz
        half_size[ni] = hs
        point_start[ni] = start
        point_end[ni] = end

        n_points = end - start

        if n_points <= max_points_per_leaf or depth >= max_depth:
            is_leaf[ni] = 1
            child_mask[ni] = 0
            child_offset[ni] = 0
            continue

        is_leaf[ni] = 0

        # Count points per octant
        counts = np.zeros(8, dtype=np.int32)
        for i in range(start, end):
            oct = _get_octant(points[i, 0], points[i, 1], points[i, 2], cx, cy, cz)
            counts[oct] += 1

        # Build child mask
        mask = np.uint8(0)
        n_occupied = 0
        for o in range(8):
            if counts[o] > 0:
                mask |= np.uint8(1 << o)
                n_occupied += 1
        child_mask[ni] = mask
        child_offset[ni] = pool_count

        # Compute offsets for rearrangement
        offsets = np.zeros(9, dtype=np.int32)
        for o in range(8):
            offsets[o + 1] = offsets[o] + counts[o]

        # Rearrange points by octant
        write_pos = np.zeros(8, dtype=np.int32)
        for o in range(8):
            write_pos[o] = offsets[o]

        for i in range(start, end):
            oct = _get_octant(points[i, 0], points[i, 1], points[i, 2], cx, cy, cz)
            dest = start + write_pos[oct]
            temp_points[dest, 0] = points[i, 0]
            temp_points[dest, 1] = points[i, 1]
            temp_points[dest, 2] = points[i, 2]
            temp_indices[dest] = indices[i]
            write_pos[oct] += 1

        for i in range(start, end):
            points[i, 0] = temp_points[i, 0]
            points[i, 1] = temp_points[i, 1]
            points[i, 2] = temp_points[i, 2]
            indices[i] = temp_indices[i]

        # Create child nodes — only for occupied octants, packed into child_pool
        child_hs = hs * 0.5
        pool_start = pool_count
        pool_count += n_occupied

        # We need to push children in reverse order so they're processed in order
        # First, assign node indices to occupied octants
        child_node_ids = np.empty(8, dtype=np.int32)
        pool_idx = pool_start
        for o in range(8):
            if counts[o] > 0:
                child_node_ids[o] = node_count
                child_pool[pool_idx] = node_count
                node_count += 1
                pool_idx += 1
            else:
                child_node_ids[o] = -1

        # Push onto stack in reverse octant order
        for o in range(7, -1, -1):
            if counts[o] == 0:
                continue
            c_start = start + offsets[o]
            c_end = start + offsets[o + 1]

            ccx = cx + child_hs * (-1.0 if (o & 1) == 0 else 1.0)
            ccy = cy + child_hs * (-1.0 if (o & 2) == 0 else 1.0)
            ccz = cz + child_hs * (-1.0 if (o & 4) == 0 else 1.0)

            stack_top += 1
            s_node[stack_top] = child_node_ids[o]
            s_start[stack_top] = c_start
            s_end[stack_top] = c_end
            s_depth[stack_top] = depth + 1
            s_cx[stack_top] = ccx
            s_cy[stack_top] = ccy
            s_cz[stack_top] = ccz
            s_hs[stack_top] = child_hs

    return node_count, pool_count


# =============================================================================
# Min distance helper
# =============================================================================

@njit(cache=True)
def _min_dist_sq_to_aabb(qx, qy, qz, cx, cy, cz, hs):
    dx = max(0.0, abs(qx - cx) - hs)
    dy = max(0.0, abs(qy - cy) - hs)
    dz = max(0.0, abs(qz - cz) - hs)
    return dx * dx + dy * dy + dz * dz


# =============================================================================
# KNN Query
# =============================================================================

@njit(cache=True)
def _svo_knn_query(query, k, points, indices,
                   child_mask, child_offset, child_pool,
                   center_x, center_y, center_z, half_size,
                   point_start, point_end, is_leaf):
    qx = float(query[0])
    qy = float(query[1])
    qz = float(query[2])

    heap_dists = np.full(k, np.inf, dtype=np.float64)
    heap_idxs = np.full(k, -1, dtype=np.int32)

    stack = np.empty(512, dtype=np.int32)
    stack_top = 0
    stack[0] = 0

    while stack_top >= 0:
        ni = stack[stack_top]
        stack_top -= 1

        if ni == -1:
            continue

        min_d = _min_dist_sq_to_aabb(qx, qy, qz,
                                     float(center_x[ni]), float(center_y[ni]),
                                     float(center_z[ni]), float(half_size[ni]))
        if min_d >= heap_dists[0]:
            continue

        if is_leaf[ni] == 1:
            for i in range(point_start[ni], point_end[ni]):
                dx = qx - float(points[i, 0])
                dy = qy - float(points[i, 1])
                dz = qz - float(points[i, 2])
                dist_sq = dx * dx + dy * dy + dz * dz
                _heap_replace_max(heap_dists, heap_idxs, k, dist_sq, indices[i])
            continue

        # Collect occupied children with their distances
        mask = child_mask[ni]
        offset = child_offset[ni]
        n_occupied = _popcount(mask)

        c_dists = np.empty(8, dtype=np.float64)
        c_ids = np.empty(8, dtype=np.int32)
        nc = 0
        pool_pos = 0
        for o in range(8):
            bit = np.uint8(1 << o)
            if (mask & bit) != 0:
                ci = child_pool[offset + pool_pos]
                pool_pos += 1
                d = _min_dist_sq_to_aabb(qx, qy, qz,
                                         float(center_x[ci]), float(center_y[ci]),
                                         float(center_z[ci]), float(half_size[ci]))
                c_dists[nc] = d
                c_ids[nc] = ci
                nc += 1

        # Insertion sort ascending
        for i in range(1, nc):
            key_d = c_dists[i]
            key_id = c_ids[i]
            j = i - 1
            while j >= 0 and c_dists[j] > key_d:
                c_dists[j + 1] = c_dists[j]
                c_ids[j + 1] = c_ids[j]
                j -= 1
            c_dists[j + 1] = key_d
            c_ids[j + 1] = key_id

        # Push farthest first
        for i in range(nc - 1, -1, -1):
            if c_dists[i] < heap_dists[0]:
                stack_top += 1
                stack[stack_top] = c_ids[i]

    _heap_sort(heap_dists, heap_idxs, k)
    return heap_dists, heap_idxs


@njit(cache=True)
def _svo_knn_batch(query_points, k, points, indices,
                   child_mask, child_offset, child_pool,
                   center_x, center_y, center_z, half_size,
                   point_start, point_end, is_leaf):
    M = len(query_points)
    all_dists = np.empty((M, k), dtype=np.float64)
    all_idxs = np.empty((M, k), dtype=np.int32)
    for i in range(M):
        dists, idxs = _svo_knn_query(
            query_points[i], k, points, indices,
            child_mask, child_offset, child_pool,
            center_x, center_y, center_z, half_size,
            point_start, point_end, is_leaf)
        all_dists[i] = dists
        all_idxs[i] = idxs
    return all_dists, all_idxs


# =============================================================================
# Radius Query
# =============================================================================

@njit(cache=True)
def _svo_radius_query(query, radius, points, indices,
                      child_mask, child_offset, child_pool,
                      center_x, center_y, center_z, half_size,
                      point_start, point_end, is_leaf):
    qx = float(query[0])
    qy = float(query[1])
    qz = float(query[2])
    r_sq = float(radius) * float(radius)
    max_results = min(len(points), 100000)
    result = np.empty(max_results, dtype=np.int32)
    count = 0

    stack = np.empty(512, dtype=np.int32)
    stack_top = 0
    stack[0] = 0

    while stack_top >= 0:
        ni = stack[stack_top]
        stack_top -= 1

        if ni == -1:
            continue

        min_d = _min_dist_sq_to_aabb(qx, qy, qz,
                                     float(center_x[ni]), float(center_y[ni]),
                                     float(center_z[ni]), float(half_size[ni]))
        if min_d > r_sq:
            continue

        if is_leaf[ni] == 1:
            for i in range(point_start[ni], point_end[ni]):
                dx = qx - float(points[i, 0])
                dy = qy - float(points[i, 1])
                dz = qz - float(points[i, 2])
                dist_sq = dx * dx + dy * dy + dz * dz
                if dist_sq <= r_sq and count < max_results:
                    result[count] = indices[i]
                    count += 1
            continue

        # Traverse occupied children
        mask = child_mask[ni]
        offset = child_offset[ni]
        pool_pos = 0
        for o in range(8):
            bit = np.uint8(1 << o)
            if (mask & bit) != 0:
                ci = child_pool[offset + pool_pos]
                pool_pos += 1
                stack_top += 1
                stack[stack_top] = ci

    return result[:count], count


@njit(cache=True)
def _svo_radius_batch(query_points, radius, points, indices,
                      child_mask, child_offset, child_pool,
                      center_x, center_y, center_z, half_size,
                      point_start, point_end, is_leaf):
    M = len(query_points)
    counts = np.empty(M, dtype=np.int32)
    for i in range(M):
        _, c = _svo_radius_query(
            query_points[i], radius, points, indices,
            child_mask, child_offset, child_pool,
            center_x, center_y, center_z, half_size,
            point_start, point_end, is_leaf)
        counts[i] = c
    return counts


# =============================================================================
# Python wrapper
# =============================================================================

class SVO:
    """Sparse Voxel Octree with Numba-accelerated build and queries."""

    def __init__(self, points, max_depth=10, max_points_per_leaf=32):
        N = len(points)
        self.max_depth = max_depth
        self.max_points_per_leaf = max_points_per_leaf

        max_nodes = max(8 * N // max_points_per_leaf + 1000, 4096)
        max_pool = max_nodes * 8  # worst case: all 8 children occupied

        self.child_mask = np.zeros(max_nodes, dtype=np.uint8)
        self.child_offset = np.zeros(max_nodes, dtype=np.int32)
        self.child_pool = np.zeros(max_pool, dtype=np.int32)
        self.center_x = np.zeros(max_nodes, dtype=np.float32)
        self.center_y = np.zeros(max_nodes, dtype=np.float32)
        self.center_z = np.zeros(max_nodes, dtype=np.float32)
        self.half_size = np.zeros(max_nodes, dtype=np.float32)
        self.point_start = np.zeros(max_nodes, dtype=np.int32)
        self.point_end = np.zeros(max_nodes, dtype=np.int32)
        self.is_leaf = np.zeros(max_nodes, dtype=np.int8)

        self.points = points.copy().astype(np.float32)
        self.indices = np.arange(N, dtype=np.int32)

        self.node_count, self.pool_count = _svo_build(
            self.points, self.indices, max_depth, max_points_per_leaf,
            self.child_mask, self.child_offset, self.child_pool,
            self.center_x, self.center_y, self.center_z,
            self.half_size, self.point_start, self.point_end, self.is_leaf)

        # Trim
        self.child_mask = self.child_mask[:self.node_count]
        self.child_offset = self.child_offset[:self.node_count]
        self.child_pool = self.child_pool[:self.pool_count]
        self.center_x = self.center_x[:self.node_count]
        self.center_y = self.center_y[:self.node_count]
        self.center_z = self.center_z[:self.node_count]
        self.half_size = self.half_size[:self.node_count]
        self.point_start = self.point_start[:self.node_count]
        self.point_end = self.point_end[:self.node_count]
        self.is_leaf = self.is_leaf[:self.node_count]

        # Compression ratio: allocated_nodes / total_possible_nodes
        # Total possible at each depth d is 8^d; sum from 0..max_depth
        total_possible = sum(8**d for d in range(max_depth + 1))
        self.compression_ratio = self.node_count / total_possible

    def knn(self, query, k):
        dists, idxs = _svo_knn_query(
            query.astype(np.float32), k,
            self.points, self.indices,
            self.child_mask, self.child_offset, self.child_pool,
            self.center_x, self.center_y, self.center_z,
            self.half_size, self.point_start, self.point_end, self.is_leaf)
        return np.sqrt(dists), idxs

    def knn_batch(self, query_points, k):
        dists, idxs = _svo_knn_batch(
            query_points.astype(np.float32), k,
            self.points, self.indices,
            self.child_mask, self.child_offset, self.child_pool,
            self.center_x, self.center_y, self.center_z,
            self.half_size, self.point_start, self.point_end, self.is_leaf)
        return np.sqrt(dists), idxs

    def radius_query(self, query, radius):
        result, count = _svo_radius_query(
            query.astype(np.float32), radius,
            self.points, self.indices,
            self.child_mask, self.child_offset, self.child_pool,
            self.center_x, self.center_y, self.center_z,
            self.half_size, self.point_start, self.point_end, self.is_leaf)
        return result

    def radius_batch(self, query_points, radius):
        counts = _svo_radius_batch(
            query_points.astype(np.float32), radius,
            self.points, self.indices,
            self.child_mask, self.child_offset, self.child_pool,
            self.center_x, self.center_y, self.center_z,
            self.half_size, self.point_start, self.point_end, self.is_leaf)
        return counts

    def memory_bytes(self):
        total = self.points.nbytes + self.indices.nbytes
        for arr in [self.child_mask, self.child_offset, self.child_pool,
                    self.center_x, self.center_y, self.center_z,
                    self.half_size, self.point_start, self.point_end, self.is_leaf]:
            total += arr.nbytes
        return total

    def structural_stats(self):
        n_leaf = int(np.sum(self.is_leaf[:self.node_count] == 1))
        n_internal = self.node_count - n_leaf
        max_d = 0
        stack = [(0, 0)]
        while stack:
            ni, d = stack.pop()
            if d > max_d:
                max_d = d
            if self.is_leaf[ni] == 0:
                mask = self.child_mask[ni]
                offset = self.child_offset[ni]
                pool_pos = 0
                for o in range(8):
                    if (mask & (1 << o)) != 0:
                        ci = self.child_pool[offset + pool_pos]
                        pool_pos += 1
                        stack.append((ci, d + 1))
        return {
            'node_count': self.node_count,
            'leaf_count': n_leaf,
            'internal_count': n_internal,
            'max_depth': max_d,
            'compression_ratio': self.compression_ratio,
            'pool_count': self.pool_count,
        }
