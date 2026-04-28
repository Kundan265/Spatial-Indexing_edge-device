"""
k-d Tree implementation using Numba @njit with flat-array (struct-of-arrays) layout.
Supports KNN and radius queries with iterative stack-based traversal.
"""

import numpy as np
from numba import njit, types
from numba.typed import Dict

# =============================================================================
# @njit helper functions
# =============================================================================

@njit(cache=True)
def _swap_points(points, indices, i, j):
    """Swap points and indices at positions i and j."""
    for d in range(3):
        tmp = points[i, d]
        points[i, d] = points[j, d]
        points[j, d] = tmp
    tmp_idx = indices[i]
    indices[i] = indices[j]
    indices[j] = tmp_idx


@njit(cache=True)
def _quickselect(points, indices, start, end, axis, k):
    """In-place quickselect: rearranges so points[start+k] is the k-th smallest
    along axis, with smaller elements before and larger after."""
    lo = start
    hi = end - 1
    while lo < hi:
        pivot_val = points[lo + (hi - lo) // 2, axis]
        i = lo
        j = hi
        while i <= j:
            while points[i, axis] < pivot_val:
                i += 1
            while points[j, axis] > pivot_val:
                j -= 1
            if i <= j:
                _swap_points(points, indices, i, j)
                i += 1
                j -= 1
        target = start + k
        if target <= j:
            hi = j
        elif target >= i:
            lo = i
        else:
            break


# =============================================================================
# Max-heap for KNN (shared across all structures)
# =============================================================================

@njit(cache=True)
def _heap_sift_down(heap_dists, heap_idxs, size, i):
    """Sift down element at position i in a max-heap."""
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
        # swap
        heap_dists[i], heap_dists[largest] = heap_dists[largest], heap_dists[i]
        heap_idxs[i], heap_idxs[largest] = heap_idxs[largest], heap_idxs[i]
        i = largest


@njit(cache=True)
def _heap_replace_max(heap_dists, heap_idxs, k, new_dist, new_idx):
    """Replace the max element if new_dist is smaller."""
    if new_dist < heap_dists[0]:
        heap_dists[0] = new_dist
        heap_idxs[0] = new_idx
        _heap_sift_down(heap_dists, heap_idxs, k, 0)


@njit(cache=True)
def _heap_sort(heap_dists, heap_idxs, k):
    """Sort a max-heap in ascending order of distance."""
    size = k
    for i in range(k - 1, 0, -1):
        heap_dists[0], heap_dists[i] = heap_dists[i], heap_dists[0]
        heap_idxs[0], heap_idxs[i] = heap_idxs[i], heap_idxs[0]
        size -= 1
        _heap_sift_down(heap_dists, heap_idxs, size, 0)


# =============================================================================
# Build
# =============================================================================

@njit(cache=True)
def _kdtree_build(points, indices, leaf_capacity,
                  split_axes, split_values, left_children, right_children,
                  point_start, point_end):
    """Build k-d tree iteratively. Returns node_count."""
    # Stack: (node_idx, start, end, depth)
    max_stack = 128
    stack_node = np.empty(max_stack, dtype=np.int32)
    stack_start = np.empty(max_stack, dtype=np.int32)
    stack_end = np.empty(max_stack, dtype=np.int32)
    stack_depth = np.empty(max_stack, dtype=np.int32)

    node_count = 1
    stack_top = 0
    stack_node[0] = 0
    stack_start[0] = 0
    stack_end[0] = len(points)
    stack_depth[0] = 0

    while stack_top >= 0:
        node_idx = stack_node[stack_top]
        start = stack_start[stack_top]
        end = stack_end[stack_top]
        depth = stack_depth[stack_top]
        stack_top -= 1

        n_points = end - start
        point_start[node_idx] = start
        point_end[node_idx] = end

        if n_points <= leaf_capacity:
            # Leaf node
            split_axes[node_idx] = -1
            split_values[node_idx] = 0.0
            left_children[node_idx] = -1
            right_children[node_idx] = -1
            continue

        # Choose axis and find median
        axis = depth % 3
        mid = n_points // 2
        _quickselect(points, indices, start, end, axis, mid)

        split_axes[node_idx] = axis
        split_values[node_idx] = points[start + mid, axis]

        # Allocate children
        left_idx = node_count
        right_idx = node_count + 1
        node_count += 2
        left_children[node_idx] = left_idx
        right_children[node_idx] = right_idx

        # Push right child first (so left is processed first)
        stack_top += 1
        stack_node[stack_top] = right_idx
        stack_start[stack_top] = start + mid
        stack_end[stack_top] = end
        stack_depth[stack_top] = depth + 1

        stack_top += 1
        stack_node[stack_top] = left_idx
        stack_start[stack_top] = start
        stack_end[stack_top] = start + mid
        stack_depth[stack_top] = depth + 1

    return node_count


# =============================================================================
# KNN Query
# =============================================================================

@njit(cache=True)
def _kdtree_knn_query(query, k, points, indices,
                      split_axes, split_values, left_children, right_children,
                      point_start, point_end):
    """KNN query returning (sorted_dists, sorted_indices). Distances are squared."""
    heap_dists = np.full(k, np.inf, dtype=np.float64)
    heap_idxs = np.full(k, -1, dtype=np.int32)

    stack = np.empty(128, dtype=np.int32)
    stack_top = 0
    stack[0] = 0  # root

    while stack_top >= 0:
        node_idx = stack[stack_top]
        stack_top -= 1

        if node_idx == -1:
            continue

        if split_axes[node_idx] == -1:
            # Leaf: check all points
            for i in range(point_start[node_idx], point_end[node_idx]):
                dx = float(query[0]) - float(points[i, 0])
                dy = float(query[1]) - float(points[i, 1])
                dz = float(query[2]) - float(points[i, 2])
                dist_sq = dx * dx + dy * dy + dz * dz
                _heap_replace_max(heap_dists, heap_idxs, k, dist_sq, indices[i])
            continue

        axis = split_axes[node_idx]
        diff = float(query[axis]) - float(split_values[node_idx])

        if diff <= 0:
            near_child = left_children[node_idx]
            far_child = right_children[node_idx]
        else:
            near_child = right_children[node_idx]
            far_child = left_children[node_idx]

        # Push far child FIRST (deeper in stack, popped LAST)
        # Push near child SECOND (on top, popped FIRST)
        # This ensures we explore the near subtree first for better pruning
        if diff * diff < heap_dists[0]:
            stack_top += 1
            stack[stack_top] = far_child

        stack_top += 1
        stack[stack_top] = near_child

    _heap_sort(heap_dists, heap_idxs, k)
    return heap_dists, heap_idxs


@njit(cache=True)
def _kdtree_knn_batch(query_points, k, points, indices,
                      split_axes, split_values, left_children, right_children,
                      point_start, point_end):
    """Batch KNN query. Returns (all_dists[M,k], all_idxs[M,k])."""
    M = len(query_points)
    all_dists = np.empty((M, k), dtype=np.float64)
    all_idxs = np.empty((M, k), dtype=np.int32)
    for i in range(M):
        dists, idxs = _kdtree_knn_query(
            query_points[i], k, points, indices,
            split_axes, split_values, left_children, right_children,
            point_start, point_end)
        all_dists[i] = dists
        all_idxs[i] = idxs
    return all_dists, all_idxs


# =============================================================================
# Radius Query
# =============================================================================

@njit(cache=True)
def _kdtree_radius_query(query, radius, points, indices,
                         split_axes, split_values, left_children, right_children,
                         point_start, point_end):
    """Radius query. Returns (result_indices, count)."""
    r_sq = float(radius) * float(radius)
    max_results = min(len(points), 100000)
    result = np.empty(max_results, dtype=np.int32)
    count = 0

    stack = np.empty(128, dtype=np.int32)
    stack_top = 0
    stack[0] = 0

    while stack_top >= 0:
        node_idx = stack[stack_top]
        stack_top -= 1

        if node_idx == -1:
            continue

        if split_axes[node_idx] == -1:
            for i in range(point_start[node_idx], point_end[node_idx]):
                dx = float(query[0]) - float(points[i, 0])
                dy = float(query[1]) - float(points[i, 1])
                dz = float(query[2]) - float(points[i, 2])
                dist_sq = dx * dx + dy * dy + dz * dz
                if dist_sq <= r_sq and count < max_results:
                    result[count] = indices[i]
                    count += 1
            continue

        axis = split_axes[node_idx]
        diff = float(query[axis]) - float(split_values[node_idx])

        if diff <= 0:
            near_child = left_children[node_idx]
            far_child = right_children[node_idx]
        else:
            near_child = right_children[node_idx]
            far_child = left_children[node_idx]

        # Push far first (popped last), near second (popped first)
        if diff * diff <= r_sq:
            stack_top += 1
            stack[stack_top] = far_child

        stack_top += 1
        stack[stack_top] = near_child

    return result[:count], count


@njit(cache=True)
def _kdtree_radius_batch(query_points, radius, points, indices,
                         split_axes, split_values, left_children, right_children,
                         point_start, point_end):
    """Batch radius query. Returns list of counts and total result size."""
    M = len(query_points)
    counts = np.empty(M, dtype=np.int32)
    for i in range(M):
        _, c = _kdtree_radius_query(
            query_points[i], radius, points, indices,
            split_axes, split_values, left_children, right_children,
            point_start, point_end)
        counts[i] = c
    return counts


# =============================================================================
# Python wrapper class
# =============================================================================

class KDTree:
    """k-d Tree with Numba-accelerated build and queries."""

    def __init__(self, points, leaf_capacity=32):
        N = len(points)
        self.leaf_capacity = leaf_capacity

        # Allocate arrays (Python level — tracemalloc can track these)
        max_nodes = max(4 * N // leaf_capacity, 1024)
        self.split_axes = np.full(max_nodes, -1, dtype=np.int32)
        self.split_values = np.zeros(max_nodes, dtype=np.float32)
        self.left_children = np.full(max_nodes, -1, dtype=np.int32)
        self.right_children = np.full(max_nodes, -1, dtype=np.int32)
        self.point_start = np.zeros(max_nodes, dtype=np.int32)
        self.point_end = np.zeros(max_nodes, dtype=np.int32)

        # Copy points (will be reordered in-place during build)
        self.points = points.copy().astype(np.float32)
        self.indices = np.arange(N, dtype=np.int32)

        self.node_count = _kdtree_build(
            self.points, self.indices, leaf_capacity,
            self.split_axes, self.split_values,
            self.left_children, self.right_children,
            self.point_start, self.point_end)

        # Trim arrays to actual size
        self.split_axes = self.split_axes[:self.node_count]
        self.split_values = self.split_values[:self.node_count]
        self.left_children = self.left_children[:self.node_count]
        self.right_children = self.right_children[:self.node_count]
        self.point_start = self.point_start[:self.node_count]
        self.point_end = self.point_end[:self.node_count]

    def knn(self, query, k):
        dists, idxs = _kdtree_knn_query(
            query.astype(np.float32), k,
            self.points, self.indices,
            self.split_axes, self.split_values,
            self.left_children, self.right_children,
            self.point_start, self.point_end)
        return np.sqrt(dists), idxs

    def knn_batch(self, query_points, k):
        dists, idxs = _kdtree_knn_batch(
            query_points.astype(np.float32), k,
            self.points, self.indices,
            self.split_axes, self.split_values,
            self.left_children, self.right_children,
            self.point_start, self.point_end)
        return np.sqrt(dists), idxs

    def radius_query(self, query, radius):
        result, count = _kdtree_radius_query(
            query.astype(np.float32), radius,
            self.points, self.indices,
            self.split_axes, self.split_values,
            self.left_children, self.right_children,
            self.point_start, self.point_end)
        return result

    def radius_batch(self, query_points, radius):
        counts = _kdtree_radius_batch(
            query_points.astype(np.float32), radius,
            self.points, self.indices,
            self.split_axes, self.split_values,
            self.left_children, self.right_children,
            self.point_start, self.point_end)
        return counts

    def memory_bytes(self):
        """Approximate memory footprint of the tree structure."""
        total = self.points.nbytes + self.indices.nbytes
        for arr in [self.split_axes, self.split_values, self.left_children,
                    self.right_children, self.point_start, self.point_end]:
            total += arr.nbytes
        return total

    def structural_stats(self):
        """Return dict of structural metrics."""
        n_leaf = int(np.sum(self.split_axes == -1))
        n_internal = self.node_count - n_leaf
        # Compute max depth via iterative traversal
        max_d = 0
        stack = [(0, 0)]
        while stack:
            ni, d = stack.pop()
            if d > max_d:
                max_d = d
            if self.split_axes[ni] != -1:
                lc = self.left_children[ni]
                rc = self.right_children[ni]
                if lc != -1:
                    stack.append((lc, d + 1))
                if rc != -1:
                    stack.append((rc, d + 1))
        return {
            'node_count': self.node_count,
            'leaf_count': n_leaf,
            'internal_count': n_internal,
            'max_depth': max_d,
        }
