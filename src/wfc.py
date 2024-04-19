import copy
import numpy as np


def all_neighbors(L, idx, depth=1, pad_value="X"):
    neighbors = copy.deepcopy(L)
    np.pad(neighbors, pad_width=depth, constant_values=pad_value)

    for i in range(len(L.shape)):
        start = idx[i]
        end = idx[i] + 2 * depth + 1
        neighbors = neighbors[start:end]

    
    # TODO remove self from neighbors
    # double check start/end idx

    K, I, J = idx
    neighbors = []

    for k in range(K - 1, K + 2):
        for i in range(I - 1, I + 2):
            for j in range(J - 1, J + 2):
                if k != K or i != I or j != J:
                    neighbors.append(L[k, i, j])
    return neighbors


def orthogonal_neighbors(L, idx):
    K, I, J = idx
    neighbors = []

    for k in range(K - 1, K + 2):
        for i in range(I - 1, I + 2):
            for j in range(J - 1, J + 2):
                if k != K or i != I or j != J:
                    neighbors.append(L[k, i, j])
    return neighbors


def train(input_levels, tile_shape, neighbor_fn=all_neighbors):
    tile_counts = {}
    full_context_counts = {}
    for L in levels:
        L = pad_3D(L, tile_shape)
        K, I, J = L.shape
        for k in range(1, K - 1):
            for i in range(1, I - 1):
                for j in range(1, J - 1):
                    t = L[k, i, j]
                    if t not in tile_counts:
                        tile_counts[t] = 0
                    tile_counts[t] += 1
                    fc = context_key(neighbor_fn(L, (k, i, j)))
                    if fc not in full_context_counts:
                        full_context_counts[fc] = {}
                    if t not in full_context_counts[fc]:
                        full_context_counts[fc][t] = 0
                    full_context_counts[fc][t] += 1

    tile_distribution = {}
    full_context_distribution = {}

    tile_distribution = normalize(tile_counts)

    for c, counts in full_context_counts.items():
        full_context_distribution[c] = normalize(counts)

    return (
        tile_distribution,
        tile_counts,
        full_context_distribution,
        full_context_counts,
    )
