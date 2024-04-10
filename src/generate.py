from src.train_levels import TRAIN_LEVELS
from src.train import (
    str_to_lvl,
    pad,
    pad_3D,
    unpad_3D,
    get_pad_value,
    get_pad_value_3D,
    encode_tiles,
    encode_tiles_3D,
    train,
    train_3D_naive,
    flip_soln,
    all_neighbors,
    all_neighbors_3D,
    normalize,
    context_key,
    decode_tiles,
    decode_tiles_3D,
)
import numpy as np
from random import random, randrange
from math import exp, log
import matplotlib.pyplot as plt
import re
import json
from pathlib import Path


def sample(distribution):
    r = random()
    sum_p = 0
    for x, px in distribution.items():
        sum_p += px
        if r < sum_p:
            return x


def log_likelihood(context_distribution, L):
    I, J = L.shape
    P = 0
    for i in range(1, I - 1):
        for j in range(1, J - 1):
            N = context_key(all_neighbors(L, (i, j)))
            t = L[i, j]
            if N not in context_distribution or t not in context_distribution[N]:
                P += log(0.0001)
            else:
                P += log(
                    context_distribution[context_key(all_neighbors(L, (i, j)))][L[i, j]]
                )
    return P


def swap_likelihood(context_distribution, L, P0, P1):
    t0 = L[P0]
    t1 = L[P1]
    N0 = context_key(all_neighbors(L, P0))
    N1 = context_key(all_neighbors(L, P1))

    P = 0
    if N0 not in context_distribution or t0 not in context_distribution[N0]:
        P += log(0.001)
    else:
        P += log(context_distribution[N0][t0])
    if N1 not in context_distribution or t1 not in context_distribution[N1]:
        P += log(0.001)
    else:
        P += log(context_distribution[N1][t1])
    return P


def swap(A, p0, p1):
    v0 = A[p0]
    A[p0] = A[p1]
    A[p1] = v0
    return A


def random_map(tile_distribution, size, tile_size):
    L = pad(np.full(size, get_pad_value(tile_size, "X")), 1, tile_size)
    I, J = L.shape
    for i in range(1, I - 1):
        for j in range(1, J - 1):
            L[i, j] = sample(tile_distribution)
    return L


def get_distribution(unfinished_level, full_context_counts, tile_size):
    I, J = unfinished_level.shape
    distribution = {}
    for i in range(1, I - 1):
        for j in range(1, J - 1):
            if unfinished_level[i, j] == get_pad_value(tile_size, "."):
                counts = {}
                n_key = context_key(all_neighbors(unfinished_level, (i, j)))
                for context, tile_counts in full_context_counts.items():
                    if re.search(n_key, context) is not None:
                        for t, count in tile_counts.items():
                            if t not in counts:
                                counts[t] = 0
                            counts[t] += count
                distribution[(i, j)] = normalize(counts)
    return distribution


def wfc(full_context_counts, size, tile_size):
    L = pad(np.full(size, get_pad_value(tile_size, ".")), 1, tile_size)

    finished = False
    while not finished:
        distribution = get_distribution(L, full_context_counts, tile_size)
        most_constrained = (0, 0)
        most_constrained_len = 1000
        most_constrained_distro = {}
        for index, distro in distribution.items():
            distro_len = len(distro)
            if distro_len < most_constrained_len:
                most_constrained = index
                most_constrained_len = distro_len
                most_constrained_distro = distro
        if most_constrained_distro == {}:
            finished = True
        else:
            L[most_constrained] = sample(most_constrained_distro)

    # I, J = L.shape
    # possible_U = {}
    # possible_D = {}
    # for i in range(1, I - 1):
    #     for j in range(1, J - 1):
    #         neighbors = context_key(all_neighbors(L, (i, j)))
    #         if neighbors in full_context_distribution:
    #             if "U" in full_context_distribution[neighbors]:
    #                 possible_U[(i, j)] = full_context_distribution[neighbors]["U"]
    #             if "D" in full_context_distribution[neighbors]:
    #                 possible_D[(i, j)] = full_context_distribution[neighbors]["D"]

    # U_distribution = normalize(possible_U)
    # D_distribution = normalize(possible_D)

    # posU = sample(U_distribution)
    # posD = sample(D_distribution)

    # if posU is not None:
    #     L[posU] = "U"
    # if posD is not None:
    #     L[posD] = "D"

    return L


def get_conditional_tile_distribution(tile_counts, regex):
    counts = {}
    for tile, count in tile_counts.items():
        if re.search(regex, tile) is not None:
            counts[tile] = count
    return normalize(counts)


def propagate(Lep, tile_shape):
    # Propagate to neighbors
    Le = unpad_3D(Lep)
    L = decode_tiles_3D(Le, tile_shape)
    Le = encode_tiles_3D(L, tile_shape)
    Lep = pad_3D(Le, tile_shape)
    return Lep


def get_most_constrained_3D(L, full_context_counts):
    K, I, J = L.shape
    distribution = {}
    idx = (0, 0, 0)
    min_len = 1000
    currT = ""
    for k in range(1, K - 1):
        for i in range(1, I - 1):
            for j in range(1, J - 1):
                currT = L[k, i, j]
                if "." in currT:
                    counts = {}
                    # Current neighborhood
                    n_key = context_key(all_neighbors_3D(L, (k, i, j)))
                    for context, tile_counts in full_context_counts.items():
                        # If the context matches the neighborhood
                        if re.search(n_key, context) is not None:
                            for candidate, count in tile_counts.items():
                                # If the candidate tile matches the current tile
                                if re.search(currT, candidate):
                                    if candidate not in counts:
                                        counts[candidate] = 0
                                    counts[candidate] += count
                    if len(counts) > 0 and len(counts) < min_len:
                        distribution = normalize(counts)
                        idx = (k, i, j)
    return idx, distribution


def wfc_3D_naive(full_context_counts, level_shape, tile_shape):
    K, I, J = level_shape
    # Initialize with randomly placed P in timestep 0.
    L = np.full(level_shape, ".")
    Pk = 0
    Pi = randrange(1, I - 1)
    Pj = randrange(1, J - 1)
    Pidx = (Pk, Pi, Pj)
    L[Pidx] = "P"
    Le = encode_tiles_3D(L, tile_shape)
    Lep = pad_3D(Le, tile_shape)

    finished = False
    while not finished:
        most_constrained_idx, most_constrained_distro = get_most_constrained_3D(
            Lep, full_context_counts
        )

        if most_constrained_distro == {}:
            finished = True
        else:
            tile = sample(most_constrained_distro)
            Lep[most_constrained_idx] = tile
            Lep = propagate(Lep, tile_shape)

    # Decode
    Le = unpad_3D(Lep)
    L = decode_tiles_3D(Le, tile_shape)
    return L


def wfc_3D(transitions, path_length, tile_size):
    # Move forward from P and backwards from D
    # How to ensure the twain will meet?
    L = pad_3D(np.full((100, 100), get_pad_value(tile_size, ".")), 1, tile_size)
    dx = randrange(100)
    dy = randrange(100)
    # idx = (dx, dy)
    # L(dx, dy) = // random tile from start distribution containing "P"
    #
    # for t in range(path_length - 1):
    #   ...
    #   idx = next path step
    # generate tile with "D" at the end of the path.
    while get_pad_value(tile_size, ".") in L:
        distribution = get_distribution(L, full_context_counts, tile_size)
        most_constrained = (0, 0)
        most_constrained_len = 1000
        most_constrained_distro = {}
        for index, distro in distribution.items():
            distro_len = len(distro)
            if distro_len < most_constrained_len:
                most_constrained = index
                most_constrained_len = distro_len
                most_constrained_distro = distro
        if most_constrained_distro == {}:
            break
        L[most_constrained] = sample(most_constrained_distro)
    return L


def mrf(tile_distribution, context_distribution, size, tile_size, generations):
    L = random_map(tile_distribution, size, tile_size)
    I, J = L.shape

    P = log_likelihood(context_distribution, L)
    Ps = [P]
    for _ in range(generations):
        selected = np.pad(np.zeros((I - 2, J - 2)), pad_width=1, constant_values=1)
        unselected = I * J
        while unselected >= 2:
            i0 = randrange(1, I - 1)
            j0 = randrange(1, J - 1)
            i1 = randrange(1, I - 1)
            j1 = randrange(1, J - 1)
            selected[i0, j0] = 1
            selected[i1, j1] = 1
            Pswap = swap_likelihood(context_distribution, L, (i0, j0), (i1, j1))
            Ltest = swap(L, (i0, j0), (i1, j1))
            Ptest = swap_likelihood(context_distribution, Ltest, (i0, j0), (i1, j1))
            p_accept = min(1, exp(Ptest - Pswap))
            r = random()
            if r < p_accept:
                L = Ltest
            unselected = np.count_nonzero(selected == 0)
        P = log_likelihood(context_distribution, L)
        Ps.append(P)
    return L, Ps


def train_generate():
    tile_size = 2

    # TRAIN_LEVELS = [TRAIN_LEVELS[0]]
    overlapping_levels = []
    non_overlapping_levels = []
    for level in TRAIN_LEVELS:
        unrolled_level = str_to_lvl(level)
        encoded_overlapping_level = encode_tiles(unrolled_level, tile_size)
        encoded_non_overlapping_level = encode_tiles(unrolled_level, tile_size, False)
        overlapping_levels.append(encoded_overlapping_level)
        non_overlapping_levels.append(encoded_non_overlapping_level)

    print("Overlap training (WFC)")
    print(overlapping_levels)

    print("Non overlap training (MRF)")
    print(non_overlapping_levels)
    (
        overlapping_tile_distribution,
        overlapping_full_context_distribution,
        overlapping_full_context_counts,
    ) = train(overlapping_levels, tile_size)

    (
        non_overlapping_tile_distribution,
        non_overlapping_full_context_distribution,
        non_overlapping_full_context_counts,
    ) = train(non_overlapping_levels, tile_size)

    # print(overlapping_full_context_counts)
    # print(non_overlapping_full_context_counts)
    print("MRF")
    encoded_level, likelihoods = mrf(
        non_overlapping_tile_distribution,
        non_overlapping_full_context_distribution,
        (2, 6),
        tile_size,
        100,
    )
    print(encoded_level)
    level = decode_tiles(encoded_level, tile_size, False)
    print(level)
    plt.clf()
    plt.plot(likelihoods)
    plt.savefig("likelihoods.png")

    print("WFC")
    encoded_level = wfc(overlapping_full_context_counts, (10, 10), tile_size)
    print(encoded_level)
    level = decode_tiles(encoded_level, tile_size)
    print(level)


def train_generate_3D(
    experiment_folder,
    solutions_folder,
    level_range,
    gen_shape=(5, 5, 5),
    tile_shape=1,
    num_levels=1,
):
    overlapping_levels = []
    for i in level_range:
        soln_file = Path(f"src/{solutions_folder}/blockdude_{i}_solution.json")
        with open(soln_file) as f:
            level = np.array(json.load(f))
            # Remove LV# row from level
            if level[0][0][1] == ".":
                level = level[:, 1:, :]
            # Replace B' with B
            level[level == "B'"] = "B"

        encoded_overlapping_level = encode_tiles_3D(level, tile_shape)
        overlapping_levels.append(encoded_overlapping_level)

        flipped_level = flip_soln(level)
        encoded_flipped_level = encode_tiles_3D(flipped_level, tile_shape)
        overlapping_levels.append(encoded_flipped_level)
    (
        overlapping_tile_distribution,
        overlapping_tile_counts,
        overlapping_full_context_distribution,
        overlapping_full_context_counts,
    ) = train_3D_naive(overlapping_levels, tile_shape)

    Path(experiment_folder).mkdir(parents=True, exist_ok=True)
    for i in range(num_levels):
        level = wfc_3D_naive(
            overlapping_full_context_counts,
            gen_shape,
            tile_shape,
        )
        print(level)
        with open(f"{experiment_folder}/{i}.txt", "w", encoding="utf-8") as f:
            f.write(str(level))
    return


if __name__ == "__main__":
    level_range = ["A", 1, 2, 3, 4]
    gen_shape = (10, 6, 10)
    tile_shape = (2, 3, 2)
    num_levels = 1
    train_generate_3D(
        f"experiments_naive_P_seededr{level_range}g{gen_shape}t{tile_shape}",
        "solutions",
        level_range,
        gen_shape,
        tile_shape,
        num_levels,
    )

    # tile_size = 2
    # L = np.array([
    #     [["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"]],
    #     [["J", "K", "L"], ["M", "N", "O"], ["P", "Q", "R"]],
    #     [["S", "T", "U"], ["V", "W", "X"], ["Y", "Z", "0"]],
    # ])
    # print(L)

    # print("Encoded: ")
    # eL = encode_tiles_3D(L, 2)
    # print(eL)

    # print("Padded:")
    # peL = pad_3D(eL, tile_size)
    # print(peL)

    # print("Unpadded:")
    # eL = unpad_3D(peL)
    # print(eL)

    # print("Decoded: ")
    # L = decode_tiles_3D(eL, 2)
    # print(L)
