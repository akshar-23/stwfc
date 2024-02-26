from train_levels import TRAIN_LEVELS
from train import (
    str_to_lvl,
    pad,
    get_pad_value,
    encode_tiles,
    train,
    all_neighbors,
    normalize,
    context_key,
    decode_tiles,
)
import numpy as np
from random import random, randrange
from math import exp, log
import matplotlib.pyplot as plt
import re


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


if __name__ == "__main__":
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
