from src.train_levels import TRAIN_LEVELS
from src.train import (
    board_to_str,
    pad,
    unpad_3D,
    encode_tiles_3D,
    train_3D,
    flip_soln,
    all_neighbors_3D,
    orthogonal_neighbors_3D,
    horizontal_neighbors_3D,
    normalize,
    decode_tiles_3D,
)
import numpy as np
from random import random, randrange
from math import exp, log
import matplotlib.pyplot as plt
import re
import json
from pathlib import Path
import time
import copy
import multiprocessing


def sample(distribution):
    r = random()
    sum_p = 0
    for x, px in distribution.items():
        sum_p += px
        if r < sum_p:
            return x


def propagate_overlapping_3D(Lpe, tile_shape, options=False):
    # Propagate to neighbors
    Lp = decode_tiles_3D(Lpe, options)
    L = unpad_3D(Lp, tile_shape)
    Lp = pad(L, tile_shape)
    Lpe = encode_tiles_3D(Lp, tile_shape)
    return Lpe


def resolve_options(candidates):
    # Find an options tile that represents all the candidates.
    collapsed = candidates[0]
    options = np.full(candidates[0].shape, "     ")
    for candidate in candidates:
        for idx in np.ndindex(candidate.shape):
            if collapsed[idx] != candidate[idx]:
                collapsed[idx] = "."
            curr_options = list(options[idx])
            curr_options.append(candidate[idx])
            options[idx] = "".join(sorted(list(set(curr_options)))).strip()
    return collapsed, options


def compatible(options, candidate):
    for idx in np.ndindex(options.shape):
        if candidate[idx] not in options[idx]:
            return False
    return True


def test_compatibility():
    options = np.array([[["AB", "AB"], ["CD", "CD"]], [["AB", "AB"], ["XY", "XY"]]])

    incompatible = np.array([[["A", "B"], ["C", "D"]], [["E", "F"], ["G", "H"]]])
    candidates = np.array([
        [[["A", "B"], ["C", "D"]], [["A", "B"], ["X", "Y"]]],
        [[["A", "A"], ["D", "C"]], [["B", "B"], ["X", "X"]]],
    ])

    print("Test compatible")
    assert not compatible(
        options, incompatible
    ), f"{incompatible} found compatible with {options}"

    for candidate in candidates:
        assert compatible(
            options, candidate
        ), f"{candidate} found compatible with {options}"

    print("Test resolve_options")
    expected_options = np.array(
        [[["A", "AB"], ["CD", "CD"]], [["AB", "B"], ["X", "XY"]]]
    )
    expected_tile = np.array([[["A", "."], [".", "."]], [[".", "B"], ["X", "."]]])
    actual_tile, actual_options = resolve_options(candidates)
    assert np.array_equal(
        expected_tile, actual_tile
    ), f"{expected_tile} != {actual_tile}"
    assert np.array_equal(
        expected_options, actual_options
    ), f"{expected_options} != {actual_options}"


def propagate_neighbors_3D(
    Lpe,
    L_options_pe,
    tile_shape,
    full_context_counts,
    fuzz=False,
    neighborhood_fn=all_neighbors_3D,
):
    K, I, J, Kt, It, Jt = Lpe.shape
    min_distribution = {}
    min_idx = (0, 0, 0)
    min_len = 1000
    did_propagate = False
    for k in range(1, K - 1):
        for i in range(1, I - 1):
            for j in range(1, J - 1):
                idx = (k, i, j)
                currT = Lpe[idx]
                if "." in currT:
                    currT_options = L_options_pe[idx]
                    counts = {}
                    # Current neighborhood
                    neighborhood_options = neighborhood_fn(L_options_pe, idx)
                    for context, tile_counts in full_context_counts.items():
                        for candidate, count in tile_counts.items():
                            # If the candidate tile matches the current tile
                            if compatible(currT_options, np.array(candidate)):
                                if candidate not in counts and fuzz:
                                    # Candidate is compatible but may not be seen in training.
                                    counts[candidate] = 1
                                # If the candidate neighborhood matches the current neighborhood
                                if (
                                    compatible(neighborhood_options, np.array(context))
                                    is not None
                                ):
                                    if candidate not in counts:
                                        counts[candidate] = 0
                                    counts[candidate] += count
                    if len(counts) < min_len:
                        min_distribution = normalize(counts)
                        min_idx = idx
                    if len(counts) > 0:
                        resolvedT, resolvedT_options = resolve_options(
                            np.array(list(counts.keys()))
                        )
                        if not np.array_equal(resolvedT_options, currT_options):
                            L_options_pe[idx] = resolvedT_options
                            # Lpe[idx] = resolvedT
                            # Lpe = propagate_overlapping_3D(Lpe, tile_shape)
                            L_options_pe = propagate_overlapping_3D(
                                L_options_pe, tile_shape, True
                            )
                            did_propagate = True
    return Lpe, L_options_pe, did_propagate, min_idx, min_distribution


def seed_t0(L, seed_vals):
    for s in seed_vals:
        sk = 0
        si = randrange(1, I - 1)
        sj = randrange(1, J - 1)
        sidx = (sk, si, sj)
        L[sidx] = s
    # Bottom row is always "W"
    K, I, J = L.shape
    L[0][I - 1] = ["W"] * J
    return L


def wfc_3D(
    full_context_counts,
    level_shape,
    tile_shape,
    max_num_trials=100,
    seed=(seed_t0, ["P"]),
    fuzz=False,
    single_char=True,
    neighborhood_fn=all_neighbors_3D,
):
    K, I, J = level_shape
    # Initialize with randomly placed P in timestep 0.
    L = np.full(level_shape, ".")
    num_trials = 0
    best = L
    best_contra_count = 1000
    best_num_samples = 0
    while "." in L and num_trials < max_num_trials:
        num_samples = 0
        L = np.full(level_shape, ".")
        L_options = np.full(level_shape, "".join(sorted(list("PDB_W"))).strip())
        seed_fn, seed_vals = seed
        L = seed_fn(L, seed_vals)

        Lp = pad(L, tile_shape)
        Lpe = encode_tiles_3D(Lp, tile_shape)

        L_options_p = pad(L_options, tile_shape)
        L_options_pe = encode_tiles_3D(L_options_p, tile_shape)

        finished = False
        while not finished:
            did_propagate = True
            while did_propagate:
                (
                    Lpe,
                    L_options_pe,
                    did_propagate,
                    most_constrained_idx,
                    most_constrained_distro,
                ) = propagate_neighbors_3D(
                    Lpe,
                    L_options_pe,
                    tile_shape,
                    full_context_counts,
                    fuzz,
                    neighborhood_fn,
                )

            if most_constrained_distro == {}:
                finished = True
            else:
                tile = np.array(sample(most_constrained_distro))
                if not single_char:
                    # Set the entire tile at once
                    Lpe[most_constrained_idx] = tile
                    L_options_pe[most_constrained_idx] = tile
                else:
                    # set a single tile
                    set_tile = Lpe[most_constrained_idx]
                    for idx in np.ndindex(set_tile.shape):
                        if set_tile[idx] == ".":
                            set_tile[idx] = tile[idx]
                            L_options_pe[most_constrained_idx][idx] = tile[idx]
                            break
                    Lpe[most_constrained_idx] = set_tile

                Lpe = propagate_overlapping_3D(Lpe, tile_shape)
                L_options_pe = propagate_overlapping_3D(L_options_pe, tile_shape, True)
                num_samples += 1

        Lp = decode_tiles_3D(Lpe)
        L = unpad_3D(Lp, tile_shape)
        count_L = np.count_nonzero(L == ".")
        if count_L < best_contra_count:
            best = copy.deepcopy(L)
            best_contra_count = count_L
            best_num_samples = num_samples
        num_trials += 1

    print(best)
    return best, num_trials, best_num_samples


def train_generate_3D(
    experiment_folder,
    solutions_folder,
    level_range,
    gen_shape=(5, 5, 5),
    tile_shape=(1, 1, 1),
    num_levels=1,
    max_num_trials=100,
    seed=(seed_t0, ["P"]),
    fuzz=False,
    single_char=True,
    neighborhood_fn=all_neighbors_3D,
):
    levels = []
    for i in level_range:
        soln_folder = f"src/{solutions_folder}/blockdude_{i}"
        soln_file = Path(f"{soln_folder}/solution.json")
        with open(soln_file) as f:
            level = np.array(json.load(f))
            # Remove LV# row from level
            if level[0][0][1] == ".":
                level = level[:, 1:, :]
            # Replace B' with B
            level[level == "B'"] = "B"
            # Remove the last timestep to ensure that P is always present
            level = level[:-1]
            fdisplay = "solutions_display"
            Path(fdisplay).mkdir(parents=True, exist_ok=True)
            fsolndisplay = f"{fdisplay}/level{i}"
            Path(fsolndisplay).mkdir(parents=True, exist_ok=True)
            for b, board in enumerate(level):
                b_str = str(b)
                while len(b_str) < 3:
                    b_str = "0" + b_str
                with open(
                    f"{fsolndisplay}/step{b_str}.lvl", "w", encoding="utf-8"
                ) as f:
                    f.write(board_to_str(board))

        levels.append(level)
        levels.append(flip_soln(level))
    (tile_counts, neighborhood_counts) = train_3D(levels, tile_shape, neighborhood_fn)

    Path(experiment_folder).mkdir(parents=True, exist_ok=True)
    num_complete = 0
    total_start = time.time()
    aggregate_trials = 0
    aggregate_samples = 0
    for i in range(num_levels):
        start_time = time.time()
        level, num_trials, num_samples = wfc_3D(
            neighborhood_counts,
            gen_shape,
            tile_shape,
            max_num_trials,
            seed,
            fuzz,
            single_char,
            neighborhood_fn,
        )
        aggregate_trials += num_trials
        aggregate_samples += num_samples
        end_time = time.time()
        elapsed = end_time - start_time
        # print(f"found level {i} in {num_trials} trials")
        # print(level)
        level_folder = f"{experiment_folder}/{i}"
        Path(level_folder).mkdir(parents=True, exist_ok=True)
        with open(f"{level_folder}/raw.json", "w", encoding="utf-8") as f:
            json.dump(level.tolist(), f, ensure_ascii=False, indent=4)
        with open(f"{level_folder}/stats.txt", "w", encoding="utf-8") as f:
            f.write(f"Num trials: {str(num_trials)}/{str(max_num_trials)}")
            f.write(f"\nNum samples: {str(num_samples)}")
            f.write(f"\nElapsed: {str(elapsed)}")
        display_folder = f"{level_folder}/display"
        Path(display_folder).mkdir(parents=True, exist_ok=True)
        for b, board in enumerate(level):
            b_str = str(b)
            while len(b_str) < 3:
                b_str = "0" + b_str
            with open(f"{display_folder}/{b_str}.lvl", "w", encoding="utf-8") as f:
                f.write(board_to_str(board))
        if "." not in level:
            num_complete += 1
    total_end = time.time()
    total_elapsed = total_end - total_start
    with open(f"{experiment_folder}/stats.txt", "w", encoding="utf-8") as f:
        f.write("3D trials")
        f.write(f"\nretry x {max_num_trials}")
        f.write(f"\nseeded with {seed}")
        f.write(f"\ntrained on {level_range}")
        f.write(f"\ngeneration shape {gen_shape}")
        f.write(f"\ntile shape {tile_shape}")
        f.write(f"\nfuzz? {str(fuzz)}")
        f.write(f"\nsingle char? {str(single_char)}")
        f.write(f"\ncomplete levels {str(num_complete)}/{num_levels}")
        f.write(f"\nelapsed time {str(total_elapsed)}")
        f.write(f"\navg num trials: {str(aggregate_trials/num_levels)}")
        f.write(f"\navg num samples: {str(aggregate_samples/num_levels)}")
        f.write(f"\nneighborhood: {str(neighborhood_fn.__name__)}")
    return


def seed_tn(L, dummy):
    dk = L.shape[0] - 1
    di = randrange(1, I - 1)
    dj = randrange(2, J - 2)
    didx = (dk, di, dj)
    L[didx] = "D"

    r = random()
    pj = dj + 1
    if r < 0.5:
        pj = dj - 1
    L[dk, di, pj] = "P"

    # Bottom row is always "W"
    K, I, J = L.shape()
    L[0][I - 1] = ["W"] * J
    return L


if __name__ == "__main__":
    level_range = ["A"]  # ["A", 1, 2, 3, 4]
    gen_shape = (4, 4, 8)  # (5, 7, 10)
    tile_shapes = [(2, 2, 2)]  # [(2, 3, 2), (2, 2, 2)]
    num_levels = 1  # 10
    max_num_trials = 1  # 10
    seeds = [
        (seed_t0, [])
    ]  # [(seed_t0, ["P"]), (seed_t0, []), (seed_t0, ["D"]), (seed_tn, [])]
    fuzzs = [False]
    single_chars = [True]
    neighborhood_fns = [
        orthogonal_neighbors_3D
    ]  # [orthogonal_neighbors_3D, horizontal_neighbors_3D]

    arg_combos = []

    for tile_shape in tile_shapes:
        for seed in seeds:
            for fuzz in fuzzs:
                for single_char in single_chars:
                    for neighborhood_fn in neighborhood_fns:
                        args = (
                            f"experiments/3D/tile{str(tile_shape)}:seed{str(seed)}:fuzz{str(fuzz)}:single_char{str(single_char)}:{str(neighborhood_fn.__name__)}:{time.time()}",
                            "solutions",
                            level_range,
                            gen_shape,
                            tile_shape,
                            num_levels,
                            max_num_trials,
                            seed,
                            fuzz,
                            single_char,
                            neighborhood_fn,
                        )
                        arg_combos.append(args)

    with multiprocessing.Pool() as pool:
        pool.starmap(train_generate_3D, arg_combos)
