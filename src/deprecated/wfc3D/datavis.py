from src.train import (
    flip_soln,
    train_3D,
    all_neighbors_3D,
    orthogonal_neighbors_3D,
    horizontal_neighbors_3D,
)
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path
import json
import os
import re
import pandas as pd


def calculate_completion_pct(level):
    total = level.size
    complete = 0
    for tile in np.nditer(level):
        if tile != ".":
            complete += 1
    return complete / total


def plot_context(tile_counts, neighborhood_counts, dir):
    N_frequency = {}
    T_frequency = {}
    T_O_frequency = {}

    T_N_counts = {}
    for N_T_counts in neighborhood_counts.values():
        # How frequently does this neighborhood occur?
        N_count = sum(N_T_counts.values())

        # How many other neighborhoods have been seen this many times?
        if N_count not in N_frequency:
            N_frequency[N_count] = 0
        N_frequency[N_count] += 1

        for tile in N_T_counts.keys():
            # How many neighborhoods has this tile been seen in?
            if tile not in T_N_counts:
                T_N_counts[tile] = 0
            T_N_counts[tile] += 1

    for T_N_count in T_N_counts.values():
        # How many other tiles have been seen in this many neighborhoods?
        if T_N_count not in T_O_frequency:
            T_O_frequency[T_N_count] = 0
        T_O_frequency[T_N_count] += 1

    for tile_count in tile_counts.values():
        # How many other tiles have been seen this frequently?
        if tile_count not in T_frequency:
            T_frequency[tile_count] = 0
        T_frequency[tile_count] += 1

    plt.clf()
    # number of unique counts of neighborhoods
    unique_N_counts = list(N_frequency.keys())
    # number of neighborhoods that have been seen that many times
    N_per_unique = list(N_frequency.values())
    plt.scatter(unique_N_counts, N_per_unique)
    plt.xlabel("Frequency of Neighborhood")
    plt.ylabel("Count of Neighborhoods with Frequency")
    plt.plot(unique_N_counts, p(unique_N_counts))
    plt.xscale("log")
    plt.savefig(f"{dir}/neighborhood_count_frequency")

    plt.clf()
    # number of unique counts of options
    unique_TO_counts = list(T_O_frequency.keys())
    # number of tiles that have that many options
    T_per_unique = list(T_O_frequency.values())
    plt.scatter(unique_TO_counts, T_per_unique)
    plt.xlabel("Number of Options per Tile")
    plt.ylabel("Count of Tiles with Number")
    plt.plot(unique_TO_counts, p(unique_TO_counts))
    plt.xscale("log")
    plt.savefig(f"{dir}/option_count_frequency")

    plt.clf()
    # number of unique counts of tiles
    unique_T_counts = list(T_frequency.keys())
    # number of tiles that have been seen that many times
    T_per_unique = list(T_frequency.values())
    plt.scatter(unique_T_counts, T_per_unique)
    plt.xlabel("Frequency of Tile")
    plt.ylabel("Count of Tiles with Frequency")
    plt.plot(unique_T_counts, p(unique_T_counts))
    plt.xscale("log")
    plt.savefig(f"{dir}/tile_count_frequency")
    return


def plot_completion(completion_pcts, filename):
    plt.clf()

    pct_min = min(completion_pcts)
    pct_max = max(completion_pcts)
    pct_avg = sum(completion_pcts) / len(completion_pcts)

    file = open(filename, "w")
    file.write("\nMin completion: " + str(pct_min))
    file.write("\nMax completion: " + str(pct_max))
    file.write("\nAvg completion: " + str(pct_avg))

    return


def plot_likelihoods(likelihoods, filename):
    plt.clf()
    x = list(range(0, len(likelihoods[0])))

    lvl_avg = []
    lvl_max = []
    lvl_min = []

    for gen in x:
        likelihood = [likelihoods[i][gen] for i in range(len(likelihoods))]
        lvl_avg.append(sum(likelihood) / len(likelihood))
        lvl_max.append(max(likelihood))
        lvl_min.append(min(likelihood))

    plt.plot(x, lvl_avg)
    plt.legend()
    plt.fill_between(x, lvl_min, lvl_max, color="blue", alpha=0.1)
    plt.savefig(filename)

    return


def save_level_tile_types(levels, filename):
    counts = {"D": [], "U": [], "#": [], ".": [], "B": [], " ": []}
    for level in levels:
        level_counts = {"D": 0, "U": 0, "#": 0, ".": 0, "B": 0, " ": 0}
        for tile in np.nditer(level):
            level_counts[str(tile)] += 1
        for tile, tile_count in level_counts.items():
            counts[tile].append(tile_count)
    file = open(filename, "w")
    for tile, tile_counts in counts.items():
        avg = 0
        if sum(tile_counts) > 0:
            avg = sum(tile_counts) / len(tile_counts)
        max = np.max(tile_counts)
        min = np.min(tile_counts)
        count, num_per_count = np.unique(tile_counts, return_counts=True)
        file.write("\n" + tile + ":\n")
        file.write("\nAVG: " + str(avg))
        file.write("\nMin: " + str(min))
        file.write("\nMax: " + str(max))
        file.write("\nCounts: \n")
        for i in range(len(count)):
            file.write("\n" + str(count[i]) + ": " + str(num_per_count[i]))


def visualize_completion(onebyfile, twobyfile):
    X = []

    full_dataset_size = 0
    for level in TRAIN_LEVELS:
        unrolled_level = str_to_lvl(level)
        (I, J) = unrolled_level.shape
        full_dataset_size += I * J
        X.append(I * J)

    X.append(full_dataset_size)

    onebyY = [
        0.667,
        0.818,
        0.798,
        0.817,
        0.811,
        0.894,
        0.823,
        0.628,
        0.703,
        0.692,
        0.791,
        0.764,
        0.986,
    ]

    twobyY = [
        0.320,
        0.395,
        0.256,
        0.451,
        0.168,
        0.343,
        0.158,
        0.179,
        0.171,
        0.161,
        0.095,
        0.124,
        0.325,
    ]

    plt.clf()
    plt.scatter(X, onebyY, label="1x1")
    plt.scatter(X, twobyY, label="2x2 overlapping")
    plt.legend()
    plt.xlabel("Dataset Size")
    plt.ylabel("Average Level Completion")
    plt.title("Level Completion by Dataset Size for 1x1 and 2x2 Tiles")
    plt.savefig(onebyfile)

    plt.clf()
    plt.scatter(X, twobyY)
    plt.savefig(twobyfile)


def visualize_training_data(dir, level_range, tile_shapes, neighborhood_fns):
    levels = []
    for i in level_range:
        soln_dir = f"{dir}/blockdude_{i}"
        soln_file = f"{soln_dir}/solution.json"
        level = []
        with open(soln_file) as f:
            level = np.array(json.load(f))
        # Remove LV# row from level
        if level[0][0][1] == ".":
            level = level[:, 1:, :]
        # Replace B' with B
        level[level == "B'"] = "B"
        # Remove the last timestep to ensure that P is always present
        level = level[:-1]
        levels.append(level)
        levels.append(flip_soln(level))

    for tile_shape in tile_shapes:
        for neighborhood_fn in neighborhood_fns:
            (tile_counts, neighborhood_counts) = train_3D(
                levels, tile_shape, neighborhood_fn
            )

            vis_dir = f"{dir}/{tile_shape}/{neighborhood_fn.__name__}"
            Path(vis_dir).mkdir(parents=True, exist_ok=True)
            plot_context(
                tile_counts,
                neighborhood_counts,
                vis_dir,
            )


def visualize_experiments(experiment_dir):
    all_level_stats = []
    for subexperiment_dir in os.listdir(experiment_dir):
        subexperiment_dir = f"{experiment_dir}/{subexperiment_dir}"
        if ".png" in subexperiment_dir:
            continue

        for level_dir in os.listdir(subexperiment_dir):
            if "." in level_dir:
                continue
            level_dir = f"{subexperiment_dir}/{level_dir}"
            stats = {}
            with open(f"{level_dir}/level_stats.json", "r") as f:
                stats = json.load(f)
            all_level_stats.append(stats)

    df = pd.DataFrame.from_records(all_level_stats)
    df = df.drop(df[df["fuzz"] == "True"].index)

    def avg_non_1(pcts):
        count = 0
        sum_pct = 0
        for pct in pcts:
            if pct < 1:
                count += 1
                sum_pct += pct
        if count == 0:
            return 1
        return sum_pct / count

    # completed_vs_static = df.groupby(
    #     ["tile_shape", "neighborhood", "seed"], as_index=False
    # )[["completed", "completed_pct", "static", "static_pct"]].agg({
    #     "completed": "mean",
    #     "static": "mean",
    #     "completed_pct": avg_non_1,
    #     "static_pct": avg_non_1,
    # })
    # completed_vs_static.plot()

    df.loc[df["neighborhood"] == "all_neighbors_3D", "neighborhood"] = "all"
    df.loc[df["neighborhood"] == "orthogonal_neighbors_3D", "neighborhood"] = "ortho"
    df.loc[df["neighborhood"] == "horizontal_neighbors_3D", "neighborhood"] = "horiz"
    df["incomplete"] = df["completed"]
    df.loc[df["completed"] == 0, "incomplete"] = 1
    df.loc[df["completed"] == 1, "incomplete"] = 0

    df.loc[df["seed"] == "seed_t0: ['P']", "seed"] = "t0(P)"
    df.loc[df["seed"] == "seed_t0: ['D']", "seed"] = "t0(D)"
    df.loc[df["seed"] == "seed_t0: []", "seed"] = "none"
    df.loc[df["seed"] == "seed_tn: []", "seed"] = "tn(PD)"

    def get_completed(x):
        d = {}
        d["completed"] = x["completed"].sum()
        d["static"] = x.loc[x["completed"] == 1, "static"].sum()
        d["incomplete"] = x["incomplete"].sum()

        return pd.Series(d, index=["completed", "static", "incomplete"])

    # plt.clf()
    # completed_vs_static = df.groupby(["tile_shape", "neighborhood"], as_index=True)[
    #     ["completed", "static", "incomplete"]
    # ].apply(get_completed)
    # completed_vs_static.plot(kind="barh")
    # plt.show()

    # plt.clf()
    # completed_vs_static = df.groupby(["seed"], as_index=True)[
    #     ["completed", "static", "incomplete"]
    # ].apply(get_completed)
    # completed_vs_static.plot(kind="barh")
    # plt.show()

    # plt.clf()
    # completed_vs_static = df.groupby(["neighborhood"], as_index=True)[
    #     ["completed", "static", "incomplete"]
    # ].apply(get_completed)
    # completed_vs_static.plot(kind="barh")
    # plt.show()

    # plt.clf()
    # completed_vs_static = df.groupby(["tile_shape"], as_index=True)[
    #     ["completed", "static", "incomplete"]
    # ].apply(get_completed)
    # completed_vs_static.plot(kind="barh")
    # plt.show()

    df = df.join(pd.DataFrame(df.pop("mechanics").values.tolist()))
    df["fall_1"] = df["fall_1"] ^ df["fall_2"]
    df["walk_1"] = df["walk_1"] ^ df["walk_2"]
    df["carry_1"] = df["carry_1"] ^ df["carry_2"]
    df["any"] = (
        df["fall_1"]
        | df["fall_2"]
        | df["walk_1"]
        | df["walk_2"]
        | df["carry_1"]
        | df["carry_2"]
        | df["door"]
        | df["jump"]
        | df["jump_with"]
        | df["block_take"]
        | df["block_drop"]
    )
    df["none"] = df["any"]
    df.loc[df["any"] == False, "none"] = True
    df.loc[df["any"] == True, "none"] = False

    mechanics = df[df["completed"] == 1][["any", "none"]].mean()
    plt.clf()
    mechanics.plot(kind="barh")
    plt.show()

    mechanics = df[df["completed"] == 1][[
        "fall_1",
        "fall_2",
        "walk_1",
        "walk_2",
        "carry_1",
        "carry_2",
        "door",
        "jump",
        "jump_with",
        "block_take",
        "block_drop",
    ]].mean()

    plt.clf()
    mechanics.plot(kind="barh")
    plt.show()

    return


def jsonify_0419(experiment_dir):
    for subexperiment_dir in os.listdir(experiment_dir):
        if ".png" in subexperiment_dir:
            continue
        subexperiment_dir = f"{experiment_dir}/{subexperiment_dir}"
        subexperiment_stats = {}
        with open(f"{subexperiment_dir}/stats.txt", "r") as f:
            stats_str = f.read()
        stats_lines = stats_str.splitlines()

        # Seed
        seed_line = stats_lines[2]
        seed_idx = seed_line.index("[")
        seed = seed_line[seed_idx:]
        subexperiment_stats["seed"] = f"seed_t0: {seed}"

        # Tile Shape
        tile_shape_line = stats_lines[5]
        tile_shape_idx = tile_shape_line.index("(")
        tile_shape_str = tile_shape_line[tile_shape_idx:]
        tile_shape_trimmed = re.sub("[(),]", "", tile_shape_str)
        tile_shape_split = tile_shape_trimmed.split()
        tile_shape = []
        for char in tile_shape_split:
            tile_shape.append(int(char))
        subexperiment_stats["tile_shape"] = str(tuple(tile_shape))

        # Fuzz
        fuzz_line = stats_lines[6]
        fuzz_idx = fuzz_line.index("?")
        fuzz = fuzz_line[fuzz_idx + 1 :].strip()
        subexperiment_stats["fuzz"] = fuzz

        # Single Char
        sc_line = stats_lines[7]
        sc_idx = sc_line.index("?")
        single_char = sc_line[sc_idx + 1 :].strip()
        subexperiment_stats["single_char"] = single_char

        # Neighborhood
        neighborhood_line = stats_lines[12]
        neighborhood_idx = neighborhood_line.index(":")
        neighborhood = neighborhood_line[neighborhood_idx + 1 :].strip()
        subexperiment_stats["neighborhood"] = neighborhood

        level_stats = subexperiment_stats
        for level_dir in os.listdir(subexperiment_dir):
            if "." in level_dir:
                continue
            level_dir = f"{subexperiment_dir}/{level_dir}"
            stats_str = ""
            stats_lines = []
            with open(f"{level_dir}/stats.txt", "r") as f:
                stats_str = f.read()
                stats_lines = stats_str.splitlines()

            # Trials
            trials_line = stats_lines[0]
            trials_idx = trials_line.index(":")
            trials_slash = trials_line[trials_idx + 1 :].strip()
            trials = trials_slash.split("/")[0]
            level_stats["trials"] = int(trials)

            # Samples
            samples_line = stats_lines[1]
            samples_idx = samples_line.index(":")
            samples = samples_line[samples_idx + 1 :].strip()
            level_stats["samples"] = int(samples)

            # Elapsed
            elapsed_line = stats_lines[2]
            elapsed_idx = elapsed_line.index(":")
            elapsed = elapsed_line[elapsed_idx + 1 :].strip()
            level_stats["elapsed"] = float(elapsed)

            # Completed
            completed, completed_pct, static, static_pct, sprite_counts, mechanics = (
                get_sprite_stats(f"{level_dir}/raw.json")
            )

            level_stats["completed"] = completed
            level_stats["completed_pct"] = completed_pct
            level_stats["static"] = static
            level_stats["static_pct"] = static_pct
            level_stats["sprite_counts"] = sprite_counts
            level_stats["mechanics"] = mechanics

            with open(f"{level_dir}/level_stats.json", "w", encoding="utf-8") as f:
                json.dump(level_stats, f)


def jsonify_0422(experiment_dir):
    for subexperiment_dir in os.listdir(experiment_dir):
        subexperiment_dir = f"{experiment_dir}/{subexperiment_dir}"

        open_idx = subexperiment_dir.index("(")
        close_idx = subexperiment_dir.index(")")
        tile_shape_str = subexperiment_dir[open_idx + 1 : close_idx]
        tile_shape_trimmed = re.sub("[,]", "", tile_shape_str)
        tile_shape_split = tile_shape_trimmed.split()
        tile_shape = []
        for char in tile_shape_split:
            tile_shape.append(int(char))

        for level_dir in os.listdir(subexperiment_dir):
            if "." in level_dir:
                continue
            level_dir = f"{subexperiment_dir}/{level_dir}"
            stats = {}
            with open(f"{level_dir}/stats.json", "r") as f:
                stats = json.load(f)

            stats["single_char"] = False
            stats["tile_shape"] = str(tuple(tile_shape))

            completed, completed_pct, static, static_pct, sprite_counts, mechanics = (
                get_sprite_stats(f"{level_dir}/raw.json")
            )

            stats["completed"] = completed
            stats["completed_pct"] = completed_pct
            stats["static"] = static
            stats["static_pct"] = static_pct
            stats["sprite_counts"] = sprite_counts
            stats["mechanics"] = mechanics

            with open(f"{level_dir}/level_stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f)


def get_sprite_stats(level_file):
    level = []
    with open(level_file) as f:
        level = np.array(json.load(f))

    T, Y, X = level.shape
    sprite_counts = {}
    contradiction_count = 0
    for sprite in np.nditer(level):
        if sprite == ".":
            contradiction_count += 1
            continue
        sprite = str(sprite)
        if sprite not in sprite_counts:
            sprite_counts[sprite] = 0
        sprite_counts[sprite] += 1

    completed = 0
    if contradiction_count == 0:
        completed = 1

    completed_pct = (level.size - contradiction_count) / level.size

    # count of timesteps with no change
    static_count = 0
    for t in range(1, T):
        if np.array_equal(level[t], level[t - 1]):
            static_count += 1
            continue

    total_transitions = T - 1
    static = 0
    if static_count == total_transitions:
        static = 1
    static_pct = static_count / total_transitions

    mechanics = {
        "walk_1": False,
        "walk_2": False,
        "fall_1": False,
        "fall_2": False,
        "carry_1": False,
        "carry_2": False,
        "jump_with": False,
        "block_take": False,
        "block_drop": False,
        "door": False,
        "jump": False,
    }

    # Flip so we only have to check mechanics in one direction.
    for L in [level, flip_soln(level)]:
        for t in range(1, T):
            P_indices = np.argwhere(L[t] == "P")
            for P_idx in P_indices:
                iy, x = P_idx

                has_B = False
                if iy > 0 and L[t, iy - 1, x] == "B":
                    has_B = True

                # if P has moved
                if L[t - 1, iy, x] != "P":

                    # Walk
                    # from solid ground
                    if iy < Y - 1 and L[t - 1, iy + 1, x - 1] in ["B", "W"]:
                        if (
                            x > 0
                            and L[t - 1, iy, x - 1] == "P"
                            and L[t, iy, x - 1] != "P"
                        ):
                            mechanics["walk_1"] = True
                            has_B_1 = False
                            if (
                                has_B
                                and L[t - 1, iy - 1, x - 1] == "B"
                                and L[t, iy - 1, x - 1] != "B"
                            ):
                                mechanics["carry_1"] = True
                                has_B_1 = True
                            if (
                                x > 1
                                and t > 1
                                and L[t - 2, iy, x - 2] == "P"
                                and L[t, iy, x - 2] != "P"
                                and L[t - 2, iy + 1, x - 2] in ["B", "W"]
                            ):
                                mechanics["walk_2"] = True
                                if (
                                    has_B_1
                                    and L[t - 2, iy - 1, x - 2] == "B"
                                    and L[t, iy - 1, x - 2] != "B"
                                ):
                                    mechanics["carry_2"] = True

                    # Jump
                    if (
                        x > 0
                        and L[t - 1, iy + 1, x - 1] == "P"
                        and L[t, iy + 1, x - 1] != "P"
                    ):
                        # from solid ground to solid ground
                        if (
                            iy < Y - 2
                            and L[t - 1, iy + 2, x - 1] in ["B", "W"]
                            and L[t, iy + 1, x] in ["B", "W"]
                        ):
                            mechanics["jump"] = True
                            if (
                                has_B
                                and L[t - 1, iy, x - 1] == "B"
                                and L[t, iy, x - 1] != "B"
                            ):
                                mechanics["jump_with"] = True

                    # Door
                    if x > 0 and iy < Y - 1 and L[t, iy + 1, x] in ["B", "W"]:
                        if L[t, iy, x - 1] == "D":
                            mechanics["door"] = True
                        if L[t, iy + 1, x - 1] == "D":
                            mechanics["door"] = True
                        if iy > 0 and L[t - 1, iy - 1, x - 1] == "D":
                            mechanics["door"] = True
                    if iy < Y - 1 and L[t - 1, iy + 1, x] == "D":
                        mechanics["door"] = True

                    # Fall
                    if iy > 0 and L[t - 1, iy - 1, x] == "P" and L[t, iy - 1, x] != "P":
                        mechanics["fall_1"] = True
                        if (
                            iy > 1
                            and t > 1
                            and L[t - 2, iy - 2, x] == "P"
                            and L[t, iy - 2, x] != "P"
                        ):
                            mechanics["fall_2"] = True

                # Block take
                if has_B and x > 0 and L[t - 1, iy - 1, x] != "B":
                    if L[t - 1, iy, x - 1] == "B" and L[t, iy, x - 1] != "B":
                        mechanics["block_take"] = True

                    if L[t - 1, iy - 1, x - 1] == "B" and L[t, iy - 1, x - 1] != "B":
                        mechanics["block_take"] = True

                # Block drop
                if not has_B and x > 0 and iy < Y - 1 and L[t - 1, iy - 1, x] == "B":
                    if L[t, iy, x - 1] == "B" and L[t - 1, iy, x - 1] != "B":
                        mechanics["block_drop"] = True

                    if L[t, iy - 1, x - 1] == "B" and L[t - 1, iy - 1, x - 1] != "B":
                        mechanics["block_drop"] = True

    return completed, completed_pct, static, static_pct, sprite_counts, mechanics


if __name__ == "__main__":

    # visualize_generated_levels(dir)

    # dir = "src/solutions"
    # level_range = ["A", 1, 2, 3, 4]
    # tile_shapes = [(1, 1, 1), (1, 2, 2), (2, 2, 2), (2, 3, 2)]
    # neighborhood_fns = [
    #     all_neighbors_3D,
    #     orthogonal_neighbors_3D,
    #     horizontal_neighbors_3D,
    # ]
    # visualize_training_data(dir, level_range, tile_shapes, neighborhood_fns)

    # jsonify_0419("experiments_0419/3D")
    # jsonify_0422("experiments_0422/3D")
    dirs = ["experiments_0419/3D", "experiments_0422/3D"]
    for dir in dirs:
        visualize_experiments(dir)
