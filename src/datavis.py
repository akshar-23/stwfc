from train_levels import TRAIN_LEVELS
from train import str_to_lvl, lvl_to_str, train, encode_tiles, decode_tiles
from generate import wfc, mrf
import matplotlib.pyplot as plt
import numpy as np
import math


def calculate_completion_pct(level):
    total = level.size
    complete = 0
    for tile in np.nditer(level):
        if tile != ".":
            complete += 1
    return complete / total


def plot_context(context_counts, filename_counts, filename_options):
    c_counts = {}
    options_counts = {}
    for tile_counts in context_counts.values():
        num_options = len(tile_counts.values())
        if num_options not in options_counts:
            options_counts[num_options] = 0
        options_counts[num_options] += 1

        count_sum = sum(tile_counts.values())
        # how many contexts have been seen sum times
        if count_sum not in c_counts:
            c_counts[count_sum] = 0
        c_counts[count_sum] += 1

    plt.clf()
    # how many times each context has been seen
    all_sums = list(c_counts.keys())
    all_counts = list(c_counts.values())
    # number of times seen vs # contexts seen that many times
    plt.scatter(all_sums, all_counts)
    plt.savefig(filename_counts)

    plt.clf()
    # how many options for each context
    all_num_options = list(options_counts.keys())
    all_count_options = list(options_counts.values())
    # options vs # contexts with that # options
    plt.scatter(all_num_options, all_count_options)
    plt.savefig(filename_options)
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


def save_levels(levels, filename):
    file = open(filename, "w")
    for level in levels:
        str_level = lvl_to_str(level)
        file.write("\n\n")
        file.write(str_level)
    return


def process(folder, tile_size):
    dimensions = (7, 20)

    overlapping_levels = []
    non_overlapping_levels = []
    for level in TRAIN_LEVELS:
        unrolled_level = str_to_lvl(level)
        encoded_overlapping_level = encode_tiles(unrolled_level, tile_size)
        encoded_non_overlapping_level = encode_tiles(unrolled_level, tile_size, False)
        overlapping_levels.append(encoded_overlapping_level)
        non_overlapping_levels.append(encoded_non_overlapping_level)

    # print("Overlap training (WFC)")
    # print(overlapping_levels)

    # print("Non overlap training (MRF)")
    # print(non_overlapping_levels)

    # ALL levels
    # (
    #     overlapping_tile_distribution,
    #     overlapping_full_context_distribution,
    #     overlapping_full_context_counts,
    # ) = train(overlapping_levels, tile_size)
    # plot_context(
    #     overlapping_full_context_counts,
    #     folder + "/overlapping_context_counts.png",
    #     folder + "/overlapping_option_counts.png",
    # )

    # (
    #     non_overlapping_tile_distribution,
    #     non_overlapping_full_context_distribution,
    #     non_overlapping_full_context_counts,
    # ) = train(non_overlapping_levels, tile_size)
    # plot_context(
    #     non_overlapping_full_context_counts,
    #     folder + "/non_overlapping_context_counts.png",
    #     folder + "/non_overlapping_option_counts.png",
    # )

    # # print(overlapping_full_context_counts)
    # # print(non_overlapping_full_context_counts)

    # mrf_levels = []
    # mrf_likelihoods = []

    # wfc_levels = []
    # wfc_completion_percentages = []

    # mrf_dimensions = (
    #     math.ceil(dimensions[0] / tile_size),
    #     math.ceil(dimensions[1] / tile_size),
    # )
    # wfc_dimensions = (dimensions[0] - (tile_size - 1), dimensions[1] - (tile_size - 1))
    # for i in range(100):
    #     encoded_level, likelihoods = mrf(
    #         non_overlapping_tile_distribution,
    #         non_overlapping_full_context_distribution,
    #         mrf_dimensions,
    #         tile_size,
    #         1000,
    #     )
    #     level = decode_tiles(encoded_level, tile_size, False)
    #     mrf_levels.append(level)
    #     mrf_likelihoods.append(likelihoods)

    #     encoded_level = wfc(overlapping_full_context_counts, wfc_dimensions, tile_size)
    #     level = decode_tiles(encoded_level, tile_size)
    #     wfc_levels.append(level)
    #     completion_pct = calculate_completion_pct(level)
    #     wfc_completion_percentages.append(completion_pct)

    # plot_completion(wfc_completion_percentages, folder + "/all_wfc_completion.txt")
    # plot_likelihoods(mrf_likelihoods, folder + "/all_mrf_likelihoods.png")
    # save_levels(wfc_levels, folder + "/all_wfc_levels.txt")
    # save_levels(mrf_levels, folder + "/all_mrf_levels.txt")
    # save_level_tile_types(wfc_levels, folder + "/all_wfc_tiles.txt")
    # save_level_tile_types(mrf_levels, folder + "/all_mrf_tiles.txt")

    # Individual levels

    for i, tlevel in enumerate(non_overlapping_levels):
        (
            non_overlapping_tile_distribution,
            non_overlapping_full_context_distribution,
            non_overlapping_full_context_counts,
        ) = train([tlevel], tile_size)
        plot_context(
            non_overlapping_full_context_counts,
            folder + "/non_overlapping_context_counts" + str(i) + ".png",
            folder + "/non_overlapping_options_counts" + str(i) + ".png",
        )

        mrf_likelihoods = []
        mrf_levels = []
        for _ in range(10):
            encoded_level, likelihoods = mrf(
                non_overlapping_tile_distribution,
                non_overlapping_full_context_distribution,
                (
                    math.ceil(tlevel.shape[0]),
                    math.ceil(tlevel.shape[1]),
                ),
                tile_size,
                1000,
            )
            level = decode_tiles(encoded_level, tile_size, False)
            mrf_levels.append(level)
            mrf_likelihoods.append(likelihoods)

        plot_likelihoods(mrf_likelihoods, folder + "/mrf_likelihoods" + str(i) + ".png")
        save_levels(mrf_levels, folder + "/mrf_levels" + str(i) + ".txt")
        save_level_tile_types(mrf_levels, folder + "/mrf_level_tiles" + str(i) + ".txt")

    # for i, tlevel in enumerate(overlapping_levels):
    #     (
    #         overlapping_tile_distribution,
    #         overlapping_full_context_distribution,
    #         overlapping_full_context_counts,
    #     ) = train([tlevel], tile_size)
    #     plot_context(
    #         overlapping_full_context_counts,
    #         folder + "/overlapping_context_counts" + str(i) + ".png",
    #         folder + "/overlapping_options_counts" + str(i) + ".png",
    #     )

    #     wfc_levels = []
    #     wfc_completion_percentages = []
    #     for _ in range(10):
    #         encoded_level = wfc(
    #             overlapping_full_context_counts,
    #             (
    #                 tlevel.shape[0],
    #                 tlevel.shape[1],
    #             ),
    #             tile_size,
    #         )
    #         level = decode_tiles(encoded_level, tile_size)
    #         wfc_levels.append(level)
    #         completion_pct = calculate_completion_pct(level)
    #         wfc_completion_percentages.append(completion_pct)

    #     plot_completion(
    #         wfc_completion_percentages, folder + "/wfc_completion" + str(i) + ".txt"
    #     )
    #     save_levels(wfc_levels, folder + "/wfc_levels" + str(i) + ".txt")
    #     save_level_tile_types(wfc_levels, folder + "/wfc_level_tiles" + str(i) + ".txt")


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
    plt.scatter(X, twobyY, label = "2x2 overlapping")
    plt.legend()
    plt.xlabel("Dataset Size")
    plt.ylabel("Average Level Completion")
    plt.title("Level Completion by Dataset Size for 1x1 and 2x2 Tiles")
    plt.savefig(onebyfile)
    
    plt.clf()
    plt.scatter(X, twobyY)
    plt.savefig(twobyfile)


if __name__ == "__main__":

    # folders = ["experiment11", "experiment12"]
    # tile_size = [1, 2]

    # for i in range(len(folders)):
    #     process(folders[i], tile_size[i])

    folder = "vis_completion"
    visualize_completion(
        folder + "/wfc_completion_1x1.png", folder + "/wfc_completion_2x2.png"
    )

    tile_size = 1
    overlapping_levels = []
    non_overlapping_levels = []
    for level in TRAIN_LEVELS:
        unrolled_level = str_to_lvl(level)
        encoded_overlapping_level = encode_tiles(unrolled_level, tile_size)
        encoded_non_overlapping_level = encode_tiles(unrolled_level, tile_size, False)
        overlapping_levels.append(encoded_overlapping_level)
        non_overlapping_levels.append(encoded_non_overlapping_level)

    (
        overlapping_tile_distribution,
        overlapping_full_context_distribution,
        overlapping_full_context_counts,
    ) = train(overlapping_levels, tile_size)
    plot_context(
        overlapping_full_context_counts,
        folder + "/overlapping_context_counts.png",
        folder + "/overlapping_option_counts.png",
    )

    (
        non_overlapping_tile_distribution,
        non_overlapping_full_context_distribution,
        non_overlapping_full_context_counts,
    ) = train(non_overlapping_levels, tile_size)
    plot_context(
        non_overlapping_full_context_counts,
        folder + "/non_overlapping_context_counts.png",
        folder + "/non_overlapping_option_counts.png",
    )
