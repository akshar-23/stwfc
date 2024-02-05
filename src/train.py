from train_levels import TRAIN_LEVELS
import numpy as np


def str_to_lvl(str):
    rows = str.split("\n")
    return np.array(list(map(list, rows)))


def get_pad_value(tile_size, pad_char="X"):
    pad_value = ""
    for _ in range(tile_size * tile_size):
        pad_value += pad_char
    return pad_value


def pad(grid, width=1, tile_size=1):
    pad_value = get_pad_value(tile_size)
    return np.pad(grid, pad_width=width, constant_values=pad_value)


def context_key(neighbors):
    key = ""
    for n in neighbors:
        key += n
    return key


def key_to_arr(key):
    return list(key)


def all_neighbors(L, pos):
    I, J = pos
    neighbors = []

    for i in range(I - 1, I + 2):
        for j in range(J - 1, J + 2):
            if i != I or j != J and i == I or j == J:
                neighbors.append(L[i, j])
    return neighbors


def wfc_neighbors(L, pos):
    I, J = pos
    neighbors = []
    for i in range(I, I + 2):
        for j in range(J - 1, J + 1):
            if i != I or j != J:
                neighbors.append(L[i, j])
    return neighbors


def train(levels, tile_size=1):

    tile_counts = {}
    full_context_counts = {}
    wfc_context_counts = {}
    for L in levels:
        L = pad(L, 1, tile_size)
        I, J = L.shape
        for i in range(1, I - 1):
            for j in range(1, J - 1):
                t = L[i, j]
                if t not in tile_counts:
                    tile_counts[t] = 0
                tile_counts[t] += 1
                fc = context_key(all_neighbors(L, (i, j)))
                wfcc = context_key(wfc_neighbors(L, (i, j)))
                if fc not in full_context_counts:
                    full_context_counts[fc] = {}
                if wfcc not in wfc_context_counts:
                    wfc_context_counts[wfcc] = {}
                if t not in full_context_counts[fc]:
                    full_context_counts[fc][t] = 0
                if t not in wfc_context_counts[wfcc]:
                    wfc_context_counts[wfcc][t] = 0
                full_context_counts[fc][t] += 1
                wfc_context_counts[wfcc][t] += 1

    tile_distribution = {}
    full_context_distribution = {}
    wfc_context_distribution = {}

    tile_distribution = normalize(tile_counts)

    for c, counts in full_context_counts.items():
        full_context_distribution[c] = normalize(counts)

    for c, counts in wfc_context_counts.items():
        wfc_context_distribution[c] = normalize(counts)

    return (
        tile_distribution,
        full_context_distribution,
        full_context_counts,
        wfc_context_distribution,
    )


def normalize(counts):
    distribution = {}
    total = sum(counts.values())
    for t, count in counts.items():
        distribution[t] = count / total
    return distribution


def encode_tiles(level, tile_size):
    I, J = level.shape

    padding = tile_size - 1

    # Final encoded level size including padding
    Ie = I + padding
    Je = J + padding

    encoding = np.full((Ie, Je), get_pad_value(tile_size))

    # Pad the level so the tiles take into account padding
    Lp = pad(level, padding)

    # Encode the level with tiles made up of overlapping tilesize x tilesize groups
    for ie in range(Ie):
        for je in range(Je):
            i = ie + padding
            j = je + padding
            tile_key = ""
            for lpi in range(i - padding, i + 1):
                for lpj in range(j - padding, j + 1):
                    tile_key += Lp[lpi, lpj]
            encoding[ie, je] = tile_key

    return encoding


if __name__ == "__main__":
    tile_size = 2

    levels = []
    for level in TRAIN_LEVELS:
        levels.append(encode_tiles(str_to_lvl(level), tile_size))

    (
        tile_distribution,
        full_context_distribution,
        full_context_counts,
        wfc_context_distribution,
    ) = train(levels, tile_size)
    print(tile_distribution)
    print(full_context_distribution)
    print(wfc_context_distribution)
