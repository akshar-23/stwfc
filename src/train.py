from train_levels import TRAIN_LEVELS
import numpy as np
import math


def str_to_lvl(str):
    rows = str.split("\n")
    return np.array(list(map(list, rows)))


def lvl_to_str(lvl):
    lvl_str = ""
    for row in lvl:
        row_str = "".join(row)
        lvl_str += row_str + "\n"
    return lvl_str


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
            if (i != I or j != J) and (i == I or j == J):
                neighbors.append(L[i, j])
    return neighbors


def train(levels, tile_size=1):

    tile_counts = {}
    full_context_counts = {}
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
        full_context_distribution,
        full_context_counts,
    )


def normalize(counts):
    distribution = {}
    total = sum(counts.values())
    for t, count in counts.items():
        distribution[t] = count / total
    return distribution


def decode_tiles(level, tile_size, overlapping=True):
    level = level[1:-1, 1:-1]
    I, J = level.shape

    # Final decoded level size
    Ie = 0
    Je = 0

    if overlapping:
        Ie = I + (tile_size - 1)
        Je = J + (tile_size - 1)
    else:
        Ie = I * tile_size
        Je = J * tile_size

    # Fill an empty array with empty values
    L = np.full((Ie, Je), ".")
    for i in range(I):
        for j in range(J):
            tiles = np.array(list(level[i, j]))
            start_i = 0
            start_j = 0
            if overlapping:
                start_i = i
                start_j = j
            else:
                start_i = i * tile_size
                start_j = j * tile_size
            ti = 0
            for li in range(start_i, start_i + tile_size):
                for lj in range(start_j, start_j + tile_size):
                    if L[li, lj] != "." and tiles[ti] != "." and L[li, lj] != tiles[ti]:
                        print("SOMETHING IS HORRIBLY BROKEN")
                    L[li, lj] = tiles[ti]
                    ti += 1
    return L


def encode_tiles(level, tile_size, overlapping=True):
    I, J = level.shape

    # Final encoded level size
    Ie = 0
    Je = 0
    if overlapping:
        Ie = I - (tile_size - 1)
        Je = J - (tile_size - 1)
    else:
        Ie = math.ceil(I / tile_size)
        Je = math.ceil(J / tile_size)

    # Fill an empty array with tiles of the appropriate size
    encoding = np.full((Ie, Je), get_pad_value(tile_size))

    # Encode the level with tilesize x tilesize groups
    for ie in range(Ie):
        for je in range(Je):
            offset = tile_size - 1
            step = 0
            if overlapping:
                step = 1
            else:
                step = tile_size
            i = ie * step + offset
            j = je * step + offset
            tile_key = ""
            for li in range(i - (tile_size - 1), i + 1):
                for lj in range(j - (tile_size - 1), j + 1):
                    if li > I - 1 or lj > J - 1:
                        # The level doesn't divide evenly by the tile size; insert wall tiles to make up.
                        tile_key += "#"
                    else:
                        tile_key += level[li, lj]
            encoding[ie, je] = tile_key
    return encoding


if __name__ == "__main__":
    tile_size = 2

    levels = []
    for level in TRAIN_LEVELS:
        unrolled_level = str_to_lvl(level)
        encoded_level = encode_tiles(unrolled_level, tile_size)
        levels.append(encoded_level)
        print(level)
        print(unrolled_level)
        print(encoded_level)

    (
        tile_distribution,
        full_context_distribution,
        full_context_counts,
    ) = train(levels, tile_size)

    # print("Tile Distribution")
    # print(tile_distribution)
    print("Context Distribution")
    print(full_context_distribution)
    print("Context Counts")
    print(full_context_counts)
