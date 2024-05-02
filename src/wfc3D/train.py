# from train_levels import TRAIN_LEVELS
import numpy as np
import math
import os
import json


def str_to_board(str):
    rows = str.split("\n")
    return np.array(list(map(list, rows)))


def board_to_str(lvl):
    lvl_str = ""
    for row in lvl:
        row_str = "".join(row)
        lvl_str += row_str + "\n"
    return lvl_str[:-1]  # trim final \n


def test_str_to_from_board():
    print("Test str_to_board")
    board_str = "ABC\nDEF\nGHI"
    board = str_to_board(board_str)
    expected = np.array([["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"]])
    assert np.array_equal(board, expected), f"{board} != {expected}"

    print("Test board_to_str")
    back_to_str = board_to_str(board)
    assert back_to_str == board_str, f"{back_to_str} != {board_str}"


def pad(grid, tile_shape=(1, 1, 1), pad_values=("X", "X", "X"), pad_widths=(1, 1, 1)):
    constant_values = []
    for dim_vals in pad_values:
        if len(dim_vals) == 1:
            constant_values.append((dim_vals[0], dim_vals[0]))
        else:
            constant_values.append(dim_vals)
    constant_values = tuplify(constant_values)

    L = grid
    for i in range(len(pad_values) - 1, -1, -1):
        expanded_pad_widths = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
        expanded_constant_values = [
            constant_values[0],
            constant_values[0],
            constant_values[0],
            constant_values[0],
            constant_values[0],
            constant_values[0],
        ]
        expanded_pad_widths[i] = (pad_widths[i], pad_widths[i])
        expanded_constant_values[i] = constant_values[i]
        L = np.pad(
            L,
            pad_width=tuplify(expanded_pad_widths),
            constant_values=tuplify(expanded_constant_values),
        )
    return L


def unpad_3D(grid, pad_widths=(1, 1, 1)):
    return grid[
        pad_widths[0] : -pad_widths[0],
        pad_widths[1] : -pad_widths[1],
        pad_widths[2] : -pad_widths[2],
    ]


def test_padding():
    print("Test pad")
    grid = np.array(
        [
            [
                ["A", "B", "C"],
                ["D", "E", "F"],
            ],
            [
                ["G", "H", "I"],
                ["J", "K", "L"],
            ],
        ],
    )
    padded = pad(grid)

    expected = np.array(
        [
            [
                ["X", "X", "X", "X", "X"],
                ["X", "X", "X", "X", "X"],
                ["X", "X", "X", "X", "X"],
                ["X", "X", "X", "X", "X"],
            ],
            [
                ["X", "X", "X", "X", "X"],
                ["X", "A", "B", "C", "X"],
                ["X", "D", "E", "F", "X"],
                ["X", "X", "X", "X", "X"],
            ],
            [
                ["X", "X", "X", "X", "X"],
                ["X", "G", "H", "I", "X"],
                ["X", "J", "K", "L", "X"],
                ["X", "X", "X", "X", "X"],
            ],
            [
                ["X", "X", "X", "X", "X"],
                ["X", "X", "X", "X", "X"],
                ["X", "X", "X", "X", "X"],
                ["X", "X", "X", "X", "X"],
            ],
        ],
    )

    assert np.array_equal(padded, expected), f"{padded} != {expected}"

    print("Test unpad_3D")
    unpadded = unpad_3D(padded)
    assert np.array_equal(unpadded, grid), f"{unpadded} != {grid}"


def islistlike(obj):
    return isinstance(obj, list) or isinstance(obj, np.ndarray)


def tuplify(listlike):
    if islistlike(listlike[0]):
        return tuple(tuplify(x) for x in listlike)
    return tuple(listlike)


def test_tuplify():
    print("Test tuple_of_tuples")
    og_list_of_lists_of_lists = np.array([
        [["A", "B", "C"], ["D", "E", "F"]],
        [["G", "H", "I"], ["J", "K", "L"]],
    ])
    expected = ((("A", "B", "C"), ("D", "E", "F")), (("G", "H", "I"), ("J", "K", "L")))
    actual = tuplify(og_list_of_lists_of_lists)
    assert expected == actual, f"{actual} != {expected}"


# above, below, left, right, diagonals, center through time
def all_neighbors_3D(L, idx):
    K, I, J = idx
    neighbors = []

    for k in range(K - 1, K + 2):
        for i in range(I - 1, I + 2):
            for j in range(J - 1, J + 2):
                if k != K or i != I or j != J:
                    neighbors.append(L[k, i, j])
    return neighbors


# above, below, left, right, center through time
def orthogonal_neighbors_3D(L, idx):
    K, I, J = idx
    neighbors = []

    for k in range(K - 1, K + 2):
        for i in range(I - 1, I + 2):
            for j in range(J - 1, J + 2):
                if (k != K or i != I or j != J) and (i == I or j == J):
                    neighbors.append(L[k, i, j])
    return neighbors


# left, right, center through time
def horizontal_neighbors_3D(L, idx):
    K, I, J = idx
    neighbors = []

    for k in range(K - 1, K + 2):
        for j in range(J - 1, J + 2):
            if k != K or j != J:
                neighbors.append(L[k, I, j])
    return neighbors


def test_neighbors():
    grid = np.array(
        [
            [
                ["0", "1", "2"],
                ["A", "B", "C"],
                ["D", "E", "F"],
            ],
            [
                ["3", "4", "5"],
                ["G", "H", "I"],
                ["J", "K", "L"],
            ],
            [
                ["6", "7", "8"],
                ["M", "N", "O"],
                ["P", "Q", "R"],
            ],
        ],
    )

    print("Test all_neighbors_3D")
    expected = [
        "0",
        "1",
        "2",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "3",
        "4",
        "5",
        "G",
        "I",
        "J",
        "K",
        "L",
        "6",
        "7",
        "8",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
    ]
    actual = all_neighbors_3D(grid, (1, 1, 1))
    assert np.array_equal(expected, actual), f"{actual} != {expected}"

    print("Test orthogonal_neighbors_3D")
    expected = ["1", "A", "B", "C", "E", "4", "G", "I", "K", "7", "M", "N", "O", "Q"]
    actual = orthogonal_neighbors_3D(grid, (1, 1, 1))
    assert np.array_equal(expected, actual), f"{actual} != {expected}"

    print("Test horizontal_neighbors_3D")
    expected = ["A", "B", "C", "G", "I", "M", "N", "O"]
    actual = horizontal_neighbors_3D(grid, (1, 1, 1))
    assert np.array_equal(expected, actual), f"{actual} != {expected}"


def normalize(counts):
    distribution = {}
    total = sum(counts.values())
    for key, count in counts.items():
        distribution[key] = count / total
    return distribution


def test_normalize():
    print("Test normalize")
    counts = {"A": 7, "B": 2, "C": 1}

    expected = {
        "A": 0.7,
        "B": 0.2,
        "C": 0.1,
    }

    actual = normalize(counts)
    assert actual == expected, f"{actual} != {expected}"


def flip_over_vertical(board):
    flipped = []
    for row in board:
        flipped.append(list(reversed(row)))
    return flipped


def flip_soln(solution):
    flipped_soln = []
    for board in solution:
        flipped_soln.append(flip_over_vertical(board))
    return np.array(flipped_soln)


def test_flip():
    print("Test flip_soln and flip_over_vertical")

    grid = np.array(
        [
            [
                ["A", "B", "C"],
                ["D", "E", "F"],
            ],
            [["G", "H", "I"], ["J", "K", "L"]],
        ],
    )

    expected = np.array(
        [
            [
                ["C", "B", "A"],
                ["F", "E", "D"],
            ],
            [
                ["I", "H", "G"],
                ["L", "K", "J"],
            ],
        ],
    )

    actual = flip_soln(grid)
    assert np.array_equal(actual, expected), f"{actual} != {expected}"


def encode_tiles_3D(L, tile_shape):
    K, I, J = L.shape
    Kt, It, Jt = tile_shape

    # Final encoded level size
    Ke = K - (Kt - 1)
    Ie = I - (It - 1)
    Je = J - (Jt - 1)

    # Fill an empty array with tiles of the appropriate size
    Le = np.full((Ke, Ie, Je, Kt, It, Jt), "#####")

    # Encode the level with tile_shape groups
    for ke in range(Ke):
        for ie in range(Ie):
            for je in range(Je):
                for kt in range(Kt):
                    for it in range(It):
                        for jt in range(Jt):
                            Le[ke, ie, je, kt, it, jt] = L[ke + kt, ie + it, je + jt]

    return Le


def decode_tiles_3D(Le, options=False, options_all="PDB_W"):
    Ke, Ie, Je, Kt, It, Jt = Le.shape

    # Final decoded level size
    K = Ke + (Kt - 1)
    I = Ie + (It - 1)
    J = Je + (Jt - 1)

    # Fill an empty array with empty values
    empty_val = "."
    if options:
        empty_val = "".join(sorted(list(options_all)))
    L = np.full((K, I, J), empty_val)

    for Leidx in np.ndindex(Le.shape):
        ke, ie, je, kt, it, jt = Leidx
        Lidx = ke + kt, ie + it, je + jt
        Lt = L[Lidx]
        Let = Le[Leidx]
        if options:
            L[Lidx] = "".join(
                sorted(list(set(list(Lt)).intersection(list(Let))))
            ).strip()
        else:
            if Lt != "." and Let != "." and Lt != Let:
                raise Exception(f"{Let} != {Lt}")
            L[Lidx] = Let

    return L


def test_encode_decode():
    grid = np.array(
        [
            [
                ["XXX", "X", "X"],
                ["X", "X", "X"],
                ["X", "X", "X"],
                ["X", "X", "X"],
            ],
            [
                ["X", "X", "X"],
                ["X", ".", "B"],
                ["X", "D", "E"],
                ["X", "X", "X"],
            ],
            [
                ["X", "X", "X"],
                ["X", "G", "H"],
                ["X", "J", "K"],
                ["X", "X", "X"],
            ],
        ],
    )
    print("Test encode_tiles_3D")
    expected_encoding = np.array([
        [
            [
                [
                    [
                        ["X", "X"],
                        ["X", "X"],
                        ["X", "X"],
                    ],
                    [
                        ["X", "X"],
                        ["X", "."],
                        ["X", "D"],
                    ],
                ],
                [
                    [
                        ["X", "X"],
                        ["X", "X"],
                        ["X", "X"],
                    ],
                    [
                        ["X", "X"],
                        [".", "B"],
                        ["D", "E"],
                    ],
                ],
            ],
            [
                [
                    [
                        ["X", "X"],
                        ["X", "X"],
                        ["X", "X"],
                    ],
                    [
                        ["X", "."],
                        ["X", "D"],
                        ["X", "X"],
                    ],
                ],
                [
                    [
                        ["X", "X"],
                        ["X", "X"],
                        ["X", "X"],
                    ],
                    [
                        [".", "B"],
                        ["D", "E"],
                        ["X", "X"],
                    ],
                ],
            ],
        ],
        [
            [
                [
                    [
                        ["X", "X"],
                        ["X", "."],
                        ["X", "D"],
                    ],
                    [
                        ["X", "X"],
                        ["X", "G"],
                        ["X", "J"],
                    ],
                ],
                [
                    [
                        ["X", "X"],
                        [".", "B"],
                        ["D", "E"],
                    ],
                    [
                        ["X", "X"],
                        ["G", "H"],
                        ["J", "K"],
                    ],
                ],
            ],
            [
                [
                    [
                        ["X", "."],
                        ["X", "D"],
                        ["X", "X"],
                    ],
                    [
                        ["X", "G"],
                        ["X", "J"],
                        ["X", "X"],
                    ],
                ],
                [
                    [
                        [".", "B"],
                        ["D", "E"],
                        ["X", "X"],
                    ],
                    [
                        ["G", "H"],
                        ["J", "K"],
                        ["X", "X"],
                    ],
                ],
            ],
        ],
    ])
    grid[0][0][0] = "X"
    encoded = encode_tiles_3D(grid, (2, 3, 2))
    assert np.array_equal(
        encoded, expected_encoding
    ), f"{encoded} != {expected_encoding}"

    print("Test decode_tiles_3D")
    expected_decoding = grid
    decoded = decode_tiles_3D(encoded)
    assert np.array_equal(
        decoded, expected_decoding
    ), f"{decoded} != {expected_decoding}"

    print("Test options")
    grid[0][0][1] = "XYZ"
    encoded = encode_tiles_3D(grid, (2, 3, 2))
    assert encoded[0][0][0][0][0][1] == "XYZ"
    assert encoded[0][0][1][0][0][0] == "XYZ"
    encoded[0][0][1][0][0][0] = "XY"
    decoded = decode_tiles_3D(encoded, True, options_all="ABCDEFGHIJKLMNOPQRSTUVWXYZ.")
    assert decoded[0][0][1] == "XY"


def train_3D(levels, tile_shape, neighborhood_fn=all_neighbors_3D):
    tile_counts = {}
    neighborhood_counts = {}
    count = 0
    for L in levels:
        Le = encode_tiles_3D(L, tile_shape)
        Lep = pad(Le, tile_shape)

        K, I, J, Kt, It, Jt = Lep.shape
        for k in range(1, K - 1):
            for i in range(1, I - 1):
                for j in range(1, J - 1):
                    T = tuplify(Lep[k, i, j])
                    if T not in tile_counts:
                        tile_counts[T] = 0
                    tile_counts[T] += 1
                    N = tuplify(neighborhood_fn(Lep, (k, i, j)))
                    if N not in neighborhood_counts:
                        neighborhood_counts[N] = {}
                        count += 1
                    if T not in neighborhood_counts[N]:
                        neighborhood_counts[N][T] = 0
                    neighborhood_counts[N][T] += 1

    return (
        tile_counts,
        neighborhood_counts,
    )


def test_train():
    print("Test train_3D")
    L = np.array(
        [
            [
                ["A", "B", "C"],
                ["D", "E", "F"],
                ["G", "H", "I"],
            ],
            [
                ["J", "K", "L"],
                ["M", "N", "O"],
                ["P", "Q", "R"],
            ],
        ],
    )
    tile_counts, neighborhood_counts = train_3D([L], (2, 3, 2))

    tile = ((("A", "B"), ("D", "E"), ("G", "H")), (("J", "K"), ("M", "N"), ("P", "Q")))
    assert tile_counts[tile] == 1

    neighborhood = (
        (
            (("X", "X"), ("X", "X"), ("X", "X")),
            (("X", "X"), ("X", "A"), ("X", "D")),
        ),
        (
            (("X", "X"), ("X", "X"), ("X", "X")),
            (("X", "X"), ("A", "B"), ("D", "E")),
        ),
        (
            (("X", "X"), ("X", "X"), ("X", "X")),
            (("X", "X"), ("B", "C"), ("E", "F")),
        ),
        (
            (("X", "X"), ("X", "X"), ("X", "X")),
            (("X", "A"), ("X", "D"), ("X", "G")),
        ),
        (
            (("X", "X"), ("X", "X"), ("X", "X")),
            (("A", "B"), ("D", "E"), ("G", "H")),
        ),
        (
            (("X", "X"), ("X", "X"), ("X", "X")),
            (("B", "C"), ("E", "F"), ("H", "I")),
        ),
        (
            (("X", "X"), ("X", "X"), ("X", "X")),
            (("X", "D"), ("X", "G"), ("X", "X")),
        ),
        (
            (("X", "X"), ("X", "X"), ("X", "X")),
            (("D", "E"), ("G", "H"), ("X", "X")),
        ),
        (
            (("X", "X"), ("X", "X"), ("X", "X")),
            (("E", "F"), ("H", "I"), ("X", "X")),
        ),
        (
            (("X", "X"), ("X", "A"), ("X", "D")),
            (("X", "X"), ("X", "J"), ("X", "M")),
        ),
        (
            (("X", "X"), ("A", "B"), ("D", "E")),
            (("X", "X"), ("J", "K"), ("M", "N")),
        ),
        (
            (("X", "X"), ("B", "C"), ("E", "F")),
            (("X", "X"), ("K", "L"), ("N", "O")),
        ),
        (
            (("X", "A"), ("X", "D"), ("X", "G")),
            (("X", "J"), ("X", "M"), ("X", "P")),
        ),
        (
            (("B", "C"), ("E", "F"), ("H", "I")),
            (("K", "L"), ("N", "O"), ("Q", "R")),
        ),
        (
            (("X", "D"), ("X", "G"), ("X", "X")),
            (("X", "M"), ("X", "P"), ("X", "X")),
        ),
        (
            (("D", "E"), ("G", "H"), ("X", "X")),
            (("M", "N"), ("P", "Q"), ("X", "X")),
        ),
        (
            (("E", "F"), ("H", "I"), ("X", "X")),
            (("N", "O"), ("Q", "R"), ("X", "X")),
        ),
        (
            (("X", "X"), ("X", "J"), ("X", "M")),
            (("X", "X"), ("X", "X"), ("X", "X")),
        ),
        (
            (("X", "X"), ("J", "K"), ("M", "N")),
            (("X", "X"), ("X", "X"), ("X", "X")),
        ),
        (
            (("X", "X"), ("K", "L"), ("N", "O")),
            (("X", "X"), ("X", "X"), ("X", "X")),
        ),
        (
            (("X", "J"), ("X", "M"), ("X", "P")),
            (("X", "X"), ("X", "X"), ("X", "X")),
        ),
        (
            (("J", "K"), ("M", "N"), ("P", "Q")),
            (("X", "X"), ("X", "X"), ("X", "X")),
        ),
        (
            (("K", "L"), ("N", "O"), ("Q", "R")),
            (("X", "X"), ("X", "X"), ("X", "X")),
        ),
        (
            (("X", "M"), ("X", "P"), ("X", "X")),
            (("X", "X"), ("X", "X"), ("X", "X")),
        ),
        (
            (("M", "N"), ("P", "Q"), ("X", "X")),
            (("X", "X"), ("X", "X"), ("X", "X")),
        ),
        (
            (("N", "O"), ("Q", "R"), ("X", "X")),
            (("X", "X"), ("X", "X"), ("X", "X")),
        ),
    )

    assert neighborhood_counts[neighborhood][tile] == 1
    assert len(neighborhood_counts[neighborhood]) == 1


def test():
    test_str_to_from_board()
    test_padding()
    test_tuplify()
    test_neighbors()
    test_normalize()
    test_flip()
    test_encode_decode()
    test_train()


if __name__ == "__main__":
    test()
