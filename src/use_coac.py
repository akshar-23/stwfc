import numpy as np
from pathlib import Path
import json
from wfc.wfc import WaveFunctionCollapse
from wfc.pattern import Pattern
from src.utils import pad


def clean_train_level__blockdude(level):
    # Remove LV# row from level
    if level[0][0][1] == ".":
        level = level[:, 1:, :]
    # Replace B' with B
    level[level == "B'"] = "B"
    # # Remove the last timestep to ensure that P is always present, and
    # # duplicate the second-to-last to allow P to stay at the door
    # level[-1] = level[-2]
    # level = np.append(level, [level[-1], level[-1]], axis=0)
    return level


def char_to_int(level):
    shape = list(level.shape)
    shape.append(1)
    level = level.reshape(shape)
    int_level = np.zeros(level.shape, dtype=int)
    key = ['.']
    for idx in np.ndindex(level.shape):
        char = level[idx]
        if char not in key:
            key.append(level[idx])
        int_level[idx] = key.index(char)
    key.append("A")
    key.append("B")
    key.append("C")
    key.append("D")
    key.append("E")
    key.append("F")
    return int_level, key


def int_to_char(level, key):
    shape = list(level.shape)[:-1]
    level = level.reshape(shape)
    char_level = np.full(level.shape, "X")

    for idx in np.ndindex(level.shape):
        i = int(level[idx])
        char_level[idx] = key[i]
    return char_level


if __name__ == "__main__":
    np.random.seed(23)

    # Pattern.set_transforms({"flipx": True})
    Pattern.set_padding({
        "pad_width": ((1, 1), (1, 1), (1, 1)),
        "constant_values": (
            ((2), (3)),
            ((4), (5)),
            ((6), (7)),
        ),
        "axis_order": (2, 1, 0),
    })

    grid_size = (5, 6, 6)
    pattern_size = (2, 2, 2)

    train_level = []
    soln_folder = f"src/training_data/static/blank"
    soln_file = Path(f"{soln_folder}/solution.json")
    with open(soln_file) as f:
        train_level = np.array(json.load(f))
    # train_level = clean_train_level__blockdude(train_level)

    # Add a dimension for coac_wfc's expected rbg
    coac_level, key = char_to_int(train_level)

    wfc = WaveFunctionCollapse(grid_size, coac_level, pattern_size)

    wfc.run()
    image = wfc.get_image()

    print(int_to_char(image, key))
