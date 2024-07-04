import numpy as np
from pathlib import Path
import json
from wfc.wfc import WaveFunctionCollapse
from wfc.pattern import Pattern
import os
import time


def initialize_path(grid_size):
    grid = np.zeros(grid_size, dtype=str)
    t0 = np.full((grid_size[1], grid_size[2]), "_")
    grid[0] = t0

    tp = 0
    yp = np.random.randint(grid_size[1])
    xp = np.random.randint(grid_size[2])
    grid[tp, yp, xp] = "P"

    td = 0
    yd = np.random.randint(grid_size[1])
    xd = np.random.randint(grid_size[2])
    grid[td, yd, xd] = "D"

    return grid


def initialize_path_noblank(grid_size):
    grid = np.zeros(grid_size, dtype=str)

    tp = 0
    yp = np.random.randint(grid_size[1])
    xp = np.random.randint(grid_size[2])
    grid[tp, yp, xp] = "P"

    td = 0
    yd = np.random.randint(grid_size[1])
    xd = np.random.randint(grid_size[2])
    grid[td, yd, xd] = "D"

    return grid


def initialize_maze(grid_size):
    grid = np.zeros(grid_size, dtype=str)

    tp = 0
    yp = np.random.randint(grid_size[1])
    xp = np.random.randint(grid_size[2])
    grid[tp, yp, xp] = "P"

    td = 0
    yd = np.random.randint(grid_size[1])
    xd = np.random.randint(grid_size[2])
    grid[td, yd, xd] = "D"

    return grid


def initialize_soko(grid_size):
    grid = np.zeros(grid_size, dtype=str)

    tb = 0
    yb = np.random.randint(grid_size[1])
    xb = np.random.randint(grid_size[2])
    grid[tb, yb, xb] = "B"

    tO = 0
    yO = np.random.randint(grid_size[1])
    xO = np.random.randint(grid_size[2])
    grid[tO, yO, xO] = "O"

    return grid


def initialize_lime(grid_size):
    grid = np.zeros(grid_size, dtype=str)

    tb = 0
    yb = grid_size[1] - 2
    xb = grid_size[2] - 1
    grid[tb, yb, xb] = "O"

    return grid


def initialize_block(grid_size):
    grid = np.zeros(grid_size, dtype=str)

    tb = 0
    yb = grid_size[1] - 2
    xb = grid_size[2] - 1
    grid[tb, yb, xb] = "O"

    return grid


def get_soko_settings():
    soko_padding = {
        "pad_width": ((0, 1), (1, 1), (1, 1)),
        "constant_values": (
            (("t"), ("T")),
            (("_"), ("_")),
            (("_"), ("_")),
        ),
        "axis_order": (2, 1, 0),
    }
    local_transforms = {"flipx": True, "flipy": True, "rotxy": [1, 2, 3]}
    global_transforms = local_transforms
    return soko_padding, local_transforms, global_transforms


def get_path_settings():
    path_padding = {
        "pad_width": ((0, 1), (1, 1), (1, 1)),
        "constant_values": (
            (("t"), ("T")),
            (("_"), ("_")),
            (("_"), ("_")),
        ),
        "axis_order": (2, 1, 0),
    }
    local_transforms = {"flipx": True, "flipy": True, "rotxy": [1, 2, 3]}
    global_transforms = local_transforms
    return path_padding, local_transforms, global_transforms


def get_maze_settings():
    maze_padding = {
        "pad_width": ((0, 1), (1, 1), (1, 1)),
        "constant_values": (
            (("t"), ("T")),
            (("W"), ("W")),
            (("W"), ("W")),
        ),
        "axis_order": (2, 1, 0),
    }
    local_transforms = {"flipx": True, "flipy": True, "rotxy": [1, 2, 3]}
    global_transforms = local_transforms
    return maze_padding, local_transforms, global_transforms


def get_lime_settings():
    lime_padding = {
        "pad_width": ((0, 1), (1, 1), (1, 1)),
        "constant_values": (
            (("t"), ("T")),
            (("_"), ("W")),
            (("_"), ("_")),
        ),
        "axis_order": (2, 1, 0),
    }
    local_transforms = {"flipx": True}
    global_transforms = local_transforms
    return lime_padding, local_transforms, global_transforms


def get_block_settings():
    block_padding = {
        "pad_width": ((0, 1), (1, 1), (1, 1)),
        "constant_values": (
            (("t"), ("T")),
            (("_"), ("W")),
            (("_"), ("_")),
        ),
        "axis_order": (2, 1, 0),
    }
    local_transforms = {}
    global_transforms = {"flip_x": True}
    return block_padding, local_transforms, global_transforms


def get_train_levels(path):
    train_levels = []
    if os.path.isfile(path):
        with open(path) as f:
            train_levels = [np.array(json.load(f))]

    else:
        for subpath in os.listdir(path):
            train_levels += get_train_levels(f"{path}/{subpath}")

    return train_levels


def board_to_str(board):
    board_str = ""
    for row in board:
        row = ["." if x == "" else x for x in row]
        row_str = " ".join(row)
        board_str += row_str + "\n"
    return board_str[:-1]  # trim final \n

def save_level(level, raw_f, display_d):
    with open(raw_f, "w", encoding="utf-8") as f:
        json.dump(level, f, ensure_ascii=False, indent=4)

    Path(display_d).mkdir(parents=True, exist_ok=True)
    for i, board in enumerate(level):
        i_str = str(i)
        while len(i_str) < 3:
            i_str = "0" + i_str
        with open(f"{display_d}/{i_str}.lvl", "w", encoding="utf-8") as f:
            f.write(board_to_str(board))


def process(
    train_paths, initialize_fn, settings_fn, grid_size, pattern_size, num_tries
):
    np.random.seed(23)
    Pattern.set_format("char")

    padding, local_transforms, global_transforms = settings_fn()
    Pattern.set_padding(padding)
    Pattern.set_local_transforms(local_transforms)
    Pattern.set_global_transforms(global_transforms)

    train_levels = []
    for path in train_paths:
        train_levels += get_train_levels(path)

    results_d = f"src/results/{time.time()}"
    Path(results_d).mkdir(parents=True, exist_ok=True)
    stats_f = f"{results_d}/stats.json"
    stats = {
        "train_paths": train_paths,
        "padding": padding,
        "local_transforms": local_transforms,
        "global_transforms": global_transforms,
        "grid_size": grid_size,
        "pattern_size": pattern_size,
        "elapsed": {},
    }
    with open(stats_f, "w", encoding="utf-8") as f:
        json.dump(stats, f)

    wfc = WaveFunctionCollapse(grid_size, train_levels, pattern_size, initialize_fn)
    num_fails = 0
    t0 = time.time()
    for i in range(num_tries):
        results_i_d = f"{results_d}/{i}"
        Path(results_i_d).mkdir(parents=True, exist_ok=True)
        ti0 = time.time()
        wfc.run()
        init_level = wfc.get_initial_image()
        level = wfc.get_image()
        if "" in level:
            num_fails += 1
        ti1 = time.time()
        elapsed = ti1 - ti0
        stats["elapsed"][i] = elapsed
        with open(stats_f, "w", encoding="utf-8") as f:
            json.dump(stats, f)

        init_raw_f = f"{results_i_d}/init_raw.json"
        final_raw_f = f"{results_i_d}/final_raw.json"
        init_display_d = f"{results_i_d}/init_display"
        final_display_d = f"{results_i_d}/final_display"

        save_level(init_level.tolist(), init_raw_f, init_display_d)
        save_level(level.tolist(), final_raw_f, final_display_d)

    t1 = time.time()
    elapsed = t1 - t0
    stats["elapsed"]["total"] = elapsed
    stats["num_fails"] = f"{num_fails}/{num_tries}"
    with open(stats_f, "w", encoding="utf-8") as f:
        json.dump(stats, f)


if __name__ == "__main__":

    grid_size = (10, 6, 6)
    pattern_size = (2, 3, 3)

    train_paths = [f"src/training_data/simple_path/path_no_walls/path_1_nw.json"]
    num_tries = 10

    initialize_fn = initialize_path_noblank
    settings_fn = get_path_settings

    process(train_paths, initialize_fn, settings_fn, grid_size, pattern_size, num_tries)
