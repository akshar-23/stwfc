from pathlib import Path
import os
import numpy as np
import pandas as pd
import json
from src.use_coac import board_to_str


def save_lvl_displays(paths):
    for path in paths:
        if os.path.isfile(path):
            if path.endswith(".json"):
                with open(path) as f:
                    level = np.array(json.load(f))
                level_str = f"{level.shape}\n"
                level = level.tolist()
                name = Path(path).stem
                dir = os.path.dirname(path)
                Path(f"{dir}/{name}").mkdir(parents=True, exist_ok=True)
                for i, board in enumerate(level):
                    board_str = board_to_str(board)
                    with open(f"{dir}/{name}/{i}.lvl", "w", encoding="utf-8") as f:
                        f.write(board_str)
                    level_str += board_str
                    level_str += "\n\n***\n\n"
                with open(f"{dir}/{name}.txt", "w", encoding="utf-8") as f:
                    f.write(level_str)

        else:
            for subpath in os.listdir(path):
                save_lvl_displays([f"{path}/{subpath}"])


def process(d_results):
    overall_num_trivial = 0
    overall_num_success = 0
    overall_num_has_extra_P = 0
    for game in os.listdir(d_results):
        game_num_trivial = 0
        game_num_success = 0
        game_num_has_extra_P = 0

        d_game = f"{d_results}/{game}"
        if os.path.isfile(d_game):
            continue
        for run in os.listdir(d_game):
            d_run = f"{d_game}/{run}"
            if os.path.isfile(d_run):
                continue
            with open(f"{d_run}/stats.json") as f:
                stats = json.load(f)

            average_time = stats["elapsed"]["total"] / 10
            effective_length = {}
            unbroken_effective_length = {}
            success = {}
            has_extra_P = {}
            num_trivial = 0
            num_trivial_unbroken = 0
            num_success = 0
            num_has_extra_P = 0

            train_paths = stats["train_paths"]
            save_lvl_displays(train_paths)

            for lvl in os.listdir(d_run):
                i = lvl[0]
                d_lvl = f"{d_run}/{lvl}"
                if os.path.isfile(d_lvl):
                    continue

                with open(f"{d_lvl}/init_raw.json") as f:
                    init_lvl = np.array(json.load(f))
                with open(f"{d_lvl}/final_raw.json") as f:
                    final_lvl = np.array(json.load(f))

                lvl_str = ""
                for _, board in enumerate(final_lvl.tolist()):
                    lvl_str += board_to_str(board)
                    lvl_str += "\n\n***\n\n"
                with open(f"{d_lvl}/final.txt", "w", encoding="utf-8") as f:
                    f.write(lvl_str)

                if np.any(final_lvl == ""):
                    success[f"{i}"] = False
                else:
                    success[f"{i}"] = True
                    num_success += 1
                    game_num_success += 1
                    overall_num_success += 1

                    unique_values, counts = np.unique(final_lvl[0], return_counts=True)
                    value_counts = dict(zip(unique_values, counts))
                    if value_counts["P"] > 1:
                        has_extra_P[f"{i}"] = True
                        num_has_extra_P += 1
                        game_num_has_extra_P += 1
                        overall_num_has_extra_P += 1
                    else:
                        has_extra_P[f"{i}"] = False
                if game == "path":
                    if np.any(init_lvl == ""):
                        stats["init_fn"] = "path_no_blank"
                    else:
                        stats["init_fn"] = "path_blank"

                if success[f"{i}"]:
                    t = len(final_lvl) - 2  # -1 for array offset, -1 for padded T
                    final_timestep = final_lvl[t]
                    while np.array_equal(final_lvl[t - 1], final_timestep) and t > 0:
                        t -= 1
                    effective_length[f"{i}"] = t + 1  # +1 for array offset
                    if effective_length[f"{i}"] <= 3:
                        num_trivial += 1
                        game_num_trivial += 1
                        overall_num_trivial += 1
                    if not has_extra_P[f"{i}"]:
                        unbroken_effective_length[f"{i}"] = effective_length[f"{i}"]
                        if effective_length[f"{i}"] <= 3:
                            num_trivial_unbroken += 1

            stats["elapsed"]["average"] = average_time
            stats["success"] = success

            if num_success > 0:
                stats["effective_length"] = effective_length
                stats["has_extra_P"] = has_extra_P
                stats["unbroken_effective_length"] = unbroken_effective_length
                stats["num_trivial_unbroken"] = num_trivial_unbroken
                stats["num_trivial"] = num_trivial
                stats["num_has_extra_P"] = num_has_extra_P
                stats["pct_trivial"] = num_trivial / num_success
                stats["pct_has_extra_P"] = num_has_extra_P / num_success
                stats["effective_length"]["average"] = sum(
                    effective_length.values()
                ) / len(effective_length.values())
                if len(stats["unbroken_effective_length"]) > 0:
                    stats["unbroken_effective_length"]["average"] = sum(
                        unbroken_effective_length.values()
                    ) / len(unbroken_effective_length.values())
                if num_trivial_unbroken > 0:
                    stats["pct_trivial_unbroken"] = num_trivial_unbroken / (
                        num_success - num_has_extra_P
                    )
            with open(f"{d_run}/stats_plus.json", "w", encoding="utf-8") as f:
                json.dump(stats, f)

        game_stats = {
            "game_num_success": game_num_success,
            "game_num_trivial": game_num_trivial,
            "game_pct_trivial": game_num_trivial / game_num_success,
            "game_num_has_extra_P": game_num_has_extra_P,
            "game_pct_has_extra_P": game_num_has_extra_P / game_num_success,
        }
        with open(f"{d_game}/stats_plus.json", "w", encoding="utf-8") as f:
            json.dump(game_stats, f)

    overall_stats = {
        "overall_num_success": overall_num_success,
        "overall_num_trivial": overall_num_trivial,
        "overall_pct_trivial": overall_num_trivial / overall_num_success,
        "overall_num_has_extra_P": overall_num_has_extra_P,
        "overall_pct_has_extra_P": overall_num_has_extra_P / overall_num_success,
    }
    with open(f"{d_results}/stats_plus.json", "w", encoding="utf-8") as f:
        json.dump(overall_stats, f)


if __name__ == "__main__":
    # The folder containing the level json files and stats.json files organized game:run:generation
    # e.g. results_by_game/game/run contains all files generated by one execution of use_coac (saved to src/results/timestamp) (individual level folders with init_raw.json and final_raw.json and the overall stats.json file) The game would be field, maze, etc
    results_folder = f"src/results_by_game"
    process(results_folder)
