import argparse
import json
import time
from pathlib import Path
from trrbt.game_agent import AgentBlockdudeProcessor
from trrbt.game_viz import ThreadedGameProcessor, run_game, run_game_input_viz
import threading
from src.train import lvl_to_str


def find_solution(n):
    start = time.time()
    level_file = f"../pyrrbt/games/blockdude_levels/blockdude_{n}.yaml"
    game = AgentBlockdudeProcessor(level_file)
    game.game_play()
    end = time.time()
    elapsed = end - start
    time_file = Path(f"src/solutions/blockdude_{n}_time.txt")
    with open(time_file, "w", encoding="utf-8") as f:
        f.write("Time: " + str(elapsed))
        f.write("\nMoves: " + str(game.move_count))
    soln = game.solution
    record_soln(n, soln)
    return soln


def find_all_solutions():
    for i in range(12):
        soln = find_solution(i)
        print(soln)


def record_play_history(n):
    level_file = f"../pyrrbt/games/blockdude_levels/blockdude_{n}.yaml"
    mtx = threading.Lock()
    game_proc = ThreadedGameProcessor(level_file, [], None, mtx)
    game_thread = threading.Thread(target=run_game, args=(game_proc,), daemon=True)

    run_game_input_viz(game_proc, game_thread, 50, None)
    soln = game_proc.history
    record_soln(n, soln)


def record_soln(n, soln):
    soln_folder = f"src/solutions/blockdude_{n}"
    Path(soln_folder).mkdir(parents=True, exist_ok=True)
    soln_file = f"{soln_folder}/solution.json"
    with open(soln_file, "w", encoding="utf-8") as f:
        json.dump(soln, f, ensure_ascii=False, indent=4)
    display_folder = f"{soln_folder}/display"
    Path(display_folder).mkdir(parents=True, exist_ok=True)
    for b, board in enumerate(soln):
        b_str = str(b)
        while len(b_str) < 3:
            b_str = "0" + b_str
        with open(f"{display_folder}/{b_str}.lvl", "w", encoding="utf-8") as f:
            f.write(lvl_to_str(board))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find Blockdude solutions.")
    parser.add_argument("--level", type=str, help="Level number for automatic solver")
    parser.add_argument("--play_level", type=str, help="Level number for manual play")
    args = parser.parse_args()
    if args.play_level is not None:
        soln = record_play_history(args.play_level)
    elif args.level is not None:
        soln = find_solution(args.level)
    else:
        find_all_solutions()
