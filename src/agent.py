import argparse
import json
import time
from pathlib import Path
from trrbt.game_agent import AgentBlockdudeProcessor


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
    soln_file = Path(f"src/solutions/blockdude_{n}_solution.json")
    with open(soln_file, "w", encoding="utf-8") as f:
        json.dump(soln, f, ensure_ascii=False, indent=4)
    return soln


def find_all_solutions():
    for i in range(12):
        soln = find_solution(i)
        print(soln)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find Blockdude solutions.")
    parser.add_argument("--level", type=int, help="Level number")
    args = parser.parse_args()
    if args.level is not None:
        soln = find_solution(args.level)
        print(soln)
    else:
        find_all_solutions()
