import numpy as np
from pathlib import Path
import json
from coac_wfc.wfc import WaveFunctionCollapse

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

if __name__ == "__main__":
    np.random.seed(23)

    grid_size = (6, 6, 6)
    pattern_size = (2, 2, 2)

    train_level = []
    soln_folder = f"src/training_data/solutions/blockdude_A"
    soln_file = Path(f"{soln_folder}/solution.json")
    with open(soln_file) as f:
        train_level = np.array(json.load(f))
        train_level = clean_train_level__blockdude(train_level)
       
    wfc = WaveFunctionCollapse(grid_size, train_level, pattern_size)

    wfc.run()

    image = wfc.get_image()

    export_voxel('../samples/output.vox', image)