# STWFC

To run:
1. Download and install the game_level branch of the wave-function-collapse project https://github.com/flaneuseh/wave-function-collapse/releases/tag/AIIDE
2. Change values in use_coac:__main__ to desired configuration.
3. ```python -m src.use_coac```

Aggregate data by game:
1. Put use_coac results into results_by_game organized by game (follow instructions in src/datavis). Remove existing results if you don't want to include them.
2. ```python -m src.datavis```

## Generated files

init_raw.json: the level after running the initialization function, before padding.
final_raw.json: the final generated level after stwfc has finished
stats.json: stats for one run of use_coac:
```
{
    # example data used for this run
    "train_paths": [
        "AIIDE/training/field/path_3_2_nw.json",
        "AIIDE/training/field/path_1_nw.json"
    ],
    # padding settings from use_coac
    "padding": {
        "pad_width": [
            [
                0,
                1
            ],
            [
                1,
                1
            ],
            [
                1,
                1
            ]
        ],
        "constant_values": [
            [
                "t",
                "T"
            ],
            [
                "_",
                "_"
            ],
            [
                "_",
                "_"
            ]
        ],
        "axis_order": [
            2,
            1,
            0
        ]
    },
    # transforms from use_coac
    "local_transforms": {
        "flipx": true,
        "flipy": true,
        "rotxy": [
            1,
            2,
            3
        ]
    },
    "global_transforms": {
        "flipx": true,
        "flipy": true,
        "rotxy": [
            1,
            2,
            3
        ]
    },
    # generation grid shape
    "grid_size": [
        6,
        4,
        4
    ],
    # pattern shape
    "pattern_size": [
        2,
        3,
        3
    ],
    # Time elapsed per level
    "elapsed": {
        "0": 10.4387948513031,
        "1": 22.81486201286316,
        "2": 3.5963542461395264,
        "3": 15.09929609298706,
        "4": 22.80034112930298,
        "5": 19.0271999835968,
        "6": 12.436919212341309,
        "7": 36.46457600593567,
        "8": 15.97426700592041,
        "9": 30.273711919784546,
        "total": 188.94412994384766
    },
    # number of failures
    "num_fails": "0/10"
}
```
stats_plus.json (in run): stats.json + additional statistics: 
```
{
    (data from stats.json)
    "init_fn": "path_no_blank", # if the game is "path", which path init fn was used.
    # success by level generated
    "success": { 
        "9": true,
        "0": true,
        "7": true,
        "6": true,
        "1": true,
        "8": true,
        "4": true,
        "3": true,
        "2": true,
        "5": true
    },
    # effective length by level
    "effective_length": {
        "9": 6,
        "0": 4,
        "7": 1,
        "6": 1,
        "1": 5,
        "8": 3,
        "4": 6,
        "3": 6,
        "2": 3,
        "5": 3,
        "average": 3.8
    },
    # whether each level has an extra P
    "has_extra_P": {
        "9": false,
        "0": false,
        "7": false,
        "6": false,
        "1": false,
        "8": false,
        "4": false,
        "3": false,
        "2": false,
        "5": false
    },
    # effective length of levels without an extra P
    "unbroken_effective_length": {
        "9": 6,
        "0": 4,
        "7": 1,
        "6": 1,
        "1": 5,
        "8": 3,
        "4": 6,
        "3": 6,
        "2": 3,
        "5": 3,
        "average": 3.8
    },
    "num_trivial_unbroken": 5, # number of "trivial" levels without an extra P
    "num_trivial": 5, # number of "trivial" levels overall
    "num_has_extra_P": 0, # number of levels with an extra P
    "pct_trivial": 0.5, # percent of levels that are trivial
    "pct_has_extra_P": 0.0, # percent with an extra P
    "pct_trivial_unbroken": 0.5 # percent without an extra P that are trivial
}
```
stats_plus.json (not in an individual run): aggregated statistics for all generated levels in contained folders.