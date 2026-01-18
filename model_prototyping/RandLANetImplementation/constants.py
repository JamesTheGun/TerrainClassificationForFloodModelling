# MODEL PARAMS:
# Large test sets because we can label as much data as we want and our positive class is pretty damn rare
TRAIN_PER_HOLDOUT = 5
TRAIN_PER_TEST_FOLD = 200
TRAIN_FOLD_TO_TEST_FOLD_RATIO = 0.6
HYPER_PARAMETERS = {}

# SLIDING WINDOW PARAMS:
STRIDE = 30
WINDOW_SIZE = 500

NIEGBBOUR_COUNT = 4

KEEP_DIMS = ["x", "y", "z", "intensity"]