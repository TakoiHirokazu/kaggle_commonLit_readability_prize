# ========================================
# library
# ========================================
from scipy.optimize import minimize
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import logging
import sys
from contextlib import contextmanager
import time


# ==================
# Constant
# ==================
ex = "_postprocess"
TRAIN_PATH = "../data/train.csv"
FOLD_PATH = "../data/fe001_train_folds.csv"
if not os.path.exists(f"../output/ex/ex{ex}"):
    os.makedirs(f"../output/ex/ex{ex}")

LOGGER_PATH = f"../output/ex/ex{ex}/ex{ex}.txt"

# ===============
# Functions
# ===============


def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s')


LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
setup_logger(out_file=LOGGER_PATH)


# ================================
# Main
# ================================
train = pd.read_csv(TRAIN_PATH)
y = train["target"]
fold_df = pd.read_csv(FOLD_PATH)
fold_array = fold_df["kfold"].values


# ================================
# exp
# ================================
ex15_svr = np.load("../output/ex/ex015/ex015_svr.npy")
ex15_ridge = np.load("../output/ex/ex015/ex015_ridge.npy")
ex15 = (ex15_svr + ex15_ridge) / 2
ex64 = np.load("../output/ex/ex064/ex064_oof.npy")
ex72 = np.load("../output/ex/ex072/ex072_oof.npy")
ex84 = np.load("../output/ex/ex084/ex084_oof.npy")
ex94 = np.load("../output/ex/ex094/ex094_oof.npy")
ex107 = np.load("../output/ex/ex107/ex107_oof.npy")
ex131 = np.load("../output/ex/ex131/ex131_oof.npy")

weight_list = [0.1,           0.05,        0.27,
               0.12,       0.12,      0.20,  0.14]
y_test = (ex15_svr + ex15_ridge) / 2 * weight_list[0] + \
    ex64 * weight_list[1] + \
    ex72 * weight_list[2] + \
    ex107 * weight_list[3] + \
    ex84 * weight_list[4] + \
    ex94 * weight_list[5] + \
    ex131 * weight_list[6]


def f(x):
    pred1 = y_test.copy()
    pred1[y_test >= 0] = y_test[y_test >= 0] * x[0]
    pred1[(y_test < 0) & (y_test >= -1)
          ] = y_test[(y_test < 0) & (y_test >= -1)] * x[1]
    pred1[(y_test < -1) & (y_test >= -2)
          ] = y_test[(y_test < -1) & (y_test >= -2)] * x[2]
    pred1[(y_test < -2)] = y_test[(y_test < -2)] * x[3]
    score = np.sqrt(mean_squared_error(y, pred1))
    return score


with timer("postprocess"):
    result = minimize(f, [1, 1, 1, 1], method="Nelder-Mead")
    LOGGER.info(f'postprocess coefficient:{result.x}')
