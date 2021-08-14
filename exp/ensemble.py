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
ex = "_ensemble"
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
ex72 = np.load("../output/ex/ex072/ex072_oof.npy")
ex107 = np.load("../output/ex/ex107/ex107_oof.npy")
ex182 = np.load("../output/ex/ex182/ex182_oof.npy")
ex190 = np.load("../output/ex/ex190/ex190_oof.npy")
ex194 = np.load("../output/ex/ex194/ex194_oof.npy")
ex216 = np.load("../output/ex/ex216/ex216_oof.npy")
ex237 = np.load("../output/ex/ex237/ex237_oof.npy")
ex272 = np.load("../output/ex/ex272/ex272_oof.npy")
ex292 = np.load("../output/ex/ex292/ex292_oof.npy")
ex384 = np.load("../output/ex/ex384/ex384_oof.npy")
ex407 = np.load("../output/ex/ex407/ex407_oof.npy")
ex429 = np.load("../output/ex/ex429/ex429_oof.npy")
ex434 = np.load("../output/ex/ex434/ex434_oof.npy")
ex448 = np.load("../output/ex/ex448/ex448_oof.npy")
ex450 = np.load("../output/ex/ex450/ex450_oof.npy")
ex450[fold_array == 2] = ex448[fold_array == 2]
ex465 = np.load("../output/ex/ex465/ex465_oof.npy")
ex497 = np.load("../output/ex/ex497/ex497_oof.npy")
ex507 = np.load("../output/ex/ex507/ex507_oof.npy")


def f(x):
    pred1 = (ex15 + ex237)/2 * x[0] + (ex72 * 0.8 + ex384 * 0.2) * x[1] + ex107 * x[2] + (ex190 + ex272) / 2 * x[3] + ex182 * x[4] +  \
        ex194 * x[5] + ex292 * x[6] + ex216 * x[7] + ex407 * x[8] + ex429 * x[9] + ex450 * x[10] + ex465 * x[11] + ex434 * x[12] + \
        ex497 * x[13] + ex507 * x[14]
    score = np.sqrt(mean_squared_error(y, pred1))
    return score


with timer("ensemble"):
    weight_init = [1 / 15 for _ in range(15)]
    result = minimize(f, weight_init, method="Nelder-Mead")
    LOGGER.info(f'ensemble_weight:{result.x}')
