# ========================================
# library
# ========================================
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel
from transformers import RobertaTokenizer
import logging
import sys
from contextlib import contextmanager
import time
import random
import os
import pickle
from sklearn.svm import SVR
from sklearn.linear_model import Ridge


# ==================
# Constant
# ==================
ex = "015"
TRAIN_PATH = "../data/train.csv"
FOLD_PATH = "../data/fe001_train_folds.csv"
if not os.path.exists(f"../output/ex/ex{ex}"):
    os.makedirs(f"../output/ex/ex{ex}")
    os.makedirs(f"../output/ex/ex{ex}/ex{ex}_model")

MODEL_PATH_BASE = f"../output/ex/ex{ex}/ex{ex}_model/ex{ex}"
LOGGER_PATH = f"../output/ex/ex{ex}/ex{ex}.txt"
OOF_RIDGE_SAVE_PATH = f"../output/ex/ex{ex}/ex{ex}_ridge.npy"
OOF_SVR_SAVE_PATH = f"../output/ex/ex{ex}/ex{ex}_svr.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============
# Settings
# ===============
SEED = 0
num_workers = 4
BATCH_SIZE = 24
max_len = 256
MODEL_PATH = '../models/roberta/roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)


# ===============
# Functions
# ===============

class CommonLitDataset(Dataset):
    def __init__(self, excerpt, tokenizer, max_len, target=None):
        self.excerpt = excerpt
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target = target

    def __len__(self):
        return len(self.excerpt)

    def __getitem__(self, item):
        text = str(self.excerpt[item])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        if self.target is not None:
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "target": torch.tensor(self.target[item], dtype=torch.float32)
            }
        else:
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)
            }


class roberta_model(nn.Module):
    def __init__(self):
        super(roberta_model, self).__init__()
        self.roberta = RobertaModel.from_pretrained(
            MODEL_PATH,
        )
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(768, 256)
        self.layernorm = nn.LayerNorm(256)
        self.drop2 = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 1)

    def forward(self, ids, mask, token_type_ids):
        # pooler
        emb = self.roberta(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            'pooler_output']
        output = self.drop(emb)
        output = self.fc(output)
        output = self.layernorm(output)
        output = self.drop2(output)
        output = self.relu(output)
        output = self.out(output)
        return output, emb


def calc_loss(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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
# train
# ================================
with timer("svr + ridge"):
    set_seed(SEED)
    oof_svr = np.zeros([len(train)])
    oof_ridge = np.zeros([len(train)])
    for fold in range(5):
        x_train, y_train = train.iloc[fold_array !=
                                      fold], y.iloc[fold_array != fold]
        x_val, y_val = train.iloc[fold_array ==
                                  fold], y.iloc[fold_array == fold]

        # dataset
        train_ = CommonLitDataset(
            x_train["excerpt"].values, tokenizer, max_len, y_train.values.reshape(-1, 1))
        val_ = CommonLitDataset(
            x_val["excerpt"].values, tokenizer, max_len, y_val.values.reshape(-1, 1))

        # loader
        train_loader = DataLoader(
            dataset=train_, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        val_loader = DataLoader(
            dataset=val_, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # model
        model = roberta_model()
        model.load_state_dict(torch.load(
            f"../output/ex/ex014/ex014_model/ex014_{fold}.pth"))
        model.to(device)
        model.eval()

        # make embedding
        train_emb = np.ndarray((0, 768))
        val_emb = np.ndarray((0, 768))

        # train
        with torch.no_grad():
            for d in train_loader:
                # =========================
                # data loader
                # =========================
                input_ids = d['input_ids']
                mask = d['attention_mask']
                token_type_ids = d["token_type_ids"]
                target = d["target"]

                input_ids = input_ids.to(device)
                mask = mask.to(device)
                token_type_ids = token_type_ids.to(device)
                target = target.to(device)
                _, emb = model(input_ids, mask, token_type_ids)
                train_emb = np.concatenate(
                    [train_emb, emb.detach().cpu().numpy()], axis=0)

        # val
        with torch.no_grad():
            for d in val_loader:
                # =========================
                # data loader
                # =========================
                input_ids = d['input_ids']
                mask = d['attention_mask']
                token_type_ids = d["token_type_ids"]
                target = d["target"]

                input_ids = input_ids.to(device)
                mask = mask.to(device)
                token_type_ids = token_type_ids.to(device)
                target = target.to(device)
                _, emb = model(input_ids, mask, token_type_ids)

                val_emb = np.concatenate(
                    [val_emb, emb.detach().cpu().numpy()], axis=0)

        x_train = pd.DataFrame(train_emb)
        x_val = pd.DataFrame(val_emb)
        # svr
        model_svr = SVR(C=10, kernel="rbf", gamma='auto')
        model_svr.fit(x_train, y_train)
        pred = model_svr.predict(x_val)
        oof_svr[fold_array == fold] = pred
        score = calc_loss(y_val, pred)
        LOGGER.info(f"fold_svr:{fold}:{score}")
        save_path = MODEL_PATH_BASE + f"_svr_roberta_emb_{fold}.pkl"
        pickle.dump(model_svr, open(save_path, 'wb'))

        # ridge
        ridge = Ridge(alpha=1)
        ridge.fit(x_train, y_train)
        pred = ridge.predict(x_val)
        oof_ridge[fold_array == fold] = pred
        score = calc_loss(y_val, pred)
        LOGGER.info(f"fold_ridge:{fold}:{score}")
        save_path = MODEL_PATH_BASE + f"_ridge_roberta_emb_{fold}.pkl"
        pickle.dump(ridge, open(save_path, 'wb'))


val_rmse = calc_loss(y, oof_svr)
LOGGER.info(f'svr_oof_score:{val_rmse}')
val_rmse = calc_loss(y, oof_ridge)
LOGGER.info(f'ridge_oof_score:{val_rmse}')
np.save(OOF_SVR_SAVE_PATH, oof_svr)
np.save(OOF_RIDGE_SAVE_PATH, oof_ridge)
