# ========================================
# library
# ========================================
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import RobertaTokenizer
import logging
import sys
from contextlib import contextmanager
import time
import random
from tqdm import tqdm
import os
import gc


# ==================
# Constant
# ==================
ex = "237"
TRAIN_PATH = "../data/train.csv"
FOLD_PATH = "../data/fe001_train_folds.csv"
if not os.path.exists(f"../output/ex/ex{ex}"):
    os.makedirs(f"../output/ex/ex{ex}")
    os.makedirs(f"../output/ex/ex{ex}/ex{ex}_model")

MODEL_PATH_BASE = f"../output/ex/ex{ex}/ex{ex}_model/ex{ex}"
OOF_SAVE_PATH = f"../output/ex/ex{ex}/ex{ex}_oof.npy"
LOGGER_PATH = f"../output/ex/ex{ex}/ex{ex}.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============
# Settings
# ===============
SEED = 0
num_workers = 4
BATCH_SIZE = 16
n_epochs = 5
es_patience = 10
max_len = 256
weight_decay = 0.1
lr = 2e-5
num_warmup_steps_rate = 0.1
eval_steps = 20

MODEL_PATH = "../output/ex/ex_mlm_roberta_base/mlm_roberta_base"
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
        super().__init__()

        config = AutoConfig.from_pretrained(MODEL_PATH)
        config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})

        self.roberta = AutoModel.from_pretrained(MODEL_PATH, config=config)

        self.attention = nn.Sequential(
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(768, 1)
        )

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)

        # There are a total of 13 layers of hidden states.
        # 1 for the embedding layer, and 12 for the 12 Roberta layers.
        # We take the hidden states from the last Roberta layer.
        last_layer_hidden_states = roberta_output.hidden_states[-1]

        # The number of cells is MAX_LEN.
        # The size of the hidden state of each cell is 768 (for roberta-base).
        # In order to condense hidden states of all cells to a context vector,
        # we compute a weighted average of the hidden states of all cells.
        # We compute the weight of each cell, using the attention neural network.
        weights = self.attention(last_layer_hidden_states)

        # weights.shape is BATCH_SIZE x MAX_LEN x 1
        # last_layer_hidden_states.shape is BATCH_SIZE x MAX_LEN x 768
        # Now we compute context_vector as the weighted average.
        # context_vector.shape is BATCH_SIZE x 768
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)

        # Now we reduce the context vector to the prediction score.
        return self.regressor(context_vector)


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
with timer("roberta"):
    set_seed(SEED)
    oof = np.zeros([len(train)])
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
            dataset=train_, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(
            dataset=val_, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # model
        model = roberta_model()
        model = model.to(device)

        # optimizer, scheduler
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=lr,
                          betas=(0.9, 0.98),
                          weight_decay=weight_decay,
                          )
        num_train_optimization_steps = int(len(train_loader) * n_epochs)
        num_warmup_steps = int(
            num_train_optimization_steps * num_warmup_steps_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        criterion = nn.MSELoss()
        best_val = None
        patience = es_patience
        for epoch in tqdm(range(n_epochs)):
            with timer(f"model_fold:{epoch}"):

                # train
                model.train()
                train_losses_batch = []

                epoch_loss = 0

                for i, d in enumerate(train_loader):

                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]
                    target = d["target"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    target = target.to(device)
                    optimizer.zero_grad()
                    output = model(input_ids, mask)
                    loss = criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    train_losses_batch.append(loss.item())

                    if i % eval_steps == 0:
                        # val
                        val_losses_batch = []
                        model.eval()  # switch model to the evaluation mode
                        val_preds = np.ndarray((0, 1))
                        with torch.no_grad():
                            # Predicting on validation set
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
                                output = model(input_ids, mask)

                                loss = criterion(output, target)
                                val_preds = np.concatenate(
                                    [val_preds, output.detach().cpu().numpy()], axis=0)
                                val_losses_batch.append(loss.item())

                        val_loss = np.mean(val_losses_batch)
                        val_rmse = calc_loss(y_val, val_preds)
                        LOGGER.info(
                            f'{fold},{epoch}:{i},val_loss:{val_loss},val_rmse:{val_rmse}')
                        # ===================
                        # early stop
                        # ===================

                        if not best_val:
                            best_val = val_loss
                            best_rmse = val_rmse
                            oof[fold_array == fold] = val_preds.reshape(-1)
                            # Saving the model
                            torch.save(model.state_dict(),
                                       MODEL_PATH_BASE + f"_{fold}.pth")
                            continue

                        if val_loss <= best_val:
                            best_val = val_loss
                            best_rmse = val_rmse
                            oof[fold_array == fold] = val_preds.reshape(-1)
                            patience = es_patience
                            # Saving current best model
                            torch.save(model.state_dict(),
                                       MODEL_PATH_BASE + f"_{fold}.pth")
                        # else:
                        #    patience -= 1
                        #    if patience == 0:
                        #        LOGGER.info(f'Early stopping. Best Val : {best_val} Best Rmse : {best_rmse}')
                        #        break
                        model.train()

                train_loss = np.mean(train_losses_batch)
        del model
        gc.collect()

val_rmse = calc_loss(y, oof)
LOGGER.info(f'oof_score:{val_rmse}')
np.save(OOF_SAVE_PATH, oof)
