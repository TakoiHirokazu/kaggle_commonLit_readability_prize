# ========================================
# library
# ========================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import transformers
from transformers import RobertaModel, RobertaTokenizer
from transformers import AlbertModel, AlbertTokenizer
from transformers import DebertaModel, DebertaTokenizer
from transformers import ElectraModel, ElectraTokenizer, ElectraForSequenceClassification
from transformers import BartModel, BertTokenizer
from transformers import MPNetModel, MPNetTokenizer
from transformers import FunnelBaseModel, FunnelTokenizer, FunnelModel
from transformers import GPT2Model, GPT2Tokenizer
from transformers import T5EncoderModel, T5Tokenizer
import logging
import sys
from contextlib import contextmanager
import time
from tqdm import tqdm
import pickle
import gc


# ==================
# Constant
# ==================
ex = "_predict"
TEST_PATH = "../data/test.csv"
SUB_PATH = "../data/sample_submission.csv"
SAVE_PATH = "../output/submission.csv"
LOGGER_PATH = f"ex{ex}.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============
# Settings
# ===============
BATCH_SIZE = 8
max_len = 256

roberta_large_MODEL_PATH = '../models/roberta/roberta-large'
roberta_large_tokenizer = RobertaTokenizer.from_pretrained(
    roberta_large_MODEL_PATH)

roberta_base_MODEL_PATH = '../models/roberta/roberta-base'
roberta_base_tokenizer = RobertaTokenizer.from_pretrained(
    roberta_base_MODEL_PATH)

roberta_base_MODEL_PATH2 = '../output/ex/ex_mlm_roberta_base/mlm_roberta_base'
roberta_base_tokenizer2 = AutoTokenizer.from_pretrained(
    roberta_base_MODEL_PATH2)

deberta_large_MODEL_PATH = "../models/deberta/large"
deberta_large_tokenizer = DebertaTokenizer.from_pretrained(
    deberta_large_MODEL_PATH)

electra_large_MODEL_PATH = "../models/electra/large-discriminator"
electra_large_tokenizer = ElectraTokenizer.from_pretrained(
    electra_large_MODEL_PATH)

bart_large_MODEL_PATH = '../models/bart/bart-large'
bart_large_tokenizer = RobertaTokenizer.from_pretrained(
    roberta_large_MODEL_PATH)

deberta_xlarge_MODEL_PATH = "../models/deberta/v2-xlarge"
deberta_xlarge_tokenizer = AutoTokenizer.from_pretrained(
    deberta_xlarge_MODEL_PATH)

mpnet_base_MODEL_PATH = 'microsoft/mpnet-base'
mpnet_base_tokenizer = MPNetTokenizer.from_pretrained(mpnet_base_MODEL_PATH)

deberta_v2_xxlarge_MODEL_PATH = "../models/deberta/v2-xxlarge"
deberta_v2_xxlarge_tokenizer = AutoTokenizer.from_pretrained(
    deberta_v2_xxlarge_MODEL_PATH)

funnel_large_base_MODEL_PATH = 'funnel-transformer/large-base'
funnel_large_base_tokenizer = FunnelTokenizer.from_pretrained(
    funnel_large_base_MODEL_PATH)

muppet_roberta_large_MODEL_PATH = 'facebook/muppet-roberta-large'
muppet_roberta_large_tokenizer = RobertaTokenizer.from_pretrained(
    muppet_roberta_large_MODEL_PATH)

funnel_large_MODEL_PATH = 'funnel-transformer/large'
funnel_large_tokenizer = FunnelTokenizer.from_pretrained(
    funnel_large_MODEL_PATH)

gpt2_medium_MODEL_PATH = "gpt2-medium"
gpt2_medium_tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2-medium", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
gpt2_medium_tokenizer.pad_token = gpt2_medium_tokenizer.eos_token

albert_v2_xxlarge_MODEL_PATH = 'albert-xxlarge-v2'
albert_v2_xxlarge_tokenizer = AlbertTokenizer.from_pretrained(
    albert_v2_xxlarge_MODEL_PATH)

electra_base_MODEL_PATH = "../models/electra/base-discriminator"
electra_base_tokenizer = ElectraTokenizer.from_pretrained(
    electra_base_MODEL_PATH)

bert_base_uncased_MODEL_PATH = 'bert-base-uncased'
bert_base_uncased_tokenizer = BertTokenizer.from_pretrained(
    bert_base_uncased_MODEL_PATH)

t5_large_MODEL_PATH = 't5-large'
t5_large_tokenizer = T5Tokenizer.from_pretrained(t5_large_MODEL_PATH)

distil_bart_MODEL_PATH = 'sshleifer/distilbart-cnn-12-6'
distil_bart_tokenizer = RobertaTokenizer.from_pretrained(
    distil_bart_MODEL_PATH)


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


class roberta_large_model(nn.Module):
    def __init__(self):
        super(roberta_large_model, self).__init__()
        self.roberta = RobertaModel.from_pretrained(
            roberta_large_MODEL_PATH,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0
        )

        # self.dropout = nn.Dropout(p=0.2)
        self.ln = nn.LayerNorm(1024)
        self.out = nn.Linear(1024, 1)

    def forward(self, ids, mask, token_type_ids):
        # pooler
        emb = self.roberta(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            "last_hidden_state"]
        emb = torch.mean(emb, axis=1)
        output = self.ln(emb)
        # output = self.dropout(output)
        output = self.out(output)
        return output


class roberta_base_model(nn.Module):
    def __init__(self):
        super(roberta_base_model, self).__init__()
        self.roberta = RobertaModel.from_pretrained(
            roberta_base_MODEL_PATH,
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


class roberta_base_model2(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(roberta_base_MODEL_PATH2)
        config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})

        self.roberta = AutoModel.from_pretrained(
            roberta_base_MODEL_PATH, config=config)

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

        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)
        return self.regressor(context_vector)


class deberta_large_model(nn.Module):
    def __init__(self):
        super(deberta_large_model, self).__init__()
        self.deberta_model = DebertaModel.from_pretrained(deberta_large_MODEL_PATH,
                                                          hidden_dropout_prob=0,
                                                          attention_probs_dropout_prob=0,
                                                          hidden_act="gelu_new")

        # self.dropout = nn.Dropout(p=0.2)
        self.ln = nn.LayerNorm(1024)
        self.out = nn.Linear(1024, 1)

    def forward(self, ids, mask, token_type_ids):
        # pooler
        emb = self.deberta_model(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            'last_hidden_state'][:, 0, :]
        output = self.ln(emb)
        # output = self.dropout(output)
        output = self.out(output)
        return output


class electra_large_model(nn.Module):
    def __init__(self):
        super(electra_large_model, self).__init__()
        self.electra = ElectraForSequenceClassification.from_pretrained(
            electra_large_MODEL_PATH,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
            summary_last_dropout=0,
            num_labels=1
        )

    def forward(self, ids, mask, token_type_ids):
        # pooler
        output = self.electra(ids, attention_mask=mask,
                              token_type_ids=token_type_ids)["logits"]
        return output


class bart_large_model(nn.Module):
    def __init__(self):
        super(bart_large_model, self).__init__()
        self.bart = BartModel.from_pretrained(
            bart_large_MODEL_PATH,
            dropout=0.0, attention_dropout=0.0
        )

        # self.dropout = nn.Dropout(p=0.2)
        self.ln = nn.LayerNorm(1024)
        self.out = nn.Linear(1024, 1)

    def forward(self, ids, mask):
        # pooler
        emb = self.bart(ids, attention_mask=mask)['last_hidden_state']
        emb = torch.mean(emb, axis=1)
        output = self.ln(emb)
        # output = self.dropout(output)
        output = self.out(output)
        return output


class deberta_xlarge_model(nn.Module):
    def __init__(self):
        super(deberta_xlarge_model, self).__init__()
        self.deberta_model = AutoModel.from_pretrained(deberta_xlarge_MODEL_PATH,
                                                       hidden_dropout_prob=0,
                                                       attention_probs_dropout_prob=0)

        # self.dropout = nn.Dropout(p=0.2)
        # self.ln = nn.LayerNorm(1536)
        self.out = nn.Linear(1536, 1)

    def forward(self, ids, mask, token_type_ids):
        # pooler
        emb = self.deberta_model(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            'last_hidden_state'][:, 0, :]
        # output = self.ln(emb)
        # output = self.dropout(output)
        output = self.out(emb)
        return output


class mpnet_base_model(nn.Module):
    def __init__(self):
        super(mpnet_base_model, self).__init__()
        self.mpnet = MPNetModel.from_pretrained(
            mpnet_base_MODEL_PATH,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0
        )

        # self.dropout = nn.Dropout(p=0.2)
        self.ln = nn.LayerNorm(768)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        # pooler
        emb = self.mpnet(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            "last_hidden_state"]
        emb = torch.mean(emb, axis=1)
        output = self.ln(emb)
        # output = self.dropout(output)
        output = self.out(output)
        return output


class deberta_v2_xxlarge_model(nn.Module):
    def __init__(self):
        super(deberta_v2_xxlarge_model, self).__init__()
        self.deberta_model = AutoModel.from_pretrained(deberta_v2_xxlarge_MODEL_PATH,
                                                       hidden_dropout_prob=0,
                                                       attention_probs_dropout_prob=0)

        # self.dropout = nn.Dropout(p=0.2)
        # self.ln = nn.LayerNorm(1536)
        self.out = nn.Linear(1536, 1)

    def forward(self, ids, mask, token_type_ids):
        # pooler
        emb = self.deberta_model(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            'last_hidden_state'][:, 0, :]
        # output = self.ln(emb)
        # output = self.dropout(output)
        output = self.out(emb)
        return output


class funnel_large_base_model(nn.Module):
    def __init__(self):
        super(funnel_large_base_model, self).__init__()
        self.funnel = FunnelBaseModel.from_pretrained(
            funnel_large_base_MODEL_PATH,
            hidden_dropout=0,
            attention_dropout=0,
            hidden_act="gelu"
        )

        # self.dropout = nn.Dropout(p=0.2)
        self.ln = nn.LayerNorm(1024)
        self.out = nn.Linear(1024, 1)

    def forward(self, ids, mask, token_type_ids):
        # pooler
        emb = self.funnel(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            "last_hidden_state"]
        emb = torch.mean(emb, axis=1)
        # output = self.ln(emb)
        # output = self.dropout(output)
        output = self.out(emb)
        return output


class muppet_roberta_large_model(nn.Module):
    def __init__(self):
        super(muppet_roberta_large_model, self).__init__()
        self.roberta = RobertaModel.from_pretrained(
            muppet_roberta_large_MODEL_PATH,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0
        )

        # self.dropout = nn.Dropout(p=0.2)
        self.ln = nn.LayerNorm(1024)
        self.out = nn.Linear(1024, 1)

    def forward(self, ids, mask, token_type_ids):
        # pooler
        emb = self.roberta(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            "last_hidden_state"]
        emb = torch.mean(emb, axis=1)
        output = self.ln(emb)
        # output = self.dropout(output)
        output = self.out(output)
        return output


class funnel_large_model(nn.Module):
    def __init__(self):
        super(funnel_large_model, self).__init__()
        self.funnel = FunnelModel.from_pretrained(
            funnel_large_MODEL_PATH,
            hidden_dropout=0,
            attention_dropout=0
        )

        # self.dropout = nn.Dropout(p=0.2)
        # self.ln = nn.LayerNorm(1024)
        self.out = nn.Linear(1024, 1)

    def forward(self, ids, mask, token_type_ids):
        # pooler
        emb = self.funnel(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            "last_hidden_state"]
        emb = torch.mean(emb, axis=1)
        # output = self.ln(emb)
        # output = self.dropout(output)
        output = self.out(emb)
        return output


class gpt2_medium_model(nn.Module):
    def __init__(self):
        super(gpt2_medium_model, self).__init__()
        self.gpt2_model = GPT2Model.from_pretrained(gpt2_medium_MODEL_PATH,
                                                    attn_pdrop=0,
                                                    embd_pdrop=0,
                                                    resid_pdrop=0,
                                                    summary_first_dropout=0)
        self.gpt2_model.resize_token_embeddings(len(gpt2_medium_tokenizer))

        # self.dropout = nn.Dropout(p=0.2)
        self.ln = nn.LayerNorm(1024)
        self.out = nn.Linear(1024, 1)

    def forward(self, ids, mask):
        # pooler
        emb = self.gpt2_model(ids, attention_mask=mask)["last_hidden_state"]
        emb = torch.mean(emb, axis=1)
        output = self.ln(emb)
        # output = self.dropout(output)
        output = self.out(output)
        return output


class albert_v2_xxlarge_model(nn.Module):
    def __init__(self):
        super(albert_v2_xxlarge_model, self).__init__()
        self.albert = AlbertModel.from_pretrained(
            albert_v2_xxlarge_MODEL_PATH,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0
        )

        # self.dropout = nn.Dropout(p=0.2)
        self.ln = nn.LayerNorm(4096)
        self.out = nn.Linear(4096, 1)

    def forward(self, ids, mask, token_type_ids):
        # pooler
        emb = self.albert(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            "last_hidden_state"]
        emb = torch.mean(emb, axis=1)
        output = self.ln(emb)
        # output = self.dropout(output)
        output = self.out(output)
        return output


class electra_base_model(nn.Module):
    def __init__(self):
        super(electra_base_model, self).__init__()
        self.electra = ElectraModel.from_pretrained(
            electra_base_MODEL_PATH,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0
        )

        # self.dropout = nn.Dropout(p=0.2)
        self.ln = nn.LayerNorm(768)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        # pooler
        emb = self.electra(ids, attention_mask=mask, token_type_ids=token_type_ids)[
            "last_hidden_state"]
        emb = torch.mean(emb, axis=1)
        output = self.ln(emb)
        # output = self.dropout(output)
        output = self.out(output)
        return output


class bert_base_uncased_model(nn.Module):
    def __init__(self):
        super(bert_base_uncased_model, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(bert_base_uncased_MODEL_PATH,
                                                           hidden_dropout_prob=0,
                                                           attention_probs_dropout_prob=0)
        # self.bert = transformers.BertForSequenceClassification.from_pretrained(BERT_MODEL,num_labels=1)
        self.ln = nn.LayerNorm(768)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        # pooler
        emb, _ = self.bert(ids, attention_mask=mask,
                           token_type_ids=token_type_ids, return_dict=False)
        emb = torch.mean(emb, axis=1)
        output = self.ln(emb)
        output = self.out(output)
        return output


class t5_large_model(nn.Module):
    def __init__(self):
        super(t5_large_model, self).__init__()
        self.t5 = T5EncoderModel.from_pretrained(t5_large_MODEL_PATH,
                                                 dropout_rate=0)

        # self.dropout = nn.Dropout(p=0.2)
        self.ln = nn.LayerNorm(1024)
        self.out = nn.Linear(1024, 1)

    def forward(self, ids, mask):
        # pooler
        emb = self.t5(ids, attention_mask=mask)['last_hidden_state']
        emb = torch.mean(emb, axis=1)
        output = self.ln(emb)
        # output = self.dropout(output)
        output = self.out(output)
        return output


class distil_bart_model(nn.Module):
    def __init__(self):
        super(distil_bart_model, self).__init__()
        self.bart = BartModel.from_pretrained(
            distil_bart_MODEL_PATH,
            activation_dropout=0.0, attention_dropout=0.0,
            classif_dropout=0, classifier_dropout=0
        )

        # self.dropout = nn.Dropout(p=0.2)
        self.ln = nn.LayerNorm(1024)
        self.out = nn.Linear(1024, 1)

    def forward(self, ids, mask):
        # pooler
        emb = self.bart(ids, attention_mask=mask)['last_hidden_state']
        emb = torch.mean(emb, axis=1)
        output = self.ln(emb)
        # output = self.dropout(output)
        output = self.out(output)
        return output


class CommonLitDataset_gpt(Dataset):
    def __init__(self, excerpt, tokenizer, max_len, target=None):
        self.excerpt = excerpt
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target = target

    def __len__(self):
        return len(self.excerpt)

    def __getitem__(self, item):
        text = str(self.excerpt[item])
        inputs = self.tokenizer('<|startoftext|>' + text + '<|endoftext|>',
                                truncation=True, max_length=self.max_len, padding="max_length")
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        # token_type_ids = inputs["token_type_ids"]
        if self.target is not None:
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                # "token_type_ids" : torch.tensor(token_type_ids, dtype=torch.long),
                "target": torch.tensor(self.target[item], dtype=torch.float32)
            }
        else:
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                # "token_type_ids" : torch.tensor(token_type_ids, dtype=torch.long)
            }


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
test = pd.read_csv(TEST_PATH)


# ================================
# roberta base -> svr + ridge
# ================================
if len(test) > 0:
    with timer("roberta base -> svr + ridge"):
        y_test_roberta_base = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, roberta_base_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        for fold in range(5):

            # model
            model = roberta_base_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex014/ex014_model/ex014_{fold}.pth"))
            model.to(device)
            model.eval()
            test_emb = np.ndarray((0, 768))

            # svr
            svr = pickle.load(
                open(f"../output/ex/ex015/ex015_model/ex015_svr_roberta_emb_{fold}.pkl", "rb"))

            # ridge
            ridge = pickle.load(
                open(f"../output/ex/ex015/ex015_model/ex015_ridge_roberta_emb_{fold}.pkl", "rb"))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    _, output = model(input_ids, mask, token_type_ids)

                    test_emb = np.concatenate(
                        [test_emb, output.detach().cpu().numpy()], axis=0)
            x_test = pd.DataFrame(test_emb)
            x_test.columns = [f"emb_{i}" for i in range(len(x_test.columns))]
            test_preds_svr = svr.predict(x_test)
            test_preds_ridge = ridge.predict(x_test)
            test_preds = (test_preds_svr + test_preds_ridge)/2
            y_test_roberta_base.append(test_preds)
            del x_test, model, test_emb
            gc.collect()

        y_test_roberta_base = np.mean(y_test_roberta_base, axis=0)
        del test_, test_loader
        gc.collect()


# ================================
# roberta base
# ================================
if len(test) > 0:
    with timer("roberta base"):
        y_test_roberta_base2 = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, roberta_base_tokenizer2, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        for fold in range(5):

            # model
            model = roberta_base_model2()
            model.load_state_dict(torch.load(
                f"../output/ex/ex237/ex237_model/ex237_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            y_test_roberta_base2.append(test_preds)
            del model
            gc.collect()

        y_test_roberta_base2 = np.mean(y_test_roberta_base2, axis=0)
        del test_, test_loader
        gc.collect()


# ================================
# roberta_large
# ================================
if len(test) > 0:
    with timer("roberta_large"):
        y_test_roberta_large = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, roberta_large_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        for fold in tqdm(range(5)):

            # model
            model = roberta_large_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex072/ex072_model/ex072_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask, token_type_ids)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            y_test_roberta_large.append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        y_test_roberta_large = np.mean(y_test_roberta_large, axis=0)


# ================================
# deberta_large
# ================================
if len(test) > 0:
    with timer("deberta_large"):
        y_test_deberta_large = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, deberta_large_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        for fold in tqdm(range(5)):

            # model
            model = deberta_large_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex182/ex182_model/ex182_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask, token_type_ids)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            y_test_deberta_large .append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        y_test_deberta_large = np.mean(y_test_deberta_large, axis=0)


# ================================
# electra_large
# ================================
if len(test) > 0:
    with timer("electra_largee"):
        y_test_electra_large = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, electra_large_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        for fold in tqdm(range(5)):

            # model
            model = electra_large_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex190/ex190_model/ex190_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask, token_type_ids)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            y_test_electra_large.append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        y_test_electra_large = np.mean(y_test_electra_large, axis=0)


# ================================
# bart_large
# ================================
if len(test) > 0:
    with timer("bart_largee"):
        y_test_bart_large = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, bart_large_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        for fold in tqdm(range(5)):

            # model
            model = bart_large_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex107/ex107_model/ex107_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            y_test_bart_large.append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        y_test_bart_large = np.mean(y_test_bart_large, axis=0)


# ================================
# deberta_xlarge
# ================================
if len(test) > 0:
    with timer("deberta_xlarge"):
        y_test_deberta_xlarge = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, deberta_xlarge_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=4, shuffle=False, num_workers=2)

        for fold in tqdm(range(5)):

            # model
            model = deberta_xlarge_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex194/ex194_model/ex194_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask, token_type_ids)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            y_test_deberta_xlarge .append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        y_test_deberta_xlarge = np.mean(y_test_deberta_xlarge, axis=0)


# ================================
# mpnet_base
# ================================
if len(test) > 0:
    with timer("mpnet_base"):
        y_test_mpnet_base = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, mpnet_base_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        for fold in tqdm(range(5)):

            # model
            model = mpnet_base_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex292/ex292_model/ex292_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]
                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask, token_type_ids)
                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            y_test_mpnet_base.append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        y_test_mpnet_base = np.mean(y_test_mpnet_base, axis=0)


# ================================
# deberta_v2_xxlarge
# ================================
if len(test) > 0:
    with timer("deberta_v2_xlarge"):
        y_test_deberta_v2_xxlarge = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, deberta_v2_xxlarge_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=4, shuffle=False, num_workers=2)

        for fold in tqdm(range(5)):

            # model
            model = deberta_v2_xxlarge_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex216/ex216_model/ex216_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask, token_type_ids)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            y_test_deberta_v2_xxlarge.append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        y_test_deberta_v2_xxlarge = np.mean(y_test_deberta_v2_xxlarge, axis=0)


# ================================
# funnel_large_base
# ================================
if len(test) > 0:
    with timer("funnel_large_base"):
        y_test_funnel_large_base = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, funnel_large_base_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=4, shuffle=False, num_workers=2)

        for fold in tqdm(range(5)):

            # model
            model = funnel_large_base_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex272/ex272_model/ex272_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask, token_type_ids)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            y_test_funnel_large_base.append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        y_test_funnel_large_base = np.mean(y_test_funnel_large_base, axis=0)

# ================================
# muppet_roberta_large
# ================================
if len(test) > 0:
    with timer("muppet_roberta_large"):
        y_test_muppet_roberta_large = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, muppet_roberta_large_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        for fold in tqdm(range(5)):

            # model
            model = muppet_roberta_large_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex384/ex384_model/ex384_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask, token_type_ids)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            y_test_muppet_roberta_large.append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        y_test_muppet_roberta_large = np.mean(
            y_test_muppet_roberta_large, axis=0)


# ================================
# funnel large
# ================================
if len(test) > 0:
    with timer("funnel_model"):
        y_test_funnel_large = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, funnel_large_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        for fold in tqdm(range(5)):

            # model
            model = funnel_large_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex407/ex407_model/ex407_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask, token_type_ids)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            y_test_funnel_large.append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        y_test_funnel_large = np.mean(y_test_funnel_large, axis=0)


# ================================
# gpt_medium
# ================================
if len(test) > 0:
    with timer("gpt_medium"):
        y_test_gpt2_medium = []
        # dataset
        test_ = CommonLitDataset_gpt(
            test["excerpt"].values, gpt2_medium_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        for fold in tqdm(range(5)):

            # model
            model = gpt2_medium_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex429/ex429_model/ex429_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    # token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    # token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            y_test_gpt2_medium.append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        y_test_gpt2_medium = np.mean(y_test_gpt2_medium, axis=0)


# ================================
# albert_v2_xxlarge_model
# ================================
if len(test) > 0:
    with timer("albert_v2_xxlarge_model"):
        y_test_albert_v2_xxlarge = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, albert_v2_xxlarge_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        for fold in tqdm(range(5)):

            # model
            model = albert_v2_xxlarge_model()
            if fold == 2:
                model.load_state_dict(torch.load(
                    f"../output/ex/ex448/ex448_model/ex448_{fold}.pth"))
            else:
                model.load_state_dict(torch.load(
                    f"../output/ex/ex450/ex450_model/ex450_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask, token_type_ids)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            y_test_albert_v2_xxlarge.append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        y_test_albert_v2_xxlarge = np.mean(y_test_albert_v2_xxlarge, axis=0)


# ================================
# ex465 electra_base_model
# ================================
if len(test) > 0:
    with timer("electra_base_model"):
        ex465_pred = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, electra_base_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        for fold in tqdm(range(5)):

            # model
            model = electra_base_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex465/ex465_model/ex465_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask, token_type_ids)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            ex465_pred.append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        ex465_pred = np.mean(ex465_pred, axis=0)


# ================================
# ex497 bert_base_uncased_model
# ================================
if len(test) > 0:
    with timer("bert_base_uncased_model"):
        ex497_pred = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, bert_base_uncased_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        for fold in tqdm(range(5)):

            # model
            model = bert_base_uncased_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex497/ex497_model/ex497_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask, token_type_ids)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            ex497_pred.append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        ex497_pred = np.mean(ex497_pred, axis=0)


# ================================
# ex434 t5_large_model
# ================================
if len(test) > 0:
    with timer("t5_large"):
        ex434_pred = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, t5_large_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        for fold in tqdm(range(5)):

            # model
            model = t5_large_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex434/ex434_model/ex434_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            ex434_pred.append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        ex434_pred = np.mean(ex434_pred, axis=0)


# ================================
# distil_bart
# ================================
if len(test) > 0:
    with timer("distil_bart"):
        ex507_pred = []
        # dataset
        test_ = CommonLitDataset(
            test["excerpt"].values, distil_bart_tokenizer, max_len, None)

        # loader
        test_loader = DataLoader(
            dataset=test_, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        for fold in tqdm(range(5)):

            # model
            model = distil_bart_model()
            model.load_state_dict(torch.load(
                f"../output/ex/ex507/ex507_model/ex507_{fold}.pth"))
            model.to(device)
            model.eval()
            test_preds = np.ndarray((0, 1))

            with torch.no_grad():
                # Predicting on validation set
                for d in test_loader:
                    # =========================
                    # data loader
                    # =========================
                    input_ids = d['input_ids']
                    mask = d['attention_mask']
                    token_type_ids = d["token_type_ids"]

                    input_ids = input_ids.to(device)
                    mask = mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    output = model(input_ids, mask)

                    test_preds = np.concatenate(
                        [test_preds, output.detach().cpu().numpy()], axis=0)
            ex507_pred.append(test_preds)
            del model
            gc.collect()
        del test_, test_loader
        gc.collect()
        ex507_pred = np.mean(ex507_pred, axis=0)


weight_list = [0.08, 0.00, 0.11, 0.09, 0.10, 0.23, 0.05,
               0.13, 0.14, 0.11, 0.17, 0.12, -0.17, -0.14, -0.11, 0.09]

if len(test) > 0:
    y_test = (y_test_roberta_base + y_test_roberta_base2.reshape(-1)) / 2 * weight_list[0] + \
             (y_test_roberta_large.reshape(-1) * 0.8 + y_test_muppet_roberta_large.reshape(-1) * 0.2) * weight_list[2] +\
        y_test_bart_large.reshape(-1) * weight_list[3] +\
             (y_test_electra_large.reshape(-1) + y_test_funnel_large_base.reshape(-1)) / 2 * weight_list[4] +\
        y_test_deberta_large.reshape(-1) * weight_list[5] +\
        y_test_deberta_xlarge.reshape(-1) * weight_list[6] +\
        y_test_mpnet_base.reshape(-1) * weight_list[7] +\
        y_test_deberta_v2_xxlarge.reshape(-1) * weight_list[8] +\
        y_test_funnel_large.reshape(-1) * weight_list[9] +\
        y_test_gpt2_medium.reshape(-1) * weight_list[10] +\
        y_test_albert_v2_xxlarge.reshape(-1) * weight_list[11] +\
        ex465_pred.reshape(-1) * weight_list[12] +\
        ex497_pred.reshape(-1) * weight_list[13] +\
        ex434_pred.reshape(-1) * weight_list[14] +\
        ex507_pred.reshape(-1) * weight_list[15]
else:
    y_test = np.zeros(len(test))
submission = pd.read_csv(
    SUB_PATH)
submission.target = y_test
submission.loc[(y_test >= 0.3), "target"] = y_test[(y_test >= 0.3)] * 1.07
submission.loc[(y_test < 0.3) & (y_test >= 0),
               "target"] = y_test[(y_test < 0.3) & (y_test >= 0)] * 1.2
submission.loc[(y_test < 0) & (y_test >= -0.7),
               "target"] = y_test[(y_test < 0) & (y_test >= -0.7)] * 0.97485814
submission.loc[(y_test < -0.7) & (y_test >= -0.9),
               "target"] = y_test[(y_test < -0.7) & (y_test >= -0.9)] * 1.01
submission.loc[(y_test < -0.9) & (y_test >= -2),
               "target"] = y_test[(y_test < -0.9) & (y_test >= -2)] * 1.02150304
submission.loc[(y_test < -2), "target"] = y_test[(y_test < -2)] * 1.02764047
submission.to_csv(SAVE_PATH, index=False)
