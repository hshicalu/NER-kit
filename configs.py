import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, linear, cosine, CosineAnnealingLR
from transformers import BertForTokenClassification, BertJapaneseTokenizer

from metrics import Accuracy, AUC
# ====================================================
# CFG
# ====================================================
class CFG:
    name = 'NER'
    seed = 2021
    dataset = ["Dummy code"]
    tokenizer = BertJapaneseTokenizer.from_pretrained
    tokenizer_params = dict(
        'cl-tohoku/bert-base-japanese-whole-word-masking',
    )
    model = BertForTokenClassification.from_pretrained
    model_params = dict(
        'cl-tohoku/bert-base-japanese-whole-word-masking',
        num_labels=["Dummy code"]+1,
        output_attentions=False,
        output_hidden_states=False
    )
    finetune = True
    max_length = 64
    scheduler = 'ReduceLROnPlateau' # ['CosineAnnealingWarmRestarts', 'ReduceLROnPlateau', 'linear', 'cosine', 'CosineAnnealingLR']
    batch_scheduler = False
    lr = 1e-3
    min_lr = 1e-5
    weight_decay = 1e-6
    optimizer = optim.Adam
    optimizer_params = dict()
    criterion = nn.BCEWithLogitsLoss()
    metric = Accuracy().torch
    # metric = AUC().torch

    print_freq = 1000
    num_workers = 4
    epochs = 200
    batch_size = 256
    apex = True
    parallel = None
    deterministic = False
    train = True
    inference = True
# <===デバックの切り替え===>
    debug = False