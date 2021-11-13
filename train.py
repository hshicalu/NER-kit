import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from pprint import pprint
import gc
from tqdm import tqdm

from collections import defaultdict
from typing import Iterable
from typing import Any
import difflib
import re

from sklearn.metrics import roc_auc_score
import torch
import torch.utils.data as D
from pytorch_lightning import Trainer
from transformers import BertJapaneseTokenizer

import warnings
warnings.filterwarnings("ignore")

from configs import *

class Preprocess():
    def __init__(self):
      self.dummy = None
    def preprocess(self):
      return None

class Accuracy():
    def __init__(self, maximize=True, verbose=False):
        super().__init__(maximize=maximize)
        self.verbose = verbose

    def _test(self, target, approx, mask):
        target = target.reshape(-1).astype(int)
        approx = approx.argmax(2).reshape(-1).astype(int)
        mask = mask.reshape(-1).astype(bool)
        return (target == approx).astype(int)[mask].mean()
    
    def torch(self, approx, target, mask):
        return self._test(target.detach().cpu().numpy(),
                          approx.detach().cpu().numpy(),
                          mask.detach().cpu().numpy())

class AUC():
    def __init__(self, maximize=True, verbose=False):
        super().__init__(maximize=maximize)
        self.verbose = verbose
    
    def _test(self, target, approx, mask):
        scores = []

        label_num = approx[0].shape[-1]
        
        target = target.reshape(-1).astype(int)
        target = np.identity(label_num)[target] # change to one-hot
        approx = approx.reshape([-1, label_num]).astype(float)
        mask = mask.reshape(-1).astype(bool)
        
        target = target[mask]
        approx = approx[mask]

        for i in range(label_num):
            try:
                scores.append(roc_auc_score(target[:, i], approx[:, i]))
            except ValueError:
                pass
            
        return np.mean(scores)

    def torch(self, approx, target, mask):
        return self._test(target.detach().cpu().numpy(),
                          approx.detach().cpu().numpy(), 
                          mask.detach().cpu().numpy())      

def main():
    ''' Set parser '''
    parser = argparse.ArgumentParser()

    ''' Preprocess '''
    data = preprocess()

    ''' train '''
    if not cfg.inference:
        train_data = ***
        valid_data = ***
        train_loader = D.DataLoader(train_data, shuffle=True, num_workers=0, batch_size=cfg.batch_size)
        valid_loader = D.DataLoader(valid_data, shuffle=True, num_workers=0, batch_size=cfg.batch_size)

        model = cfg.model(**cfg.model_params)

        if cfg.finetune:
            ***
        else:
            ***
        
        optimizer = cfg.optimizer(***, dict(lr=cfg.lr, weight_decay=cfg.weight_decay))
        scheduler = cfg.scheduler(optimizer)
        PARAMS = {
            'loader': train_loader,
            'loader_valid': valid_loader,
            ***
        }
        trainer = Trainer(model, device=***)
        trainer.fit(**PARAMS)

        ''' inference '''
        ***

if __name__ == '__main__':
    main()