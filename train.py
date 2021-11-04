import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from pprint import pprint
import gc
from tqdm import tqdm

import torch
import torch.utils.data as D
from pytorch_lightning import Trainer

import warnings
warnings.filterwarnings("ignore")

import preprocess
from metrics import Accuracy, AUC
from configs import *

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