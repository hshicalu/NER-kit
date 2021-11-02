import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from pprint import pprint
import gc
import torch
import torch.utils.data as D
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
from tqdm import tqdm

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
        model.fit(**PARAMS)

        ''' inference '''
        ***

if __name__ == '__main__':
    main()