import pandas as pd
import numpy as np

from collections import defaultdict
from typing import Iterable
from typing import Any
import difflib
import re

import torch
import torch.utils.data as D
from transformers import BertJapaneseTokenizer


MAX_LEN = 128
PAD_ID = 0
PAD_TAG = '[PAD]'

class Preprocess():
    def __init__(self):
      self.dummy = dummy
    def preprocess(self):
      return None
