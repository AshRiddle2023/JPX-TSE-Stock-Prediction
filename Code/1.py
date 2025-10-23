# libraries
import os
from decimal import ROUND_HALF_UP, Decimal#float型の計算を正確に行うため

import numpy as np
import pandas as pd
from tqdm import tqdm#処理状況の可視化
import warnings

warnings.filterwarnings('ignore')

# set base_dir to load data
base_dir = "../input/jpx-tokyo-stock-exchange-prediction"

train_files_dir = f"{base_dir}/train_files"
supplemental_files_dir = f"{base_dir}/supplemental_files"

