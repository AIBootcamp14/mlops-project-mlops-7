# utils.py

import os
import random
import numpy as np
import torch
import pandas as pd


def get_data_path(symbol, data_dir):
    mapping = {
        'Customer': 'Customer_info.csv',
        'Discount': 'Discount_info.csv',
        'Onlinesales': 'Onlinesales_info.csv',
        'Tax': 'Tax_info.csv',
        'Marketing': 'Marketing_info.csv'
    }
    if symbol not in mapping:
        raise ValueError(f"Symbol '{symbol}' is not recognized. Allowed values: {list(mapping.keys())}")
    
    path = os.path.join(data_dir, mapping[symbol])
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return path
        
    
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
    