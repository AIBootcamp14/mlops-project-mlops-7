# dataset.py
# TODO가 아닌 부분도 얼마든지 수정 가능합니다.
# 단, 수정 금지라고 쓰여있는 항목에 대해서는 수정하지 말아주세요. (불가피하게 수정이 필요할 경우 메일로 미리 문의)

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import get_data_path

print("Current Working Directory:", os.getcwd())
SYMBOLS = ['Customer', 'Discount', 'Marketing', 'Onlinesales', 'Tax']  # !!! 수정 금지 !!!


class InfoDataset(Dataset):
    def __init__(self, 
                 is_training=True, 
                 in_columns=None, 
                 out_columns=None,
                 data_dir='../data'):
        
        if in_columns is None:
            in_columns = ['고객ID', '거래ID', '거래날짜', '제품ID', '제품카테고리', '수량', '평균금액', '배송료', '쿠폰상태',
                          '성별', '고객지역', '가입기간', 'GST', '월', '쿠폰코드', '할인율', '거래금액']
        if out_columns is None:
            out_columns = ['제품카테고리']

        self.x, self.y = make_features(in_columns, out_columns, is_training, data_dir)
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_features(in_columns, out_columns, is_training, data_dir='data'):
    save_fname = 'merged_data.pkl'

    merged_path = os.path.join(data_dir, save_fname)
    if os.path.exists(merged_path):
        print(f'loading from {merged_path}')
        table = pd.read_pickle(merged_path)
    else:
        print('merging raw csv files...')
        table = merge_data(symbols=SYMBOLS, data_dir=data_dir)
        table.to_pickle(merged_path)
        print(f'saved to {merged_path}')

    # 전처리
    table.dropna(subset=out_columns, inplace=True)
    table.fillna(0, inplace=True)

    df = table[in_columns + out_columns]
    
    x = df[in_columns].to_numpy()
    y = df[out_columns].to_numpy()

    # 나중에 훈련/테스트 분할을 위해 아래처럼 자를 수도 있음
    if is_training:
        return x[:-100], y[:-100]
    else:
        return x[-100:], y[-100:]



def merge_data(symbols, data_dir):
    print("merging raw csv files...")
    dfs = {}

    for symbol in symbols:
        path = get_data_path(symbol, data_dir)
        df = pd.read_csv(path)
        print(f"[{symbol}] columns: {df.columns.tolist()}")
        dfs[symbol] = df

    # 1. Customer + Onlinesales on '고객ID'
    merged = pd.merge(dfs['Customer'], dfs['Onlinesales'], how='left', on='고객ID')

    # 2. '거래날짜'에서 '월' 추출
    if '거래날짜' in merged.columns:
        merged['월'] = pd.to_datetime(merged['거래날짜']).dt.month.astype(str).str.zfill(2)

    # 3. Tax 병합 on '제품카테고리'
    if '제품카테고리' in merged.columns and 'Tax' in dfs:
        merged = pd.merge(merged, dfs['Tax'], how='left', on='제품카테고리')
    else:
        raise KeyError("'제품카테고리'가 병합 중간에 없습니다.")

    # 4. Discount 병합 on '월'
    if '월' in merged.columns and 'Discount' in dfs:
        merged = pd.merge(merged, dfs['Discount'], how='left', on='월')
    else:
        raise KeyError("'월' 컬럼이 없어서 Discount 병합 불가")

    return merged





if __name__ == "__main__":
    dataset = InfoDataset(is_training=True)
    print(f"dataset length: {len(dataset)}")
    print(f"first sample x: {dataset[0][0]}")
    print(f"first sample y: {dataset[0][1]}")
