# dataset.py
# TODO가 아닌 부분도 얼마든지 수정 가능합니다.
# 단, 수정 금지라고 쓰여있는 항목에 대해서는 수정하지 말아주세요. (불가피하게 수정이 필요할 경우 메일로 미리 문의)

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import get_data_path


SYMBOLS = ['Customer_info', 'Discount_info', 'Marketing_info', 'Onlinesales_info', 'Tax_info.csv']  # !!! 수정 금지 !!!


class InfoDataset(Dataset):
    def __init__(self,
                 is_training=True, 
                 in_columns=['고객ID', '거래ID', '거래날짜', '제품ID', '제품카테고리', '수량', '평균금액', '배송료', '쿠폰상태',
                            '성별', '고객지역', '가입기간', 'GST', '월', '쿠폰코드', '할인율', '거래금액'], 
                 out_columns=['제품카테고리'], 
                 data_dir='data'):
        self.x, self.y = make_features(in_columns, out_columns, 
                                       is_training,  data_dir)
    
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # self.x[idx]의 사이즈는 현재 (input_days, input_dim)이므로, 이를 (input_days * input_dim)으로 flatten함
        return torch.flatten(self.x[idx]), self.y[idx]


def make_features(in_columns, out_columns, input_days, 
                  is_training, data_dir='data'):


    save_fname = f'all_{start}_{end}.pkl'

    if os.path.exists(os.path.join(data_dir, save_fname)):
        print(f'loading from {os.path.join(data_dir, save_fname)}')
        table = pd.read_pickle(os.path.join(data_dir, save_fname))
    
    else:
        print(f'making features from {start_date} to {end_date}')
        table = merge_data(start_date, end_date, symbols=SYMBOLS, data_dir=data_dir)
        table.to_pickle(os.path.join(data_dir, save_fname))
        print(f'saved to {os.path.join(data_dir, save_fname)}')


    # TODO: 데이터 클렌징 및 전처리
    table.dropna(inplace=True, subset=['제품카테고리'])
    table.fillna(0, inplace=True)


    if '제품카테고리' not in out_columns:
        raise ValueError('USD_Price must be included in out_columns')   # !!! 수정 금지 !!!
    
    use_columns = list(set(in_columns + out_columns))  # 중복 제거
    df = table[use_columns]


    # TODO: 추가적인 feature engineering이 필요하다면 아래에 작성
    # 주의 : 추가로 활용할 feature들은 in_columns에도 추가할 것
    in_columns += []


    # input_days 만큼의 과거 데이터를 사용하여 다음날의 USD_Price를 예측하도록 데이터셋 구성됨
    date_indices = sorted(table.index)
    x = np.asarray([df.loc[date_indices[i:i + input_days], in_columns] for i in range(len(df) - input_days)])
    y = np.asarray([df.loc[date_indices[i + input_days], out_columns] for i in range(len(df) - input_days)])


    # 최근 10일을 test set으로 사용
    # 주의 : 검증 및 테스트 과정에 반드시 최근 10일 데이터를 사용해야 하므로 수정하지 말 것
    training_x, test_x = x[:-10], x[-10:]  # !!! 수정 금지 !!!
    training_y, test_y = y[:-10], y[-10:]  # !!! 수정 금지 !!!

    
    return (training_x, training_y) if is_training else (test_x, test_y)



def merge_data(symbols, data_dir='data'):

    dates = pd.date_range(start_date, end_date, freq='D')
    df = pd.DataFrame(index=dates)

    if 'USD' not in symbols:
        symbols.insert(0, 'USD')

    for symbol in symbols:
        df_temp = pd.read_csv(get_data_path(symbol, data_dir), index_col="Date", parse_dates=True, na_values=['nan'])
        df_temp = df_temp.reindex(dates)
        df_temp.columns = [symbol + '_' + col for col in df_temp.columns]  # rename columns
        df = df.join(df_temp)

    return df




if __name__ == "__main__":
    dataset = InfoDataset(is_training=True)
    print(f"dataset length: {len(dataset)}")
    print(f"first sample x: {dataset[0][0]}")
    print(f"first sample y: {dataset[0][1]}")
