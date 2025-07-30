# dataset.py
# TODO가 아닌 부분도 얼마든지 수정 가능합니다.
# 단, 수정 금지라고 쓰여있는 항목에 대해서는 수정하지 말아주세요. (불가피하게 수정이 필요할 경우 메일로 미리 문의)

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import get_data_path
from sklearn.preprocessing import LabelEncoder

print("Current Working Directory:", os.getcwd())
SYMBOLS = ['Customer', 'Discount', 'Marketing', 'Onlinesales', 'Tax']  # !!! 수정 금지 !!!

CATEGORY_MAP = {
    'Fun': 'Lifestyle',
    'More Bags': 'Bags',
    'Backpacks': 'Bags',
    'Google': 'Nest'
}

MONTH_MAP = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

class InfoDataset(Dataset):
    def __init__(self, is_training=True, in_columns=None, out_columns=None, data_dir='../data'):
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
        df = pd.read_pickle(merged_path)
    else:
        print('merging raw csv files...')
        df = merge_data(SYMBOLS, data_dir)
        df.to_pickle(merged_path)
        print(f'saved to {merged_path}')

    df.dropna(subset=out_columns, inplace=True)
    df.fillna(0, inplace=True)

    # 문자열형 컬럼 인코딩
    label_cols = ['성별', '고객지역', '쿠폰코드', '월', '고객ID', '거래ID', '제품ID', '쿠폰상태']
    for col in label_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # 타겟 인코딩
    df['제품카테고리'] = LabelEncoder().fit_transform(df['제품카테고리'])

    df = df[in_columns + out_columns]
    df = df.loc[:, ~df.columns.duplicated()]

    x = df[in_columns].to_numpy().astype(float)
    y = df[out_columns].to_numpy().astype(float).squeeze()

    if is_training:
        return x[:-100], y[:-100]
    else:
        return x[-100:], y[-100:]


def merge_data(symbols, data_dir):
    print("merging raw csv files...")
    dfs = {symbol: pd.read_csv(get_data_path(symbol, data_dir)) for symbol in symbols}

    # 병합 순서: Onlinesales + Customer → Tax → Discount
    merged = pd.merge(dfs['Onlinesales'], dfs['Customer'], on='고객ID')
    merged = pd.merge(merged, dfs['Tax'], on='제품카테고리')

    # 제품카테고리 매핑
    merged['제품카테고리'] = merged['제품카테고리'].map(CATEGORY_MAP).fillna(merged['제품카테고리'])

    # Discount에서 'Notebooks' 제외
    discount_df = dfs['Discount'][dfs['Discount']['제품카테고리'] != 'Notebooks'].copy()

    # 월 파생 후 문자열로 매핑
    merged['월'] = pd.to_datetime(merged['거래날짜']).dt.month.map(MONTH_MAP)

    # Discount 병합
    merged = pd.merge(merged, discount_df, on=['제품카테고리', '월'])

    # 거래날짜 정수화
    merged['거래날짜'] = pd.to_datetime(merged['거래날짜'], errors='coerce')
    merged['거래날짜'] = merged['거래날짜'].dt.strftime('%Y%m%d').astype(float)

    # 거래금액 공식 적용
    merged['거래금액'] = merged['평균금액'] * merged['수량'] * (1 + merged['GST']) + merged['배송료']

    print("🔍 최종 병합된 컬럼 목록:", merged.columns.tolist())
    return merged






if __name__ == "__main__":
    dataset = InfoDataset(is_training=True)
    print(f"dataset length: {len(dataset)}")
    print(f"x shape: {dataset.x.shape}, y shape: {dataset.y.shape}")
    print("first sample x:", dataset[0][0])
    print("first sample y:", dataset[0][1])
    print("x[:5] (list):", dataset.x[:5].tolist())
    print("y[:5] (list):", dataset.y[:5].tolist())

    # ✅ 클래스 수 확인
    import numpy as np
    unique_classes, counts = np.unique(dataset.y.numpy(), return_counts=True)
    print("\n📊 클래스 개수:", len(unique_classes))
    print("🧩 클래스 목록:", unique_classes.tolist())
    print("🔢 클래스별 샘플 수:")
    for c, n in zip(unique_classes, counts):
        print(f" - 클래스 {int(c)}: {n}개")

    # ✅ 입력 피처 구성 확인
    in_columns = [
        '고객ID', '거래ID', '거래날짜', '제품ID', '제품카테고리', '수량', '평균금액', '배송료', '쿠폰상태',
        '성별', '고객지역', '가입기간', 'GST', '월', '쿠폰코드', '할인율', '거래금액']
    print("\n🧾 입력 피처 목록:")
    for idx, col in enumerate(in_columns):
        print(f"[{idx}] {col}")

