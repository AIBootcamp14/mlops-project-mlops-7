import os
import logging
import datetime
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils import fix_seed, get_data_path
from metric import accuracy, f1, mae
from model import MLP

# 시드 고정
fix_seed(42)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------
# 데이터셋 클래스 정의
# -------------------------
class InfoDataset(Dataset):
    def __init__(self, csv_path, is_training=True):
        self.df = pd.read_csv(csv_path)

        # 전처리
        self.df.dropna(subset=['제품카테고리'], inplace=True)
        self.df.fillna(0, inplace=True)

        # '월' 컬럼이 문자열이면 숫자로 매핑
        if self.df['월'].dtype == 'object': 
            month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
            self.df['월'] = self.df['월'].map(month_map)

        # 범주형 라벨 인코딩
        self.label_encoders = {}
        for col in ['고객ID', '거래ID', '제품ID', '제품카테고리', '쿠폰상태', '성별', '고객지역', '쿠폰코드']:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le

        # 피처와 타겟
        self.in_columns = ['고객ID', '거래ID', '제품ID', '수량', '평균금액', '배송료', '쿠폰상태',
                           '성별', '고객지역', '가입기간', 'GST', '월', '쿠폰코드', '할인율', '거래금액']
        self.out_columns = ['제품카테고리']

        X = self.df[self.in_columns].values
        y = self.df[self.out_columns].values.ravel()

        # 수치형 스케일링
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # train/val 분할
        if is_training:
            self.X = torch.tensor(X[:-100], dtype=torch.float32)
            self.y = torch.tensor(y[:-100], dtype=torch.long)
        else:
            self.X = torch.tensor(X[-100:], dtype=torch.float32)
            self.y = torch.tensor(y[-100:], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------
# 학습 루프
# -------------------------
def train():
    # 데이터 준비
    dataset = InfoDataset(csv_path='../data/merged_data.csv', is_training=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 모델 정의
    input_size = dataset.X.shape[1]
    model = MLP(input_size=input_size, n_hidden_list=[128, 64], output_size=20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 옵티마이저, 손실함수
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 학습
    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        total_acc = 0
        total_f1 = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(outputs, dim=1)
            total_loss += loss.item()
            total_acc += accuracy(pred, y_batch)
            total_f1 += f1(pred, y_batch)

        logger.info(f"Epoch {epoch} - Loss: {total_loss:.4f}, Acc: {total_acc/len(dataloader):.4f}, F1: {total_f1/len(dataloader):.4f}")

    torch.save(model.state_dict(), 'team_model.pt')
    logger.info("Model saved to team_model.pt")


if __name__ == '__main__':
    train()