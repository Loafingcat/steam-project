import sys
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 파이썬 실행 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.pipeline import load_processed

print("[DL] 스팀 흥행 예측 데이터 로딩 중(정형 데이터 전용)")

# 텍스트 피처를 제외한 83개의 핵심 정형 피처셋만 로드 (머신러닝과 동일)
X_train, X_test, y_train, y_test = load_processed("hit", with_text=False)

print(f"불균형 조정 전 학습 데이터 피처 개수: {X_train.shape[1]}개")
print(f"불균형 조정 전 레이블 분포: 실패(0)={sum(y_train==0)}개, 성공(1)={sum(y_train==1)}개")

# =========================================================================
# [다운샘플링 데이터 레벨 개선 구간] - 머신러닝 코드와 동일화
# =========================================================================
train_data = pd.concat([X_train, y_train], axis=1)

df_failure = train_data[train_data[y_train.name] == 0]
df_success = train_data[train_data[y_train.name] == 1]

df_failure_downsampled = resample(
    df_failure,
    replace=False,
    n_samples=len(df_success),
    random_state=42
)

df_train_balanced = pd.concat([df_failure_downsampled, df_success])

X_train = df_train_balanced.drop(columns=[y_train.name])
y_train = df_train_balanced[y_train.name]

print(f"다운샘플링 적용 후 레이블 분포: 실패(0)={sum(y_train==0)}개, 성공(1)={sum(y_train==1)}개 (1:1 완료)")
# =========================================================================

# =========================================================================
#  Validation 데이터 분리
# =========================================================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"Train 데이터: {len(X_train)}개")
print(f"Validation 데이터: {len(X_valid)}개")
print(f"Test 데이터: {len(X_test)}개")

# =========================================================================
#  딥러닝용 StandardScaler 적용
# =========================================================================
scaler = StandardScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

X_valid = pd.DataFrame(
    scaler.transform(X_valid),
    columns=X_valid.columns
)

X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

os.makedirs("model", exist_ok=True)

joblib.dump(
    scaler,
    "model/scaler_dl.pkl"
)

print("DL 전용 스케일러 저장 완료")
# =========================================================================

# 하이퍼파라미터 및 디바이스 설정
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 디바이스: {DEVICE}")

# PyTorch 텐서 변환 및 데이터로더 구축
y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
y_valid_arr = y_valid.values if hasattr(y_valid, 'values') else np.array(y_valid)
y_test_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_arr, dtype=torch.float32).unsqueeze(1)

X_valid_tensor = torch.tensor(X_valid.values, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid_arr, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_arr, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 데이터를 64개씩 쪼개서 공급하는 로더 복원
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

input_dim = X_train.shape[1]

# 딥러닝 모델 아키텍처 정의
class SteamSuccessMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 과적합을 막고 에폭별 편차를 줄이기 위해 드롭아웃 비율을 상향 (0.3->0.4, 0.2->0.3)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)

model = SteamSuccessMLP(input_dim).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) 

print("멀티레이어 퍼셉트론(MLP) 모델 학습 시작...")

best_valid_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_x.size(0)

    train_loss /= len(train_loader.dataset)

    # 에폭마다 검증 성능 체크 (손실 및 정확도 동시 계산)
    model.eval()
    valid_loss = 0.0
    correct = 0

    with torch.no_grad():
        for batch_x, batch_y in valid_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            valid_loss += loss.item() * batch_x.size(0)

            # 수정부 주석 추가: nn.Sigmoid() 제거에 따른 판정 기준점 동기화
            # 모델 출력(outputs)이 확률값이 아닌 Logit 레벨이므로, 성공 확률 50%를 의미하는 0.0을 임계값으로 설정
            preds = (outputs >= 0.0).float()
            correct += (preds == batch_y).sum().item()

    valid_loss /= len(valid_loader.dataset)
    valid_accuracy = (correct / len(valid_loader.dataset)) * 100

    print(
        f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Valid Loss: {valid_loss:.4f} | "
        f"Valid Acc: {valid_accuracy:.2f}%"
    )

    # 검증 손실 기준 최고 성능 모델 저장
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "model/steam_dl_model_best.pth")

# 예측 및 성능 평가 프로세스 (최고 성능 가중치 로드 후 수행)
model.load_state_dict(
    torch.load(
        "model/steam_dl_model_best.pth",
        map_location=DEVICE
    )
)

model.eval()

with torch.no_grad():
    dl_outputs = model(X_test_tensor.to(DEVICE))
    # 수정부 주석 추가: 최종 서빙 사양과 일치화
    # 최종 테스트 데이터셋 분류 시에도 Logit 스케일에 맞춰 0.0 이상을 성공(1)으로 마킹하여 스코어를 연산
    dl_pred = (dl_outputs >= 0.0).float().cpu().numpy()

print("\n=== Deep Learning (MLP) 결과 보고서 ===")
print(f"테스트 정확도(Accuracy): {accuracy_score(y_test_arr, dl_pred):.4f}")
print("\n상세 분류 지표:")
print(classification_report(y_test_arr, dl_pred))

# 스트림릿 연동용 가중치 저장
torch.save(
    model.to("cpu").state_dict(),
    "model/model_dl_weights.pth"
)

print("\nDL 모델 가중치 저장 완료")