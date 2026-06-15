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
import joblib
from sklearn.preprocessing import StandardScaler

# 파이썬 실행 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.pipeline import load_processed

print("[DL] 스팀 흥행 예측 데이터 로딩 중(정형 데이터 전용)")

# 텍스트 피처를 제외한 83개의 핵심 정형 피처셋만 로드 (머신러닝과 동일)
X_train, X_test, y_train, y_test = load_processed("hit", with_text=False)

# =========================================================================
# ⚙️ [추가 전처리] 출시 연도(release_year) 이상치 보정
# =========================================================================
if 'release_year' in X_train.columns:
    valid_years = X_train['release_year'][X_train['release_year'] > 2000]
    median_year = valid_years.median() if not valid_years.empty else 2018.0
    
    X_train.loc[X_train['release_year'] < 2000, 'release_year'] = median_year
    X_test.loc[X_test['release_year'] < 2000, 'release_year'] = median_year

print(f"불균형 조정 전 학습 데이터 피처 개수: {X_train.shape[1]}개")
print(f"불균형 조정 전 레이블 분포: 실패(0)={sum(y_train==0)}개, 성공(1)={sum(y_train==1)}개")

# =========================================================================
# 🌟 [다운샘플링 데이터 레벨 개선 구간] - 머신러닝 코드와 동일화
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
# ⚙️ [추가 피처 엔지니어링] 머신러닝과 동일한 공용 스케일러 적용
# =========================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(current_dir, "scaler.pkl")

# 이미 생성된 공용 스케일러가 있다면 로드하고, 없다면 새로 생성하여 저장
if os.path.exists(scaler_path):
    print("\n[Scaler] 기존에 생성된 공용 StandardScaler 로드 중...")
    scaler = joblib.load(scaler_path)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
else:
    print("\n[Scaler] 전처리 스케일러(Scaler) 신규 학습 및 저장 중...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)

# 하이퍼파라미터 및 디바이스 설정
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 디바이스: {DEVICE}")

# PyTorch 텐서 변환 및 데이터로더 구축 (스케일링된 데이터 주입)
y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
y_test_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_arr, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_arr, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 데이터를 64개씩 쪼개서 공급하는 로더 복원
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

input_dim = X_train.shape[1]

# 딥러닝 모델 아키텍처 정의
class SteamSuccessMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),  # 기존 0.3에서 0.4로 상향
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),  # 기존 0.2에서 0.3으로 상향
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

model = SteamSuccessMLP(input_dim).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

print("멀티레이어 퍼셉트론(MLP) 모델 학습 시작...")

best_test_loss = float('inf')

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
    
    # 에폭마다 검증 성능 체크
    model.eval()
    test_loss = 0.0
    correct = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x.size(0)
            
            preds = (outputs >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_accuracy = (correct / len(test_loader.dataset)) * 100
    
    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), os.path.join(current_dir, "steam_dl_model_best.pth"))

# 예측 및 성능 평가 프로세스 (최고 성능 가중치 로드 후 수행)
model.load_state_dict(torch.load(os.path.join(current_dir, "steam_dl_model_best.pth"), map_location=DEVICE))
model.eval()

with torch.no_grad():
    dl_outputs = model(X_test_tensor.to(DEVICE))
    dl_pred = (dl_outputs >= 0.5).float().cpu().numpy()

print("\n=== Deep Learning (MLP) 결과 보고서 ===")
print(f"테스트 정확도(Accuracy): {accuracy_score(y_test_arr, dl_pred):.4f}")
print("\n상세 분류 지표:")
print(classification_report(y_test_arr, dl_pred))

# =========================================================================
# 🌟 [스트림릿 연동을 위한 DL 가중치 최종 추출]
# =========================================================================
model.eval()
dl_path = os.path.join(current_dir, "model_dl_weights.pth")
torch.save(model.to('cpu').state_dict(), dl_path)
print(f"\n✅ [DL 완료] 스트림릿용 딥러닝 가중치 최종 저장 성공: {dl_path}")