import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st

# ── [1. 페이지 기본 설정 및 디자인] ──
st.set_page_config(page_title="Steam Success Predictor", page_icon="🎮", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #0e141b 0%, #1b2838 42%, #0b1220 100%) !important; color: #c7d5e0; }
.sub-title { color: #66c0f4; font-size: 14px; font-weight: 700; margin-bottom: 20px; text-transform: uppercase; }
.report-card { background: #16202d; border: 1px solid #2a3f5a; padding: 20px; margin-bottom: 15px; }
.score-val { color: white; font-size: 42px; font-weight: 900; font-family: sans-serif; }
</style>
""", unsafe_allow_html=True)

st.title("🎮 게임 스펙 실시간 추론 시뮬레이터")
st.markdown("---")

# ── [2. 모델 정의 및 리소스 로드] ──
MODEL_DIR = os.path.abspath("model")

class SteamSuccessMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

@st.cache_resource
def load_prediction_resources():
    lgbm = joblib.load(os.path.join(MODEL_DIR, "model_lgbm.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    dl_model = SteamSuccessMLP(len(features))
    dl_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_dl_weights.pth"), map_location="cpu"))
    dl_model.eval()
    return lgbm, dl_model, features, scaler

lgbm_model, dl_model, feature_names, scaler = load_prediction_resources()

# ── [3. 입력 UI] ──
col_input, col_report = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="sub-title">📝 게임 스펙 설정</div>', unsafe_allow_html=True)
    is_free = st.checkbox("🆓 무료 게임")
    price = st.slider("💰 가격 ($)", 0.0, 100.0, 14.99, disabled=is_free)
    achievements = st.slider("🏆 도전과제 개수", 0, 500, 0)
    dlc_count = st.number_input("📦 DLC 개수", 0, 100, 0)
    n_languages = st.slider("🌐 지원 언어 수", 1, 30, 1)
    p_win = st.checkbox("Windows", True)
    p_mac = st.checkbox("macOS", False)
    p_lin = st.checkbox("Linux", False)
    selected_genre = st.selectbox("대표 장르", ["Action", "RPG", "Adventure", "Casual", "Strategy", "Simulation", "Indie"])
    selected_tags = st.multiselect("대표 태그", ["Action", "RPG", "Adventure", "Casual", "Strategy", "Simulation", "Indie"])

    # ── [4. 버튼 기반 추론 파이프라인] ──
    if st.button("🚀 예측 결과 생성하기"):
        input_dict = {feat: 0.0 for feat in feature_names}
        input_dict.update({'price': float(price), 'is_free': 1.0 if is_free else 0.0, 
                           'achievements': float(achievements), 'dlc_count': float(dlc_count),
                           'n_languages': float(n_languages), 'windows': 1.0 if p_win else 0.0,
                           'mac': 1.0 if p_mac else 0.0, 'linux': 1.0 if p_lin else 0.0,
                           'n_platforms': float(sum([p_win, p_mac, p_lin])), 'log_price': np.log1p(price),
                           'release_year': 2024.0})
        
        # 장르/태그 매핑
        g_key = f"genre_{selected_genre.lower()}"
        if g_key in input_dict: input_dict[g_key] = 1.0
        for tag in selected_tags:
            tk = f"tag_{tag.lower()}"
            if tk in input_dict: input_dict[tk] = 1.0

        input_df = pd.DataFrame([input_dict])[feature_names]
        input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)

        # 모델 추론
        ml_prob = lgbm_model.predict_proba(input_df_scaled)[0][1] * 100
        with torch.no_grad():
            dl_prob = dl_model(torch.tensor(input_df_scaled.values, dtype=torch.float32)).item() * 100

        # 결과 저장 (세션 스테이트 사용)
        st.session_state['ml_prob'] = ml_prob
        st.session_state['dl_prob'] = dl_prob

# ── [5. 결과 출력 UI] ──
with col_report:
    st.markdown('<div class="sub-title">📊 추론 결과</div>', unsafe_allow_html=True)
    if 'ml_prob' in st.session_state:
        st.markdown(f'<div class="report-card"><div class="score-val" style="color: #4caf50;">{st.session_state.ml_prob:.1f}%</div> ML 예측</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="report-card"><div class="score-val" style="color: #2196f3;">{st.session_state.dl_prob:.1f}%</div> DL 예측</div>', unsafe_allow_html=True)
    else:
        st.info("좌측 설정을 완료하고 버튼을 눌러주세요.")