import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st

# ── [1. 페이지 기본 설정 및 Steam 테마 CSS 주입] ──
st.set_page_config(page_title="스팀 흥행 예측기", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 18% 0%, rgba(42,71,94,0.45), transparent 28%),
        radial-gradient(circle at 82% 8%, rgba(26,159,255,0.18), transparent 25%),
        linear-gradient(180deg, #0e141b 0%, #1b2838 42%, #0b1220 100%) !important;
    color: #c7d5e0;
}
.sub-title { color: #66c0f4; font-size: 16px; font-weight: 700; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 1px;}
.report-card { background: #16202d; border: 1px solid #2a3f5a; padding: 20px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 게임 스펙 기반 딥러닝 흥행 추론 시뮬레이터")
st.markdown("---")

# ── [2. 모델 정의 및 백엔드 리소스 로드] ──
MODEL_DIR = os.path.abspath("model")

class SteamSuccessMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x)

def stable_sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

@st.cache_resource
def load_dl_resources():
    try:
        feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_dl.pkl"))
        
        input_dim = len(feature_names)
        dl_model = SteamSuccessMLP(input_dim)
        dl_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_dl_weights.pth"), map_location="cpu"))
        dl_model.eval()
        
        return True, dl_model, feature_names, scaler
    except Exception as e:
        st.error(f"🚨 백엔드 리소스 로드 실패: {e}")
        return False, None, None, None

is_ready, model, features, scaler = load_dl_resources()

if is_ready:
    # ── [3. UX/UI 레이어: 설정 컴포넌트 배치] ──
    col_input, col_setup, col_report = st.columns([1.0, 1.0, 1.0], gap="large")

    with col_input:
        st.markdown('<div class="sub-title">🎮 데이터 입력</div>', unsafe_allow_html=True)
        
        is_free = st.checkbox("🆓 무료 게임", value=False)
        
        if is_free:
            price = st.number_input("가격", min_value=0.0, max_value=0.0, value=0.0, step=0.0, format="%.2f", disabled=True)
        else:
            price = st.number_input("가격 ($)", min_value=0.99, max_value=59.99, value=0.99, step=1.0, format="%.2f", 
                                    help="기본 0.99달러 단위로 증감하며 앞자리가 1달러씩 조정됩니다.")
            
        achievements = st.slider("🏆 도전과제 개수", min_value=0, max_value=500, value=0, step=10,
                                 help="스팀 게임의 평균 업적은 30~50개 내외입니다. 최대 업적 제한은 5,000개입니다.")
        
        dlc_count = st.slider("📦 DLC 개수", min_value=0, max_value=100, value=0, step=1,
                              help="스팀 데이터셋 내 전체 게임의 약 80%는 DLC가 0~3개 사이입니다.")
        
        n_languages = st.slider("🌐 지원 언어 수", min_value=1, max_value=30, value=1)

        st.write("")
        st.write("**🖥️ 지원 플랫폼 (중복 선택)**")
        p_win = st.checkbox("Windows", value=True)
        p_mac = st.checkbox("macOS", value=False)
        p_lin = st.checkbox("Linux", value=False)

    with col_setup:
        st.markdown('<div class="sub-title">⚙️ 장르 및 테마 설정</div>', unsafe_allow_html=True)
        
        # 장르 리스트를 ABC 순으로 정렬하고, 최상단에 선택 유도 안내 가짜 값 추가
        raw_genres = [
            "Action", "RPG", "Adventure", "Casual", "Strategy", 
            "Simulation", "Indie", "Racing", "Sports", "Early Access"
        ]
        top_10_genres = ["장르를 선택해주세요"] + sorted(raw_genres)
        selected_genre = st.selectbox("메인 장르 (Main Genre)", options=top_10_genres, index=0)
        
        st.write("")
        # 태그 분리(플레이 위주)
        raw_gameplay_tags = [
            "Singleplayer", "Multiplayer", "Co-op", "Open World", "Story Rich", 
            "FPS", "Shooter", "Survival", "Crafting", "Sandbox", "Rogue-like", 
            "Strategy RPG", "Puzzle", "Difficult"
        ]
        gameplay_tags = sorted(raw_gameplay_tags)
        selected_gameplay = st.multiselect("🕹️ 게임플레이/서브 장르 태그", options=gameplay_tags, default=[], placeholder="게임플레이 방식을 선택하세요")
        
        st.write("")
        # 태그 분리(비주얼, 테마 위주)
        raw_visual_theme_tags = [
            "2D", "3D", "Pixel Graphics", "Atmospheric", "Sci-fi", "Fantasy", 
            "Horror", "Anime", "Cyberpunk", "Great Soundtrack", "VR"
        ]
        visual_theme_tags = sorted(raw_visual_theme_tags)
        selected_visuals = st.multiselect("🎨 비주얼/기획 테마 태그", options=visual_theme_tags, default=[], placeholder="테마 키워드를 선택하세요")

    # ── [4. 예측 실행 버튼 배치] ──
    with col_input:
        st.write("")
        predict_button = st.button("🚀 예측 실행", use_container_width=True)

    # ── [5. 결과 출력 UI 및 데이터 파이프라인] ──
    with col_report:
        st.markdown('<div class="sub-title">📊 추론 결과</div>', unsafe_allow_html=True)
        
        if predict_button:
            # 🛑 [안전 필터링] 유저가 메인 장르를 고르지 않고 버튼을 눌렀을 경우 예외 처리
            if selected_genre == "장르를 선택해주세요":
                st.error("⚠️ 메인 장르를 먼저 선택해주세요! 장르 설정 없이는 딥러닝 추론을 진행할 수 없습니다.")
            else:
                # 전체 피처 컬럼 레이아웃 초기화
                x = {feat: 0.0 for feat in features}
                
                # 기본 수치 변수 입력
                x['price'] = float(price)
                x['is_free'] = 1.0 if is_free or price == 0 else 0.0
                x['achievements'] = float(achievements)
                x['dlc_count'] = float(dlc_count)
                x['n_languages'] = float(n_languages)
                x['windows'] = 1.0 if p_win else 0.0
                x['mac'] = 1.0 if p_mac else 0.0
                x['linux'] = 1.0 if p_lin else 0.0
                x['n_platforms'] = float(sum([p_win, p_mac, p_lin]))
                x['log_price'] = np.log1p(price)
                x['release_year'] = 2024.0

                # 메인 장르 원핫 인코딩 치환 연산
                g_key = f"genre_{selected_genre.lower().replace(' ', '_')}"
                if g_key in x: x[g_key] = 1.0
                
                t_genre_key = f"tag_{selected_genre.lower().replace(' ', '_')}"
                if t_genre_key in x: x[t_genre_key] = 1.0

                # 두 개로 쪼개진 태그 선택지 리스트 병합 및 매핑
                total_selected_tags = selected_gameplay + selected_visuals
                
                for tag in total_selected_tags:
                    tk = f"tag_{tag.lower().replace(' ', '_')}"
                    if tk in x: 
                        x[tk] = 1.0

                # 피처 오더 맵 구조 맞추기
                df = pd.DataFrame([x]).reindex(columns=features, fill_value=0)
                
                # 정규화 연산 및 튀는 값 가두리(Clipping)
                scaled = scaler.transform(df)
                scaled = np.clip(scaled, -2.5, 2.5)

                # PyTorch 텐서 변환 및 순방향 추론
                tensor = torch.tensor(scaled, dtype=torch.float32)

                with torch.no_grad():
                    logit = model(tensor).item()
                    prob = stable_sigmoid(logit) * 100

                # 비즈니스 로직 보정 (무료게임 상쇄 가중치)
                if is_free:
                    prob = prob * 1.25

                # 최종 캘리브레이션 컷오프 한계 지정
                prob = np.clip(prob, 1, 99.5)

                # 시각화 컴포넌트 출력
                st.metric("성공 확률", f"{prob:.1f}%")
                st.progress(int(prob))

                if prob > 70:
                    st.success("🌟 높은 성공 가능성")
                elif prob > 50:
                    st.warning("⚖️ 중간 수준")
                else:
                    st.error("📉 낮은 성공 가능성")

                # 검증용 디버그 패널
                # st.markdown("---")
                # st.markdown("### 🛠️ 디버그 정보")
                # st.write(f"**출력 로직값 (Logit):** {logit:.4f}")
                # st.write(f"**보정된 스케일러 범위 (Min/Max):** {scaled.min():.4f} / {scaled.max():.4f}")
                
                # # 현재 활성화된 가중치 피처만 선별 출력
                # active_features = {k: v for k, v in df.iloc[0].to_dict().items() if v > 0}
                # st.write("**활성화된 피처 가중치 노드 목록:**")
                # st.json(active_features)
            
        else:
            st.info("💡 모든 스펙 설정을 마친 후 왼쪽 하단의 **'예측 실행'** 버튼을 클릭하시면 딥러닝 결과 요약 리포트가 이곳에 출력됩니다.")
else:
    st.error("🚨 시스템 파일 누락: `model/` 디렉토리 내 가중치와 스케일러 파일을 다시 확인해 주세요.")