import streamlit as st
from style import steam_theme, steam_nav, steam_hero, section_header

st.set_page_config(page_title="흥행 예측 | Steam Analytics", page_icon="🤖", layout="wide")
steam_theme()
steam_nav(active="예측")
steam_hero(
    title="Success Prediction",
    subtitle="ML · DL 모델 기반 게임 흥행 가능성 예측",
    badge="COMING SOON"
)

st.write("")

# ── 준비 상태 배너 ──
st.markdown("""
<div style="
    background: linear-gradient(135deg, #16202d, #1a3a5c);
    border: 1px solid #2a3f5a;
    border-left: 4px solid #66c0f4;
    border-radius: 4px;
    padding: 24px 28px;
    margin-bottom: 24px;
">
    <p style="color:#66c0f4;font-size:12px;text-transform:uppercase;letter-spacing:2px;margin:0 0 6px 0;">
        STATUS
    </p>
    <p style="color:#c6d4df;font-size:16px;margin:0;">
        모델 학습 완료 후 이 페이지에 예측 결과가 자동으로 연동됩니다.
        아래 시뮬레이터는 데모용입니다.
    </p>
</div>
""", unsafe_allow_html=True)

# ── 예정 기능 카드 ──
section_header("📋 예정 기능")

c1, c2, c3 = st.columns(3)
cards = [
    ("🌲", "Random Forest", "전통 ML 앙상블 모델.\n흥행 / 비흥행 분류"),
    ("⚡", "XGBoost",        "Gradient Boosting 기반.\nFeature Importance 분석"),
    ("🧠", "Deep Learning",  "PyTorch MLP 모델.\nML vs DL 성능 비교"),
]
for col, (icon, title, desc) in zip([c1, c2, c3], cards):
    with col:
        st.markdown(f"""
        <div style="
            background:#16202d;
            border:1px solid #2a3f5a;
            border-radius:4px;
            padding:20px;
            text-align:center;
            height:140px;
        ">
            <div style="font-size:32px;">{icon}</div>
            <p style="color:#66c0f4;font-weight:700;font-size:14px;margin:8px 0 4px 0;">{title}</p>
            <p style="color:#8cbdd8;font-size:12px;margin:0;white-space:pre-line;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.write("")
c4, c5 = st.columns(2)
cards2 = [
    ("🔍", "Hidden Gem Detection", "숨은 명작 탐지.\n긍정률 ≥ 85% & Owners < 10만"),
    ("📊", "SHAP Explainability",  "어떤 피처가 흥행에 영향?\nSHAP 분석으로 근거 제시"),
]
for col, (icon, title, desc) in zip([c4, c5], cards2):
    with col:
        st.markdown(f"""
        <div style="
            background:#16202d;
            border:1px solid #2a3f5a;
            border-radius:4px;
            padding:20px;
            text-align:center;
            height:130px;
        ">
            <div style="font-size:28px;">{icon}</div>
            <p style="color:#66c0f4;font-weight:700;font-size:14px;margin:8px 0 4px 0;">{title}</p>
            <p style="color:#8cbdd8;font-size:12px;margin:0;white-space:pre-line;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── 데모 시뮬레이터 ──
section_header("🎮 Demo Simulator  —  간단 흥행 예측 체험")

col_input, col_result = st.columns([1, 1])

with col_input:
    price = st.slider("게임 가격 ($)", 0, 60, 19)
    genre = st.selectbox("장르", [
        "Action", "RPG", "Adventure", "Strategy",
        "Simulation", "Indie", "Sports", "Racing"
    ])
    platform_cnt = st.selectbox("지원 플랫폼 수", [1, 2, 3])
    is_free = st.checkbox("무료 게임")

    predict_btn = st.button("▶  예측하기", use_container_width=True)

with col_result:
    if predict_btn:
        # 단순 데모 점수
        score = 50
        if is_free:           score += 15
        if price <= 10:        score += 10
        elif price <= 20:      score += 5
        elif price >= 40:      score -= 10
        if genre in ["Action", "RPG", "Indie"]: score += 10
        score += platform_cnt * 3
        score = max(5, min(95, score))

        level = (
            ("🔥 대흥행 예상",    "#66c0f4") if score >= 75 else
            ("✅ 흥행 가능",      "#4caf50") if score >= 55 else
            ("⚠️ 보통",          "#ffc107") if score >= 40 else
            ("❌ 흥행 어려움",   "#ef5350")
        )

        st.markdown(f"""
        <div style="
            background:#16202d;
            border:1px solid #2a3f5a;
            border-radius:4px;
            padding:28px;
            text-align:center;
            margin-top:10px;
        ">
            <p style="color:#8cbdd8;font-size:12px;letter-spacing:2px;margin:0 0 8px 0;">
                예상 흥행 점수
            </p>
            <div style="font-size:56px;font-weight:900;color:{level[1]};line-height:1;">
                {score}%
            </div>
            <div style="font-size:18px;color:{level[1]};margin-top:8px;font-weight:700;">
                {level[0]}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(score / 100)
    else:
        st.markdown("""
        <div style="
            background:#16202d;
            border:1px dashed #2a3f5a;
            border-radius:4px;
            padding:60px 20px;
            text-align:center;
        ">
            <p style="color:#4d7899;font-size:14px;margin:0;">
                왼쪽에서 조건을 설정하고<br>예측하기 버튼을 누르세요
            </p>
        </div>
        """, unsafe_allow_html=True) 