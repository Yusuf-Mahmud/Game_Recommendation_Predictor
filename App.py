"""
App.py  –  Steam ML Predictor
All preprocessing lives in preprocess.py.
Inference is handled by predict.py.
This file only deals with UI.
"""

import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

import math
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from predict import predict as run_predict, PKL_PATHS
from Models.Linear_Regression_Scratch import YusufLinearRegression

import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Steam ML Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050510 0%, #0a0a1f 100%);
    border-right: 1px solid #1a1a3a;
    width: 370px !important;
}
section[data-testid="stSidebar"] * { color: #c0c0e0 !important; }

.main .block-container { background: #060612; padding: 2rem 2.5rem; max-width: 1500px; }

.app-title {
    font-family: 'Rajdhani', sans-serif; font-size: 1.8rem; font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: 0.05em;
}
.app-subtitle { font-size: 0.7rem; color: #3a3a6a !important; letter-spacing: 0.2em; text-transform: uppercase; margin-top: -2px; }

.section-header {
    font-family: 'Rajdhani', sans-serif; font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.25em; text-transform: uppercase; color: #00d4ff !important;
    padding: 0.7rem 0 0.3rem; border-bottom: 1px solid #1a2a4a;
    margin-bottom: 0.6rem; margin-top: 0.8rem;
}

.metric-card {
    background: linear-gradient(135deg, #0a0a20 0%, #0f0f28 100%);
    border: 1px solid #1a1a4a; border-radius: 4px; padding: 1.2rem;
    text-align: center; font-family: 'IBM Plex Mono', monospace;
}
.metric-card .label { color: #3a3a7a !important; font-size: 0.62rem; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0.3rem; }
.metric-card .value { color: #00d4ff !important; font-size: 1.6rem; font-weight: 500; font-family: 'Rajdhani', sans-serif; }
.metric-card .sub   { color: #4a4a8a !important; font-size: 0.7rem; margin-top: 0.2rem; }

.prediction-box {
    background: linear-gradient(135deg, #030318 0%, #080825 100%);
    border: 1px solid #00d4ff44; border-radius: 4px; padding: 2.5rem;
    text-align: center;
    box-shadow: 0 0 60px rgba(0,212,255,0.08), inset 0 0 40px rgba(0,212,255,0.03);
    position: relative; overflow: hidden;
}
.prediction-box::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
}
.prediction-box .pred-label { color: #3a6a8a !important; font-size: 0.65rem; letter-spacing: 0.25em; text-transform: uppercase; margin-bottom: 0.8rem; }
.prediction-box .pred-value { font-family: 'Rajdhani', sans-serif; color: #ffffff !important; font-size: 4rem; font-weight: 700; line-height: 1; letter-spacing: -0.02em; }

.stNumberInput input, .stSelectbox select, .stTextArea textarea {
    background: #080818 !important; border: 1px solid #1a1a3a !important;
    color: #c0c0e0 !important; border-radius: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.82rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #7c3aed22) !important;
    border: 1px solid #00d4ff !important; color: #00d4ff !important;
    border-radius: 3px !important; padding: 0.7rem 2rem !important;
    font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important;
    font-size: 1rem !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; width: 100% !important; transition: all 0.2s !important;
}
.stButton > button:hover { background: linear-gradient(135deg, #00d4ff44, #7c3aed44) !important; box-shadow: 0 0 20px rgba(0,212,255,0.2) !important; }

h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important; color: #e0e0ff !important; letter-spacing: 0.05em !important; }
hr { border-color: #1a1a3a !important; }
div[data-testid="stMarkdownContainer"] p { color: #6060a0 !important; font-size: 0.85rem; }
.stCheckbox > label { color: #8080b0 !important; font-size: 0.8rem !important; }
.stSelectbox label, .stNumberInput label, .stTextArea label { color: #5050a0 !important; font-size: 0.72rem !important; letter-spacing: 0.08em; }
</style>
""", unsafe_allow_html=True)


# ── Data / model loaders ──────────────────────────────────────────────────────

# 40 raw features used by the scratch model (no log-transform, StandardScaler)
SCRATCH_FEATURE_COLS = [
    'RequiredAge', 'DemoCount', 'DeveloperCount', 'DLCCount', 'Metacritic',
    'MovieCount', 'PackageCount', 'PublisherCount', 'ScreenshotCount',
    'SteamSpyOwners', 'SteamSpyPlayersEstimate',
    'AchievementCount', 'AchievementHighlightedCount',
    'ControllerSupport', 'IsFree', 'FreeVerAvail', 'PurchaseAvail',
    'PlatformWindows', 'PlatformLinux', 'PlatformMac',
    'CategorySinglePlayer', 'CategoryMultiplayer', 'CategoryCoop',
    'CategoryMMO', 'CategoryInAppPurchase', 'CategoryVRSupport',
    'GenreIsIndie', 'GenreIsAction', 'GenreIsAdventure', 'GenreIsCasual',
    'GenreIsStrategy', 'GenreIsRPG', 'GenreIsSimulation',
    'GenreIsEarlyAccess', 'GenreIsFreeToPlay', 'GenreIsSports',
    'GenreIsRacing', 'GenreIsMassivelyMultiplayer',
    'PriceInitial', 'PriceFinal',
]


@st.cache_data
def load_raw_data():
    df = pd.read_csv("Data/train_data.csv")
    target_col = 'RecommendationCount'
    df_clean = df[SCRATCH_FEATURE_COLS + [target_col]].dropna()
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].map(
                {'True': 1, 'False': 0, True: 1, False: 0}
            ).fillna(0)
    return df_clean.astype(float), SCRATCH_FEATURE_COLS, target_col


@st.cache_resource
def load_pkl_metadata():
    return (
        joblib.load("Models/feature_columns.pkl"),
        joblib.load("Models/feature_medians.pkl"),
        joblib.load("Models/model_metrics.pkl"),
    )


@st.cache_resource
def train_scratch_model():
    df, feature_cols, target_col = load_raw_data()
    X = df[feature_cols].values
    y = df[target_col].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = YusufLinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    return model, scaler, feature_cols, rmse, r2, len(df)


def get_pkl_metrics(model_name: str):
    _, _, model_metrics = load_pkl_metadata()
    m = model_metrics.get(model_name, {})
    df, _, _ = load_raw_data()
    return m.get("RMSE", float("nan")), m.get("R2", float("nan")), len(df)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-title">🎮 STEAM ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Recommendation Predictor</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">⚙️ Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Algorithm", [
        "Linear Regression (Scratch)",
        "Linear Regression (Project)",
        "Polynomial Regression",
        "Ridge",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
    ])

    is_pkl = model_name != "Linear Regression (Scratch)"

    # ── Pricing & Audience ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💰 Pricing & Audience</div>', unsafe_allow_html=True)
    required_age   = st.number_input("Required Age",        0,   18,    0)
    price_initial  = st.number_input("Initial Price ($)",   0.0, 200.0, 9.99,  step=0.01)
    price_final    = st.number_input("Final Price ($)",     0.0, 200.0, 9.99,  step=0.01)
    metacritic     = st.number_input("Metacritic Score (0 = none)", 0, 100, 0)
    is_free        = st.checkbox("Free to Play (IsFree)")
    purchase_avail = st.checkbox("Purchase Available", value=True)
    free_ver_avail = st.checkbox("Free Version Available")

    # ── Popularity Estimates ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Popularity Estimates</div>', unsafe_allow_html=True)
    steam_spy_owners  = st.number_input("Est. Owners (SteamSpy)",  0, 100_000_000, 500_000, step=10_000)
    steam_spy_players = st.number_input("Est. Players (SteamSpy)", 0, 100_000_000, 300_000, step=10_000)

    # ── Content & Media ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎬 Content & Media</div>', unsafe_allow_html=True)
    movie_count        = st.number_input("Trailers / Movies",        0, 50,   1)
    screenshot_count   = st.number_input("Screenshots",              0, 100, 10)
    dlc_count          = st.number_input("DLC Count",                0, 200,  0)
    package_count      = st.number_input("Packages",                 1, 200,  1)
    demo_count         = st.number_input("Demo Count",               0, 20,   0)
    achievement_count  = st.number_input("Achievements",             0, 5000, 0)
    highlighted_achiev = st.number_input("Highlighted Achievements", 0, 50,   0)

    # ── Developer / Publisher ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">👥 Developer / Publisher</div>', unsafe_allow_html=True)
    developer_count = st.number_input("Developers", 1, 50, 1)
    publisher_count = st.number_input("Publishers",  1, 50, 1)

    # ── Platforms ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🖥️ Platforms</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a: plat_win   = st.checkbox("Windows", value=True)
    with col_b: plat_linux = st.checkbox("Linux")
    with col_c: plat_mac   = st.checkbox("Mac")
    ctrl_support = st.checkbox("Controller Support")

    # ── Categories ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏷️ Categories</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        cat_single = st.checkbox("Single Player", value=True)
        cat_multi  = st.checkbox("Multiplayer")
        cat_coop   = st.checkbox("Co-op")
    with col4:
        cat_mmo = st.checkbox("MMO Category")
        cat_iap = st.checkbox("In-App Purchase")
        cat_vr  = st.checkbox("VR Support")

    # ── Genres ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎲 Genres</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        g_indie       = st.checkbox("Indie")
        g_action      = st.checkbox("Action")
        g_adventure   = st.checkbox("Adventure")
        g_casual      = st.checkbox("Casual")
        g_strategy    = st.checkbox("Strategy")
        g_rpg         = st.checkbox("RPG")
    with col2:
        g_simulation  = st.checkbox("Simulation")
        g_earlyaccess = st.checkbox("Early Access")
        g_f2p         = st.checkbox("Free to Play Genre")
        g_sports      = st.checkbox("Sports")
        g_racing      = st.checkbox("Racing")
        g_mmo_genre   = st.checkbox("Massively Multiplayer")

    st.markdown("---")
    predict_btn = st.button("🔮  RUN PREDICTION")


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("## 🎮 Steam Game Recommendation Predictor")
st.markdown("Predict how many Steam recommendations a game will receive based on all its features.")
st.markdown("---")

with st.spinner("Loading model…"):
    if model_name == "Linear Regression (Scratch)":
        scratch_model, scratch_scaler, scratch_feature_cols, rmse, r2, n_samples = train_scratch_model()
    else:
        rmse, r2, n_samples = get_pkl_metrics(model_name)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f'<div class="metric-card"><div class="label">Model</div><div class="value" style="font-size:1rem">{model_name}</div><div class="sub">Active</div></div>', unsafe_allow_html=True)
with col_m2:
    r2_display = f"{r2:.3f}" if not math.isnan(r2) else "N/A"
    st.markdown(f'<div class="metric-card"><div class="label">R² Score</div><div class="value">{r2_display}</div><div class="sub">Test set</div></div>', unsafe_allow_html=True)
with col_m3:
    rmse_display = f"{rmse:,.2f}" if not math.isnan(rmse) else "N/A"
    st.markdown(f'<div class="metric-card"><div class="label">RMSE</div><div class="value">{rmse_display}</div><div class="sub">log-scale</div></div>', unsafe_allow_html=True)
with col_m4:
    st.markdown(f'<div class="metric-card"><div class="label">Training Rows</div><div class="value">{n_samples:,}</div><div class="sub">Steam games</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:

    if model_name == "Linear Regression (Scratch)":
        # 40 raw features + StandardScaler (no preprocess.py needed)
        input_vector = [
            required_age, demo_count, developer_count, dlc_count, metacritic,
            movie_count, package_count, publisher_count, screenshot_count,
            steam_spy_owners, steam_spy_players,
            achievement_count, highlighted_achiev,
            int(ctrl_support), int(is_free), int(free_ver_avail), int(purchase_avail),
            int(plat_win), int(plat_linux), int(plat_mac),
            int(cat_single), int(cat_multi), int(cat_coop),
            int(cat_mmo), int(cat_iap), int(cat_vr),
            int(g_indie), int(g_action), int(g_adventure), int(g_casual),
            int(g_strategy), int(g_rpg), int(g_simulation),
            int(g_earlyaccess), int(g_f2p), int(g_sports),
            int(g_racing), int(g_mmo_genre),
            price_initial, price_final,
        ]
        X_input = scratch_scaler.transform([input_vector])
        raw = scratch_model.predict(X_input)
        prediction = max(0.0, float(np.ravel(raw)[0]))
        rating_label, rating_color = (
            ("Overwhelmingly Negative", "#ff3333") if prediction < 500    else
            ("Mostly Negative",         "#ff6622") if prediction < 2_000  else
            ("Mixed",                   "#ffaa00") if prediction < 5_000  else
            ("Mostly Positive",         "#88dd22") if prediction < 10_000 else
            ("Very Positive",           "#22ddaa") if prediction < 50_000 else
            ("Overwhelmingly Positive", "#00d4ff")
        )

    else:
        # PKL path: all 51 features passed to predict.py → preprocess.build_inference_row()
        inputs = {
            # core numeric
            "required_age":        required_age,
            "demo_count":          demo_count,
            "developer_count":     developer_count,
            "dlc_count":           dlc_count,
            "metacritic":          metacritic,
            "movie_count":         movie_count,
            "package_count":       package_count,
            "publisher_count":     publisher_count,
            "screenshot_count":    screenshot_count,
            "steam_spy_owners":    steam_spy_owners,
            "steam_spy_players":   steam_spy_players,
            "achievement_count":   achievement_count,
            "highlighted_achiev":  highlighted_achiev,
            "price_initial":       price_initial,
            "price_final":         price_final,
            # binary flags
            "ctrl_support":        int(ctrl_support),
            "is_free":             int(is_free),
            "free_ver_avail":      int(free_ver_avail),
            "purchase_avail":      int(purchase_avail),
            "plat_win":            int(plat_win),
            "plat_linux":          int(plat_linux),
            "plat_mac":            int(plat_mac),
            "cat_single":          int(cat_single),
            "cat_multi":           int(cat_multi),
            "cat_coop":            int(cat_coop),
            "cat_mmo":             int(cat_mmo),
            "cat_iap":             int(cat_iap),
            "cat_vr":              int(cat_vr),
            "g_indie":             int(g_indie),
            "g_action":            int(g_action),
            "g_adventure":         int(g_adventure),
            "g_casual":            int(g_casual),
            "g_strategy":          int(g_strategy),
            "g_rpg":               int(g_rpg),
            "g_simulation":        int(g_simulation),
            "g_earlyaccess":       int(g_earlyaccess),
            "g_f2p":               int(g_f2p),
            "g_sports":            int(g_sports),
            "g_racing":            int(g_racing),
            "g_mmo_genre":         int(g_mmo_genre),
        }

        result       = run_predict(model_name, inputs)
        prediction   = result["prediction"]
        rating_label = result["rating"]
        rating_color = result["color"]

    # ── TTS ───────────────────────────────────────────────────────────────────
    pred_int = int(round(prediction))
    def _n2w(n):
        if n >= 1_000_000:
            m, r = n // 1_000_000, n % 1_000_000
            return f"{m} million" + (f" {r//1000} thousand" if r//1000 else "") + (f" {r%1000}" if r%1000 else "")
        elif n >= 1_000:
            return f"{n//1000} thousand" + (f" {n%1000}" if n%1000 else "")
        return str(n)

    tts = f"Based on your input, the recommendations are: {_n2w(pred_int)} Recommends."
    st.components.v1.html(f"""<script>
        window.onload = function() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var msg = new SpeechSynthesisUtterance("{tts}");
                msg.rate = 0.95; msg.pitch = 1.0; msg.volume = 1.0;
                var v = window.speechSynthesis.getVoices();
                var pref = v.find(x => x.lang.startsWith('en') && x.localService);
                if (pref) msg.voice = pref;
                window.speechSynthesis.speak(msg);
            }}
        }};
    </script>""", height=0)

    # ── Display ───────────────────────────────────────────────────────────────
    col_pred, col_detail = st.columns([1, 1])
    with col_pred:
        st.markdown(f"""
        <div class="prediction-box">
            <div class="pred-label">Predicted Recommendations</div>
            <div class="pred-value">{prediction:,.0f}</div>
            <div style="color:{rating_color};font-family:'Rajdhani',sans-serif;font-size:1.2rem;font-weight:600;margin-top:0.8rem;letter-spacing:0.08em">
                {rating_label}
            </div>
        </div>""", unsafe_allow_html=True)

    with col_detail:
        st.markdown("#### 📋 Input Summary")
        summary_data = {
            "Feature": [
                "Price", "Owners Est.", "Players Est.", "Metacritic",
                "Achievements", "Content Volume", "Platforms",
                "Genres", "Categories",
            ],
            "Value": [
                f"${price_final:.2f} (was ${price_initial:.2f})",
                f"{steam_spy_owners:,}",
                f"{steam_spy_players:,}",
                str(metacritic) if metacritic > 0 else "N/A",
                str(achievement_count),
                str(int(screenshot_count + movie_count + dlc_count + package_count)),
                ", ".join(filter(None, [
                    "Windows" if plat_win   else "",
                    "Linux"   if plat_linux else "",
                    "Mac"     if plat_mac   else "",
                ])) or "None",
                ", ".join(filter(None, [
                    "Indie"    if g_indie       else "",
                    "Action"   if g_action      else "",
                    "Adventure"if g_adventure   else "",
                    "Casual"   if g_casual      else "",
                    "Strategy" if g_strategy    else "",
                    "RPG"      if g_rpg         else "",
                    "Sim"      if g_simulation  else "",
                    "EA"       if g_earlyaccess else "",
                    "F2P"      if g_f2p         else "",
                    "Sports"   if g_sports      else "",
                    "Racing"   if g_racing      else "",
                    "MMO"      if g_mmo_genre   else "",
                ])) or "None",
                ", ".join(filter(None, [
                    "Single"  if cat_single else "",
                    "Multi"   if cat_multi  else "",
                    "Co-op"   if cat_coop   else "",
                    "MMO"     if cat_mmo    else "",
                    "IAP"     if cat_iap    else "",
                    "VR"      if cat_vr     else "",
                ])) or "None",
            ],
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

else:
    st.markdown("""
    <div style="background:#08081a;border:1px dashed #1a1a3a;border-radius:4px;
                padding:3rem;text-align:center;font-family:'IBM Plex Mono',monospace;">
        <div style="font-size:3rem;margin-bottom:1rem">🔮</div>
        <div style="font-size:1rem;font-weight:500;color:#4a4a8a;letter-spacing:0.1em;text-transform:uppercase">
            Fill in the sidebar inputs
        </div>
        <div style="font-size:0.8rem;margin-top:0.5rem;color:#2a2a5a">
            Configure all game features in the sidebar, then click RUN PREDICTION
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature Importance ────────────────────────────────────────────────────────
with st.expander("📈 Feature Importance / Model Coefficients", expanded=False):
    import plotly.express as px

    if model_name == "Linear Regression (Scratch)":
        display_feature_cols = scratch_feature_cols
        display_model = scratch_model
    else:
        display_feature_cols, _, _ = load_pkl_metadata()
        display_model = joblib.load(PKL_PATHS[model_name])

    if hasattr(display_model, "feature_importances_"):
        imp = display_model.feature_importances_
        bar_label, color_scale = "Importance", ["#0a0a2a", "#00d4ff"]
    elif hasattr(display_model, "coef_"):
        imp = display_model.coef_
        bar_label, color_scale = "Coefficient", ["#ff3333", "#0a0a2a", "#00d4ff"]
    else:
        imp = None

    if imp is not None:
        n = min(len(display_feature_cols), len(imp))
        coefs = pd.DataFrame({"Feature": display_feature_cols[:n], bar_label: imp[:n]})
        coefs = coefs.reindex(coefs[bar_label].abs().sort_values(ascending=False).index).head(20)
        fig = px.bar(coefs, x=bar_label, y="Feature", orientation="h",
                     color=bar_label, color_continuous_scale=color_scale, template="plotly_dark")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Mono", color="#6060a0"),
            height=500, yaxis=dict(autorange="reversed"),
            showlegend=False, margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model.")

# ── Data Explorer ─────────────────────────────────────────────────────────────
with st.expander("🗂️ Training Data Explorer", expanded=False):
    df_preview, _, _ = load_raw_data()
    if model_name == "Linear Regression (Scratch)":
        st.markdown(f"**{len(df_preview):,} rows** — 40 raw features (Scratch model)")
        st.dataframe(df_preview.head(50), use_container_width=True, height=300)
    else:
        feat_cols_pkl, _, _ = load_pkl_metadata()
        st.markdown(f"**{len(feat_cols_pkl)} feature columns** used by PKL models:")
        st.write(feat_cols_pkl)
        st.markdown(f"**{len(df_preview):,} rows** of raw training data:")
        st.dataframe(df_preview.head(50), use_container_width=True, height=300)
