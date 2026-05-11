"""
App.py  –  Steam ML Predictor  (Milestone 1 + Milestone 2)
──────────────────────────────────────────────────────────
Milestone 1 tab : Regression  → predict RecommendationCount
Milestone 2 tab : Classification → predict GamePopularity (Low / Medium / High)

All 62 user-facing CSV columns are collected into `raw_inputs` (a plain dict
with exact CSV column names as keys).  Pass it to whatever test/inference
function you add later.

cols excluded from UI (text blobs / IDs):
    QueryID, ResponseID, QueryName, ResponseName, AboutText, ShortDescrip,
    DetailedDescrip, PCMinReqsText, PCRecReqsText, LinuxMinReqsText,
    LinuxRecReqsText, MacMinReqsText, MacRecReqsText, AllText,
    AllText_Cleaned, Reviews, SupportedLanguages
"""

import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

import io
import math
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
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
    width: 400px !important;
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

.class-box {
    background: linear-gradient(135deg, #030318 0%, #08251a 100%);
    border: 1px solid #00ff8844; border-radius: 4px; padding: 2.5rem;
    text-align: center;
    box-shadow: 0 0 60px rgba(0,255,136,0.08), inset 0 0 40px rgba(0,255,136,0.03);
    position: relative; overflow: hidden;
}
.class-box::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #00ff88, transparent);
}

.stNumberInput input, .stSelectbox select, .stTextArea textarea, .stTextInput input {
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
.stSelectbox label, .stNumberInput label, .stTextArea label, .stTextInput label { color: #5050a0 !important; font-size: 0.72rem !important; letter-spacing: 0.08em; }
.upload-hint { font-size:0.72rem; color:#3a3a7a !important; font-family:'IBM Plex Mono',monospace; line-height:1.6; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Constants & feature lists ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

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

# Classification model paths (Milestone 2)
CLF_PKL_PATHS = {
    "Logistic Regression":     _BASE / "saved_models" / "logistic.pkl",
    "Random Forest (Clf)":     _BASE / "saved_models" / "random_forest.pkl",
    "Gradient Boosting (Clf)": _BASE / "saved_models" / "gradient_boosting.pkl",
    "XGBoost (Clf)":           _BASE / "saved_models" / "xgboost.pkl",
}

CLASS_LABELS  = {0: "Low", 1: "Medium", 2: "High"}
CLASS_COLORS  = {0: "#ff6622", 1: "#ffaa00", 2: "#22ddaa"}


# ═══════════════════════════════════════════════════════════════════════════════
# ── Helpers ───────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def parse_txt_file(content: str) -> dict:
    """Parse key=value text file into a dict of values."""
    result = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            try:
                result[k] = float(v)
            except ValueError:
                result[k] = v  # keep string values (e.g. ReleaseDate, URLs)
    return result


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
def load_clf_metadata():
    clf_cols, clf_metrics = None, {}
    try:
        clf_cols = joblib.load("Models/clf_feature_columns.pkl")
    except Exception:
        pass
    try:
        clf_metrics = joblib.load("Models/clf_model_metrics.pkl")
    except Exception:
        pass
    return clf_cols, clf_metrics


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


def rating_from_prediction(prediction: float):
    if   prediction < 500:    return "Overwhelmingly Negative", "#ff3333"
    elif prediction < 2_000:  return "Mostly Negative",         "#ff6622"
    elif prediction < 5_000:  return "Mixed",                   "#ffaa00"
    elif prediction < 10_000: return "Mostly Positive",         "#88dd22"
    elif prediction < 50_000: return "Very Positive",           "#22ddaa"
    else:                      return "Overwhelmingly Positive", "#00d4ff"


def build_legacy_inputs(raw: dict) -> dict:
    """
    Convert raw_inputs (CSV column names) into the short-key dict that
    predict.py / preprocess.py expect.  Used by Milestone 1 inference.
    """
    return {
        "required_age":       raw["RequiredAge"],
        "demo_count":         raw["DemoCount"],
        "developer_count":    raw["DeveloperCount"],
        "dlc_count":          raw["DLCCount"],
        "metacritic":         raw["Metacritic"],
        "movie_count":        raw["MovieCount"],
        "package_count":      raw["PackageCount"],
        "publisher_count":    raw["PublisherCount"],
        "screenshot_count":   raw["ScreenshotCount"],
        "steam_spy_owners":   raw["SteamSpyOwners"],
        "steam_spy_players":  raw["SteamSpyPlayersEstimate"],
        "achievement_count":  raw["AchievementCount"],
        "highlighted_achiev": raw["AchievementHighlightedCount"],
        "price_initial":      raw["PriceInitial"],
        "price_final":        raw["PriceFinal"],
        "ctrl_support":       int(raw["ControllerSupport"]),
        "is_free":            int(raw["IsFree"]),
        "free_ver_avail":     int(raw["FreeVerAvail"]),
        "purchase_avail":     int(raw["PurchaseAvail"]),
        "plat_win":           int(raw["PlatformWindows"]),
        "plat_linux":         int(raw["PlatformLinux"]),
        "plat_mac":           int(raw["PlatformMac"]),
        "cat_single":         int(raw["CategorySinglePlayer"]),
        "cat_multi":          int(raw["CategoryMultiplayer"]),
        "cat_coop":           int(raw["CategoryCoop"]),
        "cat_mmo":            int(raw["CategoryMMO"]),
        "cat_iap":            int(raw["CategoryInAppPurchase"]),
        "cat_vr":             int(raw["CategoryVRSupport"]),
        "g_indie":            int(raw["GenreIsIndie"]),
        "g_action":           int(raw["GenreIsAction"]),
        "g_adventure":        int(raw["GenreIsAdventure"]),
        "g_casual":           int(raw["GenreIsCasual"]),
        "g_strategy":         int(raw["GenreIsStrategy"]),
        "g_rpg":              int(raw["GenreIsRPG"]),
        "g_simulation":       int(raw["GenreIsSimulation"]),
        "g_earlyaccess":      int(raw["GenreIsEarlyAccess"]),
        "g_f2p":              int(raw["GenreIsFreeToPlay"]),
        "g_sports":           int(raw["GenreIsSports"]),
        "g_racing":           int(raw["GenreIsRacing"]),
        "g_mmo_genre":        int(raw["GenreIsMassivelyMultiplayer"]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ── Sidebar ───────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="app-title">🎮 STEAM ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Milestone 1 + 2 Predictor</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Model selector ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Milestone</div>', unsafe_allow_html=True)
    milestone = st.selectbox("Task", ["Milestone 1 – Regression", "Milestone 2 – Classification"])

    if milestone.startswith("Milestone 1"):
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
    else:
        model_name = st.selectbox("Classifier", list(CLF_PKL_PATHS.keys()))

    # ══════════════════════════════════════════════════════════════════════════
    # ALL INPUT WIDGETS — one per remaining CSV column
    # Keys use the exact CSV column names for direct use in raw_inputs dict.
    # ══════════════════════════════════════════════════════════════════════════

    # ── Basic Info ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📅 Basic Info</div>', unsafe_allow_html=True)
    ReleaseDate = st.text_input("Release Date (e.g. Nov 1 2020)", value="Jan 1 2020")

    # ── Pricing & Availability ────────────────────────────────────────────────
    st.markdown('<div class="section-header">💰 Pricing & Availability</div>', unsafe_allow_html=True)
    PriceCurrency = st.selectbox("Price Currency", ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "RUB", "BRL"])
    PriceInitial  = st.number_input("Initial Price",  0.0, 500.0, 9.99,  step=0.01, format="%.2f")
    PriceFinal    = st.number_input("Final Price",    0.0, 500.0, 9.99,  step=0.01, format="%.2f")
    RequiredAge   = st.number_input("Required Age",   0,   18,    0)
    Metacritic    = st.number_input("Metacritic Score (0 = none)", 0, 100, 0)

    col_bool1, col_bool2 = st.columns(2)
    with col_bool1:
        IsFree          = st.checkbox("Is Free")
        FreeVerAvail    = st.checkbox("Free Version Available")
        PurchaseAvail   = st.checkbox("Purchase Available", value=True)
    with col_bool2:
        SubscriptionAvail = st.checkbox("Subscription Available")

    # ── Audience Estimates ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 SteamSpy Estimates</div>', unsafe_allow_html=True)
    SteamSpyOwners          = st.number_input("Owners Estimate",          0, 100_000_000, 500_000,  step=10_000)
    SteamSpyOwnersVariance  = st.number_input("Owners Variance",          0, 100_000_000, 50_000,   step=1_000)
    SteamSpyPlayersEstimate = st.number_input("Players Estimate",         0, 100_000_000, 300_000,  step=10_000)
    SteamSpyPlayersVariance = st.number_input("Players Variance",         0, 100_000_000, 30_000,   step=1_000)

    # ── Content & Media ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎬 Content & Media</div>', unsafe_allow_html=True)
    MovieCount      = st.number_input("Trailers / Movies",       0, 50,   1)
    ScreenshotCount = st.number_input("Screenshots",             0, 100, 10)
    DLCCount        = st.number_input("DLC Count",               0, 500,  0)
    PackageCount    = st.number_input("Packages",                1, 500,  1)
    DemoCount       = st.number_input("Demo Count",              0, 20,   0)

    # ── Achievements ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Achievements</div>', unsafe_allow_html=True)
    AchievementCount            = st.number_input("Total Achievements",       0, 10000, 0)
    AchievementHighlightedCount = st.number_input("Highlighted Achievements", 0, 100,   0)

    # ── Developer / Publisher ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">👥 Developer / Publisher</div>', unsafe_allow_html=True)
    DeveloperCount = st.number_input("Developer Count",  1, 100, 1)
    PublisherCount = st.number_input("Publisher Count",  1, 100, 1)

    # ── Platforms ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🖥️ Platforms</div>', unsafe_allow_html=True)
    col_plat1, col_plat2, col_plat3 = st.columns(3)
    with col_plat1: PlatformWindows = st.checkbox("Windows", value=True)
    with col_plat2: PlatformLinux   = st.checkbox("Linux")
    with col_plat3: PlatformMac     = st.checkbox("Mac")
    ControllerSupport = st.checkbox("Controller Support")

    # ── System Requirements Availability ──────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ System Req. Flags</div>', unsafe_allow_html=True)
    col_req1, col_req2 = st.columns(2)
    with col_req1:
        PCReqsHaveMin    = st.checkbox("PC Min Reqs",    value=True)
        LinuxReqsHaveMin = st.checkbox("Linux Min Reqs")
        MacReqsHaveMin   = st.checkbox("Mac Min Reqs")
    with col_req2:
        PCReqsHaveRec    = st.checkbox("PC Rec Reqs")
        LinuxReqsHaveRec = st.checkbox("Linux Rec Reqs")
        MacReqsHaveRec   = st.checkbox("Mac Rec Reqs")

    # ── Categories ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏷️ Categories</div>', unsafe_allow_html=True)
    col_cat1, col_cat2 = st.columns(2)
    with col_cat1:
        CategorySinglePlayer       = st.checkbox("Single Player", value=True)
        CategoryMultiplayer        = st.checkbox("Multiplayer")
        CategoryCoop               = st.checkbox("Co-op")
        CategoryMMO                = st.checkbox("MMO Category")
    with col_cat2:
        CategoryInAppPurchase      = st.checkbox("In-App Purchase")
        CategoryIncludeSrcSDK      = st.checkbox("Includes Src SDK")
        CategoryIncludeLevelEditor = st.checkbox("Level Editor")
        CategoryVRSupport          = st.checkbox("VR Support")

    # ── Genres ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎲 Genres</div>', unsafe_allow_html=True)
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        GenreIsNonGame           = st.checkbox("Non-Game")
        GenreIsIndie             = st.checkbox("Indie")
        GenreIsAction            = st.checkbox("Action")
        GenreIsAdventure         = st.checkbox("Adventure")
        GenreIsCasual            = st.checkbox("Casual")
        GenreIsStrategy          = st.checkbox("Strategy")
    with col_g2:
        GenreIsRPG               = st.checkbox("RPG")
        GenreIsSimulation        = st.checkbox("Simulation")
        GenreIsEarlyAccess       = st.checkbox("Early Access")
        GenreIsFreeToPlay        = st.checkbox("Free to Play Genre")
        GenreIsSports            = st.checkbox("Sports")
        GenreIsRacing            = st.checkbox("Racing")
        GenreIsMassivelyMultiplayer = st.checkbox("Massively Multiplayer")

    # ── Support & Links ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔗 Support & Links</div>', unsafe_allow_html=True)
    SupportEmail     = st.text_input("Support Email",   value="")
    SupportURL       = st.text_input("Support URL",     value="")
    Website          = st.text_input("Website URL",     value="")

    # ── Media / Metadata URLs ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🖼️ Media & Metadata</div>', unsafe_allow_html=True)
    HeaderImage      = st.text_input("Header Image URL",  value="")
    Background       = st.text_input("Background URL",    value="")

    # ── Legal / Notices ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📜 Legal & Notices</div>', unsafe_allow_html=True)
    DRMNotice        = st.text_input("DRM Notice",           value="")
    ExtUserAcctNotice= st.text_input("Ext. User Acct Notice",value="")
    LegalNotice      = st.text_input("Legal Notice",         value="")

    st.markdown("---")
    predict_btn = st.button("🔮  RUN PREDICTION")


# ═══════════════════════════════════════════════════════════════════════════════
# ── raw_inputs: single dict with ALL user-provided values, CSV column names ───
# ═══════════════════════════════════════════════════════════════════════════════
# This is the canonical object to pass to any test / inference function.
# Keys match the CSV column names exactly (sans the dropped text-blob columns).

raw_inputs: dict = {
    # ── Basic ─────────────────────────────────────────────────────────────────
    "ReleaseDate":                  ReleaseDate,

    # ── Numeric counts & scores ───────────────────────────────────────────────
    "RequiredAge":                  int(RequiredAge),
    "DemoCount":                    int(DemoCount),
    "DeveloperCount":               int(DeveloperCount),
    "DLCCount":                     int(DLCCount),
    "Metacritic":                   int(Metacritic),
    "MovieCount":                   int(MovieCount),
    "PackageCount":                 int(PackageCount),
    "PublisherCount":               int(PublisherCount),
    "ScreenshotCount":              int(ScreenshotCount),
    "AchievementCount":             int(AchievementCount),
    "AchievementHighlightedCount":  int(AchievementHighlightedCount),

    # ── SteamSpy ──────────────────────────────────────────────────────────────
    "SteamSpyOwners":               int(SteamSpyOwners),
    "SteamSpyOwnersVariance":       int(SteamSpyOwnersVariance),
    "SteamSpyPlayersEstimate":      int(SteamSpyPlayersEstimate),
    "SteamSpyPlayersVariance":      int(SteamSpyPlayersVariance),

    # ── Pricing ───────────────────────────────────────────────────────────────
    "PriceCurrency":                PriceCurrency,
    "PriceInitial":                 float(PriceInitial),
    "PriceFinal":                   float(PriceFinal),

    # ── Boolean flags ─────────────────────────────────────────────────────────
    "ControllerSupport":            bool(ControllerSupport),
    "IsFree":                       bool(IsFree),
    "FreeVerAvail":                 bool(FreeVerAvail),
    "PurchaseAvail":                bool(PurchaseAvail),
    "SubscriptionAvail":            bool(SubscriptionAvail),

    # ── Platforms ─────────────────────────────────────────────────────────────
    "PlatformWindows":              bool(PlatformWindows),
    "PlatformLinux":                bool(PlatformLinux),
    "PlatformMac":                  bool(PlatformMac),

    # ── System requirements ───────────────────────────────────────────────────
    "PCReqsHaveMin":                bool(PCReqsHaveMin),
    "PCReqsHaveRec":                bool(PCReqsHaveRec),
    "LinuxReqsHaveMin":             bool(LinuxReqsHaveMin),
    "LinuxReqsHaveRec":             bool(LinuxReqsHaveRec),
    "MacReqsHaveMin":               bool(MacReqsHaveMin),
    "MacReqsHaveRec":               bool(MacReqsHaveRec),

    # ── Categories ────────────────────────────────────────────────────────────
    "CategorySinglePlayer":         bool(CategorySinglePlayer),
    "CategoryMultiplayer":          bool(CategoryMultiplayer),
    "CategoryCoop":                 bool(CategoryCoop),
    "CategoryMMO":                  bool(CategoryMMO),
    "CategoryInAppPurchase":        bool(CategoryInAppPurchase),
    "CategoryIncludeSrcSDK":        bool(CategoryIncludeSrcSDK),
    "CategoryIncludeLevelEditor":   bool(CategoryIncludeLevelEditor),
    "CategoryVRSupport":            bool(CategoryVRSupport),

    # ── Genres ────────────────────────────────────────────────────────────────
    "GenreIsNonGame":               bool(GenreIsNonGame),
    "GenreIsIndie":                 bool(GenreIsIndie),
    "GenreIsAction":                bool(GenreIsAction),
    "GenreIsAdventure":             bool(GenreIsAdventure),
    "GenreIsCasual":                bool(GenreIsCasual),
    "GenreIsStrategy":              bool(GenreIsStrategy),
    "GenreIsRPG":                   bool(GenreIsRPG),
    "GenreIsSimulation":            bool(GenreIsSimulation),
    "GenreIsEarlyAccess":           bool(GenreIsEarlyAccess),
    "GenreIsFreeToPlay":            bool(GenreIsFreeToPlay),
    "GenreIsSports":                bool(GenreIsSports),
    "GenreIsRacing":                bool(GenreIsRacing),
    "GenreIsMassivelyMultiplayer":  bool(GenreIsMassivelyMultiplayer),

    # ── Support & links ───────────────────────────────────────────────────────
    "SupportEmail":                 SupportEmail,
    "SupportURL":                   SupportURL,
    "Website":                      Website,

    # ── Media / metadata ──────────────────────────────────────────────────────
    "HeaderImage":                  HeaderImage,
    "Background":                   Background,

    # ── Legal ─────────────────────────────────────────────────────────────────
    "DRMNotice":                    DRMNotice,
    "ExtUserAcctNotice":            ExtUserAcctNotice,
    "LegalNotice":                  LegalNotice,
}

# Legacy short-key dict required by existing predict.py / preprocess.py
inputs = build_legacy_inputs(raw_inputs)

# Also expose the scratch-model input vector using raw_inputs keys
_ev = raw_inputs   # alias for the blocks below


# ═══════════════════════════════════════════════════════════════════════════════
# ── Main area ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["📈 Milestone 1 – Regression", "🏷️ Milestone 2 – Classification"])


# ══════════════════════════════════════════════════════
# TAB 1 – REGRESSION (Milestone 1)
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown("## 🎮 Steam Game Recommendation Predictor")
    st.markdown("Predict how many Steam recommendations a game will receive based on its features.")
    st.markdown("---")

    if milestone.startswith("Milestone 2"):
        st.warning("⬅ Switch the **Milestone** selector in the sidebar to **Milestone 1 – Regression** to use this tab.")
    else:
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

        if predict_btn:
            if model_name == "Linear Regression (Scratch)":
                input_vector = [
                    _ev["RequiredAge"], _ev["DemoCount"], _ev["DeveloperCount"],
                    _ev["DLCCount"], _ev["Metacritic"], _ev["MovieCount"],
                    _ev["PackageCount"], _ev["PublisherCount"], _ev["ScreenshotCount"],
                    _ev["SteamSpyOwners"], _ev["SteamSpyPlayersEstimate"],
                    _ev["AchievementCount"], _ev["AchievementHighlightedCount"],
                    int(_ev["ControllerSupport"]), int(_ev["IsFree"]),
                    int(_ev["FreeVerAvail"]),      int(_ev["PurchaseAvail"]),
                    int(_ev["PlatformWindows"]),   int(_ev["PlatformLinux"]),
                    int(_ev["PlatformMac"]),       int(_ev["CategorySinglePlayer"]),
                    int(_ev["CategoryMultiplayer"]),int(_ev["CategoryCoop"]),
                    int(_ev["CategoryMMO"]),        int(_ev["CategoryInAppPurchase"]),
                    int(_ev["CategoryVRSupport"]),  int(_ev["GenreIsIndie"]),
                    int(_ev["GenreIsAction"]),      int(_ev["GenreIsAdventure"]),
                    int(_ev["GenreIsCasual"]),      int(_ev["GenreIsStrategy"]),
                    int(_ev["GenreIsRPG"]),         int(_ev["GenreIsSimulation"]),
                    int(_ev["GenreIsEarlyAccess"]), int(_ev["GenreIsFreeToPlay"]),
                    int(_ev["GenreIsSports"]),      int(_ev["GenreIsRacing"]),
                    int(_ev["GenreIsMassivelyMultiplayer"]),
                    _ev["PriceInitial"], _ev["PriceFinal"],
                ]
                X_input = scratch_scaler.transform([input_vector])
                raw = scratch_model.predict(X_input)
                prediction = max(0.0, float(np.ravel(raw)[0]))
                rating_label, rating_color = rating_from_prediction(prediction)
            else:
                result       = run_predict(model_name, inputs)
                prediction   = result["prediction"]
                rating_label = result["rating"]
                rating_color = result["color"]

            # TTS
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
                        "Achievements", "Content Volume", "Platforms", "Genres", "Categories",
                    ],
                    "Value": [
                        f"${_ev['PriceFinal']:.2f} (was ${_ev['PriceInitial']:.2f})",
                        f"{_ev['SteamSpyOwners']:,}",
                        f"{_ev['SteamSpyPlayersEstimate']:,}",
                        str(_ev['Metacritic']) if _ev['Metacritic'] > 0 else "N/A",
                        str(_ev['AchievementCount']),
                        str(_ev['ScreenshotCount'] + _ev['MovieCount'] + _ev['DLCCount'] + _ev['PackageCount']),
                        ", ".join(filter(None, [
                            "Windows" if _ev['PlatformWindows'] else "",
                            "Linux"   if _ev['PlatformLinux']   else "",
                            "Mac"     if _ev['PlatformMac']     else "",
                        ])) or "None",
                        ", ".join(filter(None, [
                            "Indie"      if _ev['GenreIsIndie']             else "",
                            "Action"     if _ev['GenreIsAction']            else "",
                            "Adventure"  if _ev['GenreIsAdventure']         else "",
                            "Casual"     if _ev['GenreIsCasual']            else "",
                            "Strategy"   if _ev['GenreIsStrategy']          else "",
                            "RPG"        if _ev['GenreIsRPG']               else "",
                            "Sim"        if _ev['GenreIsSimulation']        else "",
                            "EA"         if _ev['GenreIsEarlyAccess']       else "",
                            "F2P"        if _ev['GenreIsFreeToPlay']        else "",
                            "Sports"     if _ev['GenreIsSports']            else "",
                            "Racing"     if _ev['GenreIsRacing']            else "",
                            "MMO"        if _ev['GenreIsMassivelyMultiplayer'] else "",
                        ])) or "None",
                        ", ".join(filter(None, [
                            "Single"  if _ev['CategorySinglePlayer']  else "",
                            "Multi"   if _ev['CategoryMultiplayer']   else "",
                            "Co-op"   if _ev['CategoryCoop']          else "",
                            "MMO"     if _ev['CategoryMMO']           else "",
                            "IAP"     if _ev['CategoryInAppPurchase'] else "",
                            "VR"      if _ev['CategoryVRSupport']     else "",
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

        # ── Feature Importance ─────────────────────────────────────────────────
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

        # ── Data Explorer ──────────────────────────────────────────────────────
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

        # ── raw_inputs inspector ───────────────────────────────────────────────
        with st.expander("🔍 raw_inputs Object (all 62 fields)", expanded=False):
            st.markdown("This is the `raw_inputs` dict that will be passed to your test function.")
            ri_display = {k: str(v) for k, v in raw_inputs.items()}
            st.dataframe(
                pd.DataFrame(ri_display.items(), columns=["Column", "Value"]),
                hide_index=True, use_container_width=True, height=400,
            )


# ══════════════════════════════════════════════════════
# TAB 2 – CLASSIFICATION (Milestone 2)
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🏷️ Game Popularity Classifier")
    st.markdown("Classify a game into **Low / Medium / High** popularity based on its features.")
    st.markdown("---")

    selected_clf = model_name if milestone.startswith("Milestone 2") else list(CLF_PKL_PATHS.keys())[0]
    clf_path = CLF_PKL_PATHS[selected_clf]

    clf_cols, clf_metrics = load_clf_metadata()
    clf_acc = clf_metrics.get(selected_clf, {}).get("accuracy", float("nan"))

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown(f'<div class="metric-card"><div class="label">Classifier</div><div class="value" style="font-size:0.9rem">{selected_clf}</div><div class="sub">Active</div></div>', unsafe_allow_html=True)
    with col_c2:
        acc_disp = f"{clf_acc:.3f}" if not (isinstance(clf_acc, float) and math.isnan(clf_acc)) else "N/A"
        st.markdown(f'<div class="metric-card"><div class="label">Test Accuracy</div><div class="value">{acc_disp}</div><div class="sub">Held-out set</div></div>', unsafe_allow_html=True)
    with col_c3:
        st.markdown(f'<div class="metric-card"><div class="label">Classes</div><div class="value">3</div><div class="sub">Low / Medium / High</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if predict_btn:
        if not clf_path.exists():
            st.error(
                f"⚠️ Model file not found: `{clf_path}`\n\n"
                "Please run `notebook_milestone2.py` to train the classifiers."
            )
        else:
            try:
                clf_model = joblib.load(clf_path)

                from preprocess import build_inference_row
                feat_cols_pkl, feat_medians, _ = load_pkl_metadata()

                if hasattr(clf_model, "feature_names_in_"):
                    clf_feature_names = list(clf_model.feature_names_in_)
                elif clf_cols is not None:
                    clf_feature_names = clf_cols
                else:
                    clf_feature_names = feat_cols_pkl

                X_raw = build_inference_row(inputs, feat_cols_pkl, feat_medians)
                full_row = dict(zip(feat_cols_pkl, X_raw[0]))

                X_clf = np.array(
                    [[full_row.get(c, 0.0) for c in clf_feature_names]],
                    dtype=float
                )

                raw_pred = clf_model.predict(X_clf)[0]
                if isinstance(raw_pred, (int, np.integer)):
                    pop_label = CLASS_LABELS.get(int(raw_pred), str(raw_pred))
                    pop_color = CLASS_COLORS.get(int(raw_pred), "#00d4ff")
                else:
                    pop_label = str(raw_pred)
                    pop_color = {"Low": "#ff6622", "Medium": "#ffaa00", "High": "#22ddaa"}.get(pop_label, "#00d4ff")

                proba = None
                if hasattr(clf_model, "predict_proba"):
                    try:
                        proba = clf_model.predict_proba(X_clf)[0]
                    except Exception:
                        pass

                col_res, col_info = st.columns([1, 1])
                with col_res:
                    st.markdown(f"""
                    <div class="class-box">
                        <div style="color:#3a8a6a;font-size:0.65rem;letter-spacing:0.25em;text-transform:uppercase;margin-bottom:0.8rem">
                            Predicted Popularity
                        </div>
                        <div style="font-family:'Rajdhani',sans-serif;color:{pop_color};font-size:4rem;font-weight:700;line-height:1;">
                            {pop_label}
                        </div>
                        <div style="color:#3a6a4a;font-size:0.75rem;margin-top:0.6rem">
                            Game Popularity Category
                        </div>
                    </div>""", unsafe_allow_html=True)

                with col_info:
                    if proba is not None:
                        import plotly.graph_objects as go
                        n_classes = len(proba)
                        labels = [CLASS_LABELS.get(i, str(i)) for i in range(n_classes)]
                        colors = [CLASS_COLORS.get(i, "#00d4ff") for i in range(n_classes)]
                        fig_p = go.Figure(go.Bar(
                            x=labels, y=proba,
                            marker_color=colors,
                            text=[f"{p*100:.1f}%" for p in proba],
                            textposition="outside",
                        ))
                        fig_p.update_layout(
                            title="Class Probabilities",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="IBM Plex Mono", color="#8080b0"),
                            yaxis=dict(range=[0, 1.15], tickformat=".0%"),
                            height=300, margin=dict(l=10, r=10, t=40, b=10),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_p, use_container_width=True)
                    else:
                        st.markdown("#### 📋 Prediction Details")
                        st.dataframe(pd.DataFrame({
                            "Feature": ["Owners", "Players", "Price", "Metacritic", "Platforms"],
                            "Value": [
                                f"{_ev['SteamSpyOwners']:,}",
                                f"{_ev['SteamSpyPlayersEstimate']:,}",
                                f"${_ev['PriceFinal']:.2f}",
                                str(_ev['Metacritic']) if _ev['Metacritic'] > 0 else "N/A",
                                str(int(_ev['PlatformWindows']) + int(_ev['PlatformLinux']) + int(_ev['PlatformMac'])),
                            ]
                        }), hide_index=True, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        st.markdown("""
        <div style="background:#08081a;border:1px dashed #1a3a2a;border-radius:4px;
                    padding:3rem;text-align:center;font-family:'IBM Plex Mono',monospace;">
            <div style="font-size:3rem;margin-bottom:1rem">🏷️</div>
            <div style="font-size:1rem;font-weight:500;color:#3a6a4a;letter-spacing:0.1em;text-transform:uppercase">
                Configure sidebar inputs
            </div>
            <div style="font-size:0.8rem;margin-top:0.5rem;color:#2a4a3a">
                Select <b>Milestone 2 – Classification</b> in the sidebar, then click RUN PREDICTION
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Classifier feature importance ──────────────────────────────────────────
    if clf_path.exists():
        with st.expander("📈 Classifier Feature Importance", expanded=False):
            import plotly.express as px
            try:
                clf_fi = joblib.load(clf_path)
                if hasattr(clf_fi, "feature_names_in_"):
                    fi_cols = list(clf_fi.feature_names_in_)
                elif clf_cols:
                    fi_cols = clf_cols
                else:
                    fi_cols = joblib.load("Models/feature_columns.pkl")
                if hasattr(clf_fi, "feature_importances_"):
                    imp = clf_fi.feature_importances_
                    bar_label = "Importance"
                    color_scale = ["#0a0a2a", "#00ff88"]
                elif hasattr(clf_fi, "coef_"):
                    imp = clf_fi.coef_[0] if clf_fi.coef_.ndim > 1 else clf_fi.coef_
                    bar_label = "Coefficient"
                    color_scale = ["#ff3333", "#0a0a2a", "#00ff88"]
                else:
                    imp = None

                if imp is not None:
                    n = min(len(fi_cols), len(imp))
                    coefs_df = pd.DataFrame({"Feature": fi_cols[:n], bar_label: imp[:n]})
                    coefs_df = coefs_df.reindex(coefs_df[bar_label].abs().sort_values(ascending=False).index).head(20)
                    fig2 = px.bar(coefs_df, x=bar_label, y="Feature", orientation="h",
                                  color=bar_label, color_continuous_scale=color_scale, template="plotly_dark")
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="IBM Plex Mono", color="#6060a0"),
                        height=500, yaxis=dict(autorange="reversed"),
                        showlegend=False, margin=dict(l=20, r=20, t=20, b=20),
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Feature importance not available for this classifier.")
            except Exception as e:
                st.warning(f"Could not load feature importance: {e}")

    # ── raw_inputs inspector ───────────────────────────────────────────────────
    with st.expander("🔍 raw_inputs Object (all 62 fields)", expanded=False):
        st.markdown("This is the `raw_inputs` dict that will be passed to your test function.")
        ri_display = {k: str(v) for k, v in raw_inputs.items()}
        st.dataframe(
            pd.DataFrame(ri_display.items(), columns=["Column", "Value"]),
            hide_index=True, use_container_width=True, height=400,
        )