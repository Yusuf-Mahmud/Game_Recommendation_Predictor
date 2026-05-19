"""
App.py  –  Steam ML Predictor  (Milestone 1 + Milestone 2)
──────────────────────────────────────────────────────────
Milestone 1 tab : Regression  → predict RecommendationCount
Milestone 2 tab : Classification → predict GamePopularity (Low / Medium / High)

All non-text-blob CSV columns are collected into `raw_inputs` and forwarded to
TestScript.predict_game() for inference.

Columns excluded from UI (text blobs / IDs):
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

import math
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── TestScript is the unified inference entry-point ───────────────────────────
from TestScript import predict_game

# ── Milestone 1 scratch model (kept as-is) ────────────────────────────────────
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
    width: 420px !important;
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
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Constants ──────────────────────────────────────────────────────────────────
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

CLASS_LABELS = {0: "Low", 1: "Medium", 2: "High"}
CLASS_COLORS = {0: "#ff6622", 1: "#ffaa00", 2: "#22ddaa"}
POP_COLOR_MAP = {"Low": "#ff6622", "Medium": "#ffaa00", "High": "#22ddaa"}


# ═══════════════════════════════════════════════════════════════════════════════
# ── Helpers ────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

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


_PKL_ALIAS_MAP = {
    "linear":            "Linear Regression (Project)",
    "polynomial":        "Polynomial Regression",
    "ridge":             "Ridge",
    "random_forest":     "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost":           "XGBoost",
}

def get_pkl_metrics(model_name: str):
    _, _, model_metrics = load_pkl_metadata()
    canonical = _PKL_ALIAS_MAP.get(model_name, model_name)
    m = model_metrics.get(canonical, {})
    df, _, _ = load_raw_data()
    return m.get("RMSE", float("nan")), m.get("R2", float("nan")), len(df)


def rating_from_prediction(prediction: float):
    if   prediction < 500:    return "Overwhelmingly Negative", "#ff3333"
    elif prediction < 2_000:  return "Mostly Negative",         "#ff6622"
    elif prediction < 5_000:  return "Mixed",                   "#ffaa00"
    elif prediction < 10_000: return "Mostly Positive",         "#88dd22"
    elif prediction < 50_000: return "Very Positive",           "#22ddaa"
    else:                     return "Overwhelmingly Positive", "#00d4ff"


# ═══════════════════════════════════════════════════════════════════════════════
# ── Sidebar ────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="app-title">🎮 STEAM ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Milestone 1 + 2 Predictor</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Milestone / model selector ────────────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Milestone</div>', unsafe_allow_html=True)
    milestone = st.selectbox("Task", ["Milestone 1 – Regression", "Milestone 2 – Classification"])

    if milestone.startswith("Milestone 1"):
        model_name = st.selectbox("Algorithm", [
            "Linear Regression (Scratch)",
            "linear",
            "polynomial",
            "ridge",
            "random_forest",
            "gradient_boosting",
            "xgboost",
        ])
    else:
        model_name = st.selectbox("Classifier", [
            "logistic",
            "random_forest",
            "gradient_boosting",
            "xgboost",
        ])

    # ══════════════════════════════════════════════════════════════════════════
    # ALL INPUT WIDGETS — one per remaining CSV column (drops listed above)
    # Keys use the exact CSV column names.
    # ══════════════════════════════════════════════════════════════════════════

    # ── Basic Info ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📅 Basic Info</div>', unsafe_allow_html=True)
    import datetime as _dt
    _rd = st.date_input("Release Date", value=_dt.date(2020, 1, 1))
    ReleaseDate = _rd.strftime("%b %e %Y")

    # ── Pricing & Availability ────────────────────────────────────────────────
    st.markdown('<div class="section-header">💰 Pricing & Availability</div>', unsafe_allow_html=True)
    PriceCurrency = st.selectbox("Price Currency", ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "RUB", "BRL"])
    PriceInitial  = st.number_input("Initial Price",  0.0, 500.0, 9.99,  step=0.01, format="%.2f")
    PriceFinal    = st.number_input("Final Price",    0.0, 500.0, 9.99,  step=0.01, format="%.2f")
    RequiredAge   = st.number_input("Required Age",   0,   18,    0)
    Metacritic    = st.number_input("Metacritic Score (0 = none)", 0, 100, 0)

    col_bool1, col_bool2 = st.columns(2)
    with col_bool1:
        IsFree            = st.checkbox("Is Free")
        FreeVerAvail      = st.checkbox("Free Version Available")
        PurchaseAvail     = st.checkbox("Purchase Available", value=True)
    with col_bool2:
        SubscriptionAvail = st.checkbox("Subscription Available")

    # ── SteamSpy Estimates ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 SteamSpy Estimates</div>', unsafe_allow_html=True)
    SteamSpyOwners          = st.number_input("Owners Estimate",       0, 100_000_000, 500_000,  step=10_000)
    SteamSpyOwnersVariance  = st.number_input("Owners Variance",       0, 100_000_000, 50_000,   step=1_000)
    SteamSpyPlayersEstimate = st.number_input("Players Estimate",      0, 100_000_000, 300_000,  step=10_000)
    SteamSpyPlayersVariance = st.number_input("Players Variance",      0, 100_000_000, 30_000,   step=1_000)

    # ── Content & Media ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎬 Content & Media</div>', unsafe_allow_html=True)
    MovieCount      = st.number_input("Trailers / Movies",  0, 50,   1)
    ScreenshotCount = st.number_input("Screenshots",        0, 100, 10)
    DLCCount        = st.number_input("DLC Count",          0, 500,  0)
    PackageCount    = st.number_input("Packages",           1, 500,  1)
    DemoCount       = st.number_input("Demo Count",         0, 20,   0)

    # ── Achievements ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Achievements</div>', unsafe_allow_html=True)
    AchievementCount            = st.number_input("Total Achievements",       0, 10000, 0)
    AchievementHighlightedCount = st.number_input("Highlighted Achievements", 0, 100,   0)

    # ── Developer / Publisher ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">👥 Developer / Publisher</div>', unsafe_allow_html=True)
    DeveloperCount = st.number_input("Developer Count", 1, 100, 1)
    PublisherCount = st.number_input("Publisher Count", 1, 100, 1)

    # ── Platforms ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🖥️ Platforms</div>', unsafe_allow_html=True)
    col_plat1, col_plat2, col_plat3 = st.columns(3)
    with col_plat1: PlatformWindows = st.checkbox("Windows", value=True)
    with col_plat2: PlatformLinux   = st.checkbox("Linux")
    with col_plat3: PlatformMac     = st.checkbox("Mac")
    ControllerSupport = st.checkbox("Controller Support")

    # ── System Requirements ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ System Requirements</div>', unsafe_allow_html=True)
    col_req1, col_req2 = st.columns(2)
    with col_req1:
        PCReqsHaveMin    = st.checkbox("PC Min Reqs",    value=True)
        LinuxReqsHaveMin = st.checkbox("Linux Min Reqs")
        MacReqsHaveMin   = st.checkbox("Mac Min Reqs")
    with col_req2:
        PCReqsHaveRec    = st.checkbox("PC Rec Reqs")
        LinuxReqsHaveRec = st.checkbox("Linux Rec Reqs")
        MacReqsHaveRec   = st.checkbox("Mac Rec Reqs")

    st.markdown("*PC requirements text (for RAM/CPU extraction):*")
    PCMinReqsText = st.text_area("PC Min Requirements text", value="", height=60,
                                 placeholder="e.g. Minimum: RAM: 8 GB, CPU: 2.5 GHz")
    PCRecReqsText = st.text_area("PC Rec Requirements text", value="", height=60,
                                 placeholder="e.g. Recommended: RAM: 16 GB, CPU: 3.5 GHz")
    MacMinReqsText = st.text_area("Mac Min Requirements text", value="", height=50,
                                  placeholder="e.g. Minimum: RAM: 8 GB")

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
        GenreIsNonGame              = st.checkbox("Non-Game")
        GenreIsIndie                = st.checkbox("Indie")
        GenreIsAction               = st.checkbox("Action")
        GenreIsAdventure            = st.checkbox("Adventure")
        GenreIsCasual               = st.checkbox("Casual")
        GenreIsStrategy             = st.checkbox("Strategy")
    with col_g2:
        GenreIsRPG                  = st.checkbox("RPG")
        GenreIsSimulation           = st.checkbox("Simulation")
        GenreIsEarlyAccess          = st.checkbox("Early Access")
        GenreIsFreeToPlay           = st.checkbox("Free to Play Genre")
        GenreIsSports               = st.checkbox("Sports")
        GenreIsRacing               = st.checkbox("Racing")
        GenreIsMassivelyMultiplayer = st.checkbox("Massively Multiplayer")

    # ── Supported Languages ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🌐 Supported Languages</div>', unsafe_allow_html=True)
    st.markdown("*Check every language the game supports:*")
    _lang_options = ["english", "german", "french", "spanish", "italian",
                     "russian", "portuguese", "japanese", "polish", "brazil",
                     "chinese", "korean", "turkish", "arabic", "dutch"]
    col_la, col_lb = st.columns(2)
    _lang_checks = {}
    for i, lang in enumerate(_lang_options):
        col = col_la if i % 2 == 0 else col_lb
        with col:
            _lang_checks[lang] = st.checkbox(lang.capitalize(), value=(lang == "english"))

    SupportedLanguages = ", ".join(k for k, v in _lang_checks.items() if v)

    # ── NLP / Description fields ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📝 Game Description (NLP)</div>', unsafe_allow_html=True)
    st.markdown("*These fields improve prediction quality — leave blank if unavailable.*")
    DetailedDescrip = st.text_area("Detailed Description", value="", height=100,
                                   placeholder="Full game description text…")
    ShortDescrip    = st.text_area("Short Description",   value="", height=60,
                                   placeholder="One-liner summary…")
    AboutText       = st.text_area("About This Game",     value="", height=60,
                                   placeholder="About section text…")
    Reviews         = st.text_area("Reviews / Critic Text", value="", height=60,
                                   placeholder="Critic or user review excerpts…")

    # ── Support & Links ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔗 Support & Links</div>', unsafe_allow_html=True)
    SupportEmail = st.text_input("Support Email", value="")
    SupportURL   = st.text_input("Support URL",   value="")
    Website      = st.text_input("Website URL",   value="")

    # ── Media / Metadata ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🖼️ Media & Metadata</div>', unsafe_allow_html=True)
    HeaderImage = st.text_input("Header Image URL", value="")
    Background  = st.text_input("Background URL",   value="")

    # ── Legal / Notices ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📜 Legal & Notices</div>', unsafe_allow_html=True)
    DRMNotice          = st.text_input("DRM Notice",             value="")
    ExtUserAcctNotice  = st.text_input("Ext. User Acct Notice",  value="")
    LegalNotice        = st.text_input("Legal Notice",           value="")

    # ── CSV File Upload ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📂 Load Inputs from CSV File</div>', unsafe_allow_html=True)
    st.markdown("*Upload a `.csv` file with one game per row. All rows are predicted; row 1 is shown in the GUI.*")

    _BOOL_FIELDS = {
        "ControllerSupport", "IsFree", "FreeVerAvail", "PurchaseAvail", "SubscriptionAvail",
        "PlatformWindows", "PlatformLinux", "PlatformMac",
        "PCReqsHaveMin", "PCReqsHaveRec", "LinuxReqsHaveMin", "LinuxReqsHaveRec",
        "MacReqsHaveMin", "MacReqsHaveRec",
        "CategorySinglePlayer", "CategoryMultiplayer", "CategoryCoop", "CategoryMMO",
        "CategoryInAppPurchase", "CategoryIncludeSrcSDK", "CategoryIncludeLevelEditor",
        "CategoryVRSupport", "GenreIsNonGame", "GenreIsIndie", "GenreIsAction",
        "GenreIsAdventure", "GenreIsCasual", "GenreIsStrategy", "GenreIsRPG",
        "GenreIsSimulation", "GenreIsEarlyAccess", "GenreIsFreeToPlay", "GenreIsSports",
        "GenreIsRacing", "GenreIsMassivelyMultiplayer",
    }
    _INT_FIELDS = {
        "RequiredAge", "DemoCount", "DeveloperCount", "DLCCount", "Metacritic",
        "MovieCount", "PackageCount", "PublisherCount", "ScreenshotCount",
        "SteamSpyOwners", "SteamSpyOwnersVariance", "SteamSpyPlayersEstimate",
        "SteamSpyPlayersVariance", "AchievementCount", "AchievementHighlightedCount",
    }
    _FLOAT_FIELDS = {"PriceInitial", "PriceFinal"}

    def _coerce_csv_value(key: str, val):
        s = str(val).strip()
        try:
            if key in _BOOL_FIELDS:
                return int(s.lower() not in ("0", "false", "no", ""))
            elif key in _INT_FIELDS:
                return int(float(s))
            elif key in _FLOAT_FIELDS:
                return float(s)
            else:
                return s
        except (ValueError, TypeError):
            return s

    def _parse_csv_all_rows(file) -> tuple[list[dict], list]:
        """Parse all rows from a wide-format CSV. Returns (list_of_dicts, warnings)."""
        import io
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        df_csv = pd.read_csv(io.StringIO(content))
        warns = []
        if df_csv.empty:
            return [], ["CSV file is empty or has no data rows."]
        rows = []
        for i, row in df_csv.iterrows():
            parsed = {col.strip(): _coerce_csv_value(col.strip(), row[col]) for col in df_csv.columns}
            rows.append(parsed)
        return rows, warns

    uploaded_csv = st.file_uploader("Upload inputs .csv", type=["csv"], key="csv_upload")
    _txt_overrides: dict = {}
    _csv_all_rows: list = []

    if uploaded_csv is not None:
        try:
            _csv_all_rows, _csv_warns = _parse_csv_all_rows(uploaded_csv)
            if _csv_all_rows:
                _txt_overrides = _csv_all_rows[0]  # row 1 drives the GUI
                n_rows = len(_csv_all_rows)
                st.success(f"✅ Loaded **{n_rows} row{'s' if n_rows > 1 else ''}** from `{uploaded_csv.name}` — showing row 1 in GUI")
                with st.expander(f"📋 Row 1 values (of {n_rows})", expanded=False):
                    st.dataframe(
                        pd.DataFrame(_txt_overrides.items(), columns=["Field", "Value"]),
                        hide_index=True, use_container_width=True,
                    )
            else:
                st.warning("⚠️ File uploaded but no valid rows were found.")
            for w in _csv_warns:
                st.warning(f"⚠️ {w}")
        except Exception as _csv_err:
            st.error(f"Failed to read CSV: {_csv_err}")

    st.markdown("---")
    predict_btn = st.button("🔮  RUN PREDICTION")


# ═══════════════════════════════════════════════════════════════════════════════
# ── raw_inputs: single dict with ALL user-provided values (CSV column names) ───
# ── This is the object passed directly to TestScript.predict_game()          ───
# ═══════════════════════════════════════════════════════════════════════════════

raw_inputs: dict = {
    # ── Basic ──────────────────────────────────────────────────────────────────
    "ReleaseDate":                  ReleaseDate,

    # ── Numeric counts & scores ────────────────────────────────────────────────
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

    # ── SteamSpy ───────────────────────────────────────────────────────────────
    "SteamSpyOwners":               int(SteamSpyOwners),
    "SteamSpyOwnersVariance":       int(SteamSpyOwnersVariance),
    "SteamSpyPlayersEstimate":      int(SteamSpyPlayersEstimate),
    "SteamSpyPlayersVariance":      int(SteamSpyPlayersVariance),

    # ── Pricing ────────────────────────────────────────────────────────────────
    "PriceCurrency":                PriceCurrency,
    "PriceInitial":                 float(PriceInitial),
    "PriceFinal":                   float(PriceFinal),

    # ── Boolean flags ──────────────────────────────────────────────────────────
    "ControllerSupport":            int(ControllerSupport),
    "IsFree":                       int(IsFree),
    "FreeVerAvail":                 int(FreeVerAvail),
    "PurchaseAvail":                int(PurchaseAvail),
    "SubscriptionAvail":            int(SubscriptionAvail),

    # ── Platforms ──────────────────────────────────────────────────────────────
    "PlatformWindows":              int(PlatformWindows),
    "PlatformLinux":                int(PlatformLinux),
    "PlatformMac":                  int(PlatformMac),

    # ── System requirements flags ──────────────────────────────────────────────
    "PCReqsHaveMin":                int(PCReqsHaveMin),
    "PCReqsHaveRec":                int(PCReqsHaveRec),
    "LinuxReqsHaveMin":             int(LinuxReqsHaveMin),
    "LinuxReqsHaveRec":             int(LinuxReqsHaveRec),
    "MacReqsHaveMin":               int(MacReqsHaveMin),
    "MacReqsHaveRec":               int(MacReqsHaveRec),

    # ── System requirements text (for RAM/CPU extraction) ─────────────────────
    "PCMinReqsText":                PCMinReqsText,
    "PCRecReqsText":                PCRecReqsText,
    "MacMinReqsText":               MacMinReqsText,

    # ── Categories ─────────────────────────────────────────────────────────────
    "CategorySinglePlayer":         int(CategorySinglePlayer),
    "CategoryMultiplayer":          int(CategoryMultiplayer),
    "CategoryCoop":                 int(CategoryCoop),
    "CategoryMMO":                  int(CategoryMMO),
    "CategoryInAppPurchase":        int(CategoryInAppPurchase),
    "CategoryIncludeSrcSDK":        int(CategoryIncludeSrcSDK),
    "CategoryIncludeLevelEditor":   int(CategoryIncludeLevelEditor),
    "CategoryVRSupport":            int(CategoryVRSupport),

    # ── Genres ─────────────────────────────────────────────────────────────────
    "GenreIsNonGame":               int(GenreIsNonGame),
    "GenreIsIndie":                 int(GenreIsIndie),
    "GenreIsAction":                int(GenreIsAction),
    "GenreIsAdventure":             int(GenreIsAdventure),
    "GenreIsCasual":                int(GenreIsCasual),
    "GenreIsStrategy":              int(GenreIsStrategy),
    "GenreIsRPG":                   int(GenreIsRPG),
    "GenreIsSimulation":            int(GenreIsSimulation),
    "GenreIsEarlyAccess":           int(GenreIsEarlyAccess),
    "GenreIsFreeToPlay":            int(GenreIsFreeToPlay),
    "GenreIsSports":                int(GenreIsSports),
    "GenreIsRacing":                int(GenreIsRacing),
    "GenreIsMassivelyMultiplayer":  int(GenreIsMassivelyMultiplayer),

    # ── Languages ──────────────────────────────────────────────────────────────
    "SupportedLanguages":           SupportedLanguages,

    # ── NLP source text ────────────────────────────────────────────────────────
    "DetailedDescrip":              DetailedDescrip,
    "ShortDescrip":                 ShortDescrip,
    "AboutText":                    AboutText,
    "Reviews":                      Reviews if Reviews.strip() else "none",

    # ── Support & links ────────────────────────────────────────────────────────
    "SupportEmail":                 SupportEmail,
    "SupportURL":                   SupportURL,
    "Website":                      Website,

    # ── Media / metadata ───────────────────────────────────────────────────────
    "HeaderImage":                  HeaderImage if HeaderImage.strip() else None,
    "Background":                   Background  if Background.strip()  else None,

    # ── Legal ──────────────────────────────────────────────────────────────────
    "DRMNotice":                    DRMNotice         if DRMNotice.strip()         else None,
    "ExtUserAcctNotice":            ExtUserAcctNotice if ExtUserAcctNotice.strip() else None,
    "LegalNotice":                  LegalNotice       if LegalNotice.strip()       else None,
}

_ev = raw_inputs   # shorthand alias used in display blocks

# ── Apply CSV row-1 overrides to raw_inputs (drives GUI display) ────────
if _txt_overrides:
    for _k, _v in _txt_overrides.items():
        if _k in raw_inputs:
            raw_inputs[_k] = int(_v) if isinstance(_v, bool) else _v
        else:
            raw_inputs[_k] = _v
    _ev = raw_inputs  # refresh alias


def _row_to_raw_inputs(row_dict: dict, base: dict) -> dict:
    """Merge a CSV row dict onto a copy of base raw_inputs."""
    merged = base.copy()
    for k, v in row_dict.items():
        merged[k] = int(v) if isinstance(v, bool) else v
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# ── Main area ──────────────────────────────────────────────────────────────────
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
        is_scratch = (model_name == "Linear Regression (Scratch)")

        with st.spinner("Loading model…"):
            if is_scratch:
                scratch_model, scratch_scaler, scratch_feature_cols, rmse, r2, n_samples = train_scratch_model()
            else:
                try:
                    rmse, r2, n_samples = get_pkl_metrics(model_name)
                except Exception:
                    rmse, r2 = float("nan"), float("nan")
                    df, _, _ = load_raw_data()
                    n_samples = len(df)

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
            # ── Helper: predict a single raw_inputs dict ───────────────────────
            def _run_single_regression(ri, is_sc):
                if is_sc:
                    iv = [
                        ri["RequiredAge"], ri["DemoCount"], ri["DeveloperCount"],
                        ri["DLCCount"], ri["Metacritic"], ri["MovieCount"],
                        ri["PackageCount"], ri["PublisherCount"], ri["ScreenshotCount"],
                        ri["SteamSpyOwners"], ri["SteamSpyPlayersEstimate"],
                        ri["AchievementCount"], ri["AchievementHighlightedCount"],
                        ri["ControllerSupport"], ri["IsFree"],
                        ri["FreeVerAvail"], ri["PurchaseAvail"],
                        ri["PlatformWindows"], ri["PlatformLinux"],
                        ri["PlatformMac"], ri["CategorySinglePlayer"],
                        ri["CategoryMultiplayer"], ri["CategoryCoop"],
                        ri["CategoryMMO"], ri["CategoryInAppPurchase"],
                        ri["CategoryVRSupport"], ri["GenreIsIndie"],
                        ri["GenreIsAction"], ri["GenreIsAdventure"],
                        ri["GenreIsCasual"], ri["GenreIsStrategy"],
                        ri["GenreIsRPG"], ri["GenreIsSimulation"],
                        ri["GenreIsEarlyAccess"], ri["GenreIsFreeToPlay"],
                        ri["GenreIsSports"], ri["GenreIsRacing"],
                        ri["GenreIsMassivelyMultiplayer"],
                        ri["PriceInitial"], ri["PriceFinal"],
                    ]
                    X_in = scratch_scaler.transform([iv])
                    return max(0.0, float(np.ravel(scratch_model.predict(X_in))[0]))
                else:
                    return max(0.0, float(predict_game(raw=ri, model_name=model_name, mode="r")))

            try:
                # ── Row-1 / single prediction (always shown in GUI) ────────────
                prediction = _run_single_regression(raw_inputs, is_scratch)
                rating_label, rating_color = rating_from_prediction(prediction)

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
                        <div class="pred-label">Predicted Recommendations (Row 1)</div>
                        <div class="pred-value">{prediction:,.0f}</div>
                        <div style="color:{rating_color};font-family:'Rajdhani',sans-serif;font-size:1.2rem;font-weight:600;margin-top:0.8rem;letter-spacing:0.08em">
                            {rating_label}
                        </div>
                    </div>""", unsafe_allow_html=True)

                with col_detail:
                    st.markdown("#### 📋 Input Summary (Row 1)")
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
                                "Indie"    if _ev['GenreIsIndie']             else "",
                                "Action"   if _ev['GenreIsAction']            else "",
                                "Adv."     if _ev['GenreIsAdventure']         else "",
                                "Casual"   if _ev['GenreIsCasual']            else "",
                                "Strategy" if _ev['GenreIsStrategy']          else "",
                                "RPG"      if _ev['GenreIsRPG']               else "",
                                "Sim"      if _ev['GenreIsSimulation']        else "",
                                "EA"       if _ev['GenreIsEarlyAccess']       else "",
                                "F2P"      if _ev['GenreIsFreeToPlay']        else "",
                                "Sports"   if _ev['GenreIsSports']            else "",
                                "Racing"   if _ev['GenreIsRacing']            else "",
                                "MMO"      if _ev['GenreIsMassivelyMultiplayer'] else "",
                            ])) or "None",
                            ", ".join(filter(None, [
                                "Single" if _ev['CategorySinglePlayer']  else "",
                                "Multi"  if _ev['CategoryMultiplayer']   else "",
                                "Co-op"  if _ev['CategoryCoop']          else "",
                                "MMO"    if _ev['CategoryMMO']           else "",
                                "IAP"    if _ev['CategoryInAppPurchase'] else "",
                                "VR"     if _ev['CategoryVRSupport']     else "",
                            ])) or "None",
                        ],
                    }
                    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

                # ── Batch CSV: predict all rows + offer download ───────────────
                if _csv_all_rows and len(_csv_all_rows) > 1:
                    st.markdown("---")
                    st.markdown(f"#### 📦 Batch Results — {len(_csv_all_rows)} rows")
                    with st.spinner(f"Running predictions for all {len(_csv_all_rows)} rows…"):
                        batch_records = []
                        base_for_batch = {k: raw_inputs[k] for k in raw_inputs}
                        for idx, csv_row in enumerate(_csv_all_rows):
                            try:
                                ri = _row_to_raw_inputs(csv_row, base_for_batch)
                                pred_val = _run_single_regression(ri, is_scratch)
                                rl, _ = rating_from_prediction(pred_val)
                                record = dict(csv_row)
                                record["Predicted_RecommendationCount"] = round(pred_val)
                                record["Predicted_Rating"] = rl
                                batch_records.append(record)
                            except Exception as row_err:
                                record = dict(csv_row)
                                record["Predicted_RecommendationCount"] = "ERROR"
                                record["Predicted_Rating"] = str(row_err)
                                batch_records.append(record)

                    batch_df = pd.DataFrame(batch_records)
                    # Move prediction columns first
                    pred_cols = ["Predicted_RecommendationCount", "Predicted_Rating"]
                    other_cols = [c for c in batch_df.columns if c not in pred_cols]
                    batch_df = batch_df[pred_cols + other_cols]

                    st.dataframe(batch_df[pred_cols + other_cols[:6]], hide_index=True, use_container_width=True)

                    import io as _io
                    _csv_buf = _io.StringIO()
                    batch_df.to_csv(_csv_buf, index=False)
                    st.download_button(
                        label="⬇️  Download Full Results CSV",
                        data=_csv_buf.getvalue().encode("utf-8"),
                        file_name="predictions_output.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

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

        # ── raw_inputs inspector ───────────────────────────────────────────────
        with st.expander("🔍 raw_inputs Object (all fields)", expanded=False):
            st.markdown("This is the `raw_inputs` dict passed to `TestScript.predict_game()`.")
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

    selected_clf = model_name if milestone.startswith("Milestone 2") else "xgboost"

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown(f'<div class="metric-card"><div class="label">Classifier</div><div class="value" style="font-size:0.9rem">{selected_clf}</div><div class="sub">Active</div></div>', unsafe_allow_html=True)
    with col_c2:
        st.markdown(f'<div class="metric-card"><div class="label">Output</div><div class="value" style="font-size:1.2rem">Label</div><div class="sub">Low / Medium / High</div></div>', unsafe_allow_html=True)
    with col_c3:
        st.markdown(f'<div class="metric-card"><div class="label">Classes</div><div class="value">3</div><div class="sub">Low / Medium / High</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if predict_btn:
        try:
            # ── Row-1 / single classification (shown in GUI) ──────────────────
            pop_label = predict_game(raw=raw_inputs, model_name=selected_clf, mode="c")
            pop_color = POP_COLOR_MAP.get(pop_label, "#00d4ff")

            col_res, col_info = st.columns([1, 1])
            with col_res:
                st.markdown(f"""
                <div class="class-box">
                    <div style="color:#3a8a6a;font-size:0.65rem;letter-spacing:0.25em;text-transform:uppercase;margin-bottom:0.8rem">
                        Predicted Popularity (Row 1)
                    </div>
                    <div style="font-family:'Rajdhani',sans-serif;color:{pop_color};font-size:4rem;font-weight:700;line-height:1;">
                        {pop_label}
                    </div>
                    <div style="color:#3a6a4a;font-size:0.75rem;margin-top:0.6rem">
                        Game Popularity Category
                    </div>
                </div>""", unsafe_allow_html=True)

            with col_info:
                st.markdown("#### 📋 Key Inputs Used (Row 1)")
                st.dataframe(pd.DataFrame({
                    "Feature": ["Owners Est.", "Players Est.", "Price", "Metacritic",
                                "Platforms", "Languages", "Genres", "Categories"],
                    "Value": [
                        f"{_ev['SteamSpyOwners']:,}",
                        f"{_ev['SteamSpyPlayersEstimate']:,}",
                        f"${_ev['PriceFinal']:.2f}",
                        str(_ev['Metacritic']) if _ev['Metacritic'] > 0 else "N/A",
                        str(int(_ev['PlatformWindows']) + int(_ev['PlatformLinux']) + int(_ev['PlatformMac'])),
                        SupportedLanguages or "none",
                        ", ".join(filter(None, [
                            "Indie"    if _ev['GenreIsIndie']             else "",
                            "Action"   if _ev['GenreIsAction']            else "",
                            "Strategy" if _ev['GenreIsStrategy']          else "",
                            "RPG"      if _ev['GenreIsRPG']               else "",
                        ])) or "None",
                        ", ".join(filter(None, [
                            "Single" if _ev['CategorySinglePlayer'] else "",
                            "Multi"  if _ev['CategoryMultiplayer']  else "",
                            "Co-op"  if _ev['CategoryCoop']         else "",
                        ])) or "None",
                    ]
                }), hide_index=True, use_container_width=True)

            # ── Batch CSV: classify all rows + offer download ─────────────────
            if _csv_all_rows and len(_csv_all_rows) > 1:
                st.markdown("---")
                st.markdown(f"#### 📦 Batch Results — {len(_csv_all_rows)} rows")
                with st.spinner(f"Classifying all {len(_csv_all_rows)} rows…"):
                    batch_records = []
                    base_for_batch = {k: raw_inputs[k] for k in raw_inputs}
                    for idx, csv_row in enumerate(_csv_all_rows):
                        try:
                            ri = _row_to_raw_inputs(csv_row, base_for_batch)
                            label = predict_game(raw=ri, model_name=selected_clf, mode="c")
                            record = dict(csv_row)
                            record["Predicted_Popularity"] = label
                            batch_records.append(record)
                        except Exception as row_err:
                            record = dict(csv_row)
                            record["Predicted_Popularity"] = f"ERROR: {row_err}"
                            batch_records.append(record)

                batch_df = pd.DataFrame(batch_records)
                pred_cols = ["Predicted_Popularity"]
                other_cols = [c for c in batch_df.columns if c not in pred_cols]
                batch_df = batch_df[pred_cols + other_cols]

                st.dataframe(batch_df[pred_cols + other_cols[:6]], hide_index=True, use_container_width=True)

                import io as _io
                _csv_buf2 = _io.StringIO()
                batch_df.to_csv(_csv_buf2, index=False)
                st.download_button(
                    label="⬇️  Download Full Results CSV",
                    data=_csv_buf2.getvalue().encode("utf-8"),
                    file_name="classification_output.csv",
                    mime="text/csv",
                    key="clf_download",
                )

        except Exception as e:
            st.error(f"Classification failed: {e}")

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

    # ── raw_inputs inspector ───────────────────────────────────────────────────
    with st.expander("🔍 raw_inputs Object (all fields)", expanded=False):
        st.markdown("This is the `raw_inputs` dict passed to `TestScript.predict_game()`.")
        ri_display = {k: str(v) for k, v in raw_inputs.items()}
        st.dataframe(
            pd.DataFrame(ri_display.items(), columns=["Column", "Value"]),
            hide_index=True, use_container_width=True, height=400,
        )