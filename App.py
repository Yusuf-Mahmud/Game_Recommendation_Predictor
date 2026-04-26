import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from Models.Linear_Regression_Scratch import YusufLinearRegression
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import warnings
warnings.filterwarnings("ignore")

nltk.download('vader_lexicon', quiet=True)

# ── Page config ──────────────────────────────────────────────────────────────
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

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050510 0%, #0a0a1f 100%);
    border-right: 1px solid #1a1a3a;
    width: 370px !important;
}
section[data-testid="stSidebar"] * { color: #c0c0e0 !important; }

.main .block-container {
    background: #060612;
    padding: 2rem 2.5rem;
    max-width: 1500px;
}

.app-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 0.05em;
}
.app-subtitle {
    font-size: 0.7rem;
    color: #3a3a6a !important;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: -2px;
}

.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #00d4ff !important;
    padding: 0.7rem 0 0.3rem;
    border-bottom: 1px solid #1a2a4a;
    margin-bottom: 0.6rem;
    margin-top: 0.8rem;
}

.metric-card {
    background: linear-gradient(135deg, #0a0a20 0%, #0f0f28 100%);
    border: 1px solid #1a1a4a;
    border-radius: 4px;
    padding: 1.2rem;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-card .label {
    color: #3a3a7a !important;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    color: #00d4ff !important;
    font-size: 1.6rem;
    font-weight: 500;
    font-family: 'Rajdhani', sans-serif;
}
.metric-card .sub { color: #4a4a8a !important; font-size: 0.7rem; margin-top: 0.2rem; }

.prediction-box {
    background: linear-gradient(135deg, #030318 0%, #080825 100%);
    border: 1px solid #00d4ff44;
    border-radius: 4px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 0 60px rgba(0,212,255,0.08), inset 0 0 40px rgba(0,212,255,0.03);
    position: relative;
    overflow: hidden;
}
.prediction-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
}
.prediction-box .pred-label {
    color: #3a6a8a !important;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.prediction-box .pred-value {
    font-family: 'Rajdhani', sans-serif;
    color: #ffffff !important;
    font-size: 4rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.02em;
}

.stNumberInput input, .stSelectbox select, .stTextArea textarea {
    background: #080818 !important;
    border: 1px solid #1a1a3a !important;
    color: #c0c0e0 !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: #00d4ff44 !important;
    box-shadow: 0 0 0 1px #00d4ff22 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #7c3aed22) !important;
    border: 1px solid #00d4ff !important;
    color: #00d4ff !important;
    border-radius: 3px !important;
    padding: 0.7rem 2rem !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff44, #7c3aed44) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.2) !important;
}

h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important; color: #e0e0ff !important; letter-spacing: 0.05em !important; }
hr { border-color: #1a1a3a !important; }
div[data-testid="stMarkdownContainer"] p { color: #6060a0 !important; font-size: 0.85rem; }
.stCheckbox > label { color: #8080b0 !important; font-size: 0.8rem !important; }
.stSelectbox label, .stNumberInput label, .stTextArea label { color: #5050a0 !important; font-size: 0.72rem !important; letter-spacing: 0.08em; }
</style>
""", unsafe_allow_html=True)

# ── Load & Cache ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Data/train_data.csv")
    feature_cols = [
        'RequiredAge','DemoCount','DeveloperCount','DLCCount','Metacritic',
        'MovieCount','PackageCount','PublisherCount','ScreenshotCount',
        'SteamSpyOwners','SteamSpyPlayersEstimate',
        'AchievementCount','AchievementHighlightedCount',
        'ControllerSupport','IsFree','FreeVerAvail','PurchaseAvail',
        'PlatformWindows','PlatformLinux','PlatformMac',
        'CategorySinglePlayer','CategoryMultiplayer','CategoryCoop',
        'CategoryMMO','CategoryInAppPurchase','CategoryVRSupport',
        'GenreIsIndie','GenreIsAction','GenreIsAdventure','GenreIsCasual',
        'GenreIsStrategy','GenreIsRPG','GenreIsSimulation',
        'GenreIsEarlyAccess','GenreIsFreeToPlay','GenreIsSports',
        'GenreIsRacing','GenreIsMassivelyMultiplayer',
        'PriceInitial','PriceFinal',
    ]
    target_col = 'RecommendationCount'
    df_clean = df[feature_cols + [target_col]].dropna()
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].map({'True':1,'False':0,True:1,False:0}).fillna(0)
    df_clean = df_clean.astype(float)
    return df_clean, feature_cols, target_col

@st.cache_resource
def load_pkl_metadata():
    feature_columns = joblib.load("Models/feature_columns.pkl")
    feature_medians  = joblib.load("Models/feature_medians.pkl")
    model_metrics    = joblib.load("Models/model_metrics.pkl")
    return feature_columns, feature_medians, model_metrics

PKL_MODELS = {
    "Linear Regression (Project)": "Models/linear.pkl",
    "Ridge":                        "Models/ridge.pkl",
    "Random Forest":                "Models/random_forest.pkl",
    "Gradient Boosting":            "Models/gradient_boosting.pkl",
    "XGBoost":                      "Models/xgboost.pkl",
}

@st.cache_resource
def train_scratch_model():
    df, feature_cols, target_col = load_data()
    X = df[feature_cols].values
    y = df[target_col].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = YusufLinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    return model, scaler, feature_cols, rmse, r2, len(df)

def load_pkl_model(model_name):
    model = joblib.load(PKL_MODELS[model_name])
    _, _, model_metrics = load_pkl_metadata()
    metrics = model_metrics.get(model_name, {})
    df, _, _ = load_data()
    return model, metrics.get("RMSE", float("nan")), metrics.get("R2", float("nan")), len(df)

# ── NLP helpers ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_sia():
    return SentimentIntensityAnalyzer()

def about_sentiment(text):
    sia = get_sia()
    return sia.polarity_scores(str(text))['compound'] if text.strip() else 0.0

def review_word_count(text):
    return len(text.split()) if text.strip() else 0

CUSTOM_STOPWORDS = {
    "game","games","play","players","player","world","time","experience",
    "like","make","based","new","use","way","different"
}
KEYWORDS = ['multiplayer','online','co op','single player','zombie',
            'war','action','team','free','strategy','shooter','rpg',
            'indie','puzzle','horror']

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = [t for t in text.split() if t not in CUSTOM_STOPWORDS]
    return " ".join(tokens)

def has_keyword(cleaned_text, kw):
    return 1 if kw in cleaned_text else 0

def extract_ram(text):
    if not text or str(text).strip() == '': return 0
    match = re.search(r'(\d+)\s?(GB|MB)', str(text), re.IGNORECASE)
    if match:
        val, unit = match.groups()
        return int(val) if unit.upper() == 'GB' else int(val) / 1024
    return 0

def extract_proc(text):
    if not text or str(text).strip() == '': return 0
    match = re.search(r'(\d+(?:\.\d+)?)\s?(MHZ|GHZ)', str(text), re.IGNORECASE)
    if match:
        val, unit = match.groups()
        val = float(val)
        return val if unit.upper() == 'GHZ' else val / 1000
    return 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-title">🎮 STEAM ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Recommendation Predictor</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">⚙️ Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Algorithm", [
        "Linear Regression (Scratch)",
        "Linear Regression (Project)",
        "Ridge",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
    ])

    is_pkl = model_name != "Linear Regression (Scratch)"

    # ── 1. Pricing & Audience ──
    st.markdown('<div class="section-header">💰 Pricing & Audience</div>', unsafe_allow_html=True)
    required_age    = st.number_input("Required Age", 0, 18, 0)
    price_initial   = st.number_input("Initial Price ($)", 0.0, 200.0, 9.99, step=0.01)
    price_final     = st.number_input("Final Price ($)",   0.0, 200.0, 9.99, step=0.01)
    is_free         = st.checkbox("Free to Play (IsFree)")
    purchase_avail  = st.checkbox("Purchase Available", value=True)
    free_ver_avail  = st.checkbox("Free Version Available")

    # ── 2. Popularity Estimates ──
    st.markdown('<div class="section-header">📊 Popularity Estimates</div>', unsafe_allow_html=True)
    steam_spy_owners      = st.number_input("Est. Owners (SteamSpy)",       0, 100_000_000, 500_000,  step=10_000)
    steam_spy_owners_var  = st.number_input("Owners Variance (SteamSpy)",   0, 100_000_000, 200_000,  step=10_000,
                                             help="SteamSpyOwnersVariance — only used by PKL models")
    steam_spy_players     = st.number_input("Est. Players (SteamSpy)",      0, 100_000_000, 300_000,  step=10_000)
    steam_spy_players_var = st.number_input("Players Variance (SteamSpy)",  0, 100_000_000, 150_000,  step=10_000,
                                             help="SteamSpyPlayersVariance — only used by PKL models")

    # ── 3. Content & Media ──
    st.markdown('<div class="section-header">🎬 Content & Media</div>', unsafe_allow_html=True)
    movie_count       = st.number_input("Trailers / Movies",          0, 50,   1)
    screenshot_count  = st.number_input("Screenshots",                0, 100, 10)
    dlc_count         = st.number_input("DLC Count",                  0, 200,  0)
    package_count     = st.number_input("Packages",                   1, 200,  1)
    demo_count        = st.number_input("Demo Count",                 0, 20,   0)
    achievement_count     = st.number_input("Achievements",           0, 5000, 0)
    highlighted_achiev    = st.number_input("Highlighted Achievements",0, 50,  0)

    # ── 4. Team ──
    st.markdown('<div class="section-header">👥 Developer / Publisher</div>', unsafe_allow_html=True)
    developer_count = st.number_input("Developers", 1, 50, 1)
    publisher_count = st.number_input("Publishers",  1, 50, 1)

    # ── 5. Platforms ──
    st.markdown('<div class="section-header">🖥️ Platforms</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a: plat_win   = st.checkbox("Windows", value=True)
    with col_b: plat_linux = st.checkbox("Linux")
    with col_c: plat_mac   = st.checkbox("Mac")
    ctrl_support = st.checkbox("Controller Support")

    # ── 6. Categories ──
    st.markdown('<div class="section-header">🏷️ Categories</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        cat_single = st.checkbox("Single Player", value=True)
        cat_multi  = st.checkbox("Multiplayer")
        cat_coop   = st.checkbox("Co-op")
    with col4:
        cat_mmo    = st.checkbox("MMO Category")
        cat_iap    = st.checkbox("In-App Purchase")
        cat_vr     = st.checkbox("VR Support")

    # ── 7. Genres ──
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

    # ── 8. PC Requirements ── (PKL only)
    st.markdown('<div class="section-header">💻 PC Requirements</div>', unsafe_allow_html=True)
    pc_min_reqs_text = st.text_area(
        "PC Minimum Requirements Text",
        placeholder='e.g. "RAM: 8 GB  Processor: 2.4 GHz"',
        height=68,
        help="RAM (GB/MB) and CPU (GHz/MHz) are auto-extracted. Only used by PKL models.",
        disabled=not is_pkl,
    ) if is_pkl else ""

    pc_has_rec = st.checkbox("Has Recommended PC Specs?", disabled=not is_pkl) if is_pkl else False
    pc_rec_reqs_text = ""
    if is_pkl and pc_has_rec:
        pc_rec_reqs_text = st.text_area(
            "PC Recommended Requirements Text",
            placeholder='e.g. "RAM: 16 GB  Processor: 3.6 GHz"',
            height=68,
        )

    # ── 9. Mac Requirements ── (PKL only)
    st.markdown('<div class="section-header">🍎 Mac Requirements</div>', unsafe_allow_html=True)
    mac_has_min = st.checkbox("Has Mac Min Specs?", disabled=not is_pkl) if is_pkl else False
    mac_min_reqs_text = ""
    if is_pkl and mac_has_min:
        mac_min_reqs_text = st.text_area(
            "Mac Minimum Requirements Text",
            placeholder='e.g. "RAM: 4 GB"',
            height=68,
        )

    # ── 10. Support / Identity ── (PKL only)
    st.markdown('<div class="section-header">🌐 Support / Identity</div>', unsafe_allow_html=True)
    has_legal_notice  = st.checkbox("Has Legal Notice",      disabled=not is_pkl) if is_pkl else False
    has_website       = st.checkbox("Has Developer Website", disabled=not is_pkl) if is_pkl else False
    has_support_email = st.checkbox("Has Support Email",     disabled=not is_pkl) if is_pkl else False
    has_support_url   = st.checkbox("Has Support URL",       disabled=not is_pkl) if is_pkl else False
    lang_english      = st.checkbox("Supports English", value=True, disabled=not is_pkl) if is_pkl else True

    # ── 11. Game Description NLP ── (PKL only)
    st.markdown('<div class="section-header">📝 Description (NLP)</div>', unsafe_allow_html=True)
    if is_pkl:
        about_text = st.text_area(
            "About / Description Text",
            placeholder="Paste the game's About or Detailed Description here...",
            height=130,
            help="Used for sentiment (AboutSentiment) and keyword flags: has_war, has_action, has_team, AllText_len.",
        )
    else:
        st.caption("Not used by Scratch model.")
        about_text = ""

    # ── 12. Reviews NLP ── (PKL only)
    st.markdown('<div class="section-header">💬 Reviews (NLP)</div>', unsafe_allow_html=True)
    if is_pkl:
        reviews_text = st.text_area(
            "User Reviews Text",
            placeholder="Paste a sample of user reviews here...",
            height=100,
            help="Word count is multiplied by owner/player estimates → reviews_owners, reviews_players.",
        )
    else:
        st.caption("Not used by Scratch model.")
        reviews_text = ""

    st.markdown("---")
    predict_btn = st.button("🔮  RUN PREDICTION")

# ── Main area ────────────────────────────────────────────────────────────────
st.markdown("## 🎮 Steam Game Recommendation Predictor")
st.markdown("Predict how many Steam recommendations a game will receive based on all its features.")
st.markdown("---")

# Load model
with st.spinner("Loading model…"):
    if model_name == "Linear Regression (Scratch)":
        model, scaler, feature_cols_scratch, rmse, r2, n_samples = train_scratch_model()
    else:
        model, rmse, r2, n_samples = load_pkl_model(model_name)

# Stats row
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f'<div class="metric-card"><div class="label">Model</div><div class="value" style="font-size:1rem">{model_name}</div><div class="sub">Active</div></div>', unsafe_allow_html=True)
with col_m2:
    r2_display   = f"{r2:.3f}"    if not math.isnan(r2)   else "N/A"
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
        # ── Scratch: 40 raw features + StandardScaler ─────────────────────────
        input_vector = [
            required_age, demo_count, developer_count, dlc_count, 0,
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
        X_input = scaler.transform([input_vector])
        raw = model.predict(X_input)
        prediction = max(0, float(np.ravel(raw)[0]))

    else:
        # ── PKL models: all 51 features ───────────────────────────────────────
        feat_cols, feat_medians, _ = load_pkl_metadata()

        # NLP from text inputs
        cleaned_about = clean_text(about_text)
        all_text_len  = len(cleaned_about.split()) if cleaned_about.strip() else 0
        about_sent    = about_sentiment(about_text)
        review_words  = review_word_count(reviews_text)

        kw_war    = has_keyword(cleaned_about, 'war')
        kw_action = has_keyword(cleaned_about, 'action')
        kw_team   = has_keyword(cleaned_about, 'team')

        # RAM / CPU extraction
        pc_min_ram  = extract_ram(pc_min_reqs_text)
        pc_rec_ram  = extract_ram(pc_rec_reqs_text)  if pc_has_rec  else 0
        mac_min_ram = extract_ram(mac_min_reqs_text) if mac_has_min else 0
        pc_min_cpu  = extract_proc(pc_min_reqs_text)

        # Support encoding: present → training median, absent → 0
        median_email = feat_medians.get("SupportEmail", 0.00022)
        median_url   = feat_medians.get("SupportURL",   0.00342)
        median_web   = feat_medians.get("Website",      0.00022)
        support_email_enc = median_email if has_support_email else 0.0
        support_url_enc   = median_url   if has_support_url   else 0.0
        website_enc       = median_web   if has_website        else 0.0

        # Interaction features (raw before log1p)
        content_volume  = screenshot_count + movie_count + dlc_count + package_count
        platform_count  = int(plat_win) + int(plat_linux) + int(plat_mac)
        category_count  = int(cat_single) + int(cat_multi) + int(cat_coop) + int(cat_mmo) + int(cat_vr)

        owners_players      = steam_spy_owners * steam_spy_players
        price_owners        = price_final * steam_spy_owners
        price_players       = price_final * steam_spy_players
        content_owners      = content_volume * steam_spy_owners
        content_players     = content_volume * steam_spy_players
        achievement_owners  = achievement_count * steam_spy_owners
        achievement_players = achievement_count * steam_spy_players
        platform_owners     = platform_count * steam_spy_owners
        platform_players    = platform_count * steam_spy_players
        indie_price         = int(g_indie) * price_final
        category_owners     = category_count * steam_spy_owners
        category_players    = category_count * steam_spy_players
        reviews_owners      = review_words * steam_spy_owners
        reviews_players     = review_words * steam_spy_players
        highlighted_ratio   = highlighted_achiev / (achievement_count + 1)

        known = {
            # Core numeric (will be log1p'd)
            "MovieCount":                  movie_count,
            "ScreenshotCount":             screenshot_count,
            "SteamSpyOwners":              steam_spy_owners,
            "SteamSpyOwnersVariance":      steam_spy_owners_var,
            "SteamSpyPlayersEstimate":     steam_spy_players,
            "SteamSpyPlayersVariance":     steam_spy_players_var,
            "AchievementCount":            achievement_count,
            "AchievementHighlightedCount": highlighted_achiev,
            "PriceInitial":                price_initial,
            "PriceFinal":                  price_final,
            # Requirements (will be log1p'd)
            "PC_MinRam":                   pc_min_ram,
            "PC_RecRam":                   pc_rec_ram,
            "Mac_MinRam":                  mac_min_ram,
            "PC_MinCPU":                   pc_min_cpu,
            # Interaction features (will be log1p'd)
            "owners_players":              owners_players,
            "price_owners":                price_owners,
            "price_players":               price_players,
            "content_volume":              content_volume,
            "content_owners":              content_owners,
            "content_players":             content_players,
            "achievement_owners":          achievement_owners,
            "achievement_players":         achievement_players,
            "platform_count":              platform_count,
            "platform_owners":             platform_owners,
            "platform_players":            platform_players,
            "indie_price":                 indie_price,
            "category_count":              category_count,
            "category_owners":             category_owners,
            "category_players":            category_players,
            "reviews_owners":              reviews_owners,
            "reviews_players":             reviews_players,
            # Binary (no log1p)
            "ControllerSupport":           int(ctrl_support),
            "PlatformMac":                 int(plat_mac),
            "PCReqsHaveRec":               int(pc_has_rec),
            "MacReqsHaveMin":              int(mac_has_min),
            "CategoryMultiplayer":         int(cat_multi),
            "GenreIsIndie":                int(g_indie),
            "GenreIsAction":               int(g_action),
            "GenreIsAdventure":            int(g_adventure),
            "GenreIsCasual":               int(g_casual),
            "LegalNotice":                 int(has_legal_notice),
            "Lang_english":                int(lang_english),
            # Keyword binary flags (no log1p)
            "has_war":                     kw_war,
            "has_action":                  kw_action,
            "has_team":                    kw_team,
            # Freq-encoded (not log1p'd in Project.py)
            "SupportEmail":                support_email_enc,
            "SupportURL":                  support_url_enc,
            "Website":                     website_enc,
            # NLP continuous (not log1p'd in Project.py)
            "AllText_len":                 all_text_len,
            "AboutSentiment":              about_sent,
            # Ratio (not log1p'd)
            "highlighted_achievements_ratio": highlighted_ratio,
        }

        # log1p transform — same columns Project.py transformed
        log_transform_keys = [
            "MovieCount", "ScreenshotCount",
            "SteamSpyOwners", "SteamSpyOwnersVariance",
            "SteamSpyPlayersEstimate", "SteamSpyPlayersVariance",
            "AchievementCount", "AchievementHighlightedCount",
            "PriceInitial", "PriceFinal",
            "PC_MinRam", "PC_RecRam", "Mac_MinRam", "PC_MinCPU",
            "owners_players", "price_owners", "price_players",
            "content_volume", "content_owners", "content_players",
            "achievement_owners", "achievement_players",
            "platform_count", "platform_owners", "platform_players",
            "indie_price", "category_count", "category_owners", "category_players",
            "reviews_owners", "reviews_players",
        ]
        for k in log_transform_keys:
            if k in known:
                known[k] = math.log1p(max(0, known[k]))

        # Build final 51-dim vector in exact column order
        row = [known.get(col, feat_medians.get(col, 0)) for col in feat_cols]
        X_input = np.array(row).reshape(1, -1)
        raw_pred = model.predict(X_input)[0]

        # Project.py applied log1p to y → reverse with expm1
        prediction = max(0, float(np.expm1(raw_pred)))

    # ── Rating label ──────────────────────────────────────────────────────────
    if   prediction < 500:    rating, color = "Overwhelmingly Negative", "#ff3333"
    elif prediction < 2000:   rating, color = "Mostly Negative",         "#ff6622"
    elif prediction < 5000:   rating, color = "Mixed",                   "#ffaa00"
    elif prediction < 10000:  rating, color = "Mostly Positive",         "#88dd22"
    elif prediction < 50000:  rating, color = "Very Positive",           "#22ddaa"
    else:                      rating, color = "Overwhelmingly Positive", "#00d4ff"

    # ── TTS ───────────────────────────────────────────────────────────────────
    pred_int = int(round(prediction))
    def number_to_words(n):
        if n >= 1_000_000:
            m = n // 1_000_000; r = n % 1_000_000
            return f"{m} million" + (f" {r//1000} thousand" if r//1000 else "") + (f" {r%1000}" if r%1000 else "")
        elif n >= 1_000:
            return f"{n//1000} thousand" + (f" {n%1000}" if n%1000 else "")
        return str(n)

    tts_text = f"Based on your input, the recommendations are: {number_to_words(pred_int)} Recommends."
    st.components.v1.html(f"""<script>
        window.onload = function() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var msg = new SpeechSynthesisUtterance("{tts_text}");
                msg.rate = 0.95; msg.pitch = 1.0; msg.volume = 1.0;
                var voices = window.speechSynthesis.getVoices();
                var pref = voices.find(v => v.lang.startsWith('en') && v.localService);
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
            <div style="color:{color}; font-family:'Rajdhani',sans-serif; font-size:1.2rem; font-weight:600; margin-top:0.8rem; letter-spacing:0.08em">
                {rating}
            </div>
        </div>""", unsafe_allow_html=True)

    with col_detail:
        st.markdown("#### 📋 Input Summary")
        about_sent_display  = f"{about_sentiment(about_text):.3f}" if is_pkl else "N/A (Scratch)"
        review_wc_display   = str(review_word_count(reviews_text)) if is_pkl else "N/A"
        pc_ram_display      = f"{extract_ram(pc_min_reqs_text):.1f} GB" if is_pkl else "N/A"
        pc_cpu_display      = f"{extract_proc(pc_min_reqs_text):.2f} GHz" if is_pkl else "N/A"
        content_vol_display = screenshot_count + movie_count + dlc_count + package_count
        summary_data = {
            "Feature": [
                "Price", "Owners Est.", "Players Est.", "Achievements",
                "Content Volume", "Platforms", "About Sentiment",
                "Review Word Count", "PC Min RAM", "PC Min CPU",
            ],
            "Value": [
                f"${price_final:.2f}",
                f"{steam_spy_owners:,}",
                f"{steam_spy_players:,}",
                str(achievement_count),
                str(content_vol_display),
                ", ".join(filter(None, [
                    "Windows" if plat_win else "",
                    "Linux"   if plat_linux else "",
                    "Mac"     if plat_mac else ""
                ])) or "None",
                about_sent_display,
                review_wc_display,
                pc_ram_display,
                pc_cpu_display,
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

else:
    st.markdown("""
    <div style="background:#08081a; border:1px dashed #1a1a3a; border-radius:4px;
                padding:3rem; text-align:center; font-family:'IBM Plex Mono',monospace;">
        <div style="font-size:3rem; margin-bottom:1rem">🔮</div>
        <div style="font-size:1rem; font-weight:500; color:#4a4a8a; letter-spacing:0.1em; text-transform:uppercase">
            Fill in the sidebar inputs
        </div>
        <div style="font-size:0.8rem; margin-top:0.5rem; color:#2a2a5a">
            PKL models: paste game description + reviews for best NLP accuracy
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature Importance ────────────────────────────────────────────────────────
with st.expander("📈 Feature Importance / Model Coefficients", expanded=False):
    import plotly.express as px

    if model_name == "Linear Regression (Scratch)":
        display_feature_cols = feature_cols_scratch
    else:
        display_feature_cols, _, _ = load_pkl_metadata()

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
        bar_label = "Importance"
        color_scale = ["#0a0a2a", "#00d4ff"]
    elif hasattr(model, "coef_"):
        importance_values = model.coef_
        bar_label = "Coefficient"
        color_scale = ["#ff3333", "#0a0a2a", "#00d4ff"]
    else:
        importance_values = None

    if importance_values is not None:
        min_len = min(len(display_feature_cols), len(importance_values))
        coefs = pd.DataFrame({
            "Feature": display_feature_cols[:min_len],
            bar_label: importance_values[:min_len]
        }).sort_values(bar_label, key=abs, ascending=False).head(20)

        fig = px.bar(coefs, x=bar_label, y="Feature", orientation="h",
                     color=bar_label, color_continuous_scale=color_scale,
                     template="plotly_dark")
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
    df_preview, _, _ = load_data()
    if model_name == "Linear Regression (Scratch)":
        st.markdown(f"**{len(df_preview):,} rows** — 40 raw features (Scratch model, StandardScaler applied at train time)")
        st.dataframe(df_preview.head(50), use_container_width=True, height=300)
    else:
        feat_cols_pkl, _, _ = load_pkl_metadata()
        st.markdown(f"**51 feature columns** the PKL models were trained on (Project.py full pipeline):")
        st.write(feat_cols_pkl)
        st.markdown(f"**{len(df_preview):,} rows** of raw training data:")
        st.dataframe(df_preview.head(50), use_container_width=True, height=300)