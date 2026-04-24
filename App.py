import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler # <- Remove after and the one from scratch come
from Models.Linear_Regression_Scratch import YusufLinearRegression
import warnings
warnings.filterwarnings("ignore")

#Page Config
st.set_page_config(
    page_title="Steam ML Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid #2a2a4a;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0f0 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stCheckbox label {
        color: #a0a0c0 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em;
    }

    /* Main area */
    .main .block-container {
        background: #0d0d1a;
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a5a;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card .label {
        color: #6060a0;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        color: #c0c0ff;
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-card .sub {
        color: #4080ff;
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }

    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #0d2060 0%, #1a0d40 100%);
        border: 2px solid #4060ff;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(64, 96, 255, 0.3);
    }
    .prediction-box .pred-label {
        color: #8090ff;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .prediction-box .pred-value {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    /* Section headers */
    .section-header {
        color: #8090ff;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        padding: 0.6rem 0 0.3rem;
        border-bottom: 1px solid #2a2a4a;
        margin-bottom: 0.8rem;
        margin-top: 1rem;
    }

    /* Streamlit overrides */
    .stNumberInput input, .stSelectbox select {
        background: #1a1a2e !important;
        border: 1px solid #2a2a5a !important;
        color: #e0e0f0 !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4060ff, #8040ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    h1, h2, h3 { color: #e0e0ff !important; }

    .stCheckbox > label { color: #a0a0c0 !important; }

    div[data-testid="stMarkdownContainer"] p { color: #a0a0c0; }

    /* Logo / title area */
    .app-title {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6080ff, #c040ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .app-subtitle {
        color: #5050a0;
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }

    /* Divider */
    hr { border-color: #2a2a4a !important; }
</style>
""", unsafe_allow_html=True)

#Load & Cache Data
@st.cache_data
def load_data():
    df = pd.read_csv("Data/train_data.csv")
    feature_cols = [
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
    target_col = 'RecommendationCount'
    df_clean = df[feature_cols + [target_col]].dropna()
    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            df_clean[col] = df_clean[col].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0)
    df_clean = df_clean.astype(float)
    return df_clean, feature_cols, target_col

@st.cache_resource
def train_model(model_name: str):
    df, feature_cols, target_col = load_data()
    X = df[feature_cols].values
    y = df[target_col].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    #Models (Han7ot hna elmodels bta3tna)
    models = {
        "Linear Regression": YusufLinearRegression(learning_rate=0.01, epochs=1000),
        #Examples:
        # "Random Forest": RandomForestRegressor(),
        # "XGBoost": XGBRegressor(),
        # w ndefha fi el select box ta7t
    }
    model = models[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    return model, scaler, feature_cols, rmse, r2, len(df)

#Sidebar
with st.sidebar:
    st.markdown('<div class="app-title">🎮 Steam ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Recommendation Predictor</div>', unsafe_allow_html=True)
    st.markdown("---")

    #Model selector
    st.markdown('<div class="section-header">⚙️ Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox(
        "Algorithm",
        ["Linear Regression"],   # ← add more model names here when ready
        help="More models coming soon"
    )

    st.markdown('<div class="section-header">📊 Game Details</div>', unsafe_allow_html=True)

    #Numeric inputs
    required_age       = st.number_input("Required Age",              0, 18, 0)
    price_initial      = st.number_input("Initial Price ($)",         0.0, 200.0, 9.99, step=0.01)
    price_final        = st.number_input("Final Price ($)",           0.0, 200.0, 9.99, step=0.01)
    metacritic         = st.number_input("Metacritic Score",          0, 100, 75)
    achievement_count  = st.number_input("Achievements",             0, 5000, 0)
    dlc_count          = st.number_input("DLC Count",                 0, 200, 0)
    screenshot_count   = st.number_input("Screenshots",              0, 100, 10)
    movie_count        = st.number_input("Movies / Trailers",        0, 50, 1)
    developer_count    = st.number_input("Developers",               1, 50, 1)
    publisher_count    = st.number_input("Publishers",               1, 50, 1)
    package_count      = st.number_input("Packages",                 1, 200, 1)
    demo_count         = st.number_input("Demo Count",               0, 20, 0)
    steam_spy_owners   = st.number_input("Est. Owners (SteamSpy)",   0, 100_000_000, 500_000, step=10_000)
    steam_spy_players  = st.number_input("Est. Players (SteamSpy)",  0, 100_000_000, 300_000, step=10_000)
    highlighted_achiev = st.number_input("Highlighted Achievements", 0, 50, 0)

    st.markdown('<div class="section-header">🖥️ Platforms</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a: plat_win   = st.checkbox("Windows", value=True)
    with col_b: plat_linux = st.checkbox("Linux")
    with col_c: plat_mac   = st.checkbox("Mac")

    st.markdown('<div class="section-header">💰 Availability</div>', unsafe_allow_html=True)
    col_d, col_e, col_f = st.columns(3)
    with col_d: is_free       = st.checkbox("Free",    key="f1")
    with col_e: free_ver      = st.checkbox("Free Ver",key="f2")
    with col_f: purchase_avail= st.checkbox("Purchase",value=True, key="f3")
    ctrl_support = st.checkbox("Controller Support")

    st.markdown('<div class="section-header">🎲 Genres</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        g_indie      = st.checkbox("Indie")
        g_action     = st.checkbox("Action")
        g_adventure  = st.checkbox("Adventure")
        g_casual     = st.checkbox("Casual")
        g_strategy   = st.checkbox("Strategy")
        g_rpg        = st.checkbox("RPG")
    with col2:
        g_simulation = st.checkbox("Simulation")
        g_earlyaccess= st.checkbox("Early Access")
        g_f2p        = st.checkbox("Free to Play")
        g_sports     = st.checkbox("Sports")
        g_racing     = st.checkbox("Racing")
        g_mmo        = st.checkbox("MMO")

    st.markdown('<div class="section-header">🏷️ Categories</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        cat_single   = st.checkbox("Single Player", value=True)
        cat_multi    = st.checkbox("Multiplayer")
        cat_coop     = st.checkbox("Co-op")
    with col4:
        cat_mmo      = st.checkbox("MMO Category")
        cat_iap      = st.checkbox("In-App Purchase")
        cat_vr       = st.checkbox("VR Support")

    st.markdown("---")
    predict_btn = st.button("🔮  Run Prediction")

#Main Area
st.markdown("## 🎮 Steam Game Recommendation Predictor")
st.markdown("Predict how many Steam recommendations a game will receive based on its features.")
st.markdown("---")

#Load model
with st.spinner("Loading model…"):
    model, scaler, feature_cols, rmse, r2, n_samples = train_model(model_name)

# Model stats row
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Model</div>
        <div class="value" style="font-size:1.1rem">{model_name}</div>
        <div class="sub">Active</div>
    </div>""", unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">R² Score</div>
        <div class="value">{r2:.3f}</div>
        <div class="sub">Test set</div>
    </div>""", unsafe_allow_html=True)
with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">RMSE</div>
        <div class="value">{rmse:,.0f}</div>
        <div class="sub">Recommendations</div>
    </div>""", unsafe_allow_html=True)
with col_m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Training Rows</div>
        <div class="value">{n_samples:,}</div>
        <div class="sub">Steam games</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

#Prediction
if predict_btn:
    input_vector = [
        required_age, demo_count, developer_count, dlc_count, metacritic,
        movie_count, package_count, publisher_count, screenshot_count,
        steam_spy_owners, steam_spy_players,
        achievement_count, highlighted_achiev,
        int(ctrl_support), int(is_free), int(free_ver), int(purchase_avail),
        int(plat_win), int(plat_linux), int(plat_mac),
        int(cat_single), int(cat_multi), int(cat_coop),
        int(cat_mmo), int(cat_iap), int(cat_vr),
        int(g_indie), int(g_action), int(g_adventure), int(g_casual),
        int(g_strategy), int(g_rpg), int(g_simulation),
        int(g_earlyaccess), int(g_f2p), int(g_sports),
        int(g_racing), int(g_mmo),
        price_initial, price_final,
    ]
    X_input = scaler.transform([input_vector])
    prediction = max(0, model.predict(X_input)[0])

    # Rating label
    if prediction < 500:
        rating, color = "Overwhelmingly Negative", "#ff4444"
    elif prediction < 2000:
        rating, color = "Mostly Negative", "#ff7744"
    elif prediction < 5000:
        rating, color = "Mixed", "#ffaa44"
    elif prediction < 10000:
        rating, color = "Mostly Positive", "#88cc44"
    elif prediction < 50000:
        rating, color = "Very Positive", "#44cc88"
    else:
        rating, color = "Overwhelmingly Positive", "#44aaff"

    #TTS via Web Speech API
    pred_int = int(round(prediction))
    def number_to_words(n):
        if n >= 1_000_000:
            millions = n // 1_000_000
            remainder = n % 1_000_000
            if remainder == 0:
                return f"{millions} million"
            thousands = remainder // 1_000
            rest = remainder % 1_000
            parts = [f"{millions} million"]
            if thousands: parts.append(f"{thousands} thousand")
            if rest: parts.append(str(rest))
            return " ".join(parts)
        elif n >= 1_000:
            thousands = n // 1_000
            rest = n % 1_000
            if rest == 0:
                return f"{thousands} thousand"
            return f"{thousands} thousand {rest}"
        else:
            return str(n)
    tts_text = f"Based on your input, the recommendations are {number_to_words(pred_int)} recommend"
    st.components.v1.html(f"""
        <script>
            window.onload = function() {{
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel();
                    var msg = new SpeechSynthesisUtterance("{tts_text}");
                    msg.rate  = 0.2;
                    msg.pitch = 1.0;
                    msg.volume = 1.0;
                    var voices = window.speechSynthesis.getVoices();
                    var preferred = voices.find(v => v.lang.startsWith('en') && v.localService);
                    if (preferred) msg.voice = preferred;
                    window.speechSynthesis.speak(msg);
                }}
            }};
            if (window.speechSynthesis) {{
                window.speechSynthesis.onvoiceschanged = function() {{
                    window.speechSynthesis.cancel();
                    var msg = new SpeechSynthesisUtterance("{tts_text}");
                    msg.rate  = 0.95;
                    msg.pitch = 1.0;
                    msg.volume = 1.0;
                    var voices = window.speechSynthesis.getVoices();
                    var preferred = voices.find(v => v.lang.startsWith('en') && v.localService);
                    if (preferred) msg.voice = preferred;
                    window.speechSynthesis.speak(msg);
                }};
            }}
        </script>
    """, height=0)

    col_pred, col_detail = st.columns([1, 1])
    with col_pred:
        st.markdown(f"""
        <div class="prediction-box">
            <div class="pred-label">Predicted Recommendations</div>
            <div class="pred-value">{prediction:,.0f}</div>
            <div style="color:{color}; font-size:1rem; font-weight:600; margin-top:0.5rem">
                {rating}
            </div>
        </div>""", unsafe_allow_html=True)

    with col_detail:
        st.markdown("#### 📋 Input Summary")
        summary_data = {
            "Feature": ["Price", "Metacritic", "Achievements", "DLC Count",
                        "Est. Owners", "Platforms", "Genres Selected"],
            "Value": [
                f"${price_final:.2f}",
                str(metacritic),
                str(achievement_count),
                str(dlc_count),
                f"{steam_spy_owners:,}",
                ", ".join(filter(None, [
                    "Windows" if plat_win else "",
                    "Linux" if plat_linux else "",
                    "Mac" if plat_mac else ""
                ])),
                str(sum([g_indie, g_action, g_adventure, g_casual, g_strategy,
                        g_rpg, g_simulation, g_earlyaccess, g_f2p, g_sports, g_racing, g_mmo]))
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

else:
    # Placeholder state
    st.markdown("""
    <div style="background:#1a1a2e; border:1px dashed #2a2a5a; border-radius:20px;
                padding:3rem; text-align:center; color:#5050a0;">
        <div style="font-size:3rem; margin-bottom:1rem">🔮</div>
        <div style="font-size:1.2rem; font-weight:600; color:#6060a0">
            Configure game features in the sidebar
        </div>
        <div style="font-size:0.85rem; margin-top:0.5rem">
            Then click <strong>Run Prediction</strong> to see results
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

#Feature Importance (Linear Regression coefficients)
with st.expander("📈 Feature Importance / Model Coefficients", expanded=False):
    coefs = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=abs, ascending=False).head(20)

    import plotly.express as px
    fig = px.bar(
        coefs, x="Coefficient", y="Feature", orientation="h",
        color="Coefficient",
        color_continuous_scale=["#ff4444", "#2a2a4a", "#4480ff"],
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#a0a0c0"),
        height=500,
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

#Data Explorer
with st.expander("🗂️ Training Data Explorer", expanded=False):
    df_preview, _, _ = load_data()
    st.markdown(f"**{len(df_preview):,} rows × {len(df_preview.columns)} columns** (numeric features only)")
    st.dataframe(df_preview.head(50), use_container_width=True, height=300)