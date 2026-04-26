"""
preprocess.py
─────────────
Single source of truth for all feature engineering.
Imported by both Project.py (training) and predict.py → App.py (inference).

Training call:
    from preprocess import build_training_df, INFERRABLE_FEATURES

Inference call:
    from preprocess import build_inference_row
"""

import re
import math
import numpy as np
import pandas as pd
from urllib.parse import urlparse

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

# ── Constants ────────────────────────────────────────────────────────────────

BOOL_COLS = [
    "ControllerSupport", "IsFree", "FreeVerAvail", "PurchaseAvail",
    "SubscriptionAvail", "PlatformWindows", "PlatformLinux", "PlatformMac",
    "PCReqsHaveMin", "PCReqsHaveRec", "LinuxReqsHaveMin", "LinuxReqsHaveRec",
    "MacReqsHaveMin", "MacReqsHaveRec",
    "CategorySinglePlayer", "CategoryMultiplayer", "CategoryCoop", "CategoryMMO",
    "CategoryInAppPurchase", "CategoryIncludeSrcSDK", "CategoryIncludeLevelEditor",
    "CategoryVRSupport",
    "GenreIsNonGame", "GenreIsIndie", "GenreIsAction", "GenreIsAdventure",
    "GenreIsCasual", "GenreIsStrategy", "GenreIsRPG", "GenreIsSimulation",
    "GenreIsEarlyAccess", "GenreIsFreeToPlay", "GenreIsSports", "GenreIsRacing",
    "GenreIsMassivelyMultiplayer",
]

SELECTED_LANGUAGES = [
    "english", "german", "french", "spanish", "italian", "russian",
    "portuguese", "japanese", "polish", "brazil", "chinese",
]

KEYWORDS = [
    "multiplayer", "online", "co op", "single player", "zombie",
    "war", "action", "team", "free",
    "strategy", "shooter", "rpg",
    "indie", "puzzle", "horror",
]

CUSTOM_STOPWORDS = {
    "game", "games", "play", "players", "player",
    "world", "time", "experience", "like", "make",
    "based", "new", "use", "way", "different",
}

# ── The 51 features the models are trained on (must match Project.py exactly)
INFERRABLE_FEATURES = [
    # core numeric
    "RequiredAge",
    "DemoCount",
    "DeveloperCount",
    "DLCCount",
    "Metacritic",
    "MovieCount",
    "PackageCount",
    "PublisherCount",
    "ScreenshotCount",
    "SteamSpyOwners",
    "SteamSpyPlayersEstimate",
    "AchievementCount",
    "AchievementHighlightedCount",
    # binary flags
    "ControllerSupport",
    "IsFree",
    "FreeVerAvail",
    "PurchaseAvail",
    "PlatformWindows",
    "PlatformLinux",
    "PlatformMac",
    "CategorySinglePlayer",
    "CategoryMultiplayer",
    "CategoryCoop",
    "CategoryMMO",
    "CategoryInAppPurchase",
    "CategoryVRSupport",
    "GenreIsIndie",
    "GenreIsAction",
    "GenreIsAdventure",
    "GenreIsCasual",
    "GenreIsStrategy",
    "GenreIsRPG",
    "GenreIsSimulation",
    "GenreIsEarlyAccess",
    "GenreIsFreeToPlay",
    "GenreIsSports",
    "GenreIsRacing",
    "GenreIsMassivelyMultiplayer",
    # price
    "PriceInitial",
    "PriceFinal",
    # engineered interactions
    "price_discount",
    "platform_count",
    "category_count",
    "content_volume",
    "highlighted_achievements_ratio",
    "action_multiplayer",
    "rpg_achievement",
    "strategy_complexity",
    "indie_price",
    "owners_players",
    "price_owners",
    "price_players",
    "free_x_owners",
    "free_x_players",
    "content_owners",
    "content_players",
    "achievement_owners",
    "achievement_players",
    "platform_owners",
    "platform_players",
    "category_owners",
    "category_players",
]

# Columns that receive log1p during training.
# Binary cols (nunique <= 2), ratio cols, and freq-encoded cols are excluded.
LOG_TRANSFORM_COLS = {
    "RequiredAge",
    "DemoCount",
    "DeveloperCount",
    "DLCCount",
    "Metacritic",
    "MovieCount",
    "PackageCount",
    "PublisherCount",
    "ScreenshotCount",
    "SteamSpyOwners",
    "SteamSpyPlayersEstimate",
    "AchievementCount",
    "AchievementHighlightedCount",
    "PriceInitial",
    "PriceFinal",
    "price_discount",
    "platform_count",
    "category_count",
    "content_volume",
    "indie_price",
    "owners_players",
    "price_owners",
    "price_players",
    "free_x_owners",
    "free_x_players",
    "content_owners",
    "content_players",
    "action_multiplayer",
    "rpg_achievement",
    "strategy_complexity",
    "achievement_owners",
    "achievement_players",
    "platform_owners",
    "platform_players",
    "category_owners",
    "category_players",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sia():
    return SentimentIntensityAnalyzer()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = [t for t in text.split() if t not in CUSTOM_STOPWORDS]
    return " ".join(tokens)


def extract_ram(text) -> float:
    """Return RAM in GB from a requirements string, or 0."""
    if not text or str(text).strip() in ("", "nan"):
        return 0.0
    match = re.search(r"(\d+)\s?(GB|MB)", str(text), re.IGNORECASE)
    if match:
        val, unit = match.groups()
        return int(val) if unit.upper() == "GB" else int(val) / 1024
    return 0.0


def extract_proc(text) -> float:
    """Return CPU speed in GHz from a requirements string, or 0."""
    if not text or str(text).strip() in ("", "nan"):
        return 0.0
    match = re.search(r"(\d+(?:\.\d+)?)\s?(MHZ|GHZ)", str(text), re.IGNORECASE)
    if match:
        val, unit = match.groups()
        val = float(val)
        return val if unit.upper() == "GHZ" else val / 1000
    return 0.0


def _freq_encode_domain(url_series: pd.Series) -> pd.Series:
    """Extract domain then frequency-encode."""
    domains = url_series.apply(
        lambda x: urlparse(x).netloc if pd.notnull(x) and x not in ("", "none") else "none"
    )
    freq = domains.value_counts(normalize=True)
    return domains.map(freq).fillna(0)


def _sentiment(text: str) -> float:
    return _sia().polarity_scores(str(text))["compound"] if str(text).strip() else 0.0


# ── Training path ────────────────────────────────────────────────────────────

def build_training_df(df_raw: pd.DataFrame):
    """
    Apply every preprocessing step to a raw training DataFrame.
    Returns (df_features, y_log1p).

    Usage (in Project.py):
        from preprocess import build_training_df, INFERRABLE_FEATURES
        df_features, y = build_training_df(df_raw)
        X = df_features[INFERRABLE_FEATURES]
    """
    df = df_raw.copy()

    # ── basic cleaning ────────────────────────────────────────────────────────
    df = df.drop_duplicates()
    df = df.replace(r"^\s*$", np.nan, regex=True)

    df["QueryName"] = df["QueryName"].fillna(df["ResponseName"])
    df["PriceCurrency"] = df["PriceCurrency"].fillna("USD")
    df["SupportURL"] = df["SupportURL"].fillna("")
    df["SupportEmail"] = df["SupportEmail"].fillna("")
    df["Website"] = df["Website"].fillna("")
    df["Reviews"] = df["Reviews"].fillna("none")
    df["Background"] = df["Background"].fillna("none")

    best = df["DetailedDescrip"].fillna(df["AboutText"]).fillna(df["ShortDescrip"])
    for col in ("AboutText", "DetailedDescrip", "ShortDescrip"):
        df[col] = df[col].fillna(best).fillna("none")

    # ── boolean → int ─────────────────────────────────────────────────────────
    present_bools = [c for c in BOOL_COLS if c in df.columns]
    df[present_bools] = df[present_bools].astype(int)

    # ── binary from text presence ─────────────────────────────────────────────
    df["PriceCurrency"] = (df["PriceCurrency"] == "USD").astype(int)
    df["Background"] = (df["Background"] != "none").astype(int)
    df["HeaderImage"] = df["HeaderImage"].apply(lambda x: 0 if pd.isna(x) else 1)
    df["LegalNotice"] = df["LegalNotice"].apply(lambda x: 0 if pd.isna(x) else 1)
    df["DRMNotice"] = df["DRMNotice"].apply(lambda x: 0 if pd.isna(x) else 1)
    df["ExtUserAcctNotice"] = df["ExtUserAcctNotice"].apply(lambda x: 0 if pd.isna(x) else 1)

    # ── frequency encoding ────────────────────────────────────────────────────
    df["QueryName_FreqEnc"] = df["QueryName"].map(
        df["QueryName"].value_counts(normalize=True)
    )
    df["ResponseName_FreqEnc"] = df["ResponseName"].map(
        df["ResponseName"].value_counts(normalize=True)
    )
    df["SupportURL"] = _freq_encode_domain(df["SupportURL"])
    df["SupportEmail"] = df["SupportEmail"].map(
        df["SupportEmail"].value_counts(normalize=True)
    ).fillna(0)
    df["Website"] = _freq_encode_domain(df["Website"])

    # ── languages ─────────────────────────────────────────────────────────────
    def _parse_langs(text):
        if pd.isna(text) or str(text).strip() == "":
            return []
        return [p.strip().lower() for p in re.split(r",|\s{2,}", str(text)) if p.strip()]

    df["SupportedLanguages"] = df["SupportedLanguages"].fillna("").apply(_parse_langs)
    for lang in SELECTED_LANGUAGES:
        df[f"Lang_{lang}"] = df["SupportedLanguages"].apply(lambda x: int(lang in x))
    df["SupportedLanguagesCount"] = df["SupportedLanguages"].apply(len)

    # ── release date → age in years ───────────────────────────────────────────
    df["ReleaseDate"] = pd.to_datetime(df["ReleaseDate"], errors="coerce")
    df["ReleaseDate"] = (pd.Timestamp.today() - df["ReleaseDate"]).dt.days / 365.25
    df["ReleaseDate"] = df["ReleaseDate"].fillna(df["ReleaseDate"].median())

    # ── requirements extraction ───────────────────────────────────────────────
    df["PC_MinRam"] = df["PCMinReqsText"].where(df.get("PCReqsHaveMin", pd.Series(0, index=df.index)) == 1).apply(extract_ram)
    df["PC_RecRam"] = df["PCRecReqsText"].where(df.get("PCReqsHaveRec", pd.Series(0, index=df.index)) == 1).apply(extract_ram)
    df["Mac_MinRam"] = df["MacMinReqsText"].where(df.get("MacReqsHaveMin", pd.Series(0, index=df.index)) == 1).apply(extract_ram)
    df["PC_MinCPU"] = df["PCMinReqsText"].where(df.get("PCReqsHaveMin", pd.Series(0, index=df.index)) == 1).apply(extract_proc)

    # ── NLP ───────────────────────────────────────────────────────────────────
    df["AllText"] = df["DetailedDescrip"].fillna("") + " " + df["ShortDescrip"].fillna("")
    df["AllText_Cleaned"] = df["AllText"].apply(clean_text)
    df["AllText_len"] = df["AllText_Cleaned"].str.split().str.len()

    for word in KEYWORDS:
        df[f"has_{word.replace(' ', '_')}"] = (
            df["AllText_Cleaned"].str.contains(word, regex=False).astype(int)
        )

    sia = _sia()
    df["ReviewSentiment"] = df["Reviews"].fillna("").apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )
    df["AboutSentiment"] = df["AboutText"].fillna("").apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )
    df["review_words"] = df["Reviews"].fillna("").str.split().str.len()

    # ── feature interactions ──────────────────────────────────────────────────
    metacritic = df["Metacritic"].fillna(df["Metacritic"].median())

    df["owners_players"]   = df["SteamSpyOwners"] * df["SteamSpyPlayersEstimate"]
    df["price_discount"]   = (df["PriceInitial"] - df["PriceFinal"]).clip(lower=0)
    df["price_owners"]     = df["PriceFinal"] * df["SteamSpyOwners"]
    df["price_players"]    = df["PriceFinal"] * df["SteamSpyPlayersEstimate"]
    df["free_x_owners"]    = df["IsFree"] * df["SteamSpyOwners"]
    df["free_x_players"]   = df["IsFree"] * df["SteamSpyPlayersEstimate"]

    df["content_volume"]   = (
        df["ScreenshotCount"] + df["MovieCount"] + df["DLCCount"] + df["PackageCount"]
    )
    df["content_owners"]   = df["content_volume"] * df["SteamSpyOwners"]
    df["content_players"]  = df["content_volume"] * df["SteamSpyPlayersEstimate"]

    df["achievement_owners"]  = df["AchievementCount"] * df["SteamSpyOwners"]
    df["achievement_players"] = df["AchievementCount"] * df["SteamSpyPlayersEstimate"]
    df["highlighted_achievements_ratio"] = (
        df["AchievementHighlightedCount"] / (df["AchievementCount"] + 1)
    )

    df["platform_count"]   = (
        df["PlatformWindows"].astype(int)
        + df["PlatformLinux"].astype(int)
        + df["PlatformMac"].astype(int)
    )
    df["platform_owners"]  = df["platform_count"] * df["SteamSpyOwners"]
    df["platform_players"] = df["platform_count"] * df["SteamSpyPlayersEstimate"]

    df["action_multiplayer"]   = df["GenreIsAction"] * df["CategoryMultiplayer"]
    df["rpg_achievement"]      = df["GenreIsRPG"] * df["AchievementCount"]
    df["strategy_complexity"]  = df["GenreIsStrategy"] * df["AchievementCount"]
    df["indie_price"]          = df["GenreIsIndie"] * df["PriceFinal"]

    df["category_count"]   = (
        df["CategorySinglePlayer"] + df["CategoryMultiplayer"] + df["CategoryCoop"]
        + df["CategoryMMO"] + df["CategoryVRSupport"]
    )
    df["category_owners"]  = df["category_count"] * df["SteamSpyOwners"]
    df["category_players"] = df["category_count"] * df["SteamSpyPlayersEstimate"]

    # ── fill nulls on numeric cols (Metacritic, RequiredAge, DemoCount, etc.) ─
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # ── outlier clipping ──────────────────────────────────────────────────────
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    # ── log1p transform ───────────────────────────────────────────────────────
    existing_log_cols = [c for c in LOG_TRANSFORM_COLS if c in df.columns]
    df[existing_log_cols] = np.log1p(df[existing_log_cols].clip(lower=0))

    # target
    y_raw = df_raw.loc[df.index, "RecommendationCount"] if "RecommendationCount" in df_raw.columns else None
    y = np.log1p(y_raw) if y_raw is not None else None

    return df, y


# ── Inference path ───────────────────────────────────────────────────────────

def build_inference_row(inputs: dict, feat_cols: list, feat_medians: dict) -> np.ndarray:
    """
    Convert a flat dict of raw UI inputs into a 1-row numpy array ready for
    model.predict().

    The output feature vector exactly matches the columns stored in
    feature_columns.pkl (in log1p space, same as training).

    Parameters
    ----------
    inputs : dict
        Raw GUI values. All keys below are accepted.
    feat_cols : list
        Ordered list loaded from Models/feature_columns.pkl.
    feat_medians : dict
        Per-column medians (already log1p-scaled) from Models/feature_medians.pkl.

    Returns
    -------
    np.ndarray, shape (1, len(feat_cols))

    FIELD GUIDE
    ───────────
    Numeric (raw – log1p applied internally where appropriate)
        required_age         int     0
        demo_count           int     0
        developer_count      int     1
        dlc_count            int     0
        metacritic           float   0       (0 = no score)
        movie_count          int     1
        package_count        int     1
        publisher_count      int     1
        screenshot_count     int     10
        steam_spy_owners     int     500_000
        steam_spy_players    int     300_000
        achievement_count    int     0
        highlighted_achiev   int     0
        price_initial        float   9.99
        price_final          float   9.99

    Boolean (int 0/1)
        ctrl_support, is_free, free_ver_avail, purchase_avail
        plat_win, plat_linux, plat_mac
        cat_single, cat_multi, cat_coop, cat_mmo, cat_iap, cat_vr
        g_indie, g_action, g_adventure, g_casual
        g_strategy, g_rpg, g_simulation, g_earlyaccess
        g_f2p, g_sports, g_racing, g_mmo_genre
    """

    g = inputs  # shorthand

    # ── raw scalars ───────────────────────────────────────────────────────────
    required_age       = float(g.get("required_age", 0))
    demo_count         = float(g.get("demo_count", 0))
    developer_count    = float(g.get("developer_count", 1))
    dlc_count          = float(g.get("dlc_count", 0))
    metacritic         = float(g.get("metacritic", 0))
    movie_count        = float(g.get("movie_count", 1))
    package_count      = float(g.get("package_count", 1))
    publisher_count    = float(g.get("publisher_count", 1))
    screenshot_count   = float(g.get("screenshot_count", 10))
    owners             = float(g.get("steam_spy_owners", 500_000))
    players            = float(g.get("steam_spy_players", 300_000))
    achievement_count  = float(g.get("achievement_count", 0))
    highlighted_achiev = float(g.get("highlighted_achiev", 0))
    price_initial      = float(g.get("price_initial", 9.99))
    price_final        = float(g.get("price_final", 9.99))

    # ── binary ────────────────────────────────────────────────────────────────
    ctrl_support  = int(g.get("ctrl_support", 0))
    is_free       = int(g.get("is_free", 0))
    free_ver_avail= int(g.get("free_ver_avail", 0))
    purchase_avail= int(g.get("purchase_avail", 1))
    plat_win      = int(g.get("plat_win", 1))
    plat_linux    = int(g.get("plat_linux", 0))
    plat_mac      = int(g.get("plat_mac", 0))
    cat_single    = int(g.get("cat_single", 1))
    cat_multi     = int(g.get("cat_multi", 0))
    cat_coop      = int(g.get("cat_coop", 0))
    cat_mmo       = int(g.get("cat_mmo", 0))
    cat_iap       = int(g.get("cat_iap", 0))
    cat_vr        = int(g.get("cat_vr", 0))
    g_indie       = int(g.get("g_indie", 0))
    g_action      = int(g.get("g_action", 0))
    g_adventure   = int(g.get("g_adventure", 0))
    g_casual      = int(g.get("g_casual", 0))
    g_strategy    = int(g.get("g_strategy", 0))
    g_rpg         = int(g.get("g_rpg", 0))
    g_simulation  = int(g.get("g_simulation", 0))
    g_earlyaccess = int(g.get("g_earlyaccess", 0))
    g_f2p         = int(g.get("g_f2p", 0))
    g_sports      = int(g.get("g_sports", 0))
    g_racing      = int(g.get("g_racing", 0))
    g_mmo_genre   = int(g.get("g_mmo_genre", 0))

    # ── engineered features (raw, before log1p) ───────────────────────────────
    platform_count   = plat_win + plat_linux + plat_mac
    category_count   = cat_single + cat_multi + cat_coop + cat_mmo + cat_vr
    content_volume   = screenshot_count + movie_count + dlc_count + package_count
    price_discount   = max(0.0, price_initial - price_final)

    highlighted_ratio    = highlighted_achiev / (achievement_count + 1)
    indie_price          = g_indie * price_final
    action_multiplayer   = g_action * cat_multi
    rpg_achievement      = g_rpg * achievement_count
    strategy_complexity  = g_strategy * achievement_count
    owners_players       = owners * players
    price_owners         = price_final * owners
    price_players        = price_final * players
    free_x_owners        = is_free * owners
    free_x_players       = is_free * players
    content_owners       = content_volume * owners
    content_players      = content_volume * players
    achievement_owners   = achievement_count * owners
    achievement_players  = achievement_count * players
    platform_owners      = platform_count * owners
    platform_players     = platform_count * players
    category_owners      = category_count * owners
    category_players     = category_count * players

    # ── assemble raw dict (mirrors INFERRABLE_FEATURES order) ─────────────────
    raw = {
        # core numeric
        "RequiredAge":                    required_age,
        "DemoCount":                      demo_count,
        "DeveloperCount":                 developer_count,
        "DLCCount":                       dlc_count,
        "Metacritic":                     metacritic,
        "MovieCount":                     movie_count,
        "PackageCount":                   package_count,
        "PublisherCount":                 publisher_count,
        "ScreenshotCount":                screenshot_count,
        "SteamSpyOwners":                 owners,
        "SteamSpyPlayersEstimate":        players,
        "AchievementCount":               achievement_count,
        "AchievementHighlightedCount":    highlighted_achiev,
        # binary — no log1p
        "ControllerSupport":              ctrl_support,
        "IsFree":                         is_free,
        "FreeVerAvail":                   free_ver_avail,
        "PurchaseAvail":                  purchase_avail,
        "PlatformWindows":                plat_win,
        "PlatformLinux":                  plat_linux,
        "PlatformMac":                    plat_mac,
        "CategorySinglePlayer":           cat_single,
        "CategoryMultiplayer":            cat_multi,
        "CategoryCoop":                   cat_coop,
        "CategoryMMO":                    cat_mmo,
        "CategoryInAppPurchase":          cat_iap,
        "CategoryVRSupport":              cat_vr,
        "GenreIsIndie":                   g_indie,
        "GenreIsAction":                  g_action,
        "GenreIsAdventure":               g_adventure,
        "GenreIsCasual":                  g_casual,
        "GenreIsStrategy":                g_strategy,
        "GenreIsRPG":                     g_rpg,
        "GenreIsSimulation":              g_simulation,
        "GenreIsEarlyAccess":             g_earlyaccess,
        "GenreIsFreeToPlay":              g_f2p,
        "GenreIsSports":                  g_sports,
        "GenreIsRacing":                  g_racing,
        "GenreIsMassivelyMultiplayer":    g_mmo_genre,
        # price
        "PriceInitial":                   price_initial,
        "PriceFinal":                     price_final,
        # interactions
        "price_discount":                 price_discount,
        "platform_count":                 platform_count,
        "category_count":                 category_count,
        "content_volume":                 content_volume,
        "highlighted_achievements_ratio": highlighted_ratio,  # NOT log1p'd
        "action_multiplayer":             action_multiplayer,
        "rpg_achievement":                rpg_achievement,
        "strategy_complexity":            strategy_complexity,
        "indie_price":                    indie_price,
        "owners_players":                 owners_players,
        "price_owners":                   price_owners,
        "price_players":                  price_players,
        "free_x_owners":                  free_x_owners,
        "free_x_players":                 free_x_players,
        "content_owners":                 content_owners,
        "content_players":                content_players,
        "achievement_owners":             achievement_owners,
        "achievement_players":            achievement_players,
        "platform_owners":                platform_owners,
        "platform_players":               platform_players,
        "category_owners":                category_owners,
        "category_players":               category_players,
    }

    # ── apply log1p to matching keys ──────────────────────────────────────────
    for k in LOG_TRANSFORM_COLS:
        if k in raw:
            raw[k] = math.log1p(max(0.0, raw[k]))

    # ── build final ordered vector aligned to feat_cols ───────────────────────
    row = [raw.get(col, feat_medians.get(col, 0.0)) for col in feat_cols]
    return np.array(row, dtype=float).reshape(1, -1)
