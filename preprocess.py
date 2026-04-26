"""
preprocess.py
─────────────
Single source of truth for all feature engineering.
Imported by both the training notebook (Project.py / Notebook) and the
inference layer (predict.py → App.py).

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

# Columns that receive log1p during training (must match exactly)
LOG_TRANSFORM_COLS = [
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

# Final ordered feature list saved to feature_columns.pkl
INFERRABLE_FEATURES = [
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
    # binary – no log1p
    "ControllerSupport", "PlatformMac", "PCReqsHaveRec", "MacReqsHaveMin",
    "CategoryMultiplayer", "GenreIsIndie", "GenreIsAction", "GenreIsAdventure",
    "GenreIsCasual", "LegalNotice", "Lang_english",
    # keyword flags – no log1p
    "has_war", "has_action", "has_team",
    # frequency-encoded – no log1p
    "SupportEmail", "SupportURL", "Website",
    # NLP continuous – no log1p
    "AllText_len", "AboutSentiment",
    # ratio – no log1p
    "highlighted_achievements_ratio",
]


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
    Apply every preprocessing step from the notebook to a raw training
    DataFrame.  Returns (df_features, y_log1p).

    Usage (replaces the long preprocessing block in Project.py / Notebook):
        from preprocess import build_training_df, INFERRABLE_FEATURES
        df_features, y = build_training_df(df_raw)
        X = df_features[INFERRABLE_FEATURES]
    """
    df = df_raw.copy()

    # ── basic cleaning ────────────────────────────────────────────────────────
    df = df.drop_duplicates()
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # fill text nulls
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

    # review word count (used in interaction features below)
    df["review_words"] = df["Reviews"].fillna("").str.split().str.len()

    # ── feature interactions ──────────────────────────────────────────────────
    metacritic = df["Metacritic"].fillna(df["Metacritic"].median())

    df["owners_metacritic"] = df["SteamSpyOwners"] * metacritic
    df["players_metacritic"] = df["SteamSpyPlayersEstimate"] * metacritic
    df["owners_players"] = df["SteamSpyOwners"] * df["SteamSpyPlayersEstimate"]

    df["price_discount"] = df["PriceInitial"] - df["PriceFinal"]
    df["price_owners"] = df["PriceFinal"] * df["SteamSpyOwners"]
    df["price_players"] = df["PriceFinal"] * df["SteamSpyPlayersEstimate"]
    df["free_x_owners"] = df["IsFree"] * df["SteamSpyOwners"]
    df["free_x_players"] = df["IsFree"] * df["SteamSpyPlayersEstimate"]

    df["content_volume"] = (
        df["ScreenshotCount"] + df["MovieCount"] + df["DLCCount"] + df["PackageCount"]
    )
    df["content_owners"] = df["content_volume"] * df["SteamSpyOwners"]
    df["content_players"] = df["content_volume"] * df["SteamSpyPlayersEstimate"]
    df["content_metacritic"] = df["content_volume"] * metacritic

    df["achievement_owners"] = df["AchievementCount"] * df["SteamSpyOwners"]
    df["achievement_players"] = df["AchievementCount"] * df["SteamSpyPlayersEstimate"]
    df["highlighted_achievements_ratio"] = (
        df["AchievementHighlightedCount"] / (df["AchievementCount"] + 1)
    )

    df["platform_count"] = (
        df["PlatformWindows"].astype(int)
        + df["PlatformLinux"].astype(int)
        + df["PlatformMac"].astype(int)
    )
    df["platform_owners"] = df["platform_count"] * df["SteamSpyOwners"]
    df["platform_players"] = df["platform_count"] * df["SteamSpyPlayersEstimate"]

    df["action_multiplayer"] = df["GenreIsAction"] * df["CategoryMultiplayer"]
    df["rpg_achievement"] = df["GenreIsRPG"] * df["AchievementCount"]
    df["strategy_complexity"] = df["GenreIsStrategy"] * df["AchievementCount"]
    df["indie_price"] = df["GenreIsIndie"] * df["PriceFinal"]

    df["category_count"] = (
        df["CategorySinglePlayer"] + df["CategoryMultiplayer"] + df["CategoryCoop"]
        + df["CategoryMMO"] + df["CategoryVRSupport"]
    )
    df["category_owners"] = df["category_count"] * df["SteamSpyOwners"]
    df["category_players"] = df["category_count"] * df["SteamSpyPlayersEstimate"]

    df["reviews_owners"] = df["review_words"] * df["SteamSpyOwners"]
    df["reviews_players"] = df["review_words"] * df["SteamSpyPlayersEstimate"]
    df["reviews_metacritic"] = df["review_words"] * metacritic

    # ── outlier clipping ──────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=["number"]).columns
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

    Parameters
    ----------
    inputs : dict
        Keys match the raw field names collected by the GUI (see FIELD GUIDE
        below).  Missing optional keys fall back to 0 or the training median.
    feat_cols : list
        Ordered list loaded from Models/feature_columns.pkl.
    feat_medians : dict
        Per-column medians loaded from Models/feature_medians.pkl.

    Returns
    -------
    np.ndarray, shape (1, len(feat_cols))

    FIELD GUIDE (all keys, types, defaults)
    ────────────────────────────────────────
    Numeric
        required_age         int     0
        price_initial        float   9.99
        price_final          float   9.99
        steam_spy_owners     int     500_000
        steam_spy_owners_var int     200_000
        steam_spy_players    int     300_000
        steam_spy_players_var int    150_000
        movie_count          int     1
        screenshot_count     int     10
        dlc_count            int     0
        package_count        int     1
        demo_count           int     0
        achievement_count    int     0
        highlighted_achiev   int     0
        developer_count      int     1
        publisher_count      int     1
    Boolean (int 0/1)
        is_free              int     0
        purchase_avail       int     1
        free_ver_avail       int     0
        ctrl_support         int     0
        plat_win             int     1
        plat_linux           int     0
        plat_mac             int     0
        cat_single           int     1
        cat_multi            int     0
        cat_coop             int     0
        cat_mmo              int     0
        cat_iap              int     0
        cat_vr               int     0
        g_indie              int     0
        g_action             int     0
        g_adventure          int     0
        g_casual             int     0
        g_strategy           int     0
        g_rpg                int     0
        g_simulation         int     0
        g_earlyaccess        int     0
        g_f2p                int     0
        g_sports             int     0
        g_racing             int     0
        g_mmo_genre          int     0
        pc_has_rec           int     0
        mac_has_min          int     0
        has_legal_notice     int     0
        has_website          int     0
        has_support_email    int     0
        has_support_url      int     0
        lang_english         int     1
    Text (raw strings – NLP applied internally)
        about_text           str     ""
        reviews_text         str     ""
        pc_min_reqs_text     str     ""
        pc_rec_reqs_text     str     ""
        mac_min_reqs_text    str     ""
    Frequency-encoding context (pass training freq dicts when available)
        _freq_support_email  dict    {}   (value_counts from training)
        _freq_support_url    dict    {}
        _freq_website        dict    {}
    """

    g = inputs  # shorthand

    # ── raw scalars ───────────────────────────────────────────────────────────
    price_initial      = float(g.get("price_initial", 9.99))
    price_final        = float(g.get("price_final", 9.99))
    owners             = float(g.get("steam_spy_owners", 500_000))
    owners_var         = float(g.get("steam_spy_owners_var", 200_000))
    players            = float(g.get("steam_spy_players", 300_000))
    players_var        = float(g.get("steam_spy_players_var", 150_000))
    movie_count        = float(g.get("movie_count", 1))
    screenshot_count   = float(g.get("screenshot_count", 10))
    dlc_count          = float(g.get("dlc_count", 0))
    package_count      = float(g.get("package_count", 1))
    achievement_count  = float(g.get("achievement_count", 0))
    highlighted_achiev = float(g.get("highlighted_achiev", 0))

    # binary
    plat_win    = int(g.get("plat_win", 1))
    plat_linux  = int(g.get("plat_linux", 0))
    plat_mac    = int(g.get("plat_mac", 0))
    cat_single  = int(g.get("cat_single", 1))
    cat_multi   = int(g.get("cat_multi", 0))
    cat_coop    = int(g.get("cat_coop", 0))
    cat_mmo     = int(g.get("cat_mmo", 0))
    cat_vr      = int(g.get("cat_vr", 0))
    g_indie     = int(g.get("g_indie", 0))
    g_action    = int(g.get("g_action", 0))
    g_adventure = int(g.get("g_adventure", 0))
    g_casual    = int(g.get("g_casual", 0))
    g_strategy  = int(g.get("g_strategy", 0))
    g_rpg       = int(g.get("g_rpg", 0))
    pc_has_rec  = int(g.get("pc_has_rec", 0))
    mac_has_min = int(g.get("mac_has_min", 0))

    # ── NLP ───────────────────────────────────────────────────────────────────
    about_text   = str(g.get("about_text", ""))
    reviews_text = str(g.get("reviews_text", ""))
    cleaned_about = clean_text(about_text)
    all_text_len  = len(cleaned_about.split()) if cleaned_about.strip() else 0
    about_sent    = _sentiment(about_text)
    review_words  = len(reviews_text.split()) if reviews_text.strip() else 0

    kw_war    = int("war"    in cleaned_about)
    kw_action = int("action" in cleaned_about)
    kw_team   = int("team"   in cleaned_about)

    # ── requirements ─────────────────────────────────────────────────────────
    pc_min_ram  = extract_ram(g.get("pc_min_reqs_text", ""))
    pc_rec_ram  = extract_ram(g.get("pc_rec_reqs_text", "")) if pc_has_rec else 0.0
    mac_min_ram = extract_ram(g.get("mac_min_reqs_text", "")) if mac_has_min else 0.0
    pc_min_cpu  = extract_proc(g.get("pc_min_reqs_text", ""))

    # ── frequency-encoded support fields ─────────────────────────────────────
    # Use training-time freq dicts when available, else fall back to median
    support_email_enc = (
        g.get("_freq_support_email", {}).get(g.get("support_email_domain", ""), 0)
        or (feat_medians.get("SupportEmail", 0.00022) if g.get("has_support_email") else 0.0)
    )
    support_url_enc = (
        g.get("_freq_support_url", {}).get(g.get("support_url_domain", ""), 0)
        or (feat_medians.get("SupportURL", 0.00342) if g.get("has_support_url") else 0.0)
    )
    website_enc = (
        g.get("_freq_website", {}).get(g.get("website_domain", ""), 0)
        or (feat_medians.get("Website", 0.00022) if g.get("has_website") else 0.0)
    )

    # ── interaction features (raw, before log1p) ──────────────────────────────
    content_volume   = screenshot_count + movie_count + dlc_count + package_count
    platform_count   = plat_win + plat_linux + plat_mac
    category_count   = cat_single + cat_multi + cat_coop + cat_mmo + cat_vr

    owners_players      = owners * players
    price_owners        = price_final * owners
    price_players       = price_final * players
    content_owners      = content_volume * owners
    content_players     = content_volume * players
    achievement_owners  = achievement_count * owners
    achievement_players = achievement_count * players
    platform_owners     = platform_count * owners
    platform_players    = platform_count * players
    indie_price         = g_indie * price_final
    category_owners     = category_count * owners
    category_players    = category_count * players
    reviews_owners      = review_words * owners
    reviews_players     = review_words * players
    highlighted_ratio   = highlighted_achiev / (achievement_count + 1)

    # ── assemble raw dict ─────────────────────────────────────────────────────
    raw = {
        # numeric – will be log1p'd
        "MovieCount":                  movie_count,
        "ScreenshotCount":             screenshot_count,
        "SteamSpyOwners":              owners,
        "SteamSpyOwnersVariance":      owners_var,
        "SteamSpyPlayersEstimate":     players,
        "SteamSpyPlayersVariance":     players_var,
        "AchievementCount":            achievement_count,
        "AchievementHighlightedCount": highlighted_achiev,
        "PriceInitial":                price_initial,
        "PriceFinal":                  price_final,
        "PC_MinRam":                   pc_min_ram,
        "PC_RecRam":                   pc_rec_ram,
        "Mac_MinRam":                  mac_min_ram,
        "PC_MinCPU":                   pc_min_cpu,
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
        # binary – no log1p
        "ControllerSupport":           int(g.get("ctrl_support", 0)),
        "PlatformMac":                 plat_mac,
        "PCReqsHaveRec":               pc_has_rec,
        "MacReqsHaveMin":              mac_has_min,
        "CategoryMultiplayer":         cat_multi,
        "GenreIsIndie":                g_indie,
        "GenreIsAction":               g_action,
        "GenreIsAdventure":            g_adventure,
        "GenreIsCasual":               g_casual,
        "LegalNotice":                 int(g.get("has_legal_notice", 0)),
        "Lang_english":                int(g.get("lang_english", 1)),
        # keyword flags
        "has_war":                     kw_war,
        "has_action":                  kw_action,
        "has_team":                    kw_team,
        # frequency-encoded
        "SupportEmail":                support_email_enc,
        "SupportURL":                  support_url_enc,
        "Website":                     website_enc,
        # NLP continuous
        "AllText_len":                 all_text_len,
        "AboutSentiment":              about_sent,
        # ratio
        "highlighted_achievements_ratio": highlighted_ratio,
    }

    # ── apply log1p to matching keys ──────────────────────────────────────────
    for k in LOG_TRANSFORM_COLS:
        if k in raw:
            raw[k] = math.log1p(max(0.0, raw[k]))

    # ── build final ordered vector ────────────────────────────────────────────
    row = [raw.get(col, feat_medians.get(col, 0.0)) for col in feat_cols]
    return np.array(row, dtype=float).reshape(1, -1)
