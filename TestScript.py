def _predict_classification(model_name: str, raw: dict) -> str:
    name = model_name.lower()
    if name not in CLASSIFICATION_MODELS:
        raise ValueError(
            f"Unknown classification model '{model_name}'. "
            f"Valid options: {list(CLASSIFICATION_MODELS)}"
        )

    # Build the fully preprocessed row (all derived features computed)
    row = _build_row(raw)

    # ── Reconstruct the exact DataFrame the ColumnTransformer was fitted on ──
    # The preprocessor expects these named columns in the DataFrame:
    #   'AllText_Cleaned'            → text pipeline (TF-IDF word + char)
    #   log_cols (60 cols)          → numeric_transformer (imputer+log+scaler)
    #   keyword_cols (15 cols)      → passthrough
    #   'ReviewSentiment',
    #   'AboutSentiment'            → passthrough
    #   'Website'                   → url_pipeline (URLExtractor + FreqEncoder)
    #   'PriceInitial','PriceFinal',
    #   'ScreenshotCount','MovieCount',
    #   'DLCCount'                  → SteamFeatureInteractions
    #   'PCMinReqsText'             → RAMExtractor + CPUExtractor

    # ── Exact 51 columns the classification models were trained on ────────────
    # Taken directly from df.columns at training time (after correlation filter)
    # ── Exact 51 columns the classification models were trained on ────────────
    # Source: KEPT COLUMNS from correlation filter in notebook (classification mode)
    # ── Exact 51 columns the models trained on — bypass preprocessor entirely ─
    # Models were trained on raw df values after correlation filter.
    # No scaling/imputing needed here since _build_row already computes everything.
    _CLASS_51_COLS = [
        "SteamSpyPlayersVariance", "SteamSpyPlayersEstimate", "content_players",
        "category_players", "platform_players", "owners_players", "SteamSpyOwnersVariance",
        "content_owners", "reviews_players", "SteamSpyOwners", "platform_owners",
        "category_owners", "reviews_owners", "price_players", "achievement_players",
        "achievement_owners", "price_owners", "content_volume", "AchievementCount",
        "category_count", "MovieCount", "AllText_len", "CategoryMultiplayer",
        "AchievementHighlightedCount", "ScreenshotCount", "PriceInitial", "PriceFinal",
        "PC_MinCPU", "LegalNotice", "SupportEmail", "has_team", "GenreIsAction",
        "has_war", "PC_RecRam", "Mac_MinRam", "PCReqsHaveRec", "PlatformMac",
        "has_action", "platform_count", "MacReqsHaveMin", "PC_MinRam",
        "highlighted_achievements_ratio", "ControllerSupport", "GenreIsAdventure",
        "indie_price", "AboutSentiment", "GenreIsIndie", "SupportURL",
        "Lang_english", "GenreIsCasual", "Website",
    ]

    # ── Build X directly — no preprocessor needed ─────────────────────────────
    X = np.array(
        [float(row.get(col, 0)) for col in _CLASS_51_COLS],
        dtype=float
    ).reshape(1, -1)

    model = joblib.load(CLASSIFICATION_MODELS[name])

    pred = model.predict(X)[0]

    # OrdinalEncoder categories=['Low','Medium','High'] → decode back if numeric
    try:
        idx = int(round(float(pred)))
        labels = ["Low", "Medium", "High"]
        return labels[idx] if 0 <= idx < len(labels) else str(pred)
    except (ValueError, TypeError):
        return str(pred)

import re
import math
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse

import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)
warnings.filterwarnings("ignore")

# ── Custom sklearn classes ────────────────────────────────────────────────────
# production_preprocessor.pkl was pickled in the Colab notebook where these
# classes lived in __main__.  Joblib resolves class references by
# (module, qualname) stored in the pickle.  Two things are needed:
#
#   1. The classes must be defined at module level here (done below).
#   2. They must also be reachable as __main__.ClassName so that when this
#      script is IMPORTED (not run directly) unpickling still works.
#      → We inject them into sys.modules["__main__"] after definition.

import sys
from sklearn.base import BaseEstimator, TransformerMixin

class URLExtractor(BaseEstimator, TransformerMixin):
    """Extract netloc (domain) from a URL column."""
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (
            X[self.column_name]
            .apply(lambda x: urlparse(str(x)).netloc
                   if pd.notnull(x) and str(x).strip() != "" else "none")
            .values.reshape(-1, 1)
        )
"""
test_predict.py
───────────────
Unified inference function for both classification and regression Steam models.

Usage
─────
    from test_predict import predict_game

    # Regression
    result = predict_game(
        raw={"PriceFinal": 19.99, "SteamSpyOwners": 500_000, ...},
        model_name="xgboost",
        mode="r"
    )
    # → "42381"  (predicted recommendation count as string)

    # Classification
    result = predict_game(
        raw={"PriceFinal": 19.99, "SteamSpyOwners": 500_000, ...},
        model_name="random_forest",
        mode="c"
    )
    # → "High"  (predicted popularity class as string)

Parameters
──────────
raw : dict
    Raw game features.  The following columns were DROPPED in the notebook
    after feature extraction, so do NOT pass them:
        QueryName, ResponseName, QueryID, ResponseID
        AllText, AllText_Cleaned                 ← auto-built internally
        PCMinReqsText, PCRecReqsText             ← pass these if you want
        LinuxMinReqsText, LinuxRecReqsText         RAM/CPU extracted; safe
        MacMinReqsText, MacRecReqsText             to omit if not needed
        SupportedLanguages                       ← pass as comma-sep string

    The following ARE used for NLP and should be passed when available
    (they default to empty string / "none" if omitted, giving zero signal):
        "DetailedDescrip"  →  main game description text
        "ShortDescrip"     →  short description (combined with DetailedDescrip
                               to form AllText → TF-IDF + keyword features)
        "AboutText"        →  About section  (→ AboutSentiment via VADER)
        "Reviews"          →  critic/user review blurb (→ ReviewSentiment,
                               review_words, and review interaction features)

    All other original columns are accepted.  Missing keys are filled with
    sensible defaults so you can pass a sparse dict and still get a result.

model_name : str
    Regression  → "linear" | "polynomial" | "ridge" | "random_forest"
                  | "gradient_boosting" | "xgboost"
    Classification → "logistic" | "random_forest" | "gradient_boosting"
                     | "xgboost"

mode : str
    "r" or "R" → regression
    "c" or "C" → classification

Returns
───────
str  – predicted value (recommendation count for regression, class for classification)
"""


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode each value as its relative frequency seen during fit."""

    def __init__(self):
        self.counts_ = {}

    def fit(self, X, y=None):
        X_flat = pd.Series(X.ravel()) if hasattr(X, "ravel") else pd.Series(X)
        self.counts_ = X_flat.value_counts(normalize=True).to_dict()
        return self
    def transform(self, X):
        X_flat = pd.Series(X.ravel()) if hasattr(X, "ravel") else pd.Series(X)
        return X_flat.map(self.counts_).fillna(0).values.reshape(-1, 1)


class SteamFeatureInteractions(BaseEstimator, TransformerMixin):
    """Compute price_discount and content_volume interaction columns."""

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for col in ["PriceInitial", "PriceFinal", "ScreenshotCount", "MovieCount", "DLCCount"]:
            if col not in X.columns:
                X[col] = 0
        X["price_discount"] = X["PriceInitial"] - X["PriceFinal"]
        X["content_volume"] = X["ScreenshotCount"] + X["MovieCount"] + X["DLCCount"]
        return X


class RAMExtractor(BaseEstimator, TransformerMixin):
    """Extract RAM in GB from a requirements text column."""

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        def _extract(text):
            m = re.search(r"(\d+)\s?(GB|MB)", str(text), re.IGNORECASE)
            if m:
                val, unit = m.groups()
                return int(val) if unit.upper() == "GB" else int(val) / 1024
            return 0
        return X[self.column_name].apply(_extract).values.reshape(-1, 1)


class CPUExtractor(BaseEstimator, TransformerMixin):
    """Extract CPU speed in GHz from a requirements text column."""

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        def _extract(text):
            m = re.search(r"(\d+(?:\.\d+)?)\s?(MHZ|GHZ)", str(text), re.IGNORECASE)
            if m:
                val, unit = m.groups()
                val = float(val)
                return val if unit.upper() == "GHZ" else val / 1000
            return 0
        return X[self.column_name].apply(_extract).values.reshape(-1, 1)



# ── Register classes in __main__ so joblib can always find them ──────────────
# This is needed when test_predict.py is imported as a module rather than
# executed directly.  Harmless when running as __main__.
_classes_for_pickle = [
    URLExtractor, FrequencyEncoder, SteamFeatureInteractions,
    RAMExtractor, CPUExtractor,
]
_main_module = sys.modules.get("__main__")


# ── Paths ─────────────────────────────────────────────────────────────────────

if _main_module is not None:
    for _cls in _classes_for_pickle:
        if not hasattr(_main_module, _cls.__name__):
            setattr(_main_module, _cls.__name__, _cls)
BASE_DIR   = Path(__file__).resolve().parent / "saved_models"

CLASS_DIR  = BASE_DIR / "classification"

REGRESSION_MODELS = {
    "linear":             BASE_DIR / "linear.pkl",
    "polynomial":         BASE_DIR / "polynomial.pkl",
    "ridge":              BASE_DIR / "ridge.pkl",
    "random_forest":      BASE_DIR / "random_forest.pkl",
    "gradient_boosting":  BASE_DIR / "gradient_boosting.pkl",
    "xgboost":            BASE_DIR / "xgboost.pkl",
}

CLASSIFICATION_MODELS = {
    "logistic":           CLASS_DIR / "logistic.pkl",
    "random_forest":      CLASS_DIR / "random_forest.pkl",
    "gradient_boosting":  CLASS_DIR / "gradient_boosting.pkl",
    "xgboost":            CLASS_DIR / "xgboost.pkl",
}
POLY_TRANSFORMER   = BASE_DIR / "polynomial_transformer.pkl"

# ── Constants (mirrors notebook) ──────────────────────────────────────────────

CLASS_PREPROCESSOR = CLASS_DIR / "production_preprocessor.pkl"

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
    "war", "action", "team", "free", "strategy", "shooter", "rpg",
    "indie", "puzzle", "horror",
]

CUSTOM_STOPWORDS = {
    "game", "games", "play", "players", "player",
    "world", "time", "experience", "like", "make",
    "based", "new", "use", "way", "different",
}
# Columns to apply log1p on (regression path only)

LOG_TRANSFORM_COLS = {
    "RequiredAge", "DemoCount", "DeveloperCount", "DLCCount", "Metacritic",
    "MovieCount", "PackageCount", "PublisherCount", "ScreenshotCount",
    "SteamSpyOwners", "SteamSpyPlayersEstimate", "AchievementCount",
    "AchievementHighlightedCount", "PriceInitial", "PriceFinal",
    "price_discount", "platform_count", "category_count", "content_volume",
    "indie_price", "owners_players", "price_owners", "price_players",
    "free_x_owners", "free_x_players", "content_owners", "content_players",
    "action_multiplayer", "rpg_achievement", "strategy_complexity",
    "achievement_owners", "achievement_players", "platform_owners",
    "platform_players", "category_owners", "category_players",
}
# The 51 features the regression models expect (order matters)

# ── Helpers ───────────────────────────────────────────────────────────────────

INFERRABLE_FEATURES = [
    "RequiredAge", "DemoCount", "DeveloperCount", "DLCCount", "Metacritic",
    "MovieCount", "PackageCount", "PublisherCount", "ScreenshotCount",
    "SteamSpyOwners", "SteamSpyPlayersEstimate", "AchievementCount",
    "AchievementHighlightedCount", "ControllerSupport", "IsFree",
    "FreeVerAvail", "PurchaseAvail", "PlatformWindows", "PlatformLinux",
    "PlatformMac", "CategorySinglePlayer", "CategoryMultiplayer",
    "CategoryCoop", "CategoryMMO", "CategoryInAppPurchase", "CategoryVRSupport",
    "GenreIsIndie", "GenreIsAction", "GenreIsAdventure", "GenreIsCasual",
    "GenreIsStrategy", "GenreIsRPG", "GenreIsSimulation", "GenreIsEarlyAccess",
    "GenreIsFreeToPlay", "GenreIsSports", "GenreIsRacing",
    "GenreIsMassivelyMultiplayer", "PriceInitial", "PriceFinal",
    "price_discount", "platform_count", "category_count", "content_volume",
    "highlighted_achievements_ratio", "action_multiplayer", "rpg_achievement",
    "strategy_complexity", "indie_price", "owners_players", "price_owners",
    "price_players", "free_x_owners", "free_x_players", "content_owners",
    "content_players", "achievement_owners", "achievement_players",
    "platform_owners", "platform_players", "category_owners", "category_players",
]


def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = [t for t in text.split() if t not in CUSTOM_STOPWORDS]
    return " ".join(tokens)


def _extract_ram(text) -> float:
    if not text or str(text).strip() in ("", "nan"):
        return 0.0
    match = re.search(r"(\d+)\s?(GB|MB)", str(text), re.IGNORECASE)
    if match:
        val, unit = match.groups()
        return int(val) if unit.upper() == "GB" else int(val) / 1024
    return 0.0


def _extract_proc(text) -> float:
    if not text or str(text).strip() in ("", "nan"):
        return 0.0
    match = re.search(r"(\d+(?:\.\d+)?)\s?(MHZ|GHZ)", str(text), re.IGNORECASE)
    if match:
        val, unit = match.groups()
        val = float(val)
        return val if unit.upper() == "GHZ" else val / 1000
    return 0.0


def _parse_langs(text) -> list:
    if not text or str(text).strip() == "":
        return []
    return [p.strip().lower() for p in re.split(r",|\s{2,}", str(text)) if p.strip()]


# ── Default values for every raw column (used when key is absent from dict) ──

def _freq_encode_single(value: str, freq_map: dict) -> float:
    """Map a single value through a pre-built frequency dict."""
    domain = urlparse(value).netloc if value else "none"
    return freq_map.get(domain, 0.0)


# ── Core preprocessing  (mirrors notebook exactly) ───────────────────────────

COLUMN_DEFAULTS = {
    # numeric
    "RequiredAge":                0,
    "DemoCount":                  0,
    "DeveloperCount":             1,
    "DLCCount":                   0,
    "Metacritic":                 0,
    "MovieCount":                 1,
    "PackageCount":               1,
    "PublisherCount":             1,
    "ScreenshotCount":            10,
    "SteamSpyOwners":             500_000,
    "SteamSpyOwnersVariance":     0,
    "SteamSpyPlayersEstimate":    300_000,
    "SteamSpyPlayersVariance":    0,
    "AchievementCount":           0,
    "AchievementHighlightedCount":0,
    "PriceInitial":               9.99,
    "PriceFinal":                 9.99,
    # bool / binary
    "ControllerSupport":          0,
    "IsFree":                     0,
    "FreeVerAvail":               0,
    "PurchaseAvail":              1,
    "SubscriptionAvail":          0,
    "PlatformWindows":            1,
    "PlatformLinux":              0,
    "PlatformMac":                0,
    "PCReqsHaveMin":              0,
    "PCReqsHaveRec":              0,
    "LinuxReqsHaveMin":           0,
    "LinuxReqsHaveRec":           0,
    "MacReqsHaveMin":             0,
    "MacReqsHaveRec":             0,
    "CategorySinglePlayer":       1,
    "CategoryMultiplayer":        0,
    "CategoryCoop":               0,
    "CategoryMMO":                0,
    "CategoryInAppPurchase":      0,
    "CategoryIncludeSrcSDK":      0,
    "CategoryIncludeLevelEditor": 0,
    "CategoryVRSupport":          0,
    "GenreIsNonGame":             0,
    "GenreIsIndie":               0,
    "GenreIsAction":              0,
    "GenreIsAdventure":           0,
    "GenreIsCasual":              0,
    "GenreIsStrategy":            0,
    "GenreIsRPG":                 0,
    "GenreIsSimulation":          0,
    "GenreIsEarlyAccess":         0,
    "GenreIsFreeToPlay":          0,
    "GenreIsSports":              0,
    "GenreIsRacing":              0,
    "GenreIsMassivelyMultiplayer":0,
    # text that feeds NLP (kept so the function works if caller passes them)
    "SupportURL":                 "",
    "SupportEmail":               "",
    "Website":                    "",
    "SupportedLanguages":         "",
    "ReleaseDate":                None,
    # req texts (for RAM/CPU extraction)
    "PCMinReqsText":              "",
    "PCRecReqsText":              "",
    "MacMinReqsText":             "",
    "MacMinReqsCPU":              "",
    # image / notice binary sources
    "HeaderImage":                None,
    "Background":                 None,
    "LegalNotice":                None,
    "DRMNotice":                  None,
    "ExtUserAcctNotice":          None,
    # NLP source text (not passed by caller → defaults to empty)
    "DetailedDescrip":            "",
    "ShortDescrip":               "",
    "AboutText":                  "",
    "Reviews":                    "none",
}


def _build_row(raw: dict) -> pd.Series:
    """
    Turn a raw dict into a single preprocessed pandas Series whose
    index matches the training feature space used by the regression models.
    All steps mirror notebook_milestone2 in order.
    """
    # Fill missing keys with defaults
    d = {k: raw.get(k, v) for k, v in COLUMN_DEFAULTS.items()}
    # Also carry through any extra keys the caller added
    for k, v in raw.items():
        if k not in d:
            d[k] = v

    row = pd.Series(d)

    # ── boolean → int ────────────────────────────────────────────────────────
    for col in BOOL_COLS:
        if col in row.index:
            row[col] = int(bool(row[col]))

    # ── binary from text presence ─────────────────────────────────────────────
    row["PriceCurrency"] = 1   # dict won't contain currency; default USD = 1
    row["Background"]    = 0 if (row["Background"] in (None, "none", "", np.nan)) else 1
    row["HeaderImage"]   = 0 if row["HeaderImage"] is None else 1
    row["LegalNotice"]   = 0 if row["LegalNotice"]  is None else 1
    row["DRMNotice"]     = 0 if row["DRMNotice"]    is None else 1
    row["ExtUserAcctNotice"] = 0 if row["ExtUserAcctNotice"] is None else 1

    # ── frequency encoding (single-row → use 0 as fallback) ──────────────────
    # For a single inference row there is no corpus to compute frequencies from.
    # We set them to 0, which matches the "unseen" behaviour in the notebook's
    # .fillna(0) calls.
    row["QueryName_FreqEnc"]   = 0.0
    row["ResponseName_FreqEnc"]= 0.0
    row["SupportURL"]          = 0.0
    row["SupportEmail"]        = 0.0
    row["Website"]             = 0.0

    # ── languages ─────────────────────────────────────────────────────────────
    langs = _parse_langs(row.get("SupportedLanguages", ""))
    for lang in SELECTED_LANGUAGES:
        row[f"Lang_{lang}"] = int(lang in langs)
    row["SupportedLanguagesCount"] = len(langs)

    # ── release date → age in years ───────────────────────────────────────────
    rd = row.get("ReleaseDate", None)
    if rd is not None:
        try:
            ts = pd.to_datetime(rd, errors="coerce")
            row["ReleaseDate"] = (pd.Timestamp.today() - ts).days / 365.25 if pd.notnull(ts) else 5.0
        except Exception:
            row["ReleaseDate"] = 5.0
    else:
        row["ReleaseDate"] = 5.0   # median-ish fallback

    # ── requirements extraction ───────────────────────────────────────────────
    row["PC_MinRam"]  = _extract_ram(row.get("PCMinReqsText", ""))  if row["PCReqsHaveMin"]  else 0.0
    row["PC_RecRam"]  = _extract_ram(row.get("PCRecReqsText", ""))  if row["PCReqsHaveRec"]  else 0.0
    row["Mac_MinRam"] = _extract_ram(row.get("MacMinReqsText", "")) if row["MacReqsHaveMin"] else 0.0
    row["PC_MinCPU"]  = _extract_proc(row.get("PCMinReqsText", "")) if row["PCReqsHaveMin"]  else 0.0

    # ── NLP ───────────────────────────────────────────────────────────────────
    all_text = str(row.get("DetailedDescrip", "")) + " " + str(row.get("ShortDescrip", ""))
    cleaned  = _clean_text(all_text)
    row["AllText_Cleaned"] = cleaned
    row["AllText_len"]     = len(cleaned.split())

    for word in KEYWORDS:
        row[f"has_{word.replace(' ', '_')}"] = int(word in cleaned)

    sia = SentimentIntensityAnalyzer()
    row["ReviewSentiment"] = sia.polarity_scores(str(row.get("Reviews", "")))["compound"]
    row["AboutSentiment"]  = sia.polarity_scores(str(row.get("AboutText", "")))["compound"]
    row["review_words"]    = len(str(row.get("Reviews", "")).split())

    # ── feature interactions ──────────────────────────────────────────────────
    owners   = float(row["SteamSpyOwners"])
    players  = float(row["SteamSpyPlayersEstimate"])
    price_i  = float(row["PriceInitial"])
    price_f  = float(row["PriceFinal"])
    is_free  = int(row["IsFree"])
    ach      = float(row["AchievementCount"])
    ach_hi   = float(row["AchievementHighlightedCount"])
    screens  = float(row["ScreenshotCount"])
    movies   = float(row["MovieCount"])
    dlc      = float(row["DLCCount"])
    pkg      = float(row["PackageCount"])
    plat_w   = int(row["PlatformWindows"])
    plat_l   = int(row["PlatformLinux"])
    plat_m   = int(row["PlatformMac"])
    cat_s    = int(row["CategorySinglePlayer"])
    cat_mu   = int(row["CategoryMultiplayer"])
    cat_co   = int(row["CategoryCoop"])
    cat_mm   = int(row["CategoryMMO"])
    cat_vr   = int(row["CategoryVRSupport"])
    g_indie  = int(row["GenreIsIndie"])
    g_action = int(row["GenreIsAction"])
    g_rpg    = int(row["GenreIsRPG"])
    g_strat  = int(row["GenreIsStrategy"])

    row["owners_players"]   = owners * players
    row["price_discount"]   = max(0.0, price_i - price_f)
    row["price_owners"]     = price_f * owners
    row["price_players"]    = price_f * players
    row["free_x_owners"]    = is_free * owners
    row["free_x_players"]   = is_free * players

    content_volume = screens + movies + dlc + pkg
    row["content_volume"]   = content_volume
    row["content_owners"]   = content_volume * owners
    row["content_players"]  = content_volume * players

    row["achievement_owners"]             = ach * owners
    row["achievement_players"]            = ach * players
    row["highlighted_achievements_ratio"] = ach_hi / (ach + 1)

    platform_count = plat_w + plat_l + plat_m
    row["platform_count"]   = platform_count
    row["platform_owners"]  = platform_count * owners
    row["platform_players"] = platform_count * players

    row["action_multiplayer"]  = g_action * cat_mu
    row["rpg_achievement"]     = g_rpg    * ach
    row["strategy_complexity"] = g_strat  * ach
    row["indie_price"]         = g_indie  * price_f

    category_count = cat_s + cat_mu + cat_co + cat_mm + cat_vr
    row["category_count"]   = category_count
    row["category_owners"]  = category_count * owners
    row["category_players"] = category_count * players

    # metacritic interactions (kept for classification path completeness)
    metacritic = float(row.get("Metacritic", 0) or 0)
    row["owners_metacritic"]   = owners  * metacritic
    row["players_metacritic"]  = players * metacritic
    row["content_metacritic"]  = content_volume * metacritic
    row["reviews_owners"]      = row["review_words"] * owners
    row["reviews_players"]     = row["review_words"] * players
    row["reviews_metacritic"]  = row["review_words"] * metacritic

    return row


# ── Regression inference ──────────────────────────────────────────────────────

def _apply_log1p(row: pd.Series) -> pd.Series:
    """Apply log1p to the same columns the notebook transformed."""
    for col in LOG_TRANSFORM_COLS:
        if col in row.index:
            row[col] = math.log1p(max(0.0, float(row[col])))
    return row


# ── Classification inference ──────────────────────────────────────────────────

def _predict_regression(model_name: str, raw: dict) -> str:
    name = model_name.lower()
    if name not in REGRESSION_MODELS:
        raise ValueError(
            f"Unknown regression model '{model_name}'. "
            f"Valid options: {list(REGRESSION_MODELS)}"
        )

    # 1. Build & transform row
    row = _build_row(raw)
    row = _apply_log1p(row)

    # 2. Align to expected feature columns
    # Try loading feature_columns.pkl if present, else use INFERRABLE_FEATURES
    feat_col_path = BASE_DIR / "feature_columns.pkl"
    if feat_col_path.exists():
        feat_cols = joblib.load(feat_col_path)
    else:
        feat_cols = INFERRABLE_FEATURES

    feat_medians_path = BASE_DIR / "feature_medians.pkl"
    feat_medians = joblib.load(feat_medians_path) if feat_medians_path.exists() else {}

    X = np.array(
        [row.get(col, feat_medians.get(col, 0.0)) for col in feat_cols],
        dtype=float
    ).reshape(1, -1)

    # 3. Polynomial transform if needed
    model = joblib.load(REGRESSION_MODELS[name])
    if name == "polynomial":
        poly = joblib.load(POLY_TRANSFORMER)
        X = poly.transform(X)

    # 4. Predict & reverse log1p
    log_pred = float(model.predict(X)[0])
    value    = max(0.0, float(np.expm1(log_pred)))

    return str(round(value))
# Columns that the notebook's numeric_transformer (imputer→log→scaler) was
# fitted on.  Must match exactly what was passed to log_cols during training.

_CLASS_LOG_COLS = [
    "RequiredAge", "DemoCount", "DeveloperCount", "DLCCount", "Metacritic",
    "MovieCount", "PackageCount", "PublisherCount", "ScreenshotCount",
    "SteamSpyOwners", "SteamSpyOwnersVariance", "SteamSpyPlayersEstimate",
    "SteamSpyPlayersVariance", "AchievementCount", "AchievementHighlightedCount",
    "PriceInitial", "PriceFinal", "ReleaseDate",
    "PC_MinRam", "PC_RecRam", "Linux_MinRam", "Linux_RecRam",
    "Mac_MinRam", "Mac_RecRam", "PC_MinCPU", "PC_RecCPU",
    "Linux_MinCPU", "Linux_RecCPU", "Mac_MinCPU", "Mac_RecCPU",
    "owners_metacritic", "players_metacritic", "owners_players",
    "price_discount", "price_owners", "price_players",
    "free_x_owners", "free_x_players",
    "content_volume", "content_owners", "content_players", "content_metacritic",
    "achievement_owners", "achievement_players", "highlighted_achievements_ratio",
    "platform_count", "platform_owners", "platform_players",
    "action_multiplayer", "rpg_achievement", "strategy_complexity", "indie_price",
    "category_count", "category_owners", "category_players",
    "review_words", "reviews_owners", "reviews_players", "reviews_metacritic",
    "SupportedLanguagesCount",
]
# Keyword column names (same order as training)


# ── Public API ────────────────────────────────────────────────────────────────

def predict_game(raw: dict, model_name: str, mode: str) -> str:
    """
    Parameters
    ----------
    raw        : dict   Raw feature values (dropped columns are not expected).
    model_name : str    Model key (see module docstring for valid names).
    mode       : str    "r"/"R" for regression, "c"/"C" for classification.

    Returns
    -------
    str  – Predicted recommendation count (regression) or class label (classification).
    """
    if mode.lower() == "r":
        return _predict_regression(model_name, raw)
    elif mode.lower() == "c":
        return _predict_classification(model_name, raw)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Use 'r' for regression or 'c' for classification.")


# ── Quick smoke test ──────────────────────────────────────────────────────────

def _check_model_files():
    """Print a file-existence report before running predictions."""
    print("  Checking model files...")
    all_paths = {**REGRESSION_MODELS, **{"class/" + k: v for k, v in CLASSIFICATION_MODELS.items()}}
    all_paths["class/preprocessor"] = CLASS_PREPROCESSOR
    missing = []
    for name, path in all_paths.items():
        status = "OK  " if path.exists() else "MISS"
        if not path.exists():
            missing.append(name)
        print(f"    [{status}] {name:<30} {path.name}")
    if missing:
        print(f"\n  ⚠  Missing files: {missing}")
        print("  Re-run Project.py / notebook to regenerate them.\n")
    else:
        print("  All model files found.\n")


_KEYWORD_COLS = [f"has_{w.replace(' ', '_')}" for w in KEYWORDS]

if __name__ == "__main__":
    sample = {
        # ── Core numeric ──────────────────────────────────────────────────────
        "RequiredAge":               0,
        "DemoCount":                 0,
        "DeveloperCount":            1,
        "DLCCount":                  3,
        "Metacritic":                82,
        "MovieCount":                2,
        "PackageCount":              2,
        "PublisherCount":            1,
        "ScreenshotCount":           15,
        "SteamSpyOwners":            1_200_000,
        "SteamSpyPlayersEstimate":   800_000,
        "AchievementCount":          50,
        "AchievementHighlightedCount": 10,
        "PriceInitial":              29.99,
        "PriceFinal":                19.99,

        # ── Boolean flags ─────────────────────────────────────────────────────
        "ControllerSupport":         1,
        "IsFree":                    0,
        "FreeVerAvail":              0,
        "PurchaseAvail":             1,
        "PlatformWindows":           1,
        "PlatformLinux":             1,
        "PlatformMac":               0,
        "CategorySinglePlayer":      1,
        "CategoryMultiplayer":       0,
        "CategoryCoop":              0,
        "CategoryMMO":               0,
        "CategoryInAppPurchase":     0,
        "CategoryVRSupport":         0,
        "GenreIsIndie":              1,
        "GenreIsAction":             1,
        "GenreIsAdventure":          0,
        "GenreIsCasual":             0,
        "GenreIsStrategy":           0,
        "GenreIsRPG":                0,
        "GenreIsSimulation":         0,
        "GenreIsEarlyAccess":        0,
        "GenreIsFreeToPlay":         0,
        "GenreIsSports":             0,
        "GenreIsRacing":             0,
        "GenreIsMassivelyMultiplayer": 0,

        # ── Metadata ──────────────────────────────────────────────────────────
        "SupportedLanguages":        "english, french, german",
        "ReleaseDate":               "2022-03-15",

        # ── NLP source text ───────────────────────────────────────────────────
        # These were DROPPED from the DataFrame in the notebook AFTER extraction,
        # but they are the raw inputs to the NLP pipeline.  Always pass them for
        # best prediction quality; if omitted all NLP features default to 0.
        #
        # DetailedDescrip + ShortDescrip
        #   → combined into AllText_Cleaned
        #   → TF-IDF features (classification preprocessor)
        #   → 15 keyword flags: has_multiplayer, has_action, has_rpg, …
        #   → AllText_len
        #
        # AboutText  → AboutSentiment  (VADER compound score, -1 to +1)
        #
        # Reviews    → ReviewSentiment (VADER compound score)
        #           → review_words     (word count)
        #           → reviews_owners / reviews_players / reviews_metacritic
        #             (interaction features used by the classification model)
        "DetailedDescrip": (
            "An action-packed indie adventure game with online multiplayer. "
            "Fight through hordes of zombies in a post-apocalyptic war zone. "
            "Team up with friends for an intense co op shooter experience."
        ),
        "ShortDescrip": (
            "Indie action shooter with online multiplayer and zombie survival."
        ),
        "AboutText": (
            "This game offers a thrilling and immersive single player campaign "
            "alongside a robust online multiplayer mode. Great for strategy fans."
        ),
        "Reviews": (
            "An outstanding game. The multiplayer is fantastic and the indie "
            "aesthetic really shines. Highly recommended for action RPG lovers."
        ),

        # ── System requirements (optional) ────────────────────────────────────
        # Pass PCReqsHaveMin=1 together with PCMinReqsText to get RAM/CPU
        # features. Safe to omit — defaults to 0 GB / 0 GHz.
        "PCReqsHaveMin":   1,
        "PCMinReqsText":   "Minimum: OS: Windows 10, RAM: 8 GB, CPU: 2.5 GHz",
        "PCReqsHaveRec":   1,
        "PCRecReqsText":   "Recommended: OS: Windows 10, RAM: 16 GB, CPU: 3.5 GHz",
    }

    _check_model_files()
    print("─" * 56)
    print("  REGRESSION PREDICTIONS")
    print("─" * 56)
    for m in ["xgboost", "random_forest", "gradient_boosting", "ridge", "linear", "polynomial"]:
        try:
            res = predict_game(sample, m, "r")
            print(f"  {m:<24} → {res} recommendations")
        except Exception as e:
            print(f"  {m:<24} → ERROR: {e}")

    print()
    print("─" * 56)
    print("  CLASSIFICATION PREDICTIONS")
    print("─" * 56)
    for m in ["random_forest", "gradient_boosting", "xgboost", "logistic"]:
        try:
            res = predict_game(sample, m, "c")
            print(f"  {m:<24} → {res}")
        except Exception as e:
            print(f"  {m:<24} → ERROR: {e}")
    print("─" * 56)