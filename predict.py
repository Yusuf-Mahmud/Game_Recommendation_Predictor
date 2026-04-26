"""
predict.py
──────────
CLI test script and importable inference entry-point.

App.py calls predict() with the raw GUI inputs dict.

CLI usage
─────────
    python predict.py --model xgboost \
        --owners 500000 --players 300000 \
        --price 19.99 \
        --screenshots 10 --movies 1

Library usage (from App.py)
────────────────────────────
    from predict import predict

    result = predict(
        model_name="XGBoost",
        inputs={
            "price_final": 19.99,
            "steam_spy_owners": 500_000,
            ...
        }
    )
    print(result["prediction"])   # float – raw recommendation count
    print(result["rating"])       # str   – Steam-style label
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

# ── locate Models/ relative to this script ───────────────────────────────────
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "Models"

PKL_PATHS = {
    "Linear Regression (Project)": MODELS_DIR / "linear.pkl",
    "Polynomial Regression":        MODELS_DIR / "polynomial.pkl",
    "Ridge":                        MODELS_DIR / "ridge.pkl",
    "Random Forest":                MODELS_DIR / "random_forest.pkl",
    "Gradient Boosting":            MODELS_DIR / "gradient_boosting.pkl",
    "XGBoost":                      MODELS_DIR / "xgboost.pkl",
}

CLI_ALIASES = {
    "linear":      "Linear Regression (Project)",
    "poly":        "Polynomial Regression",
    "polynomial":  "Polynomial Regression",
    "ridge":       "Ridge",
    "rf":          "Random Forest",
    "forest":      "Random Forest",
    "gb":          "Gradient Boosting",
    "xgb":         "XGBoost",
    "xgboost":     "XGBoost",
}

_feat_cols    = None
_feat_medians = None
_model_cache  = {}
_poly_cache   = None


def _load_metadata():
    global _feat_cols, _feat_medians
    if _feat_cols is None:
        _feat_cols    = joblib.load(MODELS_DIR / "feature_columns.pkl")
        _feat_medians = joblib.load(MODELS_DIR / "feature_medians.pkl")
    return _feat_cols, _feat_medians


def _load_model(model_name: str):
    if model_name not in _model_cache:
        path = PKL_PATHS[model_name]
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        _model_cache[model_name] = joblib.load(path)
    return _model_cache[model_name]


def _load_poly_transformer():
    global _poly_cache
    if _poly_cache is None:
        _poly_cache = joblib.load(MODELS_DIR / "polynomial_transformer.pkl")
    return _poly_cache


def _rating(value: float) -> tuple:
    """Return (label, hex_colour) based on predicted recommendation count."""
    if   value < 500:    return "Overwhelmingly Negative", "#ff3333"
    elif value < 2_000:  return "Mostly Negative",         "#ff6622"
    elif value < 5_000:  return "Mixed",                   "#ffaa00"
    elif value < 10_000: return "Mostly Positive",         "#88dd22"
    elif value < 50_000: return "Very Positive",           "#22ddaa"
    else:                return "Overwhelmingly Positive", "#00d4ff"


def predict(model_name: str, inputs: dict) -> dict:
    """
    Parameters
    ----------
    model_name : str
        One of the keys in PKL_PATHS, or a CLI alias.
    inputs : dict
        Raw GUI values. See preprocess.build_inference_row() docstring.

    Returns
    -------
    dict with keys: prediction, rating, color, model, log_pred
    """
    canonical = CLI_ALIASES.get(model_name.lower(), model_name)
    if canonical not in PKL_PATHS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid names: {list(PKL_PATHS.keys())}"
        )

    from preprocess import build_inference_row  # noqa: PLC0415

    feat_cols, feat_medians = _load_metadata()
    X = build_inference_row(inputs, feat_cols, feat_medians)

    if canonical == "Polynomial Regression":
        poly = _load_poly_transformer()
        expected = poly.n_features_in_
        got = X.shape[1]
        if expected != got:
            raise ValueError(
                f"Polynomial transformer expects {expected} features but got {got}. "
                f"Re-run Project.py to retrain all models together."
            )
        X = poly.transform(X)

    model    = _load_model(canonical)
    log_pred = float(model.predict(X)[0])
    value    = max(0.0, float(np.expm1(log_pred)))

    rating, color = _rating(value)
    return {
        "prediction": value,
        "rating":     rating,
        "color":      color,
        "model":      canonical,
        "log_pred":   log_pred,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Predict Steam recommendation count from raw game features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="xgboost",
                   help=f"Model to use. Options: {list(PKL_PATHS)}")

    p.add_argument("--required-age",   type=int,   default=0,       dest="required_age")
    p.add_argument("--demo-count",     type=int,   default=0,       dest="demo_count")
    p.add_argument("--dev-count",      type=int,   default=1,       dest="developer_count")
    p.add_argument("--dlc",            type=int,   default=0,       dest="dlc_count")
    p.add_argument("--metacritic",     type=int,   default=0,       dest="metacritic")
    p.add_argument("--movies",         type=int,   default=1,       dest="movie_count")
    p.add_argument("--packages",       type=int,   default=1,       dest="package_count")
    p.add_argument("--pub-count",      type=int,   default=1,       dest="publisher_count")
    p.add_argument("--screenshots",    type=int,   default=10,      dest="screenshot_count")
    p.add_argument("--owners",         type=int,   default=500_000, dest="steam_spy_owners")
    p.add_argument("--players",        type=int,   default=300_000, dest="steam_spy_players")
    p.add_argument("--achievements",   type=int,   default=0,       dest="achievement_count")
    p.add_argument("--highlighted",    type=int,   default=0,       dest="highlighted_achiev")
    p.add_argument("--price-init",     type=float, default=9.99,    dest="price_initial")
    p.add_argument("--price",          type=float, default=9.99,    dest="price_final")

    p.add_argument("--ctrl",           action="store_true", dest="ctrl_support")
    p.add_argument("--free",           action="store_true", dest="is_free")
    p.add_argument("--free-ver",       action="store_true", dest="free_ver_avail")
    p.add_argument("--purchase",       action="store_true", dest="purchase_avail")
    p.add_argument("--windows",        action="store_true", dest="plat_win")
    p.add_argument("--linux",          action="store_true", dest="plat_linux")
    p.add_argument("--mac",            action="store_true", dest="plat_mac")
    p.add_argument("--single",         action="store_true", dest="cat_single")
    p.add_argument("--multi",          action="store_true", dest="cat_multi")
    p.add_argument("--coop",           action="store_true", dest="cat_coop")
    p.add_argument("--mmo-cat",        action="store_true", dest="cat_mmo")
    p.add_argument("--iap",            action="store_true", dest="cat_iap")
    p.add_argument("--vr",             action="store_true", dest="cat_vr")
    p.add_argument("--indie",          action="store_true", dest="g_indie")
    p.add_argument("--action",         action="store_true", dest="g_action")
    p.add_argument("--adventure",      action="store_true", dest="g_adventure")
    p.add_argument("--casual",         action="store_true", dest="g_casual")
    p.add_argument("--strategy",       action="store_true", dest="g_strategy")
    p.add_argument("--rpg",            action="store_true", dest="g_rpg")
    p.add_argument("--simulation",     action="store_true", dest="g_simulation")
    p.add_argument("--earlyaccess",    action="store_true", dest="g_earlyaccess")
    p.add_argument("--f2p",            action="store_true", dest="g_f2p")
    p.add_argument("--sports",         action="store_true", dest="g_sports")
    p.add_argument("--racing",         action="store_true", dest="g_racing")
    p.add_argument("--mmo-genre",      action="store_true", dest="g_mmo_genre")

    return p


def main():
    parser = _build_parser()
    args = vars(parser.parse_args())
    model_name = args.pop("model")
    args.setdefault("plat_win", 1)
    args.setdefault("cat_single", 1)
    args.setdefault("purchase_avail", 1)

    result = predict(model_name, args)

    print(f"\n{'─'*48}")
    print(f"  Model      : {result['model']}")
    print(f"  Prediction : {result['prediction']:,.0f} recommendations")
    print(f"  Rating     : {result['rating']}")
    print(f"  Log output : {result['log_pred']:.4f}")
    print(f"{'─'*48}\n")


if __name__ == "__main__":
    main()
