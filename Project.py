# Core
import numpy as np
import pandas as pd
import re
from datetime import datetime
from urllib.parse import urlparse

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

import os

# model + split
from sklearn.model_selection import train_test_split

# preprocessing
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    PolynomialFeatures,
    FunctionTransformer,
    LabelEncoder
)

# models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from textblob import TextBlob
from sklearn.pipeline import FeatureUnion
# Boosting
import xgboost as xgb


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import joblib


# Upload Dataset

df = pd.read_csv("Data/train_data.csv")


# Exploring Data

print(f"Dataset shape: {df.shape}")

print(list(df.columns))

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print(df.head())

print(df.info())

print(df.describe())


# Preprocessing

### Handel duplicates


"""
Check Duplicates
"""

df.duplicated().sum()
df = df.drop_duplicates()

print(df.shape)


### Check Nulls

print(df.isna().sum()[df.isna().sum() > 0])

"""
Check Empty Cells
"""

df = df.replace(r'^\s*$', np.nan, regex=True)

"""
Check Nulls
"""
print(df.isna().sum()[df.isna().sum() > 0])



### Replace Nulls
"""
QueryName
"""

# Fill missing QueryName values using ResponseName
df["QueryName"] = df["QueryName"].fillna(df["ResponseName"])



"""
PriceCurrency
"""

# Check unique values
print(df["PriceCurrency"].value_counts(dropna=False))

# Fill missing values with USD
df["PriceCurrency"] = df["PriceCurrency"].fillna("USD")


"""
SupportURL
"""
#fill Nulls
df['SupportURL'] = df['SupportURL'].fillna("")



"""
SupportEmail
"""

#fill Nulls
df['SupportEmail'] = df['SupportEmail'].fillna("")


"""
Website
"""

#fill Nulls
df['Website'] = df['Website'].fillna("")


"""
Description columns
"""


# fill with first found description and feed it to others, if none found make place holder
bestDescrip = df['DetailedDescrip'].fillna(df['AboutText']).fillna(df['ShortDescrip'])

df['AboutText'] = df['AboutText'].fillna(bestDescrip).fillna('none')
df['DetailedDescrip'] = df['DetailedDescrip'].fillna(bestDescrip).fillna('none')
df['ShortDescrip'] = df['ShortDescrip'].fillna(bestDescrip).fillna('none')
# nlp to be applied on them


"""
Images
"""


# i think either replace with placeholder or just turn to binary feature for now and extract domain if exist
df['Background'] = df['Background'].fillna('none')


"""
Reviews
"""

#this should be handled in nlp also, but if null should be mapped to none to give -ve effect
df['Reviews'] = df['Reviews'].fillna('none')



"""
### Split Dataset
train & test
"""


df, test_df = train_test_split(df, test_size=0.2, random_state=42)

y = df["RecommendationCount"]
# y = np.log1p(y)

df = df.drop(columns=["RecommendationCount"])


"""
Study the text feilds to use NLP
"""


"""
### Binary Encoding, Replace it with 0,1
"""

# Convert all boolean columns (True/False) to (1/0)
bool_cols = [
    "ControllerSupport","IsFree","FreeVerAvail","PurchaseAvail","SubscriptionAvail",
    "PlatformWindows","PlatformLinux","PlatformMac",
    "PCReqsHaveMin","PCReqsHaveRec","LinuxReqsHaveMin","LinuxReqsHaveRec",
    "MacReqsHaveMin","MacReqsHaveRec",
    "CategorySinglePlayer","CategoryMultiplayer","CategoryCoop","CategoryMMO",
    "CategoryInAppPurchase","CategoryIncludeSrcSDK","CategoryIncludeLevelEditor","CategoryVRSupport",
    "GenreIsNonGame","GenreIsIndie","GenreIsAction","GenreIsAdventure","GenreIsCasual",
    "GenreIsStrategy","GenreIsRPG","GenreIsSimulation","GenreIsEarlyAccess",
    "GenreIsFreeToPlay","GenreIsSports","GenreIsRacing","GenreIsMassivelyMultiplayer"
]

# True → 1 , False → 0
df[bool_cols] = df[bool_cols].astype(int)


"""
Price Currency
"""
# Encode USD as 1 (binary feature)
df["PriceCurrency"] = df["PriceCurrency"].apply(
    lambda x: 1 if x == "USD" else 0
)
# the column has only 1 value (will be dropped)
print("\n",df["PriceCurrency"].value_counts(dropna=False))


"""
Images
"""

df['Background'] = df['Background'].apply(lambda x: 0 if x == 'none' else 1)
df['HeaderImage'] = df['HeaderImage'].apply(lambda x: 0 if x == 'none' else 1)

"""
Legal Notices
"""

df['LegalNotice'] = df['LegalNotice'].apply(lambda x: 0 if pd.isna(x) else 1)

"""
DRMNotice
"""
df['DRMNotice'] = df['DRMNotice'].apply(lambda x: 0 if pd.isna(x) else 1)


"""
ExtUserAcctNotice
"""
df['ExtUserAcctNotice'] = df['ExtUserAcctNotice'].apply(lambda x: 0 if pd.isna(x) else 1)


"""
### Frequency Encoding
since all these values are text, so we're trying to extract info as much as we can
"""

"""
QueryName
"""
freq_query = df["QueryName"].value_counts(normalize=True)
df["QueryName_FreqEnc"] = df["QueryName"].map(freq_query)

"""
ResponseName
"""
freq_response = df["ResponseName"].value_counts(normalize=True)
df["ResponseName_FreqEnc"] = df["ResponseName"].map(freq_response)

"""
SupportURL
"""
# extract Domain from URL
df["SupportURL"] = df["SupportURL"].apply(
    lambda x: urlparse(x).netloc if pd.notnull(x) and x != 'none' else 'none'
)

# Apply frequency encoding
freq_support = df["SupportURL"].value_counts(normalize=True)
df["SupportURL"] = df["SupportURL"].map(freq_support).fillna(0)

"""
SupportEmail
"""
freq_email = df["SupportEmail"].value_counts(normalize=True)
df["SupportEmail"] = df["SupportEmail"].map(freq_email).fillna(0)

"""
Website
"""
# extract Domain from URL
df["Website"] = df["Website"].apply(
    lambda x: urlparse(x).netloc if pd.notnull(x) and x != 'none' else 'none'
)

# Apply frequency encoding
freq_website = df["Website"].value_counts(normalize=True)
df["Website"] = df["Website"].map(freq_website).fillna(0)



"""
### split the languages
to top 11 features and add feature count total language that game support
"""

# Languages are space-separated
def extract_languages(text):
  if pd.isna(text) or str(text).strip() == '':
    return []
  text = str(text)
  parts = re.split(r',|\s{2,}', text) # ","" or "space"
  result = []
  for part in parts:
    part = part.strip().lower()
    if part:
        result.append(part)
  return result


# Extract languages
df["SupportedLanguages"] = df["SupportedLanguages"].fillna('').apply(extract_languages)
df["SupportedLanguages"] = df["SupportedLanguages"].apply(
    lambda x: [i for i in x if pd.notnull(i)]
)

# Define important languages
selected_languages = [
    "english", "german", "french", "spanish", "italian", "russian",
    "portuguese", "japanese", "polish", "brazil", "chinese"
]

# make binary features
for lang in selected_languages:
    df["Lang_" + lang] = df["SupportedLanguages"].apply(
        lambda x: 1 if lang in x else 0
    )

df["SupportedLanguagesCount"] = df["SupportedLanguages"].apply(len)


"""
### ReleaseDate
"""
# Convert ReleaseDate column to datetime
# errors="coerce" converts invalid values to NaT (null datetime)
df["ReleaseDate"] = pd.to_datetime(df["ReleaseDate"], errors="coerce")

# Get today's date
today = pd.Timestamp.today()

# Convert ReleaseDate to age in years (difference from today)
df["ReleaseDate"] = (today - df["ReleaseDate"]).dt.days / 365.25


# Compute median age (years since release)
median_age = df["ReleaseDate"].median()

# Fill missing ReleaseDate values using median
df["ReleaseDate"] = df["ReleaseDate"].fillna(median_age)

"""
### Min/Rec Requirements
"""
# an attempt to extract info from reqs before nlp using regex
def extract_ram(text):
    if pd.isna(text): return 0
    # \d+ for one or more digits, \s for spaces, match GB or MB next to space and digit
    # -> store in 'match' variable then unpack in 'val' number and 'unit' gb or mb
    match = re.search(r'(\d+)\s?(GB|MB)', str(text), re.IGNORECASE)
    if match:
        val, unit = match.groups()
        val = int(val)
        return val if unit.upper() == 'GB' else val / 1024
    return 0

def extract_proc(text):
    if pd.isna(text): return 0
    # \d+ for one or more digits, \s for spaces, \. for decimals, match ghz or mhz next to space and digit
    # -> store in 'match' variable then unpack in 'val' number and 'unit' gb or mb
    match = re.search(r'(\d+(?:\.\d+)?)\s?(MHZ|GHZ)', str(text), re.IGNORECASE)
    if match:
        val, unit = match.groups()
        val = float(val)
        return val if unit.upper() == 'GHZ' else val / 1000
    return 0

# RAM extraction
df['PC_MinRam'] = df['PCMinReqsText'].where(df['PCReqsHaveMin'] == 1).apply(extract_ram)
df['PC_RecRam'] = df['PCRecReqsText'].where(df['PCReqsHaveRec'] == 1).apply(extract_ram)

df['Linux_MinRam'] = df['LinuxMinReqsText'].where(df['LinuxReqsHaveMin'] == 1).apply(extract_ram)
df['Linux_RecRam'] = df['LinuxRecReqsText'].where(df['LinuxReqsHaveRec'] == 1).apply(extract_ram)

df['Mac_MinRam'] = df['MacMinReqsText'].where(df['MacReqsHaveMin'] == 1).apply(extract_ram)
df['Mac_RecRam'] = df['MacRecReqsText'].where(df['MacReqsHaveRec'] == 1).apply(extract_ram)

# CPU extraction
df['PC_MinCPU'] = df['PCMinReqsText'].where(df['PCReqsHaveMin'] == 1).apply(extract_proc)
df['PC_RecCPU'] = df['PCRecReqsText'].where(df['PCReqsHaveRec'] == 1).apply(extract_proc)

df['Linux_MinCPU'] = df['LinuxMinReqsText'].where(df['LinuxReqsHaveMin'] == 1).apply(extract_proc)
df['Linux_RecCPU'] = df['LinuxRecReqsText'].where(df['LinuxReqsHaveRec'] == 1).apply(extract_proc)

df['Mac_MinCPU'] = df['MacMinReqsText'].where(df['MacReqsHaveMin'] == 1).apply(extract_proc)
df['Mac_RecCPU'] = df['MacRecReqsText'].where(df['MacReqsHaveRec'] == 1).apply(extract_proc)


# robust = RobustScaler()
# min_max = MinMaxScaler()

# # robust scaler is great to handle outliers (subtract median instead of mean then divide by iqr)
# # Minmax might fail because if a game needs very high specs it will squash the other games specs so much (think cyberpunk vs icy tower)
# req_cols = [
#     'PC_MinRam','PC_RecRam',
#     'Linux_MinRam','Linux_RecRam',
#     'Mac_MinRam','Mac_RecRam',

#     'PC_MinCPU','PC_RecCPU',
#     'Linux_MinCPU','Linux_RecCPU',
#     'Mac_MinCPU','Mac_RecCPU'
# ]

# df[[c + '_Scaled' for c in req_cols]] = robust.fit_transform(df[req_cols])

# # OR use MinMaxScaler if we want a strict 0-1 range
# # df[['MinRam_Scaled', 'MinProc_Scaled']] = min_max.fit_transform(df[['MinRam', 'MinProc']])


"""
Feature Interaction
"""

# Feature Interactions

# Market strength
df["owners_metacritic"] = df["SteamSpyOwners"] * df["Metacritic"].fillna(df["Metacritic"].median())
df["players_metacritic"] = df["SteamSpyPlayersEstimate"] * df["Metacritic"].fillna(df["Metacritic"].median())

df["owners_players"] = df["SteamSpyOwners"] * df["SteamSpyPlayersEstimate"]

# Pricing
df["price_discount"] = df["PriceInitial"] - df["PriceFinal"]

df["price_owners"] = df["PriceFinal"] * df["SteamSpyOwners"]
df["price_players"] = df["PriceFinal"] * df["SteamSpyPlayersEstimate"]

df["free_x_owners"] = df["IsFree"] * df["SteamSpyOwners"]
df["free_x_players"] = df["IsFree"] * df["SteamSpyPlayersEstimate"]

# Age effects
# df["age_owners"] = df["SteamSpyOwners"] * df["game_age_years"]
# df["age_players"] = df["SteamSpyPlayersEstimate"] * df["game_age_years"]
# df["age_metacritic"] = df["Metacritic"] * df["game_age_years"]

# Content richness
df["content_volume"] = (
    df["ScreenshotCount"] +
    df["MovieCount"] +
    df["DLCCount"] +
    df["PackageCount"]
)

df["content_owners"] = df["content_volume"] * df["SteamSpyOwners"]
df["content_players"] = df["content_volume"] * df["SteamSpyPlayersEstimate"]
df["content_metacritic"] = df["content_volume"] * df["Metacritic"].fillna(df["Metacritic"].median())

# Achievements / complexity
# df["achievement_density"] = df["AchievementCount"] / (df["game_age_years"] + 1)

df["achievement_owners"] = df["AchievementCount"] * df["SteamSpyOwners"]
df["achievement_players"] = df["AchievementCount"] * df["SteamSpyPlayersEstimate"]

df["highlighted_achievements_ratio"] = (
    df["AchievementHighlightedCount"] /
    (df["AchievementCount"] + 1)
)

# Platform reach
df["platform_count"] = (
    df["PlatformWindows"].astype(int) +
    df["PlatformLinux"].astype(int) +
    df["PlatformMac"].astype(int)
)

df["platform_owners"] = df["platform_count"] * df["SteamSpyOwners"]
df["platform_players"] = df["platform_count"] * df["SteamSpyPlayersEstimate"]

# Genre interactions
df["action_multiplayer"] = df["GenreIsAction"] * df["CategoryMultiplayer"]
df["rpg_achievement"] = df["GenreIsRPG"] * df["AchievementCount"]
df["strategy_complexity"] = df["GenreIsStrategy"] * df["AchievementCount"]
df["indie_price"] = df["GenreIsIndie"] * df["PriceFinal"]

# Category richness
df["category_count"] = (
    df["CategorySinglePlayer"] +
    df["CategoryMultiplayer"] +
    df["CategoryCoop"] +
    df["CategoryMMO"] +
    df["CategoryVRSupport"]
)

df["category_owners"] = df["category_count"] * df["SteamSpyOwners"]
df["category_players"] = df["category_count"] * df["SteamSpyPlayersEstimate"]

# Text signals
df["review_words"] = df["Reviews"].fillna("").str.split().str.len()

df["reviews_owners"] = df["review_words"] * df["SteamSpyOwners"]
df["reviews_players"] = df["review_words"] * df["SteamSpyPlayersEstimate"]
df["reviews_metacritic"] = df["review_words"] * df["Metacritic"].fillna(df["Metacritic"].median())



"""
### Description preprocessing by NLP
"""
df['AllText'] = (df['DetailedDescrip'].fillna('') + ' ' + df['ShortDescrip'].fillna('') )

custom_stopwords = set([
    "game", "games", "play", "players", "player",
    "world", "time", "experience", "like", "make",
    "based", "new", "use", "way", "different"
])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    tokens = text.split()
    tokens = [tok for tok in tokens if tok not in custom_stopwords]

    return " ".join(tokens)

df['AllText_Cleaned'] = df['AllText'].apply(clean_text)
df['AllText_len'] = df['AllText_Cleaned'].str.split().str.len()

keywords = [
    'multiplayer', 'online', 'co op',
    'single player', 'zombie',
    'war', 'action', 'team', 'free',
    'strategy', 'shooter', 'rpg',
    'indie', 'puzzle', 'horror'
]

# 1. Store the new series in a dictionary
new_cols_dict = {}

for word in keywords:
    col_name = f'has_{word.replace(" ", "_")}'
    new_cols_dict[col_name] = df['AllText_Cleaned'].str.contains(word).astype(int)

# 2. Create a temporary DataFrame from all the new columns at once
new_cols_df = pd.DataFrame(new_cols_dict)

# 3. Concatenate the new columns to the original df
df = pd.concat([df, new_cols_df], axis=1)

# List for future use
keyword_cols = list(new_cols_dict.keys())


nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

df['ReviewSentiment'] = df['Reviews'].fillna('').apply(
    lambda x: sia.polarity_scores(str(x))['compound']
)

df['AboutSentiment'] = df['AboutText'].fillna('').apply(
    lambda x: sia.polarity_scores(str(x))['compound']
)

tfidf_word = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1,2)
)

tfidf_char = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3,5),
    max_features=8000
)

vectorizer = FeatureUnion([
    ("word", tfidf_word),
    ("char", tfidf_char)
])

preprocessor = ColumnTransformer([
    ('text', vectorizer, 'AllText_Cleaned'),
    ('length', StandardScaler(), ['AllText_len']),
    ('keywords', 'passthrough', keyword_cols),
    ('sentiment', 'passthrough', ['ReviewSentiment', 'AboutSentiment'])
])


feature_cols = (
    ['AllText_Cleaned', 'AllText_len'] +
    keyword_cols +
    ['ReviewSentiment', 'AboutSentiment']
)

X_processed = preprocessor.fit_transform(df[feature_cols])

print(X_processed.shape)
print(X_processed[:5])

print(df.isna().sum()[df.isna().sum() > 0])



pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print(df.head())

"""
check nulls final time isA
"""

print(df.isna().sum()[df.isna().sum() > 0])
# language and website are hndled elsewhere
# Notices had binary feats extracted directly
# so as reqs
# maybe now these feats can be dropped

"""
### Drop Features
We already extract info from them
"""

# drop text features after extracting information
cols_to_drop = [
    "QueryName",
    "ResponseName",
    "AboutText",
    "ShortDescrip",
    "DetailedDescrip",
    "PCMinReqsText",
    "PCRecReqsText",
    "LinuxMinReqsText",
    "LinuxRecReqsText",
    "MacMinReqsText",
    "MacRecReqsText",
    "AllText",
    "AllText_Cleaned",
    "Reviews",
    "SupportedLanguages",
    'QueryID', 'ResponseID'
]

df = df.drop(columns=cols_to_drop, errors="ignore")


"""
### Handle Skew
"""


# numeric columns
num_cols = df.select_dtypes(include=["number"]).columns

# columns to exclude from log
exclude_log = set()

# binary + scaled
exclude_log.update([c for c in num_cols if df[c].nunique() <= 2])
exclude_log.update([c for c in num_cols if "_Scaled" in c])

# NLP
exclude_log.update([
    "ReviewSentiment",
    "AboutSentiment",
    "AllText_len",
])

# target
exclude_log.update([
    "RecommendationCount",
])

# freq + web encodings
exclude_log.update([
    "QueryName_FreqEnc",
    "ResponseName_FreqEnc",
    "SupportURL",
    "SupportEmail",
    "Website"
])

# language + keyword features
exclude_log.update([c for c in df.columns if c.startswith("Lang_")])
exclude_log.update([c for c in df.columns if c.startswith("has_")])

# log candidates
log_cols = [c for c in num_cols if c not in exclude_log]

print(f"Log cols: {len(log_cols)}")

# skew before
print(df[log_cols].skew().sort_values(ascending=False).head(10))

# apply log1p
df[log_cols] = np.log1p(df[log_cols])

# skew after
print(df[log_cols].skew().sort_values(ascending=False).head(10))

# apply log1p to Target
y = np.log1p(y)  # optional but recommended

"""
### Normalization
"""

# scale_cols = [
# 'SteamSpyOwners',
# 'SteamSpyOwnersVariance',
# 'SteamSpyPlayersEstimate',
# 'SteamSpyPlayersVariance',
# 'PriceInitial',
# 'PriceFinal',
# 'AllText_len'
# ]

# scaler = StandardScaler()
# df[scale_cols] = scaler.fit_transform(df[scale_cols])


"""
### Remove Outlier
"""

def clip_outliers_iqr(df, cols, k=1.5):
    df = df.copy()

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - k * IQR
        upper = Q3 + k * IQR

        df[col] = df[col].clip(lower, upper)

    return df
df = clip_outliers_iqr(df, df.columns)



"""
### Feature Extraction
"""

"""
Apply Correlation
"""

corr_target = df.corrwith(y).sort_values()

print("Correlation with target:")
print(corr_target)



"""
Visulaize Correlation
"""

# compute full correlation matrix
corr_matrix = df.corr()

plt.figure(figsize=(18, 12))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.1
)

plt.title("Feature Correlation Heatmap")
plt.show()


threshold = 0.00001 # weak correlation cutoff

# correlation series
corr_target = df.corrwith(y)

# dropped columns
cols_to_drop = corr_target[
    corr_target.isna() | (abs(corr_target) < threshold)
].index

# kept columns
cols_kept = corr_target[
    ~(corr_target.isna() | (abs(corr_target) < threshold))
]

# drop
df = df.drop(columns=cols_to_drop)

print("DROPPED COLUMNS:")
print(list(cols_to_drop))

print("\nKEPT COLUMNS + CORRELATION:")
print(cols_kept.sort_values(ascending=False))



# Modeling
"""
Linear Regression
"""
def train_linear(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return model, {
        "train": evaluate(y_train, y_train_pred),
        "test": evaluate(y_test, y_test_pred)
    }


"""
Polynomial Regression
"""
def train_polynomial(X_train, X_test, y_train, y_test, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = Ridge(alpha=1.0)  # بنستخدم Ridge مع Poly عشان نتجنب Overfitting
    model.fit(X_train_poly, y_train)

    return model, poly, {
        "train": evaluate(y_train, model.predict(X_train_poly)),
        "test": evaluate(y_test, model.predict(X_test_poly))
    }


"""
Ridge
"""
def train_ridge(X_train, X_test, y_train, y_test, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    return model, {
        "train": evaluate(y_train, model.predict(X_train)),
        "test": evaluate(y_test, model.predict(X_test))
    }


"""
Random Forest
"""
def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)

    model.fit(X_train, y_train)

    return model, {
        "train": evaluate(y_train, model.predict(X_train)),
        "test": evaluate(y_test, model.predict(X_test))
    }


"""
gradient_boosting
"""
def train_gradient_boosting(X_train, X_test, y_train, y_test):
    model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    min_samples_leaf=3,
    random_state=42
)

    model.fit(X_train, y_train)

    return model, {
        "train": evaluate(y_train, model.predict(X_train)),
        "test": evaluate(y_test, model.predict(X_test))
    }


"""
xgboost
"""
def train_xgboost(X_train, X_test, y_train, y_test):
    model = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    gamma=0.2,
    reg_alpha=0.3,
    reg_lambda=2.0,
    random_state=42
)

    model.fit(X_train, y_train)

    return model, {
        "train": evaluate(y_train, model.predict(X_train)),
        "test": evaluate(y_test, model.predict(X_test))
    }

"""
Evaluation
"""
def evaluate(y_true, y_pred):
    # reverse transform
    # y_true = np.expm1(y_true)
    # y_pred = np.expm1(y_pred)

    return {
        "MSE":  mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "R2":   r2_score(y_true, y_pred)
    }

"""
All Models
"""
def run_models(X_train, X_test, y_train, y_test, save_dir="saved_models"):

    os.makedirs(save_dir, exist_ok=True)

    results = {}

    # Linear
    lr_model, lr_res = train_linear(X_train, X_test, y_train, y_test)
    results["Linear Regression Train"] = lr_res["train"]
    results["Linear Regression Test"] = lr_res["test"]
    joblib.dump(lr_model, f"{save_dir}/linear.pkl")

    # Polynomial Regression
    poly_model, poly_transformer, poly_res = train_polynomial(X_train, X_test, y_train, y_test)
    results["Polynomial Regression Train"] = poly_res["train"]
    results["Polynomial Regression Test"] = poly_res["test"]
    joblib.dump(poly_model, f"{save_dir}/polynomial.pkl")
    joblib.dump(poly_transformer, f"{save_dir}/polynomial_transformer.pkl")
    
    # Ridge
    ridge_model, ridge_res = train_ridge(X_train, X_test, y_train, y_test)
    results["Ridge Train"] = ridge_res["train"]
    results["Ridge Test"] = ridge_res["test"]
    joblib.dump(ridge_model, f"{save_dir}/ridge.pkl")

    # Random Forest
    rf_model, rf_res = train_random_forest(X_train, X_test, y_train, y_test)
    results["Random Forest Train"] = rf_res["train"]
    results["Random Forest Test"] = rf_res["test"]
    joblib.dump(rf_model, f"{save_dir}/random_forest.pkl")

    # Gradient Boosting
    gb_model, gb_res = train_gradient_boosting(X_train, X_test, y_train, y_test)
    results["Gradient Boosting Train"] = gb_res["train"]
    results["Gradient Boosting Test"] = gb_res["test"]
    joblib.dump(gb_model, f"{save_dir}/gradient_boosting.pkl")

    # XGBoost
    xgb_model, xgb_res = train_xgboost(X_train, X_test, y_train, y_test)
    results["XGBoost Train"] = xgb_res["train"]
    results["XGBoost Test"] = xgb_res["test"]
    joblib.dump(xgb_model, f"{save_dir}/xgboost.pkl")

    # Save test metrics for each model so App.py can display them in the cards
    model_metrics = {
        "Linear Regression (Project)": lr_res["test"],
        "Ridge":                        ridge_res["test"],
        "Random Forest":                rf_res["test"],
        "Gradient Boosting":            gb_res["test"],
        "XGBoost":                      xgb_res["test"],
    }
    joblib.dump(model_metrics, f"{save_dir}/model_metrics.pkl")
    print("Saved model_metrics.pkl")

    # Final table
    results_df = pd.DataFrame(results).T

    print("\nResults (Train vs Test):")
    print(results_df)

    # Best model based on TEST R2 only
    test_rows = results_df.loc[results_df.index.str.contains("Test")]
    best_model = test_rows["R2"].idxmax()

    print(f"\nBest Model (Test R2): {best_model}")

    return results_df


"""
Save feature metadata for App.py inference
"""
# ── IMPORTANT: only keep columns that App.py can fully reconstruct from UI inputs.
# NLP features (AllText_len, has_*, sentiment), frequency-encoded text columns
# (SupportEmail, SupportURL, Website, QueryName_FreqEnc, ResponseName_FreqEnc),
# hardware-req columns (PC_MinRam, etc.), and variance columns cannot be
# reconstructed at inference time, so they are excluded here.
# Training and inference must use exactly the same feature set.

INFERRABLE_FEATURES = [
    # ── core numeric ──────────────────────────────────────────────────────────
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
    # ── binary flags ──────────────────────────────────────────────────────────
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
    # ── price ─────────────────────────────────────────────────────────────────
    "PriceInitial",
    "PriceFinal",
    # ── engineered interactions (all computable from the inputs above) ────────
    "price_discount",
    "platform_count",
    "category_count",
    "content_volume",
    "highlighted_achievements_ratio",
    "action_multiplayer",
    "rpg_achievement",
    "strategy_complexity",
    "indie_price",
    "owners_metacritic",
    "players_metacritic",
    "owners_players",
    "price_owners",
    "price_players",
    "free_x_owners",
    "free_x_players",
    "content_owners",
    "content_players",
    "content_metacritic",
    "achievement_owners",
    "achievement_players",
    "platform_owners",
    "platform_players",
    "category_owners",
    "category_players",
]

# Keep only the columns that actually exist in df after all preprocessing
inferrable_cols = [c for c in INFERRABLE_FEATURES if c in df.columns]
df_model = df[inferrable_cols].copy()

print(f"\nTraining on {len(inferrable_cols)} inferrable features (dropped NLP / text columns).")
print(inferrable_cols)

# Save the exact column order the models were trained on
joblib.dump(inferrable_cols, "Models/feature_columns.pkl")

# Save per-column medians so App.py can impute any missing values
joblib.dump(df_model.median(numeric_only=True).to_dict(), "Models/feature_medians.pkl")

print(f"Saved {len(inferrable_cols)} feature columns and medians to Models/")

"""
Split & Run
"""
X_train, X_test, y_train, y_test = train_test_split(
    df_model, y,
    test_size=0.2,
    random_state=42
)

run_models(X_train, X_test, y_train, y_test, save_dir="Models")
