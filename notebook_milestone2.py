# Core
import numpy as np
import pandas as pd
import re
from datetime import datetime
from urllib.parse import urlparse
import os
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# model + split
from sklearn.model_selection import train_test_split

# preprocessing
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    PolynomialFeatures,
    FunctionTransformer,
    LabelEncoder,
    OrdinalEncoder
)
from sklearn.base import BaseEstimator, TransformerMixin

# models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
# classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
# NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from textblob import TextBlob
from sklearn.pipeline import FeatureUnion
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# ====================================================================
# zyad
mode = input("enter target mode: classification/regression [c/r]")
if mode == "c" or mode == "C":
  df = pd.read_csv("Data/train_data_class.csv")
else:
  df = pd.read_csv("Data/train_data.csv")

"""# Exploring Data"""

print(f"Dataset shape: {df.shape}")

print(list(df.columns))

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df.head()

df.info()

df.describe()

# Preprocessing

"""
### Handel duplicates

Check Duplicates
"""

df.duplicated().sum()

df = df.drop_duplicates()

df.shape

"""### Check Nulls

Check Nulls
"""

df.isna().sum()[df.isna().sum() > 0]

"""Check Empty Cells"""

df = df.replace(r'^\s*$', np.nan, regex=True)

"""Check Nulls"""

df.isna().sum()[df.isna().sum() > 0]

"""### Replace Nulls

QueryName
"""

# Fill missing QueryName values using ResponseName
# If QueryName is null, copy the value from ResponseName
df["QueryName"] = df["QueryName"].fillna(df["ResponseName"])

"""PriceCurrency"""

# Check unique values
print(df["PriceCurrency"].value_counts(dropna=False))

# Fill missing values with USD
df["PriceCurrency"] = df["PriceCurrency"].fillna("USD")

"""SupportURL"""

#fill Nulls
df['SupportURL'] = df['SupportURL'].fillna("")

"""SupportEmail"""

#fill Nulls
df['SupportEmail'] = df['SupportEmail'].fillna("")

"""Website"""

#fill Nulls
df['Website'] = df['Website'].fillna("")

"""Description columns"""

# fill wirth first found description and feed it to others, if none found make place holder
# fill nulls with other descriptions not lose columsn, still to be actually preprocessed
bestDescrip = df['DetailedDescrip'].fillna(df['AboutText']).fillna(df['ShortDescrip'])

df['AboutText'] = df['AboutText'].fillna(bestDescrip).fillna('none')
df['DetailedDescrip'] = df['DetailedDescrip'].fillna(bestDescrip).fillna('none')
df['ShortDescrip'] = df['ShortDescrip'].fillna(bestDescrip).fillna('none')
# nlp to be applied on them, if none is found give -ve effect?

"""Images"""

# i think either replace with placeholder or just turn to binary feature for now and extract domain if exist
df['Background'] = df['Background'].fillna('none')

"""Reviews"""

#this should be handled in nlp also, but if null should be mapped to none to give -ve effect
df['Reviews'] = df['Reviews'].fillna('none')

"""### Split Dataset
train & test
"""

df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# IMPORTANT: encode new target (classification)
if mode == "c" or mode == "C":
  encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
  df['GamePopularity'] = encoder.fit_transform(df[['GamePopularity']])
if mode == "c" or mode == "C":
  y = df["GamePopularity"]
  df = df.drop(columns=["GamePopularity"])
else:
  y = df["RecommendationCount"]
  # y = np.log1p(y)

  df = df.drop(columns=["RecommendationCount"])


# Encoding
"""
### Binary Encoding
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

# True -> 1 , False -> 0
df[bool_cols] = df[bool_cols].astype(int)

"""Price Currency"""

# Encode USD as 1 (binary feature)
df["PriceCurrency"] = df["PriceCurrency"].apply(
    lambda x: 1 if x == "USD" else 0
)
# the column has only 1 value (will be dropped)
print("\n",df["PriceCurrency"].value_counts(dropna=False))

"""Images"""

# again if exist -> better, if not -ve effect

# a lot of null replacing can be used directly with np.nan instead of none, but maybe this is cleaner
df['Background'] = df['Background'].apply(lambda x: 0 if x == 'none' else 1)

# there is no nulls, all  values are 1 (will be dropped)
df['HeaderImage'] = df['HeaderImage'].apply(lambda x: 0 if x == 'none' else 1)

"""Legal Notices"""

# one way to work around notices, another is to check for certain keywords (more on that later, needs search)
df['LegalNotice'] = df['LegalNotice'].apply(lambda x: 0 if pd.isna(x) else 1)

"""DRMNotice"""

df['DRMNotice'] = df['DRMNotice'].apply(lambda x: 0 if pd.isna(x) else 1)

"""ExtUserAcctNotice"""

df['ExtUserAcctNotice'] = df['ExtUserAcctNotice'].apply(lambda x: 0 if pd.isna(x) else 1)

"""
### Frequency Encoding
since all these values are text, so we're trying to extract info as much as we can

QueryName
"""

# Apply frequency Encoding
freq_query = df["QueryName"].value_counts(normalize=True)
df["QueryName_FreqEnc"] = df["QueryName"].map(freq_query)

"""ResponseName"""

# Apply frequency Encoding
freq_response = df["ResponseName"].value_counts(normalize=True)
df["ResponseName_FreqEnc"] = df["ResponseName"].map(freq_response)

"""SupportURL"""

# extract Domain from URL
df["SupportURL"] = df["SupportURL"].apply(
    lambda x: urlparse(x).netloc if pd.notnull(x) and x != 'none' else 'none'
)

# Apply frequency Encoding
freq_support = df["SupportURL"].value_counts(normalize=True)
df["SupportURL"] = df["SupportURL"].map(freq_support).fillna(0)

"""SupportEmail"""

# Apply frequency Encoding
freq_email = df["SupportEmail"].value_counts(normalize=True)
df["SupportEmail"] = df["SupportEmail"].map(freq_email).fillna(0)

"""Website"""

# extract Domain from URL
df["Website"] = df["Website"].apply(
    lambda x: urlparse(x).netloc if pd.notnull(x) and x != 'none' else 'none'
)

# Apply frequency Encoding
freq_website = df["Website"].value_counts(normalize=True)
df["Website"] = df["Website"].map(freq_website).fillna(0)


# ====================================================================================
# mahmoud

"""
### split the languages
to top 11 features and add feature count total language that game support
"""

# Languages are space-separated
def extract_languages(text):
  if pd.isna(text) or str(text).strip() == '':
    return []
  text = str(text)
  parts = re.split(r',|\s{2,}', text) #, or space
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

"""### ReleaseDate"""

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

"""### Min/Rec Requirements"""

# an attempt to extract info from reqs before nlp using regex
# info is to be the number of memory required, if too much maybe -ve effect
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

"""Feature Interaction"""

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


# =================================================================================================
# sadek
"""### Description preprocessing by NLP"""

df['AllText'] = (
    df['DetailedDescrip'].fillna('') + ' ' +
    df['ShortDescrip'].fillna('')
)

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
    tokens = [t for t in tokens if t not in custom_stopwords]

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

# 1. Store the new series in a dictionary instead of the DataFrame
new_cols_dict = {}

for word in keywords:
    col_name = f'has_{word.replace(" ", "_")}'
    # We calculate the series and store it in the dict
    new_cols_dict[col_name] = df['AllText_Cleaned'].str.contains(word).astype(int)

# 2. Create a temporary DataFrame from all the new columns at once
new_cols_df = pd.DataFrame(new_cols_dict)

# 3. Concatenate the new columns to the original df in one single operation
df = pd.concat([df, new_cols_df], axis=1)

# List for future use
keyword_cols = list(new_cols_dict.keys())

# Sentiment Analysis
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

df.head()
# QueryName	ResponseName AboutText ShortDescrip ShortDescrip PCMinReqsText PCRecReqsText	LinuxMinReqsText	LinuxRecReqsText	MacMinReqsText	MacRecReqsText	AllText	AllText_Cleaned

"""check nulls final time isA"""

df.isna().sum()[df.isna().sum() > 0]
# language and website are hndled elsewhere
# Notices had binary feats extracted directly
# so as reqs
# maybe now these feats can be dropped

"""### Drop Features
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
df_full = df.copy() 
df = df.drop(columns=cols_to_drop, errors="ignore")


# =========================================================================
# Yusuf

"""### Handle Skew"""

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
if mode == "r" or mode == "R":
  y = np.log1p(y)  # optional but recommended


# """### Normalization"""
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


"""### Remove Outlier"""

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
#Remove outlier
df = clip_outliers_iqr(df, df.columns)

"""
### Feature Extraction

Apply Correlation
"""

corr_target = df.corrwith(y).sort_values()

print("Correlation with target:")
print(corr_target)

"""Visulaize Correlation"""

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




# 1. Create a sub-pipeline for numeric columns (Handles Nulls + Skew + Scaling)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # saves null handeling
    ('log', FunctionTransformer(np.log1p)),        # saves skeweess fix
    ('scaler', StandardScaler())                   # saves scaling
])

# 2. Update ColumnTransformer
preprocessor = ColumnTransformer([
    ('text', vectorizer, 'AllText_Cleaned'),
    ('num', numeric_transformer, log_cols),        # Apply the sub-pipeline to numeric columns
    ('keywords', 'passthrough', keyword_cols),
    ('sentiment', 'passthrough', ['ReviewSentiment', 'AboutSentiment'])
])


joblib.dump(preprocessor, 'preprocessor.pkl')

# ============================================================================================
# abdelhakim regression part
# ahmed classification part

"""# Modeling

Linear Regression
"""

if mode.lower() == "r":
    def train_linear(X_train, X_test, y_train, y_test):
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        return model, {
            "train": evaluate(y_train, y_train_pred),
            "test": evaluate(y_test, y_test_pred)
        }
    
    """Polynomial Regression"""
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
    
    """Ridge"""
    def train_ridge(X_train, X_test, y_train, y_test, alpha=1.0):
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        return model, {
            "train": evaluate(y_train, model.predict(X_train)),
            "test": evaluate(y_test, model.predict(X_test))
        }
    
    """Random Forest"""
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

    """gradient_boosting"""
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
    
    """xgboost"""
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
        random_state=42)
        model.fit(X_train, y_train)

        return model, {
            "train": evaluate(y_train, model.predict(X_train)),
            "test": evaluate(y_test, model.predict(X_test))
        }

"""Classification models"""

if mode.lower() == "c":
    # logistic regression
    def train_logistic(X_train, X_test, y_train, y_test):
        C_values = [0.01, 0.1, 1.0, 10.0]







        
        results = []

        for c in C_values:
            model = LogisticRegression(C=c,max_iter=1000, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            results.append({
                "C": c,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "model": model
                })
        return results
    
    # random forest
    def train_rf_classifier(X_train, X_test, y_train, y_test):
        n_est_values = [100, 200, 300]
        results = []

        for n in n_est_values:
            model = RandomForestClassifier(
                n_estimators=n,
                max_depth=None,
                random_state=42,
                min_samples_split=5, min_samples_leaf=2,
                max_features='sqrt', class_weight='balanced'
                )
            model.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))

            results.append({
                "n_estimators": n,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "model": model
                })

        return results
    
    # Gradient Boosting
    def train_gb_classifier(X_train, X_test, y_train, y_test):
        model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            min_samples_leaf=3,
            random_state=42
        )
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        return {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "model": model
        }
    # XGBoost
    def train_xgb_classifier(X_train, X_test, y_train, y_test):
        learning_rates = [0.01, 0.05, 0.1]
        results = []


        le = LabelEncoder()
        y_tr = le.fit_transform(y_train)
        y_te = le.transform(y_test)

        for lr in learning_rates:
            model = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=lr,
                max_depth=4,
                subsample=0.8,
                random_state=42,
                colsample_bytree=0.8, use_label_encoder=False,
                eval_metric='mlogloss'
                )
            model.fit(X_train, y_tr)

            train_pred = le.inverse_transform(model.predict(X_train))
            test_pred = le.inverse_transform(model.predict(X_test))

            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)

            results.append({
                "learning_rate": lr,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "model": model
                })

        return results

"""Evaluation"""

# Commented out IPython magic to ensure Python compatibility.
if mode.lower() == "r":
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

"""Regression Models"""

if mode.lower() == "r":
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

        # Final table
        results_df = pd.DataFrame(results).T

        print("\nResults (Train vs Test):")
        print(results_df)

        # Best model based on TEST R2 only
        test_rows = results_df.loc[results_df.index.str.contains("Test")]
        best_model = test_rows["R2"].idxmax()

        print(f"\nBest Model (Test R2): {best_model}")

        return results_df
    

"""run classification models"""

if mode.lower() == "c":
    def run_classification(X_train, X_test, y_train, y_test, save_dir='saved_models/'):
        os.makedirs(save_dir, exist_ok=True)
        results = {}

        lg_res = train_logistic(X_train, X_test, y_train, y_test)
        best_lg = max(lg_res, key=lambda x: x['test_acc'])
        results['Logistic Regression Train'] = best_lg['train_acc']
        results['Logistic Regression Test']  = best_lg['test_acc']
        lg = best_lg['model']
        joblib.dump(lg, f'{save_dir}/logistic.pkl')

        rf_res = train_rf_classifier(X_train, X_test, y_train, y_test)
        best_rf = max(rf_res, key=lambda x: x['test_acc'])
        results['Random Forest Train'] = best_rf['train_acc']
        results['Random Forest Test']  = best_rf['test_acc']
        rf = best_rf['model']
        joblib.dump(rf, f'{save_dir}/random_forest.pkl')

        gb_res = train_gb_classifier(X_train, X_test, y_train, y_test)
        results['Gradient Boosting Train'] = gb_res['train_acc']
        results['Gradient Boosting Test']  = gb_res['test_acc']
        gb = gb_res['model']
        joblib.dump(gb, f'{save_dir}/gradient_boosting.pkl')

        xgb_res = train_xgb_classifier(X_train, X_test, y_train, y_test)
        best_xgb = max(xgb_res, key=lambda x: x['test_acc'])
        results['XGBoost Train'] = best_xgb['train_acc']
        results['XGBoost Test']  = best_xgb['test_acc']
        xm = best_xgb['model']
        joblib.dump(xm, f'{save_dir}/xgboost.pkl')

        res_df = pd.DataFrame(results, index=[0]).T
        res_df.columns = ['Score']
        print('\n[CLASSIFICATION] Results (Train vs Test):')
        print(res_df)

        # Confusion matrix for best model (Random Forest by default)
        y_pred_best = rf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_best, labels=rf.classes_)
        disp = ConfusionMatrixDisplay(cm, display_labels=rf.classes_)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title('Random Forest — Confusion Matrix (Test)')
        plt.tight_layout()
        plt.show()

        test_rows = res_df.loc[res_df.index.str.contains('Test')]
        best = test_rows['Score'].idxmax()
        print(f'\n Best Model (Test Accuracy): {best}')
        return res_df


    print('run_regression and run_classification defined')

if mode.lower() == "c":
    def evaluate(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            # "report": classification_report(y_true, y_pred)
        }

X_train, X_test, y_train, y_test = train_test_split(
    df, y,
    test_size=0.2,
    random_state=42
)
if mode.lower() == "c":
    run_classification(X_train, X_test, y_train, y_test)
    rf_clf = joblib.load('saved_models/random_forest.pkl')
    y_pred = rf_clf.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
else:
    run_models(X_train, X_test, y_train, y_test)

# ==================================================================================
# test script
class RAMExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        def extract_ram(text):
            if pd.isna(text): return 0
            match = re.search(r'(\d+)\s?(GB|MB)', str(text), re.IGNORECASE)
            if match:
                val, unit = match.groups()
                val = int(val)
                return val if unit.upper() == 'GB' else val / 1024
            return 0
        return X[self.column_name].apply(extract_ram).values.reshape(-1, 1)

class CPUExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        def extract_proc(text):
            if pd.isna(text): return 0
            match = re.search(r'(\d+(?:\.\d+)?)\s?(MHZ|GHZ)', str(text), re.IGNORECASE)
            if match:
                val, unit = match.groups()
                val = float(val)
                return val if unit.upper() == 'GHZ' else val / 1000
            return 0
        return X[self.column_name].apply(extract_proc).values.reshape(-1, 1)

from urllib.parse import urlparse

class URLExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.column_name].apply(
            lambda x: urlparse(str(x)).netloc if pd.notnull(x) and str(x).strip() != '' else 'none'
        ).values.reshape(-1, 1)

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.counts_ = {}

    def fit(self, X, y=None):
        # Expects X as a 1D-like structure or single column DataFrame
        X_flat = pd.Series(X.ravel()) if hasattr(X, 'ravel') else pd.Series(X)
        self.counts_ = X_flat.value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X_flat = pd.Series(X.ravel()) if hasattr(X, 'ravel') else pd.Series(X)
        return X_flat.map(self.counts_).fillna(0).values.reshape(-1, 1)

class SteamFeatureInteractions(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Assuming X is a DataFrame with specific columns
        X = X.copy()
        # Logic for automated interactions
        X['price_discount'] = X['PriceInitial'] - X['PriceFinal']
        X['content_volume'] = X['ScreenshotCount'] + X['MovieCount'] + X['DLCCount']
        # Drop the source columns if they are no longer needed
        return X


# Define internal pipelines
url_pipeline = Pipeline([
    ('extract', URLExtractor('Website')),
    ('freq', FrequencyEncoder())
])


def preprocess_raw(df):
    # Duplicates & nulls
    df = df.drop_duplicates()
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df["QueryName"] = df["QueryName"].fillna(df["ResponseName"])
    df["PriceCurrency"] = df["PriceCurrency"].fillna("USD")
    df['SupportURL'] = df['SupportURL'].fillna("")
    df['SupportEmail'] = df['SupportEmail'].fillna("")
    df['Website'] = df['Website'].fillna("")
    bestDescrip = df['DetailedDescrip'].fillna(df['AboutText']).fillna(df['ShortDescrip'])
    df['AboutText'] = df['AboutText'].fillna(bestDescrip).fillna('none')
    df['DetailedDescrip'] = df['DetailedDescrip'].fillna(bestDescrip).fillna('none')
    df['ShortDescrip'] = df['ShortDescrip'].fillna(bestDescrip).fillna('none')
    df['Background'] = df['Background'].fillna('none')
    df['Reviews'] = df['Reviews'].fillna('none')

    # Binary encoding
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
    df[bool_cols] = df[bool_cols].astype(int)
    df["PriceCurrency"] = df["PriceCurrency"].apply(lambda x: 1 if x == "USD" else 0)
    df['Background'] = df['Background'].apply(lambda x: 0 if x == 'none' else 1)
    df['HeaderImage'] = df['HeaderImage'].apply(lambda x: 0 if x == 'none' else 1)
    df['LegalNotice'] = df['LegalNotice'].apply(lambda x: 0 if pd.isna(x) else 1)
    df['DRMNotice'] = df['DRMNotice'].apply(lambda x: 0 if pd.isna(x) else 1)
    df['ExtUserAcctNotice'] = df['ExtUserAcctNotice'].apply(lambda x: 0 if pd.isna(x) else 1)

    # Frequency encoding
    freq_query = df["QueryName"].value_counts(normalize=True)
    df["QueryName_FreqEnc"] = df["QueryName"].map(freq_query)
    freq_response = df["ResponseName"].value_counts(normalize=True)
    df["ResponseName_FreqEnc"] = df["ResponseName"].map(freq_response)
    df["SupportURL"] = df["SupportURL"].apply(
        lambda x: urlparse(x).netloc if pd.notnull(x) and x != 'none' else 'none'
    )
    freq_support = df["SupportURL"].value_counts(normalize=True)
    df["SupportURL"] = df["SupportURL"].map(freq_support).fillna(0)
    freq_email = df["SupportEmail"].value_counts(normalize=True)
    df["SupportEmail"] = df["SupportEmail"].map(freq_email).fillna(0)
    df["Website"] = df["Website"].apply(
        lambda x: urlparse(x).netloc if pd.notnull(x) and x != 'none' else 'none'
    )
    freq_website = df["Website"].value_counts(normalize=True)
    df["Website"] = df["Website"].map(freq_website).fillna(0)

    # Languages
    def extract_languages(text):
        if pd.isna(text) or str(text).strip() == '': return []
        parts = re.split(r',|\s{2,}', str(text))
        return [p.strip().lower() for p in parts if p.strip()]

    df["SupportedLanguages"] = df["SupportedLanguages"].fillna('').apply(extract_languages)
    selected_languages = ["english","german","french","spanish","italian","russian",
                          "portuguese","japanese","polish","brazil","chinese"]
    for lang in selected_languages:
        df["Lang_" + lang] = df["SupportedLanguages"].apply(lambda x: 1 if lang in x else 0)
    df["SupportedLanguagesCount"] = df["SupportedLanguages"].apply(len)

    # Release date
    df["ReleaseDate"] = pd.to_datetime(df["ReleaseDate"], errors="coerce")
    today = pd.Timestamp.today()
    df["ReleaseDate"] = (today - df["ReleaseDate"]).dt.days / 365.25
    df["ReleaseDate"] = df["ReleaseDate"].fillna(df["ReleaseDate"].median())

    # Requirements
    df['PC_MinRam']    = df['PCMinReqsText'].where(df['PCReqsHaveMin'] == 1).apply(extract_ram)
    df['PC_RecRam']    = df['PCRecReqsText'].where(df['PCReqsHaveRec'] == 1).apply(extract_ram)
    df['Linux_MinRam'] = df['LinuxMinReqsText'].where(df['LinuxReqsHaveMin'] == 1).apply(extract_ram)
    df['Linux_RecRam'] = df['LinuxRecReqsText'].where(df['LinuxReqsHaveRec'] == 1).apply(extract_ram)
    df['Mac_MinRam']   = df['MacMinReqsText'].where(df['MacReqsHaveMin'] == 1).apply(extract_ram)
    df['Mac_RecRam']   = df['MacRecReqsText'].where(df['MacReqsHaveRec'] == 1).apply(extract_ram)
    df['PC_MinCPU']    = df['PCMinReqsText'].where(df['PCReqsHaveMin'] == 1).apply(extract_proc)
    df['PC_RecCPU']    = df['PCRecReqsText'].where(df['PCReqsHaveRec'] == 1).apply(extract_proc)
    df['Linux_MinCPU'] = df['LinuxMinReqsText'].where(df['LinuxReqsHaveMin'] == 1).apply(extract_proc)
    df['Linux_RecCPU'] = df['LinuxRecReqsText'].where(df['LinuxReqsHaveRec'] == 1).apply(extract_proc)
    df['Mac_MinCPU']   = df['MacMinReqsText'].where(df['MacReqsHaveMin'] == 1).apply(extract_proc)
    df['Mac_RecCPU']   = df['MacRecReqsText'].where(df['MacReqsHaveRec'] == 1).apply(extract_proc)

    # Feature interactions
    df["owners_metacritic"] = df["SteamSpyOwners"] * df["Metacritic"].fillna(df["Metacritic"].median())
    df["players_metacritic"] = df["SteamSpyPlayersEstimate"] * df["Metacritic"].fillna(df["Metacritic"].median())
    df["owners_players"] = df["SteamSpyOwners"] * df["SteamSpyPlayersEstimate"]
    df["price_discount"] = df["PriceInitial"] - df["PriceFinal"]
    df["price_owners"] = df["PriceFinal"] * df["SteamSpyOwners"]
    df["price_players"] = df["PriceFinal"] * df["SteamSpyPlayersEstimate"]
    df["free_x_owners"] = df["IsFree"] * df["SteamSpyOwners"]
    df["free_x_players"] = df["IsFree"] * df["SteamSpyPlayersEstimate"]
    df["content_volume"] = df["ScreenshotCount"] + df["MovieCount"] + df["DLCCount"] + df["PackageCount"]
    df["content_owners"] = df["content_volume"] * df["SteamSpyOwners"]
    df["content_players"] = df["content_volume"] * df["SteamSpyPlayersEstimate"]
    df["content_metacritic"] = df["content_volume"] * df["Metacritic"].fillna(df["Metacritic"].median())
    df["achievement_owners"] = df["AchievementCount"] * df["SteamSpyOwners"]
    df["achievement_players"] = df["AchievementCount"] * df["SteamSpyPlayersEstimate"]
    df["highlighted_achievements_ratio"] = df["AchievementHighlightedCount"] / (df["AchievementCount"] + 1)
    df["platform_count"] = df["PlatformWindows"].astype(int) + df["PlatformLinux"].astype(int) + df["PlatformMac"].astype(int)
    df["platform_owners"] = df["platform_count"] * df["SteamSpyOwners"]
    df["platform_players"] = df["platform_count"] * df["SteamSpyPlayersEstimate"]
    df["action_multiplayer"] = df["GenreIsAction"] * df["CategoryMultiplayer"]
    df["rpg_achievement"] = df["GenreIsRPG"] * df["AchievementCount"]
    df["strategy_complexity"] = df["GenreIsStrategy"] * df["AchievementCount"]
    df["indie_price"] = df["GenreIsIndie"] * df["PriceFinal"]
    df["category_count"] = (df["CategorySinglePlayer"] + df["CategoryMultiplayer"] +
                            df["CategoryCoop"] + df["CategoryMMO"] + df["CategoryVRSupport"])
    df["category_owners"] = df["category_count"] * df["SteamSpyOwners"]
    df["category_players"] = df["category_count"] * df["SteamSpyPlayersEstimate"]
    df["review_words"] = df["Reviews"].fillna("").str.split().str.len()
    df["reviews_owners"] = df["review_words"] * df["SteamSpyOwners"]
    df["reviews_players"] = df["review_words"] * df["SteamSpyPlayersEstimate"]
    df["reviews_metacritic"] = df["review_words"] * df["Metacritic"].fillna(df["Metacritic"].median())

    # NLP
    df['AllText'] = df['DetailedDescrip'].fillna('') + ' ' + df['ShortDescrip'].fillna('')
    df['AllText_Cleaned'] = df['AllText'].apply(clean_text)
    df['AllText_len'] = df['AllText_Cleaned'].str.split().str.len()

    keyword_cols_local = []
    new_cols_dict = {}
    for word in keywords:
        col_name = f'has_{word.replace(" ", "_")}'
        new_cols_dict[col_name] = df['AllText_Cleaned'].str.contains(word).astype(int)
        keyword_cols_local.append(col_name)
    df = pd.concat([df, pd.DataFrame(new_cols_dict)], axis=1)

    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    df['ReviewSentiment'] = df['Reviews'].fillna('').apply(
        lambda x: sia.polarity_scores(str(x))['compound']
    )
    df['AboutSentiment'] = df['AboutText'].fillna('').apply(
        lambda x: sia.polarity_scores(str(x))['compound']
    )

    return df

# Unified Preprocessor
production_preprocessor = ColumnTransformer([
    ('text', vectorizer, 'AllText_Cleaned'),
    ('num', numeric_transformer, log_cols),
    ('keywords', 'passthrough', keyword_cols),
    ('sentiment', 'passthrough', ['ReviewSentiment', 'AboutSentiment']),
    ('website_logic', url_pipeline, ['Website']),
    ('interactions', SteamFeatureInteractions(), ['PriceInitial', 'PriceFinal', 'ScreenshotCount', 'MovieCount', 'DLCCount']),
    ('ram_min', RAMExtractor('PCMinReqsText'), ['PCMinReqsText']),
    ('cpu_min', CPUExtractor('PCMinReqsText'), ['PCMinReqsText'])
])
production_preprocessor.fit(df_full)
# Save the production-ready preprocessor
joblib.dump(production_preprocessor, 'production_preprocessor.pkl')
print("Production preprocessor bundled and downloaded.")


mode  =input("classification or regression [c/r]: ")

cols_to_drop = [
    "QueryName", "ResponseName", "AboutText", "ShortDescrip", "DetailedDescrip",
    "PCMinReqsText", "PCRecReqsText", "LinuxMinReqsText", "LinuxRecReqsText",
    "MacMinReqsText", "MacRecReqsText", "AllText", "AllText_Cleaned",
    "Reviews", "SupportedLanguages", 'QueryID', 'ResponseID'
]

if mode.lower() == "c":
  for model in ['logistic', 'random_forest', 'gradient_boosting', 'xgboost']:
    loaded_model = joblib.load(f'saved_models/{model}.pkl')
    loaded_preprocessor = joblib.load('production_preprocessor.pkl')
    # unseen data
    new_df = pd.read_csv("Data/train_data_class.csv")
    new_df = preprocess_raw(new_df)
    new_df = new_df.drop(columns=["GamePopularity"], errors="ignore")
    new_df = new_df.drop(columns=cols_to_drop, errors="ignore")
    new_df = new_df[X_train.columns]
    print(f"Transformed shape: {new_df.shape}")

    preds = loaded_model.predict(new_df)
    print(f"{model}: Classification Predictions: {preds[:5]}")

# loaded_model = joblib.load('saved_models/random_forest.pkl') # TO DO: for loop for all models + reg/class control....
# loaded_preprocessor = joblib.load('production_preprocessor.pkl')
else:
  for model in ['random_forest', 'gradient_boosting', 'linear', 'polynomial', 'ridge', 'xgboost']:
    loaded_model = joblib.load(f'saved_models/{model}.pkl')
    loaded_preprocessor = joblib.load('production_preprocessor.pkl')
    # unseen data
    new_df = pd.read_csv("Data/train_data.csv")
    new_df = preprocess_raw(new_df)
    new_df = new_df.drop(columns=["RecommendationCount"], errors="ignore")
    new_df = new_df.drop(columns=cols_to_drop, errors="ignore")
    new_df = new_df[X_train.columns]

    preds = loaded_model.predict(new_df)
    print(f"{model}: Regression Predictions: {preds[:5]}")