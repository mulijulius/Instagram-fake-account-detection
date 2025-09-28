import pandas as pd
from typing import List, Tuple

# Updated feature columns to match instagram_description.csv
DEFAULT_FEATURE_COLUMNS: List[str] = [
    "profile pic",             # user has profile picture or not
    "nums/length username",    # ratio of number of numerical chars in username to its length
    "fullname words",          # full name in word tokens
    "nums/length fullname",    # ratio of number of numerical characters in full name to its length
    "name==username",          # are username and full name literally the same
    "description length",      # bio length in characters
    "external URL",            # has external URL or not
    "private",                 # private or not
    "#posts",                  # number of posts
    "#followers",              # number of followers
]

def select_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]
    return cols

def split_features_labels(df: pd.DataFrame, feature_cols: List[str], has_label: bool = True) -> Tuple[pd.DataFrame, pd.Series | None]:
    X = df[feature_cols].copy()
    y = df["label"].astype(int) if has_label and "label" in df.columns else None
    return X, y
