import pandas as pd
from typing import List, Tuple


DEFAULT_FEATURE_COLUMNS: List[str] = [
    "followers",
    "following",
    "bio_length",
    "has_profile_pic",
    "account_age_days",
    "verified",
    "posts_count",
    "post_frequency",
    "avg_caption_len",
    "hashtag_count",
    "avg_likes",
    "avg_comments",
    "engagement_rate",
    "posting_variance",
]


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]
    return cols


def split_features_labels(df: pd.DataFrame, feature_cols: List[str], has_label: bool = True) -> Tuple[pd.DataFrame, pd.Series | None]:
    X = df[feature_cols].copy()
    y = df["label"].astype(int) if has_label and "label" in df.columns else None
    return X, y
