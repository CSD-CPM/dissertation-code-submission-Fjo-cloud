from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class AppDataConfig:
    target_col: str = "enrolled"


def basic_feature_engineering(df: pd.DataFrame, config: AppDataConfig) -> pd.DataFrame:
    """
    Basic, non-leaky feature engineering for the appdata10 dataset.

    Operations:
    - Convert 'hour' from string 'HH:MM:SS' to integer hour [0, 23].
    - Drop identifier and obvious leakage columns.
    """
    df = df.copy()

    # --- Robust hour conversion ---
    if "hour" in df.columns:
        # If already numeric, leave it. Otherwise parse safely.
        if not pd.api.types.is_numeric_dtype(df["hour"]):
            df["hour"] = (
                pd.to_datetime(
                    df["hour"].astype(str).str.strip(),  # remove leading/trailing spaces
                    format="%H:%M:%S",
                    errors="coerce",                     # invalid -> NaT instead of error
                )
                .dt.hour
            )
            # Fill any parsing failures with the most frequent hour
            if df["hour"].isna().any():
                df["hour"] = df["hour"].fillna(df["hour"].mode()[0])

    # Columns we do NOT want to use as predictors
    leakage_cols = ["user", "first_open", "enrolled_date", "screen_list"]
    cols_to_drop = [c for c in leakage_cols if c in df.columns]

    df = df.drop(columns=cols_to_drop)

    return df

def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced behavioral engineering for dissertation rigor.
    """
    df = df.copy()
    
    # 1. Engagement Intensity
    # If age is 0, we've got bad data, so we handle that logic
    df['engagement_intensity'] = df['numscreens'] / (df['age'] + 1)
    
    # 2. Helpful vs Action Ratio
    # We hypothesize that users who play the minigame are more likely to convert
    df['is_active_user'] = (df['minigame'] + df['used_premium_feature']).clip(0, 1)
    
    return df


def train_test_split_df(
    df: pd.DataFrame,
    config: AppDataConfig,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Convenience wrapper to split a dataframe into X_train, X_test, y_train, y_test
    using the configured target column.
    """
    from sklearn.model_selection import train_test_split  # local import to avoid cycles

    assert config.target_col in df.columns, f"Target column {config.target_col!r} not in dataframe."

    X = df.drop(columns=[config.target_col])
    y = df[config.target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
