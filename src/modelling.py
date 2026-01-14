from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer for the app dataset.
    Currently all features are numeric; if we add categoricals later,
    we can extend this function.
    """
    numeric_features: List[str] = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )

    return preprocessor


def build_logistic_pipeline(X: pd.DataFrame, random_state: int = 42) -> Pipeline:
    """
    Return a sklearn Pipeline with preprocessing + Logistic Regression.
    """
    preprocessor = build_preprocessor(X)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )
    return pipe


def build_random_forest_pipeline(X: pd.DataFrame, random_state: int = 42) -> Pipeline:
    """
    Return a sklearn Pipeline with preprocessing + RandomForestClassifier.
    """
    preprocessor = build_preprocessor(X)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )
    return pipe
