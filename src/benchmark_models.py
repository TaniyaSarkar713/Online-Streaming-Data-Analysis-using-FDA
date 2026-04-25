from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import calculate_robust_metrics, classification_metrics


def regression_model_from_name(name: str):
    if "Random Forest" in name:
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=10,
            random_state=713,
        )
    if "Gradient Boosting" in name:
        return GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            random_state=713,
        )
    if "MLP" in name:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        alpha=0.01,
                        early_stopping=True,
                        max_iter=1500,
                        random_state=713,
                    ),
                ),
            ]
        )
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=4.0))])


def classification_model_from_name(name: str):
    if "Random Forest" in name:
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=10,
            random_state=713,
        )
    if "Gradient Boosting" in name:
        return GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            random_state=713,
        )
    return Pipeline(
        [("scaler", StandardScaler()), ("logit", LogisticRegression(max_iter=2000))]
    )


def run_regression_benchmarks(
    feature_sets: Dict[str, pd.DataFrame],
    y: np.ndarray,
    test_size: float = 0.30,
    random_state: int = 713,
) -> pd.DataFrame:
    idx_train, idx_test = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
    )
    rows = []
    for model_name, feat in feature_sets.items():
        X_train = feat.iloc[idx_train]
        X_test = feat.iloc[idx_test]
        y_train = y[idx_train]
        y_test = y[idx_test]

        model = regression_model_from_name(model_name)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(calculate_robust_metrics(y_test, pred, model_name))
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def run_classification_benchmarks(
    feature_sets: Dict[str, pd.DataFrame],
    y: np.ndarray,
    test_size: float = 0.30,
    random_state: int = 713,
) -> pd.DataFrame:
    idx_train, idx_test = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    rows = []
    for model_name, feat in feature_sets.items():
        X_train = feat.iloc[idx_train]
        X_test = feat.iloc[idx_test]
        y_train = y[idx_train]
        y_test = y[idx_test]

        model = classification_model_from_name(model_name)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append(classification_metrics(y_test, prob, pred, model_name))
    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
