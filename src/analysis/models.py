"""Statistical tests and simple predictive models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import statsmodels.api as sm

from ..finance.surprise import compute_beat_miss_flag


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def make_negative_positive_groups(df: pd.DataFrame, feature: str, quantile: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Split into negative/positive groups based on a feature quantile."""
    clean = df.dropna(subset=[feature])
    threshold = clean[feature].quantile(quantile)
    negative = clean[clean[feature] <= threshold]
    positive = clean[clean[feature] >= threshold]
    return negative, positive, threshold


def compare_groups_ttest(df: pd.DataFrame, feature: str, outcome: str) -> dict:
    """Run a Welch t-test on the outcome across low/high feature groups."""
    negative, positive, threshold = make_negative_positive_groups(df, feature)
    res = stats.ttest_ind(
        negative[outcome].dropna(),
        positive[outcome].dropna(),
        equal_var=False,
    )
    return {
        "feature": feature,
        "outcome": outcome,
        "threshold": threshold,
        "t_stat": res.statistic,
        "p_value": res.pvalue,
        "n_negative": len(negative),
        "n_positive": len(positive),
    }


def run_linear_regression(df: pd.DataFrame, outcome: str, predictors: Iterable[str]):
    """Run an OLS regression on the specified predictors."""
    cols = [outcome] + list(predictors)
    clean = df.dropna(subset=cols)
    X = sm.add_constant(clean[list(predictors)])
    y = clean[outcome]
    model = sm.OLS(y, X).fit()
    return model


def summarize_regression(model) -> str:
    return model.summary().as_text()


def ensure_beat_miss_flag(df: pd.DataFrame, ret_col: str = "ret_1d") -> pd.DataFrame:
    """Add beat_miss_flag column if missing, using price reaction proxy."""
    if "beat_miss_flag" not in df.columns or df["beat_miss_flag"].isna().all():
        if ret_col in df.columns:
            df = df.copy()
            df["beat_miss_flag"] = compute_beat_miss_flag(df, ret_col=ret_col)
    return df


def run_logistic_downdrift_model(df: pd.DataFrame) -> dict:
    """Predict >5% abnormal downside within 5 days using logistic regression."""
    df = ensure_beat_miss_flag(df.copy())
    df["label"] = (df["abn_ret_5d"] < -0.05).astype(int)
    features = [
        "qa_sent_score",
        "prep_sent_score",
        "tone_shift",
        "qa_hedge_rate",
        "qa_risk_rate",
        "beat_miss_flag",
    ]
    missing = [feat for feat in features if feat not in df.columns]
    if missing:
        return {"error": f"Missing features for logistic model: {missing}"}

    cols = features + ["earnings_date", "label"]
    clean = df.dropna(subset=cols).sort_values("earnings_date")
    if clean.empty:
        return {"error": "No data available for logistic model"}

    split_idx = int(len(clean) * 0.8) or 1
    train = clean.iloc[:split_idx]
    test = clean.iloc[split_idx:]
    if test.empty:
        test = train.copy()

    X_train, y_train = train[features], train["label"]
    X_test, y_test = test[features], test["label"]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    probas = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    metrics = {
        "train_size": len(train),
        "test_size": len(test),
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probas) if len(set(y_test)) > 1 else float("nan"),
        "classification_report": classification_report(y_test, preds),
        "features": features,
    }
    return {"model": clf, "metrics": metrics}


def load_features(config_path: Path = DEFAULT_CONFIG) -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    path = Path(config.get("events_with_features_path", "data_processed/events_with_features.parquet"))
    return pd.read_parquet(path)


if __name__ == "__main__":
    raise SystemExit("Use run_all_models.py to execute analyses.")
