"""Run simple statistical tests and models on the earnings NLP dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from .models import (
    compare_groups_ttest,
    ensure_beat_miss_flag,
    load_features,
    run_linear_regression,
    run_logistic_downdrift_model,
    summarize_regression,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run t-tests, OLS, and logistic models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    args = parser.parse_args()

    df = load_features(Path(args.config))
    df = ensure_beat_miss_flag(df)

    print("=== Welch t-test: Q&A sentiment vs 5d abnormal return ===")
    ttest_res = compare_groups_ttest(df, feature="qa_sent_score", outcome="abn_ret_5d")
    for k, v in ttest_res.items():
        print(f"{k}: {v}")

    print("\n=== OLS: returns explained by language features ===")
    predictors = [
        "prep_sent_score",
        "qa_sent_score",
        "tone_shift",
        "qa_hedge_rate",
        "qa_risk_rate",
        "beat_miss_flag",
    ]
    ols_model = run_linear_regression(df, outcome="abn_ret_5d", predictors=predictors)
    print(summarize_regression(ols_model))

    print("\n=== OLS: QA sentiment/tone shift controlling for beat/miss ===")
    qa_predictors = ["qa_sent_score", "tone_shift", "beat_miss_flag"]
    qa_ols = run_linear_regression(df, outcome="abn_ret_5d", predictors=qa_predictors)
    print(summarize_regression(qa_ols))

    print("\n=== Logistic: predict >5% downside (abn_ret_5d < -5%) ===")
    log_res = run_logistic_downdrift_model(df)
    if "metrics" in log_res:
        for k, v in log_res["metrics"].items():
            print(f"{k}: {v}")
    else:
        print(log_res.get("error", "Unknown error"))


if __name__ == "__main__":
    main()
