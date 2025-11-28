"""Generate plots and tables summarizing sentiment vs returns."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .models import (
    ensure_beat_miss_flag,
    load_features,
    run_linear_regression,
    run_logistic_downdrift_model,
    summarize_regression,
)

matplotlib.use("Agg")  # headless


def save_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for col in [c for c in ["abn_ret_1d", "abn_ret_5d", "ret_1d", "ret_5d"] if c in df.columns]:
        plt.figure(figsize=(5, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(col)
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{col}.png", dpi=150)
        plt.close()


def save_scatter_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ret_col in [c for c in ["abn_ret_1d", "abn_ret_5d"] if c in df.columns]:
        plt.figure(figsize=(5, 4))
        sns.scatterplot(data=df, x="qa_sent_score", y=ret_col)
        plt.axhline(0, color="grey", linestyle="--", linewidth=1)
        plt.axvline(0, color="grey", linestyle="--", linewidth=1)
        plt.title(f"qa_sent_score vs {ret_col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter_qa_sent_vs_{ret_col}.png", dpi=150)
        plt.close()


def save_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.dropna(subset=["qa_sent_score"]).copy()
    if df.empty:
        return
    df["sent_bucket"] = pd.qcut(df["qa_sent_score"], 3, labels=["Low", "Mid", "High"])
    for ret_col in [c for c in ["abn_ret_1d", "abn_ret_5d"] if c in df.columns]:
        plt.figure(figsize=(5, 4))
        sns.boxplot(data=df, x="sent_bucket", y=ret_col)
        plt.title(f"{ret_col} by sentiment tercile")
        plt.tight_layout()
        plt.savefig(out_dir / f"box_sent_bucket_{ret_col}.png", dpi=150)
        plt.close()


def save_regression_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictors = [
        "prep_sent_score",
        "qa_sent_score",
        "tone_shift",
        "qa_hedge_rate",
        "qa_risk_rate",
        "beat_miss_flag",
    ]
    qa_predictors = ["qa_sent_score", "tone_shift", "beat_miss_flag"]

    def _summary_df(model):
        return pd.DataFrame({"coef": model.params, "p_value": model.pvalues})

    ols_full = run_linear_regression(df, outcome="abn_ret_5d", predictors=predictors)
    _summary_df(ols_full).to_csv(out_dir / "ols_full_coeffs.csv")

    ols_qa = run_linear_regression(df, outcome="abn_ret_5d", predictors=qa_predictors)
    _summary_df(ols_qa).to_csv(out_dir / "ols_qa_coeffs.csv")

    # Save text summaries for quick inspection.
    (out_dir / "ols_full_summary.txt").write_text(summarize_regression(ols_full))
    (out_dir / "ols_qa_summary.txt").write_text(summarize_regression(ols_qa))


def save_logistic_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_res = run_logistic_downdrift_model(df)
    if "metrics" not in log_res:
        (out_dir / "logistic_metrics.txt").write_text(log_res.get("error", "No data"))
        return

    metrics = log_res["metrics"]
    pd.DataFrame([metrics]).to_csv(out_dir / "logistic_metrics.csv", index=False)

    model = log_res.get("model")
    features = metrics.get("features", [])
    if model is not None:
        coefs = pd.DataFrame({"feature": ["intercept", *features], "coef": [model.intercept_[0], *model.coef_[0]]})
        coefs.to_csv(out_dir / "logistic_coeffs.csv", index=False)


def save_summary_table(df: pd.DataFrame, out_path: Path) -> None:
    cols = [c for c in ["prep_sent_score", "qa_sent_score", "tone_shift", "abn_ret_1d", "abn_ret_5d", "beat_miss_flag"] if c in df.columns]
    summary = df[cols].describe().T
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Save sentiment/returns plots and tables.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "config" / "config.yaml"))
    parser.add_argument("--plots-dir", default="assets/plots", help="Directory to save plots")
    parser.add_argument("--tables-dir", default="data_processed", help="Directory to save tables")
    args = parser.parse_args()

    df = load_features(Path(args.config))
    df = ensure_beat_miss_flag(df)

    plots_dir = Path(args.plots_dir)
    tables_dir = Path(args.tables_dir)

    save_summary_table(df, tables_dir / "summary_stats.csv")
    save_histograms(df, plots_dir)
    save_scatter_plots(df, plots_dir)
    save_boxplots(df, plots_dir)
    save_regression_tables(df, tables_dir)
    save_logistic_metrics(df, tables_dir)

    print(f"Saved plots to {plots_dir} and tables to {tables_dir}")


if __name__ == "__main__":
    main()
