"""Compute event-window returns and abnormal returns for earnings calls."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import yaml

from .returns import compute_event_window_returns, download_price_history
from .surprise import compute_beat_miss_flag


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute returns and abnormal returns for events.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    parser.add_argument("--windows", nargs="*", type=int, default=[1, 5], help="Forward day windows to compute")
    parser.add_argument("--refresh-cache", action="store_true", help="Redownload prices instead of using cache")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    events_path = Path(config.get("events_base_path", "data_processed/events_base.parquet"))
    output_path = Path(config.get("events_with_returns_path", "data_processed/events_with_returns.parquet"))
    benchmark = config.get("benchmark_ticker", "XBI")
    start = config.get("price_start_date", "2013-01-01")
    end = config.get("price_end_date", "2025-12-31")
    price_cache_dir = config.get("price_cache_dir")

    events = pd.read_parquet(events_path)
    tickers: List[str] = sorted(set(events["ticker"].unique().tolist()) | {benchmark})

    prices = download_price_history(
        tickers,
        start=start,
        end=end,
        price_cache_dir=price_cache_dir,
        refresh_cache=args.refresh_cache,
    )
    with_returns = compute_event_window_returns(events, prices, benchmark_ticker=benchmark, window_days=args.windows)
    with_returns["beat_miss_flag"] = compute_beat_miss_flag(with_returns, ret_col="ret_1d")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with_returns.to_parquet(output_path, index=False)
    print(f"Saved events with returns to {output_path}")


if __name__ == "__main__":
    main()
