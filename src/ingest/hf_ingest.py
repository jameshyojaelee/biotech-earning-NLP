"""Ingest earnings call transcripts from HuggingFace and build the base events table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import yaml
from datasets import load_dataset


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sp500_earnings_dataset(config_path: str) -> Tuple[dict, object]:
    """Load the configured HuggingFace dataset of S&P 500 earnings calls."""
    config = load_config(Path(config_path))
    dataset_name = config.get("hf_dataset_name", "glopardo/sp500-earnings-transcripts")
    revision = config.get("hf_dataset_revision") or None
    ds = load_dataset(dataset_name, revision=revision, split="train")
    print(f"Loaded dataset {dataset_name} at revision {revision or 'latest'}")
    return config, ds


def flatten_segments_field(segments: object) -> pd.DataFrame:
    """Convert a raw segments field into a DataFrame with speaker metadata."""
    if segments is None:
        return pd.DataFrame(columns=["speaker_role", "speaker_name", "text"])
    if isinstance(segments, pd.DataFrame):
        return segments
    if not isinstance(segments, list):
        return pd.DataFrame(columns=["speaker_role", "speaker_name", "text"])

    df = pd.DataFrame(segments)
    if df.empty:
        return pd.DataFrame(columns=["speaker_role", "speaker_name", "text"])

    text_col = None
    for candidate in ["text", "content", "segment_text", "body"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col and "text" not in df.columns:
        df["text"] = df[text_col]
    df["text"] = df.get("text", pd.Series([None] * len(df), index=df.index))

    if "speaker_role" not in df.columns:
        for candidate in ["speaker", "speaker_name", "role", "speaker_title"]:
            if candidate in df.columns:
                df["speaker_role"] = df[candidate]
                break
    if "speaker_name" not in df.columns:
        for candidate in ["speaker_name", "speaker", "speaker_role"]:
            if candidate in df.columns:
                df["speaker_name"] = df[candidate]
                break

    keep_cols: List[str] = [col for col in ["speaker_role", "speaker_name", "text"] if col in df.columns]
    return df[keep_cols].copy()


def filter_healthcare_calls(ds, sector: str = "Health Care") -> pd.DataFrame:
    """Convert the dataset to pandas and keep only Health Care earnings calls."""
    # Earnings calls are scheduled events where executives share results; the
    # Q&A section often surfaces new information that can move stocks.
    df = ds.to_pandas()
    # Focus on biotech/health care names so the analysis stays within one sector.
    filtered = df[df["sector"] == sector].copy()
    optional_segment_cols = [col for col in ["segments", "speaker_segments", "turns"] if col in filtered.columns]
    keep_cols = ["ticker", "company", "sector", "earnings_date", "year", "quarter", "transcript", *optional_segment_cols]
    filtered = filtered[keep_cols]
    # The earnings_date anchors the event window used for return calculations.
    filtered["earnings_date"] = pd.to_datetime(filtered["earnings_date"], errors="coerce")
    return filtered.dropna(subset=["earnings_date"])


def save_events_base(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest HF earnings dataset and build base events table.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml")
    args = parser.parse_args()

    config, ds = load_sp500_earnings_dataset(args.config)
    events_df = filter_healthcare_calls(ds, sector=config.get("sector_filter", "Health Care"))
    output_path = Path(config.get("events_base_path", "data_processed/events_base.parquet"))
    save_path = save_events_base(events_df, output_path)
    print(f"Saved {len(events_df)} events to {save_path}")


if __name__ == "__main__":
    main()
