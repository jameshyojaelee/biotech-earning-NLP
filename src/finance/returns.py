"""Utilities for downloading prices and computing event-window returns."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# Stock returns measure percentage price change; a benchmark ETF (here XBI) is
# used to represent the sector so abnormal returns (stock minus benchmark)
# isolate firm-specific moves around the event window (the days immediately
# after the earnings date).

def _normalize_price_df(data, tickers: List[str]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                prices = data["Adj Close"].copy()
            else:
                prices = data["Close"].copy()
        else:
            if len(tickers) == 1 and tickers[0] in data.columns:
                prices = data[[tickers[0]]]
            elif "Adj Close" in data.columns:
                prices = data["Adj Close"].to_frame(name=tickers[0])
            elif "Close" in data.columns:
                prices = data["Close"].to_frame(name=tickers[0])
            else:
                prices = data.copy()
    else:  # Series
        prices = data.to_frame(name=tickers[0])

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{ticker}.parquet"


def _load_cached_prices(cache_dir: Path, ticker: str) -> Optional[pd.DataFrame]:
    path = _cache_path(cache_dir, ticker)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    if ticker not in df.columns and len(df.columns) == 1:
        df = df.rename(columns={df.columns[0]: ticker})
    return df[[ticker]].sort_index()


def _save_prices_to_cache(cache_dir: Path, ticker: str, prices: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    prices[[ticker]].to_parquet(_cache_path(cache_dir, ticker))


def download_price_history(
    tickers: Iterable[str],
    start: str,
    end: str,
    price_cache_dir: Optional[Path] = None,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Download adjusted close prices for tickers using yfinance with caching."""
    tickers = list(tickers)
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    cache_dir = Path(price_cache_dir) if price_cache_dir else None

    cached_frames: List[pd.DataFrame] = []
    missing: List[str] = []

    if cache_dir and not refresh_cache:
        for ticker in tickers:
            cached = _load_cached_prices(cache_dir, ticker)
            if cached is not None:
                coverage_ok = (cached.index.min() <= start_dt) and (cached.index.max() >= end_dt)
                if coverage_ok:
                    cached_frames.append(cached)
                    continue
            missing.append(ticker)
    else:
        missing = tickers.copy()

    downloaded = pd.DataFrame()
    if missing:
        data = yf.download(missing, start=start_dt, end=end_dt, auto_adjust=True, progress=False)
        downloaded = _normalize_price_df(data, missing)
        for ticker in missing:
            if ticker in downloaded.columns and cache_dir:
                _save_prices_to_cache(cache_dir, ticker, downloaded)

    all_frames = []
    if not downloaded.empty:
        all_frames.append(downloaded)
    all_frames.extend(cached_frames)

    if not all_frames:
        return pd.DataFrame()

    prices = pd.concat(all_frames, axis=1)
    prices = prices.loc[(prices.index >= start_dt) & (prices.index <= end_dt)]
    prices = prices.sort_index()
    return prices


def _price_on_or_before(prices: pd.DataFrame, ticker: str, date: pd.Timestamp) -> Optional[float]:
    if ticker not in prices.columns:
        return None
    series = prices[ticker].dropna().sort_index()
    available = series.loc[series.index <= date]
    if available.empty:
        return None
    return float(available.iloc[-1])


def _price_on_or_after(
    prices: pd.DataFrame,
    ticker: str,
    date: pd.Timestamp,
    offset_to_next_business_day: bool = False,
) -> Optional[float]:
    """Return the earliest available price on or after `date`.

    If `offset_to_next_business_day` is True, the search starts from the next
    business day rather than the provided date.
    """
    if offset_to_next_business_day:
        date = pd.Timestamp(date) + pd.offsets.BDay()

    if ticker not in prices.columns:
        return None

    series = prices[ticker].dropna().sort_index()
    available = series.loc[series.index >= date]
    if available.empty:
        return None
    return float(available.iloc[0])


def compute_event_window_returns(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str,
    window_days: List[int],
) -> pd.DataFrame:
    """Compute stock and abnormal returns for specified forward windows."""
    events = events.copy()
    events["earnings_date"] = pd.to_datetime(events["earnings_date"], errors="coerce")

    for window in window_days:
        ret_col = f"ret_{window}d"
        bench_col = f"bench_ret_{window}d"
        abn_col = f"abn_ret_{window}d"
        stock_returns: List[Optional[float]] = []
        bench_returns: List[Optional[float]] = []
        abn_returns: List[Optional[float]] = []

        for row in events.itertuples():
            event_date = pd.to_datetime(row.earnings_date)
            end_date = event_date + pd.Timedelta(days=window)

            base_price = _price_on_or_before(prices, row.ticker, event_date)
            end_price = _price_on_or_after(prices, row.ticker, end_date)
            bench_base = _price_on_or_before(prices, benchmark_ticker, event_date)
            bench_end = _price_on_or_after(prices, benchmark_ticker, end_date)

            if base_price is None or end_price is None:
                stock_returns.append(np.nan)
            else:
                stock_returns.append((end_price - base_price) / base_price)

            if bench_base is None or bench_end is None:
                bench_returns.append(np.nan)
            else:
                bench_returns.append((bench_end - bench_base) / bench_base)

            if pd.isna(stock_returns[-1]) or pd.isna(bench_returns[-1]):
                abn_returns.append(np.nan)
            else:
                abn_returns.append(stock_returns[-1] - bench_returns[-1])

        events[ret_col] = stock_returns
        events[bench_col] = bench_returns
        events[abn_col] = abn_returns

    return events


if __name__ == "__main__":
    raise SystemExit("Use src/finance/compute_returns_for_events.py instead.")
