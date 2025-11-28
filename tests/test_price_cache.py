# Run tests: pytest -q
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from finance.returns import download_price_history  # noqa: E402


def test_download_uses_cache_without_refresh(monkeypatch, tmp_path):
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    cached = pd.DataFrame({"AAA": [1.0, 1.1, 1.2]}, index=dates)
    cache_path = tmp_path / "AAA.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached.to_parquet(cache_path)

    called = False

    def fake_download(*args, **kwargs):
        nonlocal called
        called = True
        return pd.DataFrame()

    monkeypatch.setattr("finance.returns.yf.download", fake_download)

    prices = download_price_history(["AAA"], start=str(dates[0]), end=str(dates[-1]), price_cache_dir=tmp_path)

    assert not called, "Should not hit yfinance when cache covers range"
    assert "AAA" in prices.columns
    assert len(prices) == 3


def test_download_refreshes_and_saves(monkeypatch, tmp_path):
    dates = pd.date_range("2024-02-01", periods=2, freq="D")
    returned = pd.DataFrame({"AAA": [2.0, 2.1]}, index=dates)

    called = False

    def fake_download(tickers, start, end, auto_adjust, progress):
        nonlocal called
        called = True
        return returned

    monkeypatch.setattr("finance.returns.yf.download", fake_download)

    prices = download_price_history(
        ["AAA"],
        start=str(dates[0]),
        end=str(dates[-1]),
        price_cache_dir=tmp_path,
        refresh_cache=True,
    )

    cache_file = tmp_path / "AAA.parquet"
    assert called
    assert cache_file.exists(), "Downloaded prices should be cached"
    assert prices.loc[dates[0], "AAA"] == 2.0
