# Run tests: pytest -q
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from finance.returns import compute_event_window_returns  # noqa: E402
from finance.surprise import compute_beat_miss_flag  # noqa: E402


def test_compute_event_window_returns():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame(
        {
            "AAA": [10, 11, 11.5, 11.5, 12, 12],
            "XBI": [10, 10.5, 10.5, 10.6, 10.7, 10.8],
        },
        index=dates,
    )

    events = pd.DataFrame({"ticker": ["AAA"], "earnings_date": [dates[0]]})
    out = compute_event_window_returns(events, prices, benchmark_ticker="XBI", window_days=[1, 5])

    assert "ret_1d" in out.columns
    assert abs(out.loc[0, "ret_1d"] - 0.1) < 1e-6
    assert abs(out.loc[0, "abn_ret_1d"] - 0.05) < 1e-6


def test_forward_window_uses_next_trading_day():
    dates = pd.bdate_range("2024-01-05", periods=3)  # Fri, Mon, Tue
    prices = pd.DataFrame(
        {
            "AAA": [10.0, 12.0, 12.5],
            "XBI": [20.0, 21.0, 21.5],
        },
        index=dates,
    )

    events = pd.DataFrame({"ticker": ["AAA"], "earnings_date": [dates[0]]})
    out = compute_event_window_returns(events, prices, benchmark_ticker="XBI", window_days=[1])

    # Forward 1d window (Saturday) should use Monday's price, not Friday's.
    assert abs(out.loc[0, "ret_1d"] - 0.2) < 1e-6
    assert abs(out.loc[0, "abn_ret_1d"] - 0.15) < 1e-6


def test_beat_miss_flag_values():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {
            "AAA": [10.0, 11.0, 11.0, 10.0, 10.0],
            "XBI": [10.0, 10.0, 10.0, 10.0, 10.0],
        },
        index=dates,
    )

    events = pd.DataFrame({"ticker": ["AAA", "AAA", "AAA"], "earnings_date": dates[:3]})
    out = compute_event_window_returns(events, prices, benchmark_ticker="XBI", window_days=[1])
    flags = compute_beat_miss_flag(out, ret_col="ret_1d")

    assert set(flags.dropna().unique()) == {1, 0, -1}
    assert list(flags.iloc[:3]) == [1, 0, -1]
