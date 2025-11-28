"""Beat/miss proxies derived from price reactions or consensus data."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_beat_miss_flag(
    events: pd.DataFrame,
    ret_col: str = "ret_1d",
    consensus_col: Optional[str] = None,
) -> pd.Series:
    """Return a {-1, 0, 1} beat/miss proxy.

    Uses 1-day price reaction as a fallback proxy: positive moves are treated
    as beats (1), negatives as misses (-1), and zero moves as neutral (0).
    If a consensus surprise column exists and contains any non-null
    values, that series is used instead so the flag remains nullable when
    future data is added.
    """

    base_series = None

    if consensus_col and consensus_col in events.columns:
        series = events[consensus_col]
        if series.notna().any():
            base_series = series

    if base_series is None:
        if ret_col in events.columns:
            base_series = events[ret_col]
        else:
            # No usable source; return all NaNs to keep pipeline intact.
            return pd.Series([np.nan] * len(events), index=events.index, name="beat_miss_flag")

    def _flag(val):
        if pd.isna(val):
            return np.nan
        if val > 0:
            return 1
        if val < 0:
            return -1
        return 0

    flags = base_series.apply(_flag)
    flags.name = "beat_miss_flag"
    return flags
