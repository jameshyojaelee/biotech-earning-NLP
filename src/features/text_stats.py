"""Text stats for hedging and biotech risk language in Q&A sections."""

from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd

HEDGE_TERMS = [
    "may",
    "might",
    "could",
    "can",
    "uncertain",
    "uncertainty",
    "visibility",
    "approximately",
    "around",
    "potentially",
    "possible",
    "expect",
    "believe",
    "should",
    "plan to",
    "expect to",
]

RISK_TERMS = [
    "fda",
    "fda approved",
    "fda-approved",
    "trial hold",
    "clinical hold",
    "enrollment delay",
    "enrollment delays",
    "enrollment paused",
    "data cutoff",
    "adverse event",
    "adverse events",
    "serious adverse event",
    "serious adverse events",
    "safety events",
    "safety signal",
    "black box",
    "recall",
    "delay",
    "setback",
    "pdufa",
    "pdufa date",
    "crl",
    "complete response letter",
    "phase i",
    "phase ii",
    "phase iii",
    "enrollment",
    "dropout",
    "serious adverse",
    "type c meeting",
    "breakthrough designation",
]


def preprocess_text(text: str) -> List[str]:
    """Lowercase and tokenize, normalizing punctuation and hyphens."""
    if not text:
        return []
    normalized = text.lower()
    normalized = re.sub(r"[-]+", " ", normalized)  # keep hyphenated words searchable
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return tokens


def count_terms(text: str, term_list: Iterable[str]) -> int:
    """Count occurrences allowing hyphenated and multi-word matches."""
    tokens = preprocess_text(text)
    if not tokens:
        return 0
    joined = " ".join(tokens)
    count = 0
    for term in term_list:
        term_tokens = preprocess_text(term)
        if not term_tokens:
            continue
        pattern = r"(?<!\S)" + re.escape(" ".join(term_tokens)) + r"(?!\S)"
        count += len(re.findall(pattern, joined))
    return count


def compute_qa_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hedging/risk counts and rates for Q&A text."""
    df = df.copy()
    qa_word_counts = []
    hedge_counts = []
    risk_counts = []

    texts = df["qa_text"] if "qa_text" in df else pd.Series([], dtype=str)
    for text in texts:
        tokens = preprocess_text(text)
        qa_word_counts.append(len(tokens))
        hedge_counts.append(count_terms(text, HEDGE_TERMS))
        risk_counts.append(count_terms(text, RISK_TERMS))

    df["qa_word_count"] = qa_word_counts
    df["qa_hedge_terms"] = hedge_counts
    df["qa_risk_terms"] = risk_counts

    df["qa_hedge_rate"] = df["qa_hedge_terms"] / df["qa_word_count"].replace(0, pd.NA)
    df["qa_risk_rate"] = df["qa_risk_terms"] / df["qa_word_count"].replace(0, pd.NA)
    df["qa_hedge_rate"] = df["qa_hedge_rate"].fillna(0.0)
    df["qa_risk_rate"] = df["qa_risk_rate"].fillna(0.0)

    return df
