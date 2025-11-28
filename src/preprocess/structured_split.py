"""Split transcripts using structured speaker metadata when available."""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd

from .transcript_splitter import split_prepared_and_qa


MANAGEMENT_TITLES = (
    "ceo",
    "cfo",
    "coo",
    "cio",
    "cto",
    "president",
    "vice president",
    "vp",
    "chairman",
    "chief",
    "executive",
    "founder",
    "investor relations",
    "ir",
)


def _normalize_segments(segments: Iterable[object]) -> pd.DataFrame:
    df = pd.DataFrame(segments)
    if df.empty:
        return df

    text_col = None
    for candidate in ["text", "content", "segment_text", "body"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col and "text" not in df.columns:
        df["text"] = df[text_col]
    df["text"] = df.get("text", pd.Series([None] * len(df), index=df.index)).fillna("")

    if "speaker_role" not in df.columns:
        for candidate in ["speaker", "speaker_name", "role", "speaker_title"]:
            if candidate in df.columns:
                df["speaker_role"] = df[candidate]
                break
    df["speaker_role"] = df.get("speaker_role", pd.Series([None] * len(df), index=df.index)).fillna("")

    return df


def _classify_role(role: str) -> str:
    role_lower = role.lower()
    if "analyst" in role_lower:
        return "analyst"
    if "operator" in role_lower:
        return "operator"
    if any(title in role_lower for title in MANAGEMENT_TITLES):
        return "management"
    return "other"


def _split_by_segments(segments: Iterable[object]) -> Tuple[str, str]:
    df = _normalize_segments(segments)
    prepared_parts, qa_parts = [], []
    qa_started = False

    for _, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        role = str(row.get("speaker_role", ""))
        role_type = _classify_role(role)

        if role_type == "analyst":
            qa_started = True
            qa_parts.append(text)
            continue

        if role_type == "operator":
            if prepared_parts:
                qa_started = True
            # Operator remarks mark boundaries; skip adding to the text blocks.
            continue

        if qa_started:
            qa_parts.append(text)
        else:
            prepared_parts.append(text)

    return "\n".join(prepared_parts).strip(), "\n".join(qa_parts).strip()


def extract_sections(record) -> Tuple[str, str]:
    """Return (prepared_text, qa_text) using structured segments when available."""
    segments = None
    if hasattr(record, "get"):
        segments = record.get("segments")
    elif isinstance(record, dict):
        segments = record.get("segments")

    if isinstance(segments, pd.DataFrame):
        if not segments.empty:
            return _split_by_segments(segments)
    elif isinstance(segments, Iterable) and not isinstance(segments, (str, bytes)):
        segments_list = list(segments)
        if segments_list:
            return _split_by_segments(segments_list)

    transcript = ""
    if hasattr(record, "get"):
        transcript = record.get("transcript", "")
    elif isinstance(record, dict):
        transcript = record.get("transcript", "")
    return split_prepared_and_qa(transcript)
