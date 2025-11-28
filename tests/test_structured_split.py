# Run tests: pytest -q
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from preprocess.structured_split import extract_sections  # noqa: E402


def test_structured_split_uses_operator_as_boundary():
    segments = [
        {"speaker_role": "CEO", "text": "Welcome to the call."},
        {"speaker_role": "CFO", "text": "Here are the results."},
        {"speaker_role": "Operator", "text": "We will now begin the Q&A."},
        {"speaker_role": "Analyst", "text": "Can you talk about guidance?"},
        {"speaker_role": "CEO", "text": "Sure, happy to."},
    ]

    record = pd.Series({"segments": segments, "transcript": ""})
    prepared, qa = extract_sections(record)

    assert "Welcome" in prepared and "results" in prepared
    assert "guidance" in qa and "happy" in qa
    assert "Operator" not in prepared


def test_structured_split_falls_back_to_regex():
    text = "Prepared remarks. Q&A Analyst: Hello there."
    record = pd.Series({"transcript": text})
    prepared, qa = extract_sections(record)

    assert "Prepared remarks" in prepared
    assert "Analyst" in qa


def test_speaker_name_detection():
    segments = [
        {"speaker_name": "Operator", "text": "We will start questions."},
        {"speaker_name": "Analyst One", "text": "Question here."},
        {"speaker_name": "CFO", "text": "Answer here."},
    ]

    record = pd.Series({"segments": segments, "transcript": ""})
    prepared, qa = extract_sections(record)

    assert prepared == ""
    assert "Question" in qa and "Answer" in qa
