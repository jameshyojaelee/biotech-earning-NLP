# Run tests: pytest -q
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from features.text_stats import compute_qa_text_features  # noqa: E402


def test_compute_qa_text_features_counts_terms():
    df = pd.DataFrame(
        {
            "qa_text": [
                "We may experience an FDA clinical hold and might face an adverse event.",
                "Clear outlook with no issues mentioned.",
            ]
        }
    )
    out = compute_qa_text_features(df)
    assert out.loc[0, "qa_hedge_terms"] >= 2  # may, might
    assert out.loc[0, "qa_risk_terms"] >= 2  # FDA, clinical hold, adverse event
    assert out.loc[1, "qa_hedge_terms"] == 0
    assert out.loc[1, "qa_risk_terms"] == 0


def test_hyphenated_and_multiword_terms():
    df = pd.DataFrame(
        {
            "qa_text": [
                "This is an FDA-approved therapy with enrollment delays and a clinical-hold risk.",
                "We received a complete response letter after the type C meeting; PDUFA date moved.",
            ]
        }
    )

    out = compute_qa_text_features(df)

    # Hyphenated terms and plural forms should match.
    assert out.loc[0, "qa_risk_terms"] >= 3  # fda-approved, enrollment delays, clinical hold
    # Multi-word phrases should match.
    assert out.loc[1, "qa_risk_terms"] >= 3  # complete response letter, type c meeting, pdufa date
