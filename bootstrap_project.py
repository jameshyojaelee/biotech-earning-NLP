"""Bootstrap the earnings NLP project structure.

This script creates the directory tree, placeholder files, and config needed to
run the biotech earnings-call NLP workflow without overwriting any existing
files. Run from the repository root: `python bootstrap_project.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def ensure_directories(base: Path, dirs: Iterable[Path]) -> None:
    """Create each directory if it does not already exist."""
    for d in dirs:
        full = base / d
        full.mkdir(parents=True, exist_ok=True)
        print(f"created dir: {full}")


def write_file_if_missing(path: Path, content: str) -> None:
    """Write `content` only when the file is absent to avoid overwriting."""
    if path.exists():
        print(f"skip existing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote file: {path}")


def main(base_dir: str = ".") -> None:
    base = Path(base_dir)

    dirs = [
        Path("config"),
        Path("data_raw"),
        Path("data_processed"),
        Path("notebooks"),
        Path("src/ingest"),
        Path("src/preprocess"),
        Path("src/features"),
        Path("src/finance"),
        Path("src/analysis"),
        Path("tests"),
    ]

    files = {
        Path("README.md"): "# Earnings NLP for Biotech\n",
        Path("requirements.txt"): "pandas\n",
        Path("pyproject.toml"): "[project]\nname = \"earnings-nlp\"\nversion = \"0.1.0\"\n",
        Path("config/config.yaml"): "hf_dataset_name: glopardo/sp500-earnings-transcripts\n",
        Path("notebooks/01_exploration.ipynb"): "{}",  # placeholder minimal JSON
        Path("src/__init__.py"): "",
        Path("src/ingest/__init__.py"): "",
        Path("src/preprocess/__init__.py"): "",
        Path("src/features/__init__.py"): "",
        Path("src/finance/__init__.py"): "",
        Path("src/analysis/__init__.py"): "",
        Path("tests/__init__.py"): "",
    }

    ensure_directories(base, dirs)

    for rel_path, content in files.items():
        write_file_if_missing(base / rel_path, content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap earnings NLP project structure without overwriting existing files.")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Root directory for the project structure (default: current directory)",
    )
    args = parser.parse_args()
    main(args.base_dir)
