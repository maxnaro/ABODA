"""Load and save ground-truth annotation JSON files."""

import json
from pathlib import Path


def load_annotations(path: Path) -> dict:
    """Load existing annotations from a JSON file, or return an empty dict."""
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_annotations(path: Path, data: dict):
    """Write annotations to a JSON file and print a confirmation."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved annotations to {path}")
