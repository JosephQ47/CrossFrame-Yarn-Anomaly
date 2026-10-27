"""Inventory timestamp-named local July source candidates without relabeling them."""

import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent
DATASET = Path(r"D:\dataset")
CANONICAL = DATASET / "black_fabric"
OUT = ROOT / "data/black_local_history_inventory_2026-10-27.json"
RAW_STYLE = re.compile(r"^DI2_Cam[1-6]_(\d{4}-\d\d-\d\d)_.*\.jpg$", re.IGNORECASE)


def scan(root):
    matches = []
    for directory, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in {".git", "node_modules", ".venv"}]
        for name in files:
            if RAW_STYLE.match(name):
                matches.append((name, str(Path(directory) / name)))
    return matches


def main():
    research = scan(RESEARCH)
    dataset = scan(DATASET)
    canonical_names = {path.name for path in CANONICAL.glob("DI2_Cam*.jpg")}
    july_research = [name for name, _ in research if "2026-07-18" in name]
    july_dataset = [name for name, _ in dataset if "2026-07-18" in name]
    other_date_dataset = Counter(RAW_STYLE.match(name).group(1)
                                 for name, _ in dataset if "2026-07-18" not in name)
    result = {
        "scope": "filename inventory only; same-state normal status cannot be inferred from filename",
        "roots": [str(RESEARCH), str(DATASET)],
        "july_research_path_count": len(july_research),
        "july_research_distinct_names": len(set(july_research)),
        "july_dataset_path_count": len(july_dataset),
        "july_dataset_distinct_names": len(set(july_dataset)),
        "canonical_black_fabric_names": len(canonical_names),
        "july_raw_style_names_not_in_canonical": sorted(
            (set(july_research) | set(july_dataset)) - canonical_names),
        "other_date_dataset_raw_style_names": dict(sorted(other_date_dataset.items())),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
