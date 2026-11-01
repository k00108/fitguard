#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def clean(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True
    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear outputs from packaged notebooks")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    count = sum(clean(path) for path in args.root.rglob("*.ipynb"))
    print(f"cleaned_notebooks={count}")


if __name__ == "__main__":
    main()
