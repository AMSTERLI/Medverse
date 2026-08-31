"""Create a formal-training gate only after all tiny-set cases overfit."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--minimum-mean-dice", type=float, default=0.80)
    parser.add_argument("--minimum-case-dice", type=float, default=0.60)
    parser.add_argument("--expected-cases", type=int, default=8)
    args = parser.parse_args()
    with args.metrics_csv.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    dice = [float(row["dice"]) for row in rows]
    report = {
        "cases": len(rows), "mean_dice": sum(dice) / len(dice) if dice else 0.0,
        "minimum_dice": min(dice) if dice else 0.0,
        "required_cases": args.expected_cases, "required_mean_dice": args.minimum_mean_dice,
        "required_minimum_case_dice": args.minimum_case_dice,
    }
    passed = (
        len(rows) == args.expected_cases and report["mean_dice"] >= args.minimum_mean_dice
        and report["minimum_dice"] >= args.minimum_case_dice
    )
    report["passed"] = passed
    if not passed:
        print(json.dumps(report, indent=2))
        raise SystemExit(6)
    args.gate.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.gate.with_suffix(args.gate.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, args.gate)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
