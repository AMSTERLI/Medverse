"""Snapshot nnU-Net v2 planner output for reuse by the Medverse arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-spacing", action="store_true")
    args = parser.parse_args()
    plans = json.loads(args.plans.read_text(encoding="utf-8"))
    try:
        configuration = plans["configurations"][args.configuration]
        spacing = [float(value) for value in configuration["spacing"]]
        patch_size = [int(value) for value in configuration["patch_size"]]
    except KeyError as exc:
        raise KeyError(f"missing nnU-Net configuration field: {exc}") from exc
    snapshot = {
        "planner": "nnU-Net v2",
        "configuration": args.configuration,
        "spacing_xyz_mm": spacing,
        "nnunet_patch_size": patch_size,
        "nnunet_batch_size": configuration.get("batch_size"),
        "plans_file": str(args.plans),
        "plans_sha256": sha256(args.plans),
        "fingerprint_file": str(args.fingerprint),
        "fingerprint_sha256": sha256(args.fingerprint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    if args.print_spacing:
        print(",".join(str(value) for value in spacing))
    else:
        print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
