"""Install the versioned mixed-channel preprocessor into the active nnU-Net package."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import nnunetv2


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "medverse"
        / "data"
        / "nnunet_mixed_channel_preprocessor.py",
    )
    args = parser.parse_args()
    destination = (
        Path(nnunetv2.__file__).resolve().parent
        / "preprocessing"
        / "preprocessors"
        / "mixed_channel_preprocessor.py"
    )
    if not destination.is_file() or digest(destination) != digest(args.source):
        shutil.copy2(args.source, destination)
    print(f"installed={destination} sha256={digest(destination)}")


if __name__ == "__main__":
    main()
