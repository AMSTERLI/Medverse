"""Install the prompted full-CT preprocessor and trainers into nnU-Net v2."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import nnunetv2


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or digest(destination) != digest(source):
        shutil.copy2(source, destination)
    print(f"installed={destination} sha256={digest(destination)}")


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    package = Path(nnunetv2.__file__).resolve().parent
    install(
        project / "medverse" / "data" / "nnunet_prompted_fullct_preprocessor.py",
        package / "preprocessing" / "preprocessors" / "prompted_fullct_preprocessor.py",
    )
    install(
        project / "medverse" / "nnunet" / "nnUNetTrainerPromptedFullCT.py",
        package / "training" / "nnUNetTrainer" / "nnUNetTrainerPromptedFullCT.py",
    )


if __name__ == "__main__":
    main()
