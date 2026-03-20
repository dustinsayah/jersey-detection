from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external"
PUBLIC_MODELS_ROOT = PROJECT_ROOT / "app" / "model" / "public"

REPOS: tuple[tuple[str, str, Path], ...] = (
    (
        "uncertainty-jnr",
        "https://github.com/lukaszgrad/uncertainty-jnr.git",
        EXTERNAL_ROOT / "uncertainty-jnr",
    ),
    (
        "jersey-number-pipeline",
        "https://github.com/mkoshkina/jersey-number-pipeline.git",
        EXTERNAL_ROOT / "jersey-number-pipeline",
    ),
)

WEIGHTS: tuple[tuple[str, str, Path], ...] = (
    (
        "Grad ViT-B SoccerNet checkpoint",
        "https://drive.google.com/file/d/16npJY-gyboRE_HNTQI1dC_fIxh3oxa0S/view?usp=drive_link",
        PUBLIC_MODELS_ROOT / "uncertainty_jnr_vitb.pth",
    ),
    (
        "Koshkina SoccerNet legibility checkpoint",
        "https://drive.google.com/uc?id=18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw",
        PUBLIC_MODELS_ROOT / "koshkina_legibility_soccer.pth",
    ),
    (
        "Koshkina SoccerNet PARSeq checkpoint",
        "https://drive.google.com/uc?id=1uRln22tlhneVt3P6MePmVxBWSLMsL3bm",
        PUBLIC_MODELS_ROOT / "koshkina_parseq_soccer.ckpt",
    ),
)


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(command)
    print(f"+ {printable}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def _require_command(name: str, *, reason: str) -> None:
    if shutil.which(name):
        return
    raise RuntimeError(f"Missing required command '{name}' ({reason}).")


def _clone_or_update_repo(
    *,
    name: str,
    url: str,
    destination: Path,
    update_existing: bool,
    shallow: bool,
) -> None:
    if destination.exists():
        if not (destination / ".git").exists():
            raise RuntimeError(
                f"{destination} already exists but is not a git checkout for {name}."
            )
        if update_existing:
            _run(["git", "-C", str(destination), "pull", "--ff-only"])
        else:
            print(f"Skipping existing repo: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["git", "clone"]
    if shallow:
        command.extend(["--depth", "1"])
    command.extend([url, str(destination)])
    _run(command)


def _download_weight(
    *,
    name: str,
    url: str,
    destination: Path,
    overwrite: bool,
) -> None:
    try:
        import gdown
    except Exception as error:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "gdown is required to download public checkpoints. "
            "Install requirements.txt first."
        ) from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(f"{destination.suffix}.part")
    if overwrite:
        destination.unlink(missing_ok=True)
        temp_path.unlink(missing_ok=True)
    elif destination.exists():
        print(f"Skipping existing weight: {destination}")
        return

    print(f"+ download {name} -> {destination}")
    downloaded = gdown.download(
        url,
        str(temp_path),
        quiet=False,
        fuzzy=True,
        resume=temp_path.exists() and not overwrite,
    )
    if not downloaded or not temp_path.exists():
        raise RuntimeError(f"Failed to download {name} from {url}")
    temp_path.replace(destination)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone external repos and download public-reader checkpoints.",
    )
    parser.add_argument(
        "--skip-repos",
        action="store_true",
        help="Do not clone or update the external research repos.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Do not download the public-reader model checkpoints.",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Leave existing repo checkouts untouched instead of pulling.",
    )
    parser.add_argument(
        "--no-shallow",
        action="store_true",
        help="Use full git clones instead of --depth 1.",
    )
    parser.add_argument(
        "--overwrite-models",
        action="store_true",
        help="Redownload model checkpoints even if files already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.skip_repos:
        _require_command("git", reason="needed to clone/update external research repos")
        for name, url, destination in REPOS:
            _clone_or_update_repo(
                name=name,
                url=url,
                destination=destination,
                update_existing=not args.no_update,
                shallow=not args.no_shallow,
            )

    if not args.skip_models:
        for name, url, destination in WEIGHTS:
            _download_weight(
                name=name,
                url=url,
                destination=destination,
                overwrite=args.overwrite_models,
            )

    print("Public-reader bootstrap complete.")
    print("Next step: install dependencies with `pip install -r requirements.txt` if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
