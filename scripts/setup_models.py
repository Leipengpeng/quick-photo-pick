"""Fetch the tested Facet revision and verify official portrait weights."""

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.request import urlopen


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "burst_engine"))
from paths import FACET, FACET_ARCHIVE, FACET_COMMIT, MODEL_DIR, MODEL_FILES, sha256_file


def facet_revision(directory):
    return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()


def setup_facet():
    if FACET.exists():
        marker = FACET / ".quick-photo-pick-revision"
        if marker.is_file() and marker.read_text(encoding="ascii").strip() == FACET_COMMIT:
            return
        if (FACET / ".git").exists() and facet_revision(FACET) == FACET_COMMIT:
            return
        raise RuntimeError(f"Facet checkout {FACET} must be at {FACET_COMMIT}.")

    FACET.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=FACET.parent) as temp:
        temp = Path(temp)
        archive = temp / "facet.tar.gz"
        print(f"Downloading pinned Facet {FACET_COMMIT[:7]}", flush=True)
        with urlopen(FACET_ARCHIVE, timeout=60) as response, archive.open("wb") as target:
            shutil.copyfileobj(response, target)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(temp, filter="data")
        checkout = temp / f"facet-{FACET_COMMIT}"
        if not (checkout / "analyzers" / "face_permissive.py").is_file():
            raise RuntimeError("The pinned Facet archive has an unexpected layout.")
        (checkout / ".quick-photo-pick-revision").write_text(FACET_COMMIT + "\n", encoding="ascii")
        checkout.rename(FACET)


def setup_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in MODEL_FILES.items():
        model = MODEL_DIR / name
        if model.is_file() and sha256_file(model) == expected:
            print(f"Verified {name}", flush=True)
            continue
        print(f"Downloading {name}", flush=True)
        with NamedTemporaryFile(dir=MODEL_DIR, suffix=".part", delete=False) as opened:
            partial = Path(opened.name)
        try:
            with urlopen(url, timeout=60) as response, partial.open("wb") as target:
                shutil.copyfileobj(response, target)
            if sha256_file(partial) != expected:
                raise RuntimeError(f"Checksum mismatch for {name}.")
            partial.replace(model)
        finally:
            partial.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        setup_facet()
        setup_models()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))
    print("Facet and the three verified models are ready.")
