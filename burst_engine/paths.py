"""Resolve external Facet code and verify the three runtime models."""

import hashlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / ".runtime"
FACET = Path(os.environ.get("QUICK_PHOTO_PICK_FACET", RUNTIME / "facet")).resolve()
MODEL_DIR = Path(os.environ.get("QUICK_PHOTO_PICK_MODELS", RUNTIME / "models")).resolve()
FACET_COMMIT = "f7d22dc807b1e106381858d60612c1d3b27f08dc"
FACET_ARCHIVE = f"https://codeload.github.com/ncoevoet/facet/tar.gz/{FACET_COMMIT}"
MODEL_FILES = {
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/f12e12798e8314f7c074a6656816c048dcc95b7a/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/ba91a3b91d00d76e86540d4013f944bd6b514e39/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
}


def load_facet():
    if not (FACET / "analyzers" / "face_permissive.py").is_file():
        raise RuntimeError("Facet is missing. Run python scripts/setup_models.py first.")
    if str(FACET) not in sys.path:
        sys.path.append(str(FACET))


def sha256_file(path):
    with path.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def verify_models():
    for name, (_, expected) in MODEL_FILES.items():
        model = MODEL_DIR / name
        if not model.is_file() or sha256_file(model) != expected:
            raise RuntimeError(f"Invalid or missing model {model.name}. Run python scripts/setup_models.py.")
    return MODEL_DIR
