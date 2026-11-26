"""Reuse Facet's portrait signals and add group-relative eye geometry."""

import atexit
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

import cv2
import numpy as np

from engine import technical_metrics
from paths import MODEL_FILES, load_facet, verify_models

load_facet()

from analyzers import face_blendshapes, face_permissive
from analyzers.composition import CompositionAnalyzer
from analyzers.image_cache import laplacian_variance
from analyzers.technical import TechnicalAnalyzer


_model_temp = None


def close_models():
    global _model_temp
    scorer = face_blendshapes._scorer
    if scorer and scorer._landmarker:
        scorer._landmarker.close()
        face_blendshapes._scorer = None
    if _model_temp is not None:
        _model_temp.cleanup()
        _model_temp = None


class FaceWithEyeGeometry(face_permissive.PermissiveFaceAnalyzer):
    def analyze_faces(self, image):
        self.last_eyes = []
        result = super().analyze_faces(image)
        result["eye_geometry"] = self.last_eyes[0] if len(self.last_eyes) == 1 else None
        return result

    def _mesh_analysis(self, image, bbox):
        result = super()._mesh_analysis(image, bbox)
        if result is None or result[3] is None:
            return result
        _, _, names, mesh = result

        def ratio(pts):
            a, b, c, d, e, f = mesh[list(pts)]
            return float((np.linalg.norm(b - f) + np.linalg.norm(c - e)) /
                         max(2 * np.linalg.norm(a - d), 1e-6))

        x1, y1, x2, y2 = bbox
        self.last_eyes.append({
            "aperture": [ratio((362, 385, 387, 263, 373, 380)),
                         ratio((33, 160, 158, 133, 153, 144))],
            "activation": [float(names["eyeBlinkLeft"]), float(names["eyeBlinkRight"])],
            "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
            "scale": float(max(x2 - x1, y2 - y1)),
        })
        return result


def face_analyzer():
    files = verify_models()
    face_permissive._ensure_weights = lambda: tuple(str(files / name) for name in list(MODEL_FILES)[:2])
    if not str(files).isascii():
        # The path-based ONNX loader can fail on Unicode Windows paths.
        face_permissive._create_detector = buffer_detector
        face_permissive._create_recognizer = buffer_recognizer
    landmarker_file = landmarker_path(files / "face_landmarker.task")
    landmarker = face_blendshapes.get_blendshape_scorer
    face_permissive.get_blendshape_scorer = lambda **kwargs: landmarker(
        **kwargs, model_path=str(landmarker_file)
    )
    analyzer = FaceWithEyeGeometry(min_confidence=0.5, min_face_size=20, enable_ofiq=False)
    if not analyzer.ensure_models():
        raise RuntimeError("Facet face models did not load. Check the output of scripts/setup_models.py.")
    return analyzer


def landmarker_path(model):
    """MediaPipe's Windows binary opens its task through an ASCII path."""
    global _model_temp
    if os.name != "nt" or str(model).isascii():
        return model
    if _model_temp is None:
        candidates = (Path(gettempdir()), Path(os.environ.get("PUBLIC", "C:/Users/Public")) / "Documents")
        for directory in candidates:
            if str(directory).isascii() and directory.is_dir():
                try:
                    _model_temp = TemporaryDirectory(dir=directory, prefix="quick-photo-pick-", ignore_cleanup_errors=True)
                    shutil.copyfile(model, Path(_model_temp.name) / model.name)
                    break
                except OSError:
                    if _model_temp is not None:
                        _model_temp.cleanup()
                        _model_temp = None
        if _model_temp is None:
            raise RuntimeError("MediaPipe requires an ASCII model or TEMP path on Windows. Set QUICK_PHOTO_PICK_MODELS or TEMP accordingly.")
    return Path(_model_temp.name) / model.name


def onnx_buffer(path):
    with open(path, "rb") as stream:
        return np.frombuffer(stream.read(), dtype=np.uint8)


def buffer_detector(path, size, score, nms, top_k):
    return cv2.FaceDetectorYN.create("onnx", onnx_buffer(path), np.empty(0, np.uint8), size, score, nms, top_k)


def buffer_recognizer(path):
    return cv2.FaceRecognizerSF.create("onnx", onnx_buffer(path), np.empty(0, np.uint8))


atexit.register(close_models)


def face_metrics(rgb, analyzer, *, include_iqa=False):
    face = analyzer.analyze_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    metrics = {
        "faces": face["face_count"],
        "eye_detail": float(face["raw_eye_sharpness"]) if face["face_count"] else None,
        "blink": bool(face["is_blink"]) if face["is_blink"] is not None and face["face_count"] else None,
        "eye_geometry": face["eye_geometry"] if face["face_count"] == 1 else None,
    }
    if include_iqa:
        metrics["iqa_faces"] = [
            {"box": detail["bbox"], "smile": None if detail["smile_score"] is None else float(detail["smile_score"])}
            for detail in face["face_details"]
        ]
    return metrics


def measure_row(rgb, analyzer, *, include_iqa=False):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    native = TechnicalAnalyzer.get_sharpness_data(bgr)
    metrics = technical_metrics(rgb) | face_metrics(rgb, analyzer, include_iqa=include_iqa)
    if include_iqa:
        metrics["iqa_laplacian"] = native["raw_variance"]
    return metrics | {"facet_sharpness": native["normalized"]}
