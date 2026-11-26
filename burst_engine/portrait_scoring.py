"""Explainable technical scoring for one burst, using the local Facet face pass.

Scalars are relative to the current burst. They express technical preference
and have no calibrated interpretation across bursts, cameras, or genres.
"""

from math import isfinite, log1p, tanh
from statistics import median

import cv2
import numpy as np

from engine import contextual_blinks, technical_metrics
from iqa import iqa_report, measure_iqa


_PROFILES = {
    "portrait": {"detail": 0.48, "eye_focus": 0.42, "exposure": 0.08, "motion": 0.02},
    "general": {"detail": 0.87, "exposure": 0.10, "motion": 0.03},
    "landscape": {"detail": 0.83, "exposure": 0.15, "motion": 0.02},
    "street": {"detail": 0.86, "exposure": 0.11, "motion": 0.03},
}
_MAX_DIRECTION_EDGE = 384


def measure_signals(rgb, analyzer=None, *, include_iqa=False):
    """Measure one uint8 RGB frame. Reuse one initialized analyzer serially.

    An unavailable face pass leaves the face measurements unknown. Invalid
    image arrays raise ValueError instead of producing usable-looking scores.
    """
    if (not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 3 or rgb.dtype != np.uint8):
        raise ValueError("scoring requires a uint8 RGB image of at least 3×3 pixels")

    row = {
        "faces": 0, "blink": None, "eye_detail": None, "eye_geometry": None,
        "face_analysis_available": False, "face_analysis_error": None,
    }
    if analyzer is not None:
        # Run Facet once. Its eye sharpness uses the eye region selected by
        # YuNet/MediaPipe and excludes most of the background.
        try:
            from face_runtime import measure_row

            row.update(measure_row(rgb, analyzer, include_iqa=include_iqa))
            row["face_analysis_available"] = True
        except (cv2.error, RuntimeError, OSError) as exc:
            row.update(technical_metrics(rgb))
            row["face_analysis_error"] = str(exc)
    else:
        row.update(technical_metrics(rgb))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    row["clipped_shadows"] = float(histogram[:3].sum() / gray.size)
    row["clipped_highlights"] = float(histogram[253:].sum() / gray.size)
    row["luma_median"] = int(np.searchsorted(np.cumsum(histogram), (gray.size + 1) // 2))
    if include_iqa:
        row["iqa_observations"] = measure_iqa(rgb, gray, row)
        row.pop("iqa_faces", None)
        row.pop("iqa_laplacian", None)

    h, w = gray.shape
    if max(h, w) > _MAX_DIRECTION_EDGE:
        scale = _MAX_DIRECTION_EDGE / max(h, w)
        gray = cv2.resize(gray, (max(3, round(w * scale)), max(3, round(h * scale))),
                          interpolation=cv2.INTER_AREA)
    filtered = cv2.GaussianBlur(gray, (3, 3), 0.8)
    gx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1)
    xx, yy, xy = float(np.mean(gx * gx)), float(np.mean(gy * gy)), float(np.mean(gx * gy))
    # Edge anisotropy is a weak cue. Panning, repeated lines, and composition
    # can produce it. Scoring uses it only relative to comparable burst frames.
    row["edge_directionality"] = (
        min(1.0, ((xx - yy) ** 2 + 4 * xy ** 2) ** 0.5 / (xx + yy))
        if xx + yy else None
    )
    return row


def _number(value):
    if value is None:
        return None
    value = float(value)
    return value if isfinite(value) else None


def _relative(values, scale):
    observed = [value for value in values if value is not None]
    if not observed:
        return [None] * len(values)
    reference = median(observed)
    return [None if value is None else 0.5 + 0.5 * tanh((value - reference) / scale)
            for value in values]


def _motion_preferences(directions, details):
    observed = [value for value in directions if value is not None]
    shared = [value for value in details if value is not None]
    if len(observed) < 2 or len(shared) < 2:
        return [None] * len(directions)
    direction_reference = median(observed)
    detail_reference = median(shared)
    preferences = []
    for direction, detail in zip(directions, details):
        if detail is None or direction is None:
            preferences.append(None)
            continue
        anisotropy = max(0.0, (direction - direction_reference - 0.08) / 0.3)
        softness = max(0.0, (detail_reference - detail - 0.06) / 0.4)
        preferences.append(0.5 * (1.0 - min(1.0, anisotropy, softness)))
    return preferences


def score_group(rows, profile="auto"):
    """Return ranked records and technical reasons for an existing group.

    Index addresses the supplied frame row. Each percentage-like component
    is relative to group peers; None means the observation is unavailable.
    The profile can be auto, portrait, general, landscape, or street.
    """
    if profile != "auto" and profile not in _PROFILES:
        raise ValueError(f"unknown scoring profile: {profile}")
    if not rows:
        return []
    if profile == "auto":
        face_frames = sum(int(row.get("faces") or 0) > 0 for row in rows)
        profile = "portrait" if face_frames and 2 * face_frames >= len(rows) else "general"
    weights = _PROFILES[profile]
    portrait = profile == "portrait"
    details = [_number(row.get("detail")) for row in rows]
    if any(value is None for value in details):
        raise ValueError("one or more frames have no valid detail measurement")
    eyes = [_number(row.get("eye_detail")) for row in rows]
    eyes = [log1p(value) if value is not None and value >= 0 else None for value in eyes]
    clipping = [_number(row.get("clip")) for row in rows]
    directions = [_number(row.get("edge_directionality")) for row in rows]
    by_signal = {
        "detail": _relative(details, 0.30),
        "eye_focus": _relative(eyes, 0.72),
        # Bright and dark styles can both be intentional. The weak exposure
        # preference measures clipping, with no preferred average luminance.
        "exposure": _relative([None if v is None else -v for v in clipping], 0.16),
        "motion": _motion_preferences(directions, details),
    }
    observations = [
        {"blink": row.get("blink"), "faces": row.get("faces") or 0,
         "eye_geometry": row.get("eye_geometry")}
        for row in rows
    ]
    states = contextual_blinks(observations) if portrait else [row["blink"] for row in observations]
    any_open = any(state is False for state in states)
    iqa_values = [row["iqa_observations"]["sharpness"] for row in rows if "iqa_observations" in row]
    iqa_ranks = {}
    for rank, value in enumerate(sorted(iqa_values, reverse=True), 1):
        iqa_ranks.setdefault(value, rank)
    graded = []
    for i, (row, blink) in enumerate(zip(rows, states)):
        used = {key: by_signal[key][i] for key in weights}
        active = {key: weight for key, weight in weights.items() if used[key] is not None}
        total = sum(active.values())
        contributions = {key: weight * used[key] / total for key, weight in active.items()}
        technical_score = sum(contributions.values())
        review = []
        reasons = []
        if portrait:
            if blink is True:
                reasons.append("eyes_closed_by_model" if row.get("blink") is True else "eyes_closed_relative_to_burst")
                score = 0.30 * technical_score
            elif blink is None:
                review.append("eye_state_unverified")
                score = 0.34 + 0.29 * technical_score
            else:
                reasons.append("eyes_open")
                score = 0.67 + 0.33 * technical_score
            if used["eye_focus"] is not None and used["eye_focus"] < 0.30:
                reasons.append("eye_detail_lower_than_peers")
            if not any_open:
                review.append("no_verified_open_eye_frame")
        else:
            score = technical_score
        if used["detail"] < 0.30:
            reasons.append("frame_detail_lower_than_peers")
        if used["exposure"] is not None and used["exposure"] < 0.30:
            reasons.append("more_clipping_than_peers")
        if used["motion"] is not None and used["motion"] < 0.40:
            reasons.append("directional_softness_cue")
        if row.get("face_analysis_error"):
            review.append("face_analysis_failed")
        result = {
            "index": i,
            "profile": profile,
            "score": round(100 * score, 4),
            "technical_score": round(100 * technical_score, 4),
            "components": {key: None if value is None else round(100 * value, 4)
                           for key, value in used.items()},
            "contributions": {key: round(100 * value, 4) for key, value in contributions.items()},
            "effective_blink": blink,
            "review": bool(review),
            "review_reasons": review,
            "reasons": reasons,
            "_unrounded_score": score,
        }
        if "iqa_observations" in row:
            raw_rank = iqa_ranks[row["iqa_observations"]["sharpness"]]
            result["iqa"] = iqa_report(row, raw_rank, blink, portrait)
        graded.append(result)
    graded.sort(key=lambda item: item["_unrounded_score"], reverse=True)
    for result in graded:
        del result["_unrounded_score"]
    return graded
