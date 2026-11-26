"""Optional per-frame IQA evidence. The default selection score is independent of it."""

import cv2
import numpy as np


def measure_iqa(rgb, gray, signals):
    from face_runtime import CompositionAnalyzer, laplacian_variance

    height, width = gray.shape
    stride = max(1, (max(height, width) + 255) // 256)
    pixels = rgb[::stride, ::stride]
    luma = gray[::stride, ::stride]
    low_chroma = (pixels.max(axis=2) - pixels.min(axis=2) <= 45) & (luma >= 32) & (luma <= 223)
    sample = pixels[low_chroma].astype(np.int16)
    cast = None
    if len(sample) >= max(32, 0.05 * luma.size):
        cast = {
            "red_minus_green": float(np.median(sample[:, 0] - sample[:, 1])),
            "blue_minus_green": float(np.median(sample[:, 2] - sample[:, 1])),
        }

    faces = signals.get("iqa_faces", [])
    subject = faces[0]["box"] if len(faces) == 1 else None
    source = "face_box" if subject is not None else None
    if not faces:
        subject = CompositionAnalyzer.detect_subject_region(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if subject is not None:
            source = "edge_contour"
    placement = None
    if subject is not None:
        placement = CompositionAnalyzer.get_placement_data(subject, width, height)
        # Expose the geometric measurements individually.
        placement = {key: placement[key] for key in ("power_point_score", "line_score", "center_score")}

    variance = signals.get("iqa_laplacian")
    return {
        "sharpness": variance if variance is not None else laplacian_variance(gray),
        "exposure": {
            "near_black_luma_share": signals["clipped_shadows"],
            "near_white_luma_share": signals["clipped_highlights"],
            "near_white_rgb_share": [float(v) for v in (rgb >= 253).mean(axis=(0, 1))],
            "median_luma": signals["luma_median"],
        },
        "white_balance": {"low_chroma_sample_share": float(low_chroma.mean()), "channel_offsets": cast},
        "composition": {"subject_box_px": subject, "source": source, "placement_proxies": placement},
        "expression": {"smile_signals": [face["smile"] for face in faces]},
    }


def iqa_report(measured, raw_rank, blink, portrait):
    observations = measured["iqa_observations"]
    return {
        "narrow": {
            "relative_sharpness": {
                "status": "validated_on_dataset",
                "scope": "camera_shake_and_mixed_synthetic_degradation_pairs",
                "laplacian_variance": observations["sharpness"],
                "rank_in_group": raw_rank,
            },
            "motion_blur_attribution": {
                "status": "possibly_effective", "edge_directionality": measured["edge_directionality"],
            },
            "missed_focus": {
                "status": "possibly_effective", "eye_region_sharpness": measured.get("eye_detail"),
            },
            "exposure_error": {"status": "possibly_effective", **observations["exposure"]},
            "white_balance_error": {"status": "possibly_effective", **observations["white_balance"]},
        },
        "broad": {
            "blink": {
                "status": "validated_on_dataset",
                "scope": "single_subject_full_closure_with_known_group",
                "closed": blink if portrait else None,
                "single_frame_closed": measured.get("blink") if portrait else None,
            },
            "composition": {"status": "possibly_effective", **observations["composition"]},
            "expression_quality": {"status": "possibly_effective", **observations["expression"]},
            "overall_quality": {"status": "possibly_effective", "score": None},
        },
    }
