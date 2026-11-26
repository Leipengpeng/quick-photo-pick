"""Run local metadata-aware grouping, technical scoring, and review export."""

import csv
import json
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import rawpy

from face_runtime import face_analyzer, laplacian_variance
from engine import make_preview, read_image, read_image_variants
from grouping_workflow import RAW_EXTENSIONS, build_capture_groups
from portrait_scoring import measure_signals, score_group


PROFILES = ("auto", "portrait", "general", "landscape", "street")


def scene_preview_indices(previews):
    """Choose temporal representatives plus the sharpest unused preview."""
    if not previews:
        return []
    primary = sorted({0, len(previews) // 2})
    if len(previews) > len(primary):
        remaining = (index for index in range(len(previews)) if index not in primary)
        primary.append(max(remaining, key=lambda index: laplacian_variance(previews[index].gray)))
    return primary


def _frame_result(frame, versions, tags, signals, score, status, error=None):
    record = {
        "path": str(frame.path), "versions": [str(path) for path in versions],
        "timestamp": tags.get("DateTimeOriginal"), "camera": frame.camera,
        "lens": frame.lens, "focal_length": frame.focal_length,
        "rank": None, "decision": status,
        "recommended": status in ("select", "preserve_sequence"),
        "score": score["score"] if score else None,
        "technical_score": score["technical_score"] if score else None,
        "profile": score["profile"] if score else None,
        "components": score["components"] if score else {},
        "contributions": score["contributions"] if score else {},
        "blink": score["effective_blink"] if score else None,
        "reasons": (list(score["reasons"]) + score["review_reasons"]) if score else [],
        "metrics": signals,
        "error": error,
    }
    if score and "iqa" in score:
        record["iqa"] = score["iqa"]
        record["metrics"] = {key: value for key, value in signals.items() if key != "iqa_observations"}
    return record


def run_selection(
    paths, tags, *, profile="auto", keepers=1, semantic_review=None, review_budget=0, iqa_report=False,
    scene_clustering=True,
):
    if profile not in PROFILES:
        raise ValueError(f"unsupported scoring profile: {profile}")
    if keepers < 1:
        raise ValueError("keepers must be at least one")

    analyzer = None
    scoring_image = None
    per_frame_signals = {}
    shared_decode_count = 0
    early_measurement_seconds = 0.0
    scene_feature_seconds = 0.0
    grouped_scene_features = {}
    scene_representatives = {}
    if scene_clustering:
        from scene_clustering import SceneFeatureExtractor, cluster_scenes

        extract_scene_features = SceneFeatureExtractor()

    def describe_representative(path, preview):
        nonlocal scene_feature_seconds
        begun = perf_counter()
        features = extract_scene_features(path, preview)
        scene_feature_seconds += perf_counter() - begun
        return features

    def keep_group_previews(group, previews):
        grouped_scene_features[group] = [
            describe_representative(*previews[index])
            for index in scene_preview_indices([preview for _, preview in previews])
        ]

    def measure(image):
        nonlocal analyzer
        if analyzer is None:
            analyzer = face_analyzer()
        if iqa_report:
            return measure_signals(image, analyzer, include_iqa=True)
        return measure_signals(image, analyzer)

    def prepare_preview(path):
        nonlocal scoring_image
        if path.suffix.lower() not in RAW_EXTENSIONS:
            return make_preview(read_image(path, 640))
        preview_image, image = read_image_variants(path, 640, 960)
        preview = make_preview(preview_image)
        scoring_image = image
        return preview

    def measure_shared_decode(path):
        nonlocal scoring_image, shared_decode_count, early_measurement_seconds
        if scoring_image is None:
            return
        started = perf_counter()
        per_frame_signals[path] = measure(scoring_image)
        scoring_image = None
        shared_decode_count += 1
        early_measurement_seconds += perf_counter() - started

    start = perf_counter()
    grouped = build_capture_groups(
        paths, tags, preview_factory=prepare_preview, on_preview_read=measure_shared_decode,
        on_group_previews=keep_group_previews if scene_clustering else None,
        semantic_review=semantic_review, review_budget=review_budget,
    )
    grouping_seconds = perf_counter() - start - early_measurement_seconds - scene_feature_seconds
    warnings = list(grouped.warnings)
    boundary_paths = {path for pair in grouped.review_boundaries for path in pair}
    output = []
    decisions = Counter()
    for number, group in enumerate(grouped.groups, 1):
        valid = []
        rejected_reads = []
        representatives = grouped_scene_features.pop(group, None)
        needs_scene_previews = scene_clustering and representatives is None and not group.preserve_all
        sharpest_preview = None
        if representatives is None:
            representatives = []
        for index, frame in enumerate(group.frames):
            versions = grouped.versions[frame.path]
            analyzed_path = grouped.preview_sources.get(frame.path, frame.path)
            if analyzed_path in per_frame_signals:
                valid.append((frame, per_frame_signals.pop(analyzed_path), analyzed_path))
                continue
            try:
                try:
                    image = read_image(analyzed_path, 960)
                except (OSError, ValueError, RuntimeError, rawpy.LibRawError):
                    if analyzed_path != frame.path or len(versions) < 2:
                        raise
                    analyzed_path = versions[1]
                    image = read_image(analyzed_path, 960)
            except (OSError, ValueError, RuntimeError, rawpy.LibRawError) as exc:
                warnings.append(f"{frame.path.name} 无法读取，已保留为待复核")
                rejected_reads.append(_frame_result(
                    frame, versions, grouped.tags[frame.path], None, None,
                    "review", f"{type(exc).__name__}: {exc}",
                ))
                continue
            if needs_scene_previews:
                preview = make_preview(image)
                if not representatives or (index >= len(group.frames) // 2 and len(representatives) == 1):
                    representatives.append(describe_representative(analyzed_path, preview))
                else:
                    detail = laplacian_variance(preview.gray)
                    if sharpest_preview is None or detail > sharpest_preview[0]:
                        sharpest_preview = (detail, analyzed_path, preview)
            valid.append((frame, measure(image), analyzed_path))
        if scene_clustering:
            if sharpest_preview is not None:
                _, path, preview = sharpest_preview
                representatives.append(describe_representative(path, preview))
            scene_representatives[number] = representatives

        result_rows = []
        signals = [row for _, row, _ in valid]
        ranked = score_group(signals, profile=profile) if signals else []
        case_requires_review = any(frame.path in boundary_paths for frame in group.frames)
        selected_count = 0
        for rank, candidate in enumerate(ranked, 1):
            frame, measured, analyzed_path = valid[candidate["index"]]
            if group.preserve_all:
                decision = "preserve_sequence"
            elif candidate["review"] or (case_requires_review and rank <= keepers):
                decision = "review"
            elif not case_requires_review and selected_count < keepers and not (
                candidate["profile"] == "portrait" and candidate["effective_blink"] is True
            ):
                decision = "select"
                selected_count += 1
            else:
                decision = "lower_priority"
            row = _frame_result(
                frame, grouped.versions[frame.path], grouped.tags[frame.path], measured, candidate, decision,
            )
            row["rank"] = rank
            row["analyzed_path"] = str(analyzed_path)
            if analyzed_path != frame.path:
                row["reasons"].append("used_paired_jpeg_after_raw_decode_failure")
            if frame.path in boundary_paths:
                row["reasons"].append("uncertain_group_boundary")
            result_rows.append(row)
        result_rows.extend(rejected_reads)
        if result_rows and all(not row["recommended"] for row in result_rows):
            if result_rows[0]["blink"] is True:
                result_rows[0]["reasons"].append("no_eligible_open_eye_frame")
            result_rows[0]["decision"] = "review"
        decisions.update(row["decision"] for row in result_rows)
        output.append({
            "group": number,
            "basis": group.basis,
            "keep_as_sequence": group.preserve_all,
            "uncertain_boundary": case_requires_review,
            "frames": result_rows,
        })
    scene_result = {}
    timings = {"grouping": round(grouping_seconds, 3)}
    if scene_clustering:
        scene_start = perf_counter()
        scene_result = cluster_scenes(scene_representatives)
        timings["scene_clustering"] = round(perf_counter() - scene_start + scene_feature_seconds, 3)
    timings["total"] = round(perf_counter() - start, 3)
    result = {
        "groups": output,
        "capture_frames": sum(len(group.frames) for group in grouped.groups),
        "source_files": len(paths),
        "untimed_singletons": sum(group.basis == "missing_timestamp" for group in grouped.groups),
        "profile": profile, "keepers_per_group": keepers,
        "decisions": dict(decisions),
        "review_boundaries": [[str(path) for path in pair] for pair in grouped.review_boundaries],
        "warnings": warnings,
        "geometry_pairs": grouped.geometry_calls,
        "semantic_reviews": grouped.reviewed,
        "shared_raw_decodes": shared_decode_count,
        "timing_seconds": timings,
    }
    if iqa_report:
        result["iqa_report"] = True
    return result | scene_result


def _write_text_atomically(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8-sig" if path.suffix == ".csv" else "utf-8",
                            newline="", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer(stream)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_results(result, json_path, csv_path=None):
    def write_csv(stream):
        fields = [
            "group", "basis", "rank", "decision", "score", "technical_score", "profile", "path", "versions", "analyzed_path",
            "timestamp", "camera", "lens", "focal_length", "blink", "components", "contributions", "reasons", "error",
        ]
        if result.get("iqa_report"):
            fields.insert(fields.index("reasons"), "iqa")
        scene_for_group = {
            group: scene["scene_id"]
            for scene in result.get("scene_clusters", []) for group in scene["groups"]
        }
        groups = result["groups"]
        if scene_for_group:
            fields.insert(0, "scene_cluster")
            groups = sorted(groups, key=lambda group: scene_for_group[group["group"]])
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for group in groups:
            for frame in group["frames"]:
                record = frame | {
                    "group": group["group"], "basis": group["basis"],
                    "versions": " | ".join(frame["versions"]), "reasons": " | ".join(frame["reasons"]),
                    "components": json.dumps(frame["components"], ensure_ascii=False),
                    "contributions": json.dumps(frame["contributions"], ensure_ascii=False),
                }
                if scene_for_group:
                    record["scene_cluster"] = scene_for_group[group["group"]]
                if "iqa" in frame:
                    record["iqa"] = json.dumps(frame["iqa"], ensure_ascii=False, allow_nan=False)
                writer.writerow(record)

    _write_text_atomically(json_path, lambda stream: json.dump(
        result, stream, ensure_ascii=False, allow_nan=False, indent=2,
    ))
    if csv_path:
        _write_text_atomically(csv_path, write_csv)
