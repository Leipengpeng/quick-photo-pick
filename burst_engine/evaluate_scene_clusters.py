"""Audit scene clustering on the local HDR+ subset and manually checked sentinels."""

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path, PureWindowsPath


MANIFEST = Path(__file__).resolve().parent.parent / "benchmark_data" / "hdrplus_manifest.json"
CHAPEL_2014 = "0218_20141102_112123_601"
CHAPEL_2016 = "0127_20161018_111029_303"
DISTINCT_VIEWS = {
    CHAPEL_2014: "chapel interior, 2014",
    CHAPEL_2016: "chapel interior, 2016",
    "47L8_20150518_144020_604": "valley",
    "0919_20150910_131720_137": "building entrance",
    "0543_20150918_175650_498": "outdoor portrait",
}
HARD_NEGATIVE_VIEWS = {
    "6G7M_20150404_132319_709": "dark room with stained glass",
    "J6ZC_20150408_183238_027": "stone structure with wooden beams",
    "6G7M_20150417_063859_191": "yard silhouette at sunset",
    "33TJ_20150612_201525_012": "street silhouette at sunset",
    "33TJ_20150810_194922_367": "waterfront at twilight",
    "33TJ_20150822_185205_562": "palms silhouetted against twilight",
    "0919_20150906_130312_427": "dry branches on rocky ground",
    "4742_20150918_190218_179": "mountains on the sunset horizon",
}
HARD_NEGATIVE_PAIRS = (
    (CHAPEL_2014, "6G7M_20150404_132319_709"),
    (CHAPEL_2016, "6G7M_20150404_132319_709"),
    ("6G7M_20150404_132319_709", "J6ZC_20150408_183238_027"),
    ("6G7M_20150417_063859_191", "33TJ_20150612_201525_012"),
    ("33TJ_20150612_201525_012", "4742_20150918_190218_179"),
    ("33TJ_20150810_194922_367", "33TJ_20150822_185205_562"),
    ("0919_20150906_130312_427", "0543_20150918_175650_498"),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def frame_key(path):
    source = PureWindowsPath(path)
    return source.parent.name, source.name


def audit(selection, manifest, baseline=None):
    bursts = {row["group"]: row["frames"] for row in selection["groups"]}
    capture_clusters = selection["scene_clusters"]
    links = selection["scene_links"]
    memberships = Counter(group for scene in capture_clusters for group in scene["groups"])
    scene_pairings = {
        tuple(sorted(pair))
        for scene in capture_clusters
        for pair in combinations(scene["groups"], 2)
    }
    scene_for_group = {group: scene["scene_id"] for scene in capture_clusters for group in scene["groups"]}
    evidence_counts = Counter(tuple(sorted(link["groups"])) for link in links)
    paths_by_group = {group: {frame_key(row["path"]) for row in frames} for group, frames in bursts.items()}
    evidence_sources = {
        group: {frame_key(path) for row in frames for path in [row["path"], *row.get("versions", [])]}
        for group, frames in bursts.items()
    }
    valid_evidence_paths = all(
        len(link["groups"]) == 2 and len(link["paths"]) == 2
        and all(frame_key(path) in evidence_sources.get(group, set()) for group, path in zip(link["groups"], link["paths"]))
        for link in links
    )

    truth_bursts = defaultdict(set)
    for row in manifest:
        truth_bursts[row["group"]].add(frame_key(row["path"]))
    partition_matches_truth = (
        Counter(frozenset(paths) for paths in paths_by_group.values())
        == Counter(frozenset(paths) for paths in truth_bursts.values())
    )
    complete_burst = {
        next(iter(paths))[0]: group
        for group, paths in paths_by_group.items()
        if paths and paths == truth_bursts.get(next(iter(paths))[0])
    }
    positive = (CHAPEL_2014, CHAPEL_2016)
    distinct_views = [source for source in DISTINCT_VIEWS if source not in positive]
    negatives = list(combinations(distinct_views, 2)) + [
        (chapel, view) for chapel in positive for view in distinct_views
    ]
    descriptions = DISTINCT_VIEWS | HARD_NEGATIVE_VIEWS

    def scene_membership(first, second):
        a, b = complete_burst.get(first), complete_burst.get(second)
        scene_a, scene_b = scene_for_group.get(a), scene_for_group.get(b)
        return {
            "source_bursts": [first, second],
            "subjects": [descriptions[first], descriptions[second]],
            "burst_groups": [a, b],
            "scene_ids": [scene_a, scene_b],
        }

    def condition(pair, expected_same):
        member = scene_membership(*pair)
        left, right = member["scene_ids"]
        member["correct"] = left is not None and right is not None and (left == right) == expected_same
        return member

    positive_result = condition(positive, True)
    negative_results = [condition(pair, False) for pair in [*negatives, *HARD_NEGATIVE_PAIRS]]
    checks = {
        "scene_partition_covers_every_burst_once": memberships == Counter({group: 1 for group in bursts}),
        "unique_scene_ids_and_no_empty_scenes": (
            len({scene["scene_id"] for scene in capture_clusters}) == len(capture_clusters)
            and all(scene["groups"] for scene in capture_clusters)
        ),
        "every_coclustered_burst_pair_has_one_direct_evidence_link": (
            evidence_counts == Counter({pair: 1 for pair in scene_pairings})
        ),
        "evidence_paths_belong_to_their_bursts": valid_evidence_paths,
        "burst_partition_matches_hdrplus_labels": partition_matches_truth,
        "same_composition_across_dates_clustered": positive_result["correct"],
        "visually_distinct_examples_separate": all(pair["correct"] for pair in negative_results),
    }
    if baseline is not None:
        checks["burst_frames_scores_ranks_and_decisions_identical_to_baseline"] = selection["groups"] == baseline["groups"]

    checked_pairings = {
        tuple(sorted(pair["burst_groups"]))
        for pair in [positive_result, *negative_results]
        if None not in pair["burst_groups"]
    }
    positive_pair = set(positive_result["burst_groups"])
    return {
        "passed": all(checks.values()),
        "data_scope": "HDR+ labels are for capture bursts. Scene labels only cover the manually checked pairs below.",
        "counts": {
            "source_frames": sum(len(frames) for frames in bursts.values()),
            "burst_groups": len(bursts),
            "scene_clusters": len(capture_clusters),
            "scene_clusters_with_multiple_bursts": sum(len(scene["groups"]) > 1 for scene in capture_clusters),
            "unlabelled_coclustered_pairs": len(scene_pairings - checked_pairings),
        },
        "checks": checks,
        "labelled_positive": positive_result,
        "labelled_positive_evidence": [link for link in links if set(link["groups"]) == positive_pair],
        "labelled_negatives": negative_results,
    }


def audit_gimbalblur(manifest_path):
    """Verify same-view reference shots at different shutter settings."""
    import sys
    from time import perf_counter

    import cv2

    sys.path.insert(0, str(Path(__file__).resolve().parent / ".deps"))
    from engine import make_preview, read_image
    from scene_clustering import SceneFeatureExtractor, _candidate_pairs, _matching_evidence, cluster_scenes

    cv2.setNumThreads(1)
    data_root = manifest_path.parent
    references = [row for row in read_json(manifest_path) if row["label"] == "sharp_reference"]
    describe = SceneFeatureExtractor()
    by_shot = {}
    ground_truth = {}
    for shot, row in enumerate(references, 1):
        path = data_root / row["path"]
        by_shot[shot] = [describe(path, make_preview(read_image(path, 640)))]
        ground_truth[shot] = (row["scene"], row["device"])
    started = perf_counter()
    baseline = cluster_scenes(by_shot, contrast_recheck=False)
    baseline_seconds = perf_counter() - started
    started = perf_counter()
    prediction = cluster_scenes(by_shot)
    contrast_seconds = perf_counter() - started
    same_pairs = {pair for pair in combinations(ground_truth, 2) if ground_truth[pair[0]] == ground_truth[pair[1]]}

    def clustered_pairs(result):
        return {
            tuple(sorted(pair))
            for cluster in result["scene_clusters"]
            for pair in combinations(cluster["groups"], 2)
        }

    baseline_pairs = clustered_pairs(baseline)
    predicted_pairs = clustered_pairs(prediction)
    seed_pairs = set(_candidate_pairs(by_shot))
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def items(pairs):
        return [[references[shot - 1]["path"] for shot in pair] for pair in sorted(pairs)]

    missed = []
    for a, b in sorted(same_pairs - predicted_pairs):
        missed.append({
            "paths": items([(a, b)])[0],
            "retrieved_by_index": (a, b) in seed_pairs,
            "passes_direct_geometry": _matching_evidence(by_shot[a][0], by_shot[b][0], matcher) is not None,
        })

    return {
        "data_scope": "GimbalBlur sharp references with controlled framing. Scene and device tags define repeated views.",
        "reference_images": len(references),
        "reference_scenes": len(set(ground_truth.values())),
        "predicted_scenes": len(prediction["scene_clusters"]),
        "candidate_pairs": prediction["candidate_pairs"],
        "geometry_checks": prediction["geometry_checks"],
        "contrast_geometry_checks": prediction["contrast_geometry_checks"],
        "expected_same_pairs": len(same_pairs),
        "expected_distinct_pairs": len(references) * (len(references) - 1) // 2 - len(same_pairs),
        "correct_same_pairs": len(same_pairs & predicted_pairs),
        "baseline_correct_same_pairs": len(same_pairs & baseline_pairs),
        "baseline_unexpected_same_pairs": items(baseline_pairs - same_pairs),
        "baseline_cluster_seconds": round(baseline_seconds, 3),
        "contrast_cluster_seconds": round(contrast_seconds, 3),
        "missed_same_pairs": missed,
        "unexpected_same_pairs": items(predicted_pairs - same_pairs),
    }


def audit_chapel_framing(manifest_path):
    """Repeat a four-scene fixture while changing the second real chapel view."""
    import cv2
    import numpy as np

    from engine import make_preview, read_image
    from scene_clustering import SceneFeatureExtractor, cluster_scenes

    cv2.setNumThreads(1)
    data_root = manifest_path.parent
    subjects = {**DISTINCT_VIEWS, **HARD_NEGATIVE_VIEWS}
    identifiers = {source: index for index, source in enumerate(subjects, 1)}
    source_paths = defaultdict(list)
    for row in read_json(manifest_path):
        if row["group"] in subjects:
            source_paths[row["group"]].append((row["frame"], data_root / row["path"]))

    describe = SceneFeatureExtractor()
    by_shot = {}
    second_chapel = []
    for source, identifier in identifiers.items():
        paths = sorted(source_paths[source])
        by_shot[identifier] = []
        for index in sorted({0, len(paths) // 2}):
            _, path = paths[index]
            image = read_image(path, 960)
            by_shot[identifier].append(describe(path, make_preview(image)))
            if source == CHAPEL_2016:
                second_chapel.append((path, image))

    def reframing(image, zoom, x, y, angle, gain):
        height, width = image.shape[:2]
        center = ((0.5 + x) * width, (0.5 + y) * height)
        matrix = cv2.getRotationMatrix2D(center, angle, zoom)
        matrix[:, 2] += (width * 0.5 - center[0], height * 0.5 - center[1])
        edited = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
        # Test crops stay within the captured pixels, without reflected borders.
        if gain != 1:
            edited = np.clip(edited.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return edited

    # Offsets report the displacement of the crop center as a fraction of the original image.
    cases = (
        ("original", 1.00, 0.00, 0.00, 0, 1.00),
        ("slight_shift_right", 1.12, 0.04, -0.02, 1, 1.00),
        ("slight_shift_left", 1.12, -0.04, 0.02, -1, 1.00),
        ("centered_zoom", 1.20, 0.00, 0.00, 0, 1.00),
        ("nearby_composition", 1.20, 0.03, -0.03, 2, 1.00),
        ("nearby_composition_darker", 1.12, -0.03, 0.02, -1, 0.80),
        ("reframe_brighter", 1.25, 0.07, -0.05, 2, 1.30),
        ("reframe_darker", 1.25, -0.07, 0.05, -2, 0.66),
        ("medium_zoom", 1.40, 0.07, 0.06, 3, 1.00),
        ("strong_zoom", 1.65, -0.11, 0.07, -4, 1.00),
        ("tight_detail", 2.00, 0.12, -0.08, 5, 1.00),
    )
    core_ids = {identifiers[source] for source in DISTINCT_VIEWS}
    chapel_ids = {identifiers[source] for source in (CHAPEL_2014, CHAPEL_2016)}
    expected = {frozenset(chapel_ids), *(frozenset((value,)) for value in core_ids - chapel_ids)}
    results = []
    for label, zoom, x, y, angle, gain in cases:
        current = by_shot | {
            identifiers[CHAPEL_2016]: [
                describe(path, make_preview(reframing(image, zoom, x, y, angle, gain)))
                for path, image in second_chapel
            ]
        }
        core = cluster_scenes({value: current[value] for value in sorted(core_ids)})
        extended = cluster_scenes(current)
        memberships = {
            group: scene["scene_id"] for scene in extended["scene_clusters"] for group in scene["groups"]
        }
        core_groups = {frozenset(scene["groups"]) for scene in core["scene_clusters"]}
        confusion = [
            [subjects[left], subjects[right]] for left, right in HARD_NEGATIVE_PAIRS
            if memberships[identifiers[left]] == memberships[identifiers[right]]
        ]
        results.append({
            "case": label,
            "zoom": zoom, "crop_offset_xy": [x, y], "rotation_degrees": angle, "brightness_gain": gain,
            "four_scene_partition_correct": core_groups == expected,
            "core_scene_count": len(core_groups),
            "chapel_pair_retained_with_distractors": len({memberships[value] for value in chapel_ids}) == 1,
            "hard_negative_false_merges": confusion,
            "chapel_alignment": next((
                evidence for evidence in core["scene_links"] if set(evidence["groups"]) == chapel_ids
            ), None),
        })
    return {
        "data_scope": "Two actual HDR+ chapel captures. Transforms of the second capture are controlled perturbations, not independent scene labels.",
        "four_expected_scenes": ["chapel interior", *list(DISTINCT_VIEWS.values())[2:]],
        "core_capture_groups": len(core_ids),
        "capture_groups_with_distractors": len(by_shot),
        "hard_negative_pairs_per_case": len(HARD_NEGATIVE_PAIRS),
        "cases": results,
    }


def _perturb_view(image, side, offset, angle, jitter, gain):
    """Apply an in-bounds view transform and return the actual retained area."""
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    # Exposed borders cannot become an artificial matching cue.
    corners = np.asarray([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]], dtype=np.float32)
    t = np.radians(angle)
    rotate = np.asarray([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], dtype=np.float32)
    crop = side * corners * [width, height] @ rotate
    crop += jitter * [width, height]
    minimum = -crop.min(axis=0)
    maximum = [width - 1, height - 1] - crop.max(axis=0)
    crop += minimum + offset * (maximum - minimum)
    destination = (corners + .5) * [width - 1, height - 1]
    mapping = cv2.getPerspectiveTransform(crop.astype(np.float32), destination.astype(np.float32))
    result = cv2.warpPerspective(image, mapping, (width, height))
    if gain != 1:
        result = np.clip(result.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    area = cv2.contourArea(crop.astype(np.float32)) / (width * height)
    return result, area


def audit_random_framing(manifest_path, *, trials=24, seed=114382):
    """Measure reproducible viewpoint perturbations on four RAW scenes."""
    import cv2
    import numpy as np

    from engine import make_preview, read_image
    from scene_clustering import SceneFeatureExtractor, cluster_scenes
    from workflow import scene_preview_indices

    cv2.setNumThreads(1)
    data_root = manifest_path.parent
    sources = [*DISTINCT_VIEWS, "6G7M_20150404_132319_709"]
    source_paths = defaultdict(list)
    for row in read_json(manifest_path):
        if row["group"] in sources:
            source_paths[row["group"]].append((row["frame"], data_root / row["path"]))
    describe = SceneFeatureExtractor()

    def features(images):
        previews = [(path, make_preview(rgb)) for path, rgb in images]
        return [
            describe(*previews[index]) for index in scene_preview_indices([item[1] for item in previews])
        ]

    lookup = {}
    for identifier, source in enumerate(sources, 1):
        paths = [path for _, path in sorted(source_paths[source])]
        if identifier == 2:
            second_chapel = [(path, read_image(path, 960)) for path in paths]
        else:
            indices = sorted({0, len(paths) // 2})
            lookup[identifier] = features([(paths[index], read_image(paths[index], 960)) for index in indices])

    truth = {frozenset((1, 2)), frozenset((3,)), frozenset((4,)), frozenset((5,)), frozenset((6,))}
    cases_by_variant = {}
    for label, area, exposure in (("retain_80_pct", .80, False), ("retain_80_pct_with_exposure", .80, True),
                                  ("retain_64_pct", .64, False)):
        rng = np.random.default_rng(seed)
        cases = []
        for trial in range(1, trials + 1):
            offset = rng.random(2)
            angle = rng.uniform(-3, 3)
            jitter = rng.uniform(-.009, .009, size=(4, 2))
            exposure_draw = rng.uniform(.65, 1.25)
            gain = exposure_draw if exposure else 1.0
            crops = [_perturb_view(im, np.sqrt(area), offset, angle, jitter, gain) for _, im in second_chapel]
            changed = features([
                (path, crop) for (path, _), (crop, _) in zip(second_chapel, crops)
            ])
            current = lookup | {2: changed}
            baseline = cluster_scenes({key: frames[:2] for key, frames in current.items()}, contrast_recheck=False)
            upgraded = cluster_scenes(current)
            case = {
                "trial": trial, "retained_area": round(crops[0][1], 4),
                "rotation_degrees": round(angle, 4), "offset_fractions": np.round(offset, 4).tolist(),
                "brightness_gain": round(gain, 4),
            }
            for version, result in (("baseline", baseline), ("upgraded", upgraded)):
                groups = {frozenset(scene["groups"]) for scene in result["scene_clusters"]}
                chapel_evidence = next((item for item in result["scene_links"] if item["groups"] == [1, 2]), None)
                case[version] = {
                    "entire_partition_correct": groups == truth,
                    "chapel_joined": frozenset((1, 2)) in groups,
                    "false_merge_pairs": [
                        pair for cluster in result["scene_clusters"]
                        for pair in combinations(cluster["groups"], 2) if pair != (1, 2)
                    ],
                    "chapel_evidence": chapel_evidence,
                }
            cases.append(case)
        cases_by_variant[label] = {
            "baseline_correct_partitions": sum(case["baseline"]["entire_partition_correct"] for case in cases),
            "upgraded_correct_partitions": sum(case["upgraded"]["entire_partition_correct"] for case in cases),
            "upgraded_false_merge_pairs": sum(len(case["upgraded"]["false_merge_pairs"]) for case in cases),
            "trials": cases,
        }
    return {
        "data_scope": "Controlled perturbations of one real HDR+ chapel capture. Six bursts per trial: two chapel captures, three distinct scenes, and a different stained-glass interior.",
        "seed": seed, "trials_per_variant": trials, "cases": cases_by_variant,
    }


def audit_independent_crops(manifest_path, *, trials=24, seed=114383):
    """Cluster two independently jittered crops of each of four REAL RAW photos."""
    import cv2
    import numpy as np

    from engine import make_preview, read_image
    from scene_clustering import SceneFeatureExtractor, cluster_scenes

    cv2.setNumThreads(1)
    describe = SceneFeatureExtractor()
    root = manifest_path.parent
    first_frames = {row["group"]: row["path"] for row in read_json(manifest_path) if row["frame"] == 0}
    sources = {key: name for key, name in DISTINCT_VIEWS.items() if key != CHAPEL_2016}
    originals = [(root / first_frames[key], read_image(root / first_frames[key], 960)) for key in sources]
    gt = {(2 * i + 1, 2 * i + 2) for i in range(len(originals))}
    rng = np.random.default_rng(seed)
    cases = []
    for trial in range(1, trials + 1):
        features = {}
        edits = []
        for path, image in originals:
            for _ in range(2):
                group = len(features) + 1
                offset = rng.random(2)
                angle = rng.uniform(-3, 3)
                jitter = rng.uniform(-.009, .009, size=(4, 2))
                variant, area = _perturb_view(image, np.sqrt(.80), offset, angle, jitter, 1.0)
                features[group] = [describe(path.with_stem(f"{path.stem}_crop{group}"), make_preview(variant))]
                edits.append({
                    "group": group, "source": path.relative_to(root).as_posix(), "retained_area": round(area, 4),
                    "rotation_degrees": round(angle, 4), "offset_fractions": np.round(offset, 4).tolist(),
                })
        case = {"trial": trial, "image_edits": edits}
        for version, contrast in (("baseline", False), ("upgraded", True)):
            result = cluster_scenes(features, contrast_recheck=contrast)
            scenes = [scene["groups"] for scene in result["scene_clusters"]]
            observed = {pair for scene in scenes for pair in combinations(scene, 2)}
            case[version] = {
                "scenes": scenes, "partition_correct": observed == gt,
                "false_merge_pairs": sorted(observed - gt), "missed_pairs": sorted(gt - observed),
            }
        cases.append(case)
    return {
        "data_scope": "Two independent, randomly reframed crops per original DNG, four different real subjects. Tests the complete four-cluster partition.",
        "seed": seed, "trial_count": trials, "scene_labels": list(sources.values()),
        "expected_scene_pairs": sorted(gt), "trials": cases,
        "summary": {
            version: {
                "correct_partitions": sum(case[version]["partition_correct"] for case in cases),
                "false_merge_pairs": sum(len(case[version]["false_merge_pairs"]) for case in cases),
                "missed_pairs": sum(len(case[version]["missed_pairs"]) for case in cases),
            }
            for version in ("baseline", "upgraded")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path, help="Scene-enabled selection JSON")
    parser.add_argument("--data", type=Path, default=MANIFEST.parent, help="Directory with benchmark manifests and images")
    parser.add_argument("--baseline", type=Path, help="Prior selection JSON with the same input and options")
    parser.add_argument("--output", type=Path, help="Write audit JSON for independent review")
    parser.add_argument("--gimbalblur", action="store_true", help="Also check repeated GimbalBlur reference shots")
    parser.add_argument("--chapel-framing", action="store_true", help="Audit changed framing on the two real chapel captures")
    parser.add_argument("--random-framing", action="store_true", help="Audit seeded RAW crops and slight camera jitter")
    parser.add_argument("--independent-crops", action="store_true", help="Audit two randomized crops per RAW across four scenes")
    args = parser.parse_args()
    manifest = args.data / MANIFEST.name
    baseline = read_json(args.baseline) if args.baseline else None
    result = audit(read_json(args.selection), read_json(manifest), baseline)
    if args.gimbalblur:
        result["gimbalblur_repeated_views"] = audit_gimbalblur(manifest.with_name("gimbalblur_manifest.json"))
    if args.chapel_framing:
        result["chapel_framing"] = audit_chapel_framing(manifest)
    if args.random_framing:
        result["random_framing"] = audit_random_framing(manifest)
    if args.independent_crops:
        result["independent_crops"] = audit_independent_crops(manifest)
    report = json.dumps(result, ensure_ascii=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
