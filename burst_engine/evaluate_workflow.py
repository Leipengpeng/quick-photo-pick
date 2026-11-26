"""Score the labelled cases in the existing benchmark using the production scorer."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

from benchmark import DATA, RESULTS, face_analyzer, facet_rank, load_manifest
from engine import ranking_keys, read_image
from portrait_scoring import measure_signals, score_group

import cv2


_analyzer = None
LABELLED = ("eyeblink8", "gimbalblur", "modelscope_portrait")


def _setup():
    global _analyzer
    cv2.setNumThreads(1)
    _analyzer = face_analyzer()


def _evaluate_group(task):
    data, dataset, group_id, rows, include_iqa = task
    # An arbitrary manifest order should not decide which class wins an exact tie.
    rows = sorted(rows, key=lambda row: hashlib.sha256(row["path"].encode()).digest())
    images = [read_image(data / row["path"], 960) for row in rows]
    if dataset == "modelscope_portrait":
        images = [cv2.resize(rgb, (512, 512), interpolation=(
            cv2.INTER_AREA if rgb.shape[0] > 512 else cv2.INTER_LINEAR)) for rgb in images]
    measured = [measure_signals(rgb, _analyzer, include_iqa=include_iqa) for rgb in images]
    ranked = score_group(measured)
    previous_keys = ranking_keys(measured)
    facet_indices, facet_scores = facet_rank(measured)
    previous_indices = sorted(range(len(rows)), key=previous_keys.__getitem__, reverse=True)
    previous_positions = {index: rank for rank, index in enumerate(previous_indices, 1)}
    facet_positions = {index: rank for rank, index in enumerate(facet_indices, 1)}
    results = []
    for rank, candidate in enumerate(ranked, 1):
        source = rows[candidate["index"]]
        record = candidate | {
            "path": source["path"],
            "label": source["label"],
            "rank": rank,
            "facet_blink": measured[candidate["index"]]["blink"],
            "facet_technical_rank": facet_positions[candidate["index"]],
            "facet_technical_score": facet_scores[candidate["index"]],
            "prototype_rank": previous_positions[candidate["index"]],
            "prototype_key": previous_keys[candidate["index"]],
        }
        if include_iqa:
            record["facet_capped_sharpness"] = measured[candidate["index"]]["facet_sharpness"]
        results.append(record)
    return dataset, {"group": group_id, "ranking": results}


def _preferred(dataset, frame):
    if dataset == "eyeblink8":
        return frame["label"] != "fully_closed_in_at_least_one_eye"
    return frame["label"] in {"high_quality_reference", "sharp_reference"}


def _counts(dataset, groups):
    counts = Counter()
    profiles = Counter()
    for group in groups:
        ranked = group["ranking"]
        profiles[ranked[0]["profile"]] += 1
        counts["groups"] += 1
        counts["images"] += len(ranked)
        counts["top1_correct"] += _preferred(dataset, ranked[0])
        counts["facet_technical_top1"] += _preferred(dataset, min(ranked, key=lambda f: f["facet_technical_rank"]))
        counts["prototype_top1"] += _preferred(dataset, min(ranked, key=lambda f: f["prototype_rank"]))
        positives = [frame for frame in ranked if _preferred(dataset, frame)]
        negatives = [frame for frame in ranked if not _preferred(dataset, frame)]
        for good in positives:
            for bad in negatives:
                counts["rank_pairs"] += 1
                counts["strict_rank_correct"] += good["score"] > bad["score"]
                counts["tied_pairs"] += good["score"] == bad["score"]
                counts["facet_technical_strict_pairs"] += good["facet_technical_score"] > bad["facet_technical_score"]
                counts["prototype_strict_pairs"] += good["prototype_key"] > bad["prototype_key"]
                if "iqa" in good and dataset != "eyeblink8":
                    sharp = good["iqa"]["narrow"]["relative_sharpness"]["laplacian_variance"]
                    degraded = bad["iqa"]["narrow"]["relative_sharpness"]["laplacian_variance"]
                    counts["iqa_raw_sharpness_correct"] += sharp > degraded
                    counts["iqa_raw_sharpness_ties"] += sharp == degraded
                    capped_good = good["facet_capped_sharpness"]
                    capped_bad = bad["facet_capped_sharpness"]
                    counts["facet_capped_sharpness_correct"] += capped_good > capped_bad
                    counts["facet_capped_sharpness_ties"] += capped_good == capped_bad
        if dataset == "eyeblink8":
            counts["blink_positive"] += len(negatives)
            counts["blink_negative"] += len(positives)
            counts["blink_true_positive"] += sum(frame["effective_blink"] is True for frame in negatives)
            counts["blink_false_positive"] += sum(frame["effective_blink"] is True for frame in positives)
            counts["blink_unclassified"] += sum(frame["effective_blink"] is None for frame in ranked)
            if "iqa" in ranked[0]:
                counts["iqa_blink_true_positive"] += sum(f["iqa"]["broad"]["blink"]["closed"] is True for f in negatives)
                counts["iqa_blink_false_positive"] += sum(f["iqa"]["broad"]["blink"]["closed"] is True for f in positives)
    return dict(counts) | {"profiles": dict(profiles)}


def evaluate(data, out, workers, datasets, *, include_iqa=False):
    start = perf_counter()
    tasks = []
    for dataset in datasets:
        grouped = defaultdict(list)
        for row in load_manifest(data, dataset):
            grouped[row["group"]].append(row)
        tasks.extend((data, dataset, name, rows, include_iqa) for name, rows in grouped.items())
    results = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers, initializer=_setup) as pool:
        for index, (dataset, group) in enumerate(pool.map(_evaluate_group, tasks, chunksize=4), 1):
            results[dataset].append(group)
            if index % 100 == 0 or index == len(tasks):
                print(f"{index}/{len(tasks)} groups in {perf_counter() - start:.1f}s", flush=True)
    summary = {dataset: _counts(dataset, results[dataset]) for dataset in datasets}
    summary["elapsed_seconds"] = round(perf_counter() - start, 3)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scoring_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "scoring_details.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=RESULTS / "workflow_v2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dataset", action="append", choices=LABELLED)
    parser.add_argument("--iqa-report", action="store_true", help="Measure optional IQA evidence on the labelled data")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    evaluate(args.data, args.out, args.workers, tuple(dict.fromkeys(args.dataset or LABELLED)),
             include_iqa=args.iqa_report)
