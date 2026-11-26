"""Evaluate the local engine against checked-in Facet and labelled fixtures."""

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
from face_runtime import close_models, face_analyzer, measure_row
from paths import load_facet

load_facet()

import cv2
import numpy as np
from PIL import Image

from db import get_connection, init_database
from processing.burst_score import compute_burst_score
from processing.scorer import process_bursts

from engine import Frame, Segmenter, contextual_blinks, make_preview, rank_group, read_image, technical_metrics


DATA = WORKSPACE / "benchmark_data"
RESULTS = HERE / "results"


def load_manifest(data, dataset):
    return json.loads((data / f"{dataset}_manifest.json").read_text(encoding="utf-8"))


def capture_time(exif):
    value = exif.get("DateTimeOriginal")
    if not value:
        return None
    timestamp = datetime.strptime(value, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    subsec = str(exif.get("SubSecTimeOriginal", ""))
    return timestamp + (float("0." + subsec) if subsec.isdecimal() else 0.0)


def create_frame(data, row, order):
    exif = row.get("exif", {})
    focal = exif.get("FocalLength")
    if isinstance(focal, (list, tuple)):
        focal = focal[0] / focal[1] if focal[1] else None
    return Frame(
        data / row["path"], capture_time(exif), order, exif.get("Model"),
        exif.get("LensSerialNumber") or exif.get("LensModel"), focal,
    )


def facet_groups(frames, previews):
    origin = datetime(1970, 1, 1)
    with tempfile.TemporaryDirectory(prefix="burst_facet_") as temp:
        database = str(Path(temp) / "facet.db")
        init_database(database)
        ids = [f"frame_{i:06}" for i in range(len(frames))]
        records = [
            (key, key, (origin + timedelta(seconds=frame.timestamp)).isoformat(timespec="microseconds")
             if frame.timestamp is not None else None, f"{preview.phash:016x}")
            for key, frame, preview in zip(ids, frames, previews)
        ]
        with get_connection(database) as conn:
            conn.executemany("INSERT INTO photos(path, filename, date_taken, phash) VALUES (?,?,?,?)", records)
            conn.commit()
        process_bursts(database)
        with get_connection(database) as conn:
            found = {row["path"]: row["burst_group_id"] for row in
                     conn.execute("SELECT path, burst_group_id FROM photos")}
    output = defaultdict(list)
    for index, key in enumerate(ids):
        group = found[key]
        output[("burst", group) if group is not None else ("single", index)].append(index)
    return list(output.values())


def group_metrics(expected, predicted):
    if sorted(index for group in predicted for index in group) != list(range(len(expected))):
        raise ValueError("nonpartition output")
    result = {index: i for i, group in enumerate(predicted) for index in group}
    truth_members = defaultdict(set)
    for i, group in enumerate(expected):
        truth_members[group].add(i)
    fp = fn = same = predicted_same = 0
    for i, j in combinations(range(len(expected)), 2):
        truth, together = expected[i] == expected[j], result[i] == result[j]
        same += truth
        predicted_same += together
        fp += together and not truth
        fn += truth and not together
    return {
        "predicted_groups": len(predicted),
        "exact_truth_groups": len(set(map(frozenset, predicted)) & set(map(frozenset, truth_members.values()))),
        "truth_groups": len(truth_members),
        "false_merge_pairs": fp, "predicted_same_pairs": predicted_same,
        "false_merge_rate": fp / predicted_same if predicted_same else None,
        "false_split_pairs": fn, "true_same_pairs": same, "false_split_rate": fn / same,
        "mixed_predicted_groups": sum(len({expected[i] for i in group}) > 1 for group in predicted),
        "split_truth_groups": sum(len({result[i] for i in group}) > 1 for group in truth_members.values()),
    }


def evaluate_segmentation(data, out):
    rows = load_manifest(data, "hdrplus")
    records = sorted(enumerate(rows), key=lambda item: (
        capture_time(item[1]["exif"]) or float("inf"), item[0]
    ))
    originals = [create_frame(data, row, i) for i, row in records]
    labels = [row["group"] for _, row in records]
    previews = [make_preview(read_image(frame.path, 640)) for frame in originals]
    cases = {"original_exif": originals}
    for name, gap in (("compressed_160ms", 0.16), ("equal_cadence", 0.033)):
        clock = 1_700_000_000.0
        flat = []
        for i, frame in enumerate(originals):
            if i:
                clock += gap if labels[i] != labels[i - 1] else 0.033
            flat.append(replace(frame, timestamp=round(clock, 6), camera=None, lens=None))
        cases[name] = flat
    summary = {}
    for name, inputs in cases.items():
        start = perf_counter()
        baseline = facet_groups(inputs, previews)
        baseline_sec = perf_counter() - start
        segmenter = Segmenter(inputs, previews)
        start = perf_counter()
        prediction = segmenter.run()
        segment_sec = perf_counter() - start
        summary[name] = {
            "facet": group_metrics(labels, baseline) | {"seconds": round(baseline_sec, 4)},
            "engine": group_metrics(labels, prediction) | {
                "seconds": round(segment_sec, 4), "geometric_pairs": segmenter.geometry_calls,
                "ambiguous_edges": len(segmenter.ambiguous), "vlm_calls": segmenter.reviewed,
            },
        }
        details = {"truth": labels, "images": [row["path"] for _, row in records],
                   "facet": baseline, "engine": prediction, "ambiguous_edges": segmenter.ambiguous}
        (out / f"groups_{name}.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        print("groups", name, summary[name], flush=True)
    (out / "groups.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def blink_metrics(groups, use_context=False):
    rows = [
        row | {"blink": state}
        for group in groups.values()
        for row, state in zip(group, contextual_blinks(group) if use_context else [r["blink"] for r in group])
    ]
    pos = [r for r in rows if r["target"]]
    neg = [r for r in rows if not r["target"]]
    tp = sum(r["blink"] is True for r in pos)
    fp = sum(r["blink"] is True for r in neg)
    return {
        "true_positives": tp, "false_positives": fp, "positives": len(pos), "negatives": len(neg),
        "recall": tp / len(pos), "false_positive_rate": fp / len(neg),
        "unknown_positive": sum(r["blink"] is None for r in pos),
        "unknown_negative": sum(r["blink"] is None for r in neg),
        "faces_detected": sum(r["faces"] > 0 for r in rows),
    }


def facet_rank(rows):
    scores = [compute_burst_score({
        "tech_sharpness": r["facet_sharpness"], "is_blink": r["blink"],
        "face_count": r["faces"],
    }) for r in rows]
    return sorted(range(len(rows)), key=lambda i: (-scores[i], i)), scores


def evaluate_ranking(data, out, dataset):
    analyzer = face_analyzer()
    rows = load_manifest(data, dataset)
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    ranking = []
    wins = defaultdict(int)
    pair_wins = defaultdict(int)
    pair_count = 0
    start = perf_counter()
    measures = {}
    for count, (name, members) in enumerate(groups.items(), 1):
        # Match the faces' pixel scales within controlled portrait reference pairs.
        # The low-quality partner is resampled without a restoration model.
        images = [read_image(data / row["path"], 640) for row in members]
        if dataset == "modelscope_portrait":
            images = [cv2.resize(rgb, (512, 512), interpolation=(
                cv2.INTER_AREA if rgb.shape[0] > 512 else cv2.INTER_LINEAR)) for rgb in images]
        metrics = [measure_row(rgb, analyzer) for rgb in images]
        if dataset != "hdrplus":
            for row, score in zip(members, metrics):
                score["target"] = (
                    row["label"] == "fully_closed_in_at_least_one_eye" if dataset == "eyeblink8"
                    else row["label"] in {"sharp_reference", "high_quality_reference"}
                )
        measures[name] = metrics
        group_blinks = contextual_blinks(metrics)
        orders = {"engine": rank_group(metrics)}
        orders["facet_technical"], baseline_scores = facet_rank(metrics)
        if dataset != "hdrplus":
            prefer = [i for i, score in enumerate(metrics) if score["target"] != (dataset == "eyeblink8")]
            other = set(range(len(metrics))) - set(prefer)
            pair_count += len(prefer) * len(other)
            for key, order in orders.items():
                positions = {member: i for i, member in enumerate(order)}
                wins[key] += order[0] in prefer
                pair_wins[key] += sum(positions[i] < positions[j] for i in prefer for j in other)
        ranking.append({
            "group": name,
            "frames": [
                {"path": row["path"], "label": row.get("label"), "metrics": score,
                 "facet_technical_score": baseline_scores[i], "engine_blink": group_blinks[i]}
                for i, (row, score) in enumerate(zip(members, metrics))
            ],
            **orders,
        })
        if count % 100 == 0 or count == len(groups):
            print(dataset, f"{count}/{len(groups)} groups, {perf_counter() - start:.1f}s", flush=True)
    stats = {"images": len(rows), "groups": len(groups), "seconds": round(perf_counter() - start, 3)}
    if dataset != "hdrplus":
        for key in ("engine", "facet_technical"):
            stats[key] = {
                "top1_correct": wins[key], "top1_total": len(groups), "top1": wins[key] / len(groups),
                "ordered_pairs": pair_wins[key], "pairs": pair_count, "pair_order": pair_wins[key] / pair_count,
            }
    if dataset == "eyeblink8":
        stats["facet_blink"] = blink_metrics(measures)
        stats["engine_blink"] = blink_metrics(measures, use_context=True)
    (out / f"ranking_{dataset}.json").write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    (out / f"score_{dataset}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(dataset, stats, flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("groups", "hdrplus", "eyeblink8", "modelscope_portrait", "gimbalblur"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.task == "groups":
        evaluate_segmentation(args.data, args.out)
    else:
        evaluate_ranking(args.data, args.out, args.task)
