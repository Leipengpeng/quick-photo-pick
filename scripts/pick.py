"""Group local captures, rank each burst, and export a browser review."""

import argparse
import json
import shutil
import sys
import webbrowser
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4


ENGINE = Path(__file__).resolve().parent.parent / "burst_engine"
sys.path.insert(0, str(ENGINE))


def pick(source, output, *, metadata=None, guide=None, exiftool=None, profile="auto"):
    """Run analysis and publish a completed, independently numbered run."""
    import cv2
    from export_photos import apply_best_picture_rule, export_photos
    from face_runtime import close_models
    from report import build_payload, read_json, save_json
    from select_photos import RAW, select
    from workflow import write_results

    source = source.resolve(strict=True)
    output = output.resolve()
    if not source.is_dir():
        raise ValueError(f"Photo input must be a directory: {source}")
    if output.is_relative_to(source):
        raise ValueError("The output directory must be outside the source photo folder.")
    notes = read_json(guide) if guide else {}
    template = (ENGINE / "review.html").read_text(encoding="utf-8")
    if template.count("__PHOTO_DATA__") != 1:
        raise ValueError("The report template must contain one data placeholder.")

    now = datetime.now().astimezone()
    run_id = now.strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8]
    destination = output / run_id
    stage = output / f".{run_id}.partial"
    stage.mkdir(parents=True)
    cv2.setNumThreads(1)
    start = perf_counter()
    try:
        result = select(source, sidecar=metadata, exiftool=exiftool, profile=profile, keepers=2, iqa_report=True)
    finally:
        close_models()
    apply_best_picture_rule(result)
    write_results(result, stage / "selection.json", stage / "selection.csv")
    engine_seconds = perf_counter() - start
    exported = export_photos(result, source, stage, notes)
    save_json(stage / "照片核验.json", exported["records"])
    formats = Counter()
    for record in exported["records"]:
        extension = Path(record["source"]).suffix.lower()
        formats["RAW" if extension in RAW else extension[1:].upper().replace("JPEG", "JPG")] += 1
    raw_analyzed = sum(
        Path(frame.get("analyzed_path") or "").suffix.lower() in RAW
        for group in result["groups"] for frame in group["frames"]
    )
    summary = {
        "run_id": run_id, "generated_at": now.isoformat(timespec="seconds"), "rehearsal": False,
        "frames": result["capture_frames"], "capture_groups": len(result["groups"]),
        "scene_groups": len(result["scene_clusters"]), "decisions": result["decisions"],
        "engine_seconds": round(engine_seconds, 3), "pipeline_seconds": result["timing_seconds"],
        "report": str(destination / "选片报告.html"),
        "csv": str(destination / "selection.csv"), "json": str(destination / "selection.json"),
        "grouped_folder": str(destination / exported["folder"].name),
        "grouped_photos": len(exported["records"]), "input_formats": dict(formats), "raw_analyzed": raw_analyzed,
        "best_picture_count": result["decisions"].get("select", 0),
        "best_picture_files": exported["best_picture_files"],
        "export_manifest": str(destination / "照片核验.json"),
    }
    payload = build_payload(result, stage, summary, notes, exported)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    (stage / "选片报告.html").write_text(template.replace("__PHOTO_DATA__", data), encoding="utf-8")
    summary["end_to_end_seconds"] = round(perf_counter() - start, 3)
    save_json(stage / "summary.json", summary)
    stage.rename(destination)
    pending = output / f".{run_id}.receipt.tmp"
    save_json(pending, summary)
    pending.replace(output / "latest.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos", type=Path, help="Input photo directory")
    parser.add_argument("--output", type=Path, required=True, help="Separate directory for completed runs")
    parser.add_argument("--metadata", type=Path, help="Optional ExifTool-style JSON with SourceFile entries")
    parser.add_argument("--guide", type=Path, help="Optional photo captions and provenance metadata")
    parser.add_argument("--exiftool", default=shutil.which("exiftool"), help="Optional ExifTool executable")
    parser.add_argument("--profile", choices=("auto", "portrait", "general", "landscape", "street"), default="auto")
    parser.add_argument("--open", action="store_true", help="Open the finished report in the default browser")
    args = parser.parse_args()
    try:
        summary = pick(args.photos, args.output, metadata=args.metadata, guide=args.guide,
                       exiftool=args.exiftool, profile=args.profile)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    if args.open:
        webbrowser.open(Path(summary["report"]).as_uri())


if __name__ == "__main__":
    main()
