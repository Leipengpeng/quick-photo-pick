"""Read-only command line entry point for selecting photos from a local folder."""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from PIL import ExifTags, Image
from tifffile import TiffFile

RAW = {".dng", ".arw", ".nef", ".cr2", ".cr3", ".raf", ".orf", ".rw2"}
RASTER = {".jpg", ".jpeg", ".png"}
KEYS = (
    "DateTimeOriginal", "SubSecTimeOriginal", "OffsetTimeOriginal", "Make", "Model",
    "SerialNumber", "BodySerialNumber", "LensMake", "LensModel", "LensSerialNumber", "FocalLength",
    "BurstUUID", "SequenceKind", "SequenceGroupId",
)
ALIASES = {name.casefold(): name for name in KEYS}


def _standard_tags(tags):
    return {ALIASES[key.casefold()]: value for key, value in tags.items() if key.casefold() in ALIASES}


def _embedded(path):
    if path.suffix.lower() in RASTER:
        try:
            with Image.open(path) as image:
                exif = image.getexif()
                ifd = exif.get_ifd(34665)
                top = {ExifTags.TAGS.get(key, str(key)): value for key, value in exif.items()}
                nested = {ExifTags.TAGS.get(key, str(key)): value for key, value in ifd.items()}
            return _standard_tags(top | nested)
        except (OSError, ValueError, SyntaxError):
            return {}
    try:
        with TiffFile(path) as tiff:
            tags = tiff.pages[0].tags
            top = {tag.name: tag.value for tag in tags.values() if tag.name.casefold() in ALIASES}
            nested = tags[34665].value if 34665 in tags else {}
            return _standard_tags(top | (nested if isinstance(nested, dict) else {}))
    except (OSError, ValueError, KeyError):
        # Non-TIFF RAW containers can optionally be scanned by ExifTool.
        return {}


def _index_metadata(rows, base):
    result = {}
    for row in rows:
        source = Path(row["SourceFile"])
        if not source.is_absolute():
            source = base / source
        result[source.resolve()] = _standard_tags(row)
    return result


def _metadata(paths, root, sidecar, exiftool):
    from_tool = {}
    if exiftool:
        command = [exiftool, "-j", "-n", "-charset", "filename=UTF8", "-r"]
        command.extend("-" + key for key in KEYS if key not in {"SequenceKind", "SequenceGroupId"})
        command.append(str(root))
        process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
        from_tool = _index_metadata(json.loads(process.stdout), root)
    from_sidecar = _index_metadata(json.loads(sidecar.read_text(encoding="utf-8-sig")), root) if sidecar else {}
    return {
        path: (from_tool.get(path) or _embedded(path)) | from_sidecar.get(path, {})
        for path in paths
    }


def _filename_order(path):
    return tuple(int(word) if word.isdecimal() else word.casefold()
                 for word in re.split(r"(\d+)", str(path)))


def select(root, sidecar=None, exiftool=None, *, profile="auto", keepers=1,
           semantic_review=None, review_budget=0, iqa_report=False, scene_clustering=True):
    from workflow import run_selection

    paths = sorted(
        (path.resolve() for path in root.rglob("*") if path.is_file()
         and path.suffix.lower() in (RAW | RASTER)), key=_filename_order,
    )
    if not paths:
        raise ValueError("input folder has no supported pictures")
    return run_selection(
        paths, _metadata(paths, root, sidecar, exiftool), profile=profile, keepers=keepers,
        semantic_review=semantic_review, review_budget=review_budget, iqa_report=iqa_report,
        scene_clustering=scene_clustering,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", type=Path, help="Optional sortable CSV for review")
    parser.add_argument("--metadata", type=Path, help="Optional ExifTool-style JSON with SourceFile entries")
    parser.add_argument("--exiftool", default=shutil.which("exiftool"))
    parser.add_argument("--profile", choices=("auto", "portrait", "general", "landscape", "street"), default="auto")
    parser.add_argument("--keepers", type=int, default=1, help="Maximum automatically recommended frames per group")
    parser.add_argument("--iqa-report", action="store_true", help="Export quality evidence with validation status")
    parser.add_argument("--no-scene-clustering", action="store_true", help="Export capture groups only")
    params = parser.parse_args()
    root = params.folder.resolve(strict=True)
    if not root.is_dir():
        parser.error("folder must be a directory")
    if params.output.suffix.lower() != ".json":
        parser.error("output must use the .json extension")
    if params.csv and params.csv.suffix.lower() != ".csv":
        parser.error("CSV output must use the .csv extension")
    if params.keepers < 1:
        parser.error("--keepers must be at least one")
    import cv2

    from workflow import write_results

    cv2.setNumThreads(1)
    answer = select(root, params.metadata, params.exiftool, profile=params.profile,
                    keepers=params.keepers, iqa_report=params.iqa_report, scene_clustering=not params.no_scene_clustering)
    write_results(answer, params.output, params.csv)
    summary = {
        "source_files": answer["source_files"], "capture_frames": answer["capture_frames"],
        "groups": len(answer["groups"]), "decisions": answer["decisions"],
        "untimed_singletons": answer["untimed_singletons"], "output": str(params.output.resolve()),
    }
    if "scene_clusters" in answer:
        summary["scenes"] = len(answer["scene_clusters"])
    print(json.dumps(summary, ensure_ascii=True))
