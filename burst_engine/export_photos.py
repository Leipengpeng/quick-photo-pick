"""Copy the evaluated original files into grouped and recommended folders."""

import hashlib
import re
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import quote


def local_link(path, root, *, directory=False):
    return quote(path.relative_to(root).as_posix()) + ("/" if directory else "")


def group_title(group, notes):
    title = next((notes.get(Path(row["path"]).name, {}).get("title") for row in group["frames"]
                  if notes.get(Path(row["path"]).name, {}).get("title")), "")
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", str(title)).strip(". ")[:32]


def capture_order(frame):
    return (frame.get("timestamp") or "9999", Path(frame["path"]).name.casefold(), frame["path"])


def apply_best_picture_rule(result):
    """Choose from the engine's two eligible recommendations for each group."""
    totals = Counter()
    for group in result["groups"]:
        target = 2 if len(group["frames"]) > 10 else 1
        group["keeper_target"] = target
        kept = 0
        for frame in group["frames"]:
            if frame["decision"] == "select":
                kept += 1
                if kept > target:
                    frame["decision"] = "lower_priority"
                    frame["recommended"] = False
            totals[frame["decision"]] += 1
    result.pop("keepers_per_group", None)
    result["best_picture_rule"] = {"up_to_10": 1, "over_10": 2}
    result["decisions"] = dict(totals)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def allocated_names(versions, reserved):
    names = [source.name for source in versions]
    suffix = 2
    while any(name.casefold() in reserved for name in names):
        names = [f"{source.stem}__{suffix:02}{source.suffix}" for source in versions]
        suffix += 1
    reserved.update(name.casefold() for name in names)
    return names


def export_photos(result, source, output, guide):
    """Export a byte-identical copy of every analyzed input, including variants."""
    collection = output / "已分组照片"
    collection.mkdir()
    notes = guide.get("photos", {})
    copied = set()
    by_group = {}
    by_file = {}
    manifest = []
    num_best_files = 0
    for group in result["groups"]:
        group_id = group["group"]
        label = group_title(group, notes)
        folder = collection / (f"第{group_id:02}组" + (f" {label}" if label else ""))
        best_folder = folder / "最佳图片"
        best_folder.mkdir(parents=True)
        by_group[group_id] = {"folder": local_link(folder, output, directory=True),
                              "best_folder": local_link(best_folder, output, directory=True)}
        reserved = set()
        for frame in sorted(group["frames"], key=capture_order):
            versions = [Path(value) for value in frame["versions"]]
            if any(not path.is_relative_to(source) for path in versions):
                raise ValueError("照片的实际路径必须在所选文件夹内")
            if any(path in copied for path in versions) or len(set(versions)) != len(versions):
                raise ValueError("同一文件出现在多个拍摄项")
            names = allocated_names(versions, reserved)
            for path, name in zip(versions, names):
                grouped_copy = folder / name
                shutil.copy2(path, grouped_copy)
                sha256 = digest(path)
                if digest(grouped_copy) != sha256:
                    raise OSError(f"分组照片校验失败 {grouped_copy}")
                best_copy = None
                if frame["decision"] == "select":
                    best_copy = best_folder / name
                    shutil.copy2(grouped_copy, best_copy)
                    if digest(best_copy) != sha256:
                        raise OSError(f"最佳图片校验失败 {best_copy}")
                    num_best_files += 1
                manifest.append({
                    "group": group_id, "rank": frame["rank"], "decision": frame["decision"],
                    "source": str(path), "sha256": sha256, "bytes": grouped_copy.stat().st_size,
                    "grouped_copy": grouped_copy.relative_to(output).as_posix(),
                    "best_copy": best_copy.relative_to(output).as_posix() if best_copy else None,
                })
                by_file[str(path)] = grouped_copy
                copied.add(path)
    if len(copied) != result["source_files"]:
        raise ValueError(f"导出文件数量不符，预计 {result['source_files']}，实际 {len(copied)}")
    return {"groups": by_group, "files": by_file, "records": manifest,
            "folder": collection, "best_picture_files": num_best_files}
