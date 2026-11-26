"""Build capture groups from local photo paths and EXIF metadata."""

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Callable, Iterable, Mapping

import cv2
import rawpy

from engine import Frame, Preview, Segmenter, make_preview, read_image


RAW_EXTENSIONS = frozenset({".dng", ".arw", ".nef", ".cr2", ".cr3", ".raf", ".orf", ".rw2"})
JPEG_EXTENSIONS = frozenset({".jpg", ".jpeg"})
PRESERVE_KINDS = frozenset({"bracket", "panorama", "hdr_panorama", "focus_stack"})
UNKNOWN_VALUES = frozenset({"", "unknown", "undefined", "none", "n/a", "null", "-", "not available"})
MAX_INFERRED_GAP_SECONDS = 15
PREVIEW_ERRORS = (OSError, ValueError, RuntimeError, rawpy.LibRawError, cv2.error)
IDENTIFIER_TAGS = frozenset({"BodySerialNumber", "SerialNumber", "LensSerialNumber", "BurstUUID"})


@dataclass(frozen=True)
class CaptureGroup:
    frames: tuple[Frame, ...]
    basis: str
    preserve_all: bool


@dataclass
class GroupingResult:
    groups: list[CaptureGroup]
    versions: dict[Path, tuple[Path, ...]]
    tags: dict[Path, dict]
    review_boundaries: list[tuple[Path, Path]]
    warnings: list[str]
    geometry_calls: int = 0
    reviewed: int = 0
    preview_sources: dict[Path, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class Identity:
    make: str | None = None
    model: str | None = None
    serial: str | None = None

    def conflicts(self, other):
        return any(
            left and right and left != right
            for left, right in zip((self.make, self.model, self.serial),
                                   (other.make, other.model, other.serial))
        )

    def merged(self, other):
        return Identity(self.make or other.make, self.model or other.model, self.serial or other.serial)

    def label(self):
        parts = [part for part in (self.make, self.model, self.serial) if part]
        return "/".join(parts) if parts else None


@dataclass(frozen=True)
class Capture:
    path: Path
    version_paths: tuple[Path, ...]
    tags: dict
    timestamp: float | None
    order: int
    camera: Identity
    lens: Identity
    focal_length: float | None
    burst_id: str | None
    sequence_kind: str | None
    sequence_id: str | None

    def authority(self):
        if self.sequence_kind and self.sequence_id:
            return ("exposure_sequence", self.path.parent, self.sequence_kind, self.sequence_id)
        if self.burst_id:
            return ("native_burst", self.burst_id)
        return None

    def as_frame(self, camera, lens):
        return Frame(
            self.path, self.timestamp, self.order, camera.label(), lens.label(),
            self.focal_length, self.burst_id,
        )


def _text(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = unicodedata.normalize("NFKC", str(value).rstrip("\x00")).casefold()
    text = " ".join(text.split())
    return None if text in UNKNOWN_VALUES else text


def _identifier(value):
    text = _text(value)
    if text is None or re.fullmatch(r"[0-]+", text):
        return None
    return text


def _camera(tags):
    make, model = _text(tags.get("Make")), _text(tags.get("Model"))
    serial = _identifier(tags.get("BodySerialNumber")) or _identifier(tags.get("SerialNumber"))
    return Identity(make, model, serial)


def _lens(tags):
    return Identity(
        _text(tags.get("LensMake")), _text(tags.get("LensModel")),
        _identifier(tags.get("LensSerialNumber")),
    )


def _capture_stamp(tags):
    original = str(tags.get("DateTimeOriginal") or "").strip()
    if len(original) >= 10 and original[4] == ":":
        original = original[:10].replace(":", "-") + original[10:]
    # ExifTool supplies the timezone and fractional seconds separately, while
    # some sidecars put both on DateTimeOriginal.
    match = re.fullmatch(r"(.{19})(\.\d+)?(Z|[+-]\d{2}:?\d{2})?", original)
    if not match:
        return None, False
    base, fraction, zone = match.groups()
    subsec = str(tags.get("SubSecTimeOriginal", "")).strip()
    if fraction is None and subsec.isascii() and subsec.isdecimal():
        fraction = "." + subsec
    offset = str(tags.get("OffsetTimeOriginal") or "")
    if zone is None and re.fullmatch(r"[+-]\d{2}:?\d{2}", offset):
        zone = offset
    try:
        stamp = datetime.fromisoformat(base + (fraction or "") + (zone or ""))
    except ValueError:
        return None, False
    return stamp.replace(tzinfo=stamp.tzinfo or timezone.utc).timestamp(), bool(fraction)


def _focal_length(tags):
    value = tags.get("FocalLength")
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            value = float(value[0]) / float(value[1])
        result = float(value.removesuffix(" mm") if isinstance(value, str) else value)
        return result if isfinite(result) and result > 0 else None
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        return None


def _natural_path(path):
    return tuple(
        (0, int(part)) if part.isascii() and part.isdecimal() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.as_posix())
    )


def _same_capture(raw_tags, jpeg_tags):
    raw_time, raw_subsec = _capture_stamp(raw_tags)
    jpeg_time, jpeg_subsec = _capture_stamp(jpeg_tags)
    if raw_time is None or jpeg_time is None:
        return False
    if raw_subsec and jpeg_subsec:
        if raw_time != jpeg_time:
            return False
    elif raw_time // 1 != jpeg_time // 1:
        return False
    raw_camera, jpeg_camera = _camera(raw_tags), _camera(jpeg_tags)
    same_model = raw_camera.model and raw_camera.model == jpeg_camera.model
    same_serial = raw_camera.serial and raw_camera.serial == jpeg_camera.serial
    if not (same_model or same_serial) or raw_camera.conflicts(jpeg_camera):
        return False
    raw_focal, jpeg_focal = _focal_length(raw_tags), _focal_length(jpeg_tags)
    zoom_disagrees = (
        raw_focal and jpeg_focal
        and abs(raw_focal - jpeg_focal) > 0.01 * max(raw_focal, jpeg_focal)
    )
    return not zoom_disagrees and not _lens(raw_tags).conflicts(_lens(jpeg_tags))


def _versions(paths, tags):
    families = defaultdict(list)
    for path in paths:
        families[(path.parent, path.stem.casefold())].append(path)
    for files in families.values():
        raw = [path for path in files if path.suffix.lower() in RAW_EXTENSIONS]
        jpeg = [path for path in files if path.suffix.lower() in JPEG_EXTENSIONS]
        if len(raw) == 1 and len(jpeg) == 1 and _same_capture(tags.get(raw[0], {}), tags.get(jpeg[0], {})):
            merged = dict(tags.get(jpeg[0], {}))
            merged.update({
                key: value for key, value in tags.get(raw[0], {}).items()
                if (_identifier(value) if key in IDENTIFIER_TAGS else _text(value))
            })
            yield raw[0], (raw[0], jpeg[0]), merged
            for path in files:
                if path != raw[0] and path != jpeg[0]:
                    yield path, (path,), dict(tags.get(path, {}))
        else:
            for path in files:
                yield path, (path,), dict(tags.get(path, {}))


def _make_captures(paths, tags):
    variants = sorted(_versions(paths, tags), key=lambda row: _natural_path(row[0]))
    captures = []
    for order, (path, aliases, info) in enumerate(variants):
        kind = (_text(info.get("SequenceKind")) or "").replace(" ", "_").replace("-", "_")
        captures.append(Capture(
            path, aliases, info, _capture_stamp(info)[0], order,
            _camera(info), _lens(info), _focal_length(info), _identifier(info.get("BurstUUID")),
            kind if kind in PRESERVE_KINDS else None,
            _text(info.get("SequenceGroupId")) or _text(info.get("SequenceGroupID")),
        ))
    return captures


def _chronology(capture):
    return (capture.timestamp is None, capture.timestamp or 0, capture.order)


def _compatible_spans(captures):
    current = []
    camera, lens = Identity(), Identity()
    for capture in captures:
        if current and (camera.conflicts(capture.camera) or lens.conflicts(capture.lens)
                        or capture.sequence_kind != current[-1].sequence_kind):
            yield current, camera, lens
            current = []
            camera, lens = Identity(), Identity()
        current.append(capture)
        camera = camera.merged(capture.camera)
        lens = lens.merged(capture.lens)
    if current:
        yield current, camera, lens


def _default_preview(path):
    return make_preview(read_image(path, 640))


def build_capture_groups(
    paths: Iterable[Path],
    tags: Mapping[Path, Mapping],
    *,
    preview_factory: Callable[[Path], Preview] | None = None,
    on_preview_read: Callable[[Path], None] | None = None,
    on_group_previews: Callable[[CaptureGroup, list[tuple[Path, Preview]]], None] | None = None,
    semantic_review=None,
    review_budget: int = 0,
) -> GroupingResult:
    """Use trusted capture IDs, metadata constraints, and timestamped matching.

    The preview factory receives the primary path or its verified JPEG
    companion if decoding the primary fails. The optional on_preview_read
    callback runs after a preview was read successfully. Remaining images without
    trusted IDs require a timestamp for automatic grouping. The group preview
    callback can reuse visual previews before they are released.
    """
    preview_factory = preview_factory or _default_preview
    captures = _make_captures(sorted(set(paths), key=_natural_path), tags)
    result = GroupingResult(
        [], {capture.path: capture.version_paths for capture in captures},
        {capture.path: capture.tags for capture in captures}, [], [],
    )
    ordered_captures = sorted(captures, key=_chronology)
    authority = defaultdict(list)
    for capture in ordered_captures:
        key = capture.authority()
        if key is not None:
            authority[key].append(capture)
    anchored = {}
    for key, members in authority.items():
        spans = list(_compatible_spans(members))
        if len(spans) > 1:
            result.warnings.append(f"{members[0].path.name} 的相机、镜头或序列元数据存在冲突")
        for span, camera, lens in spans:
            group = CaptureGroup(
                tuple(capture.as_frame(camera, lens) for capture in span),
                "exposure_sequence" if span[0].sequence_kind else key[0],
                bool(span[0].sequence_kind),
            )
            anchored.update({capture.path: group for capture in span})

    def single(capture, basis):
        return CaptureGroup(
            (capture.as_frame(capture.camera, capture.lens),), basis, bool(capture.sequence_kind),
        )

    candidates = []
    camera, lens = Identity(), Identity()

    def review(frames):
        try:
            return semantic_review(frames)
        except Exception as error:
            result.warnings.append(f"{frames[0].path.name} 的边界辅助复核失败（{type(error).__name__}）")
            return None

    def flush():
        nonlocal camera, lens
        if not candidates:
            return
        frames = [capture.as_frame(camera, lens) for capture, _ in candidates]
        previews = [preview for _, preview in candidates]
        segmenter = Segmenter(
            frames, previews, semantic_review=review if semantic_review else None,
            review_budget=max(0, review_budget - result.reviewed),
        )
        preserve = bool(candidates[0][0].sequence_kind)
        basis = "exposure_sequence" if preserve else "visual_timeline"
        for indices in segmenter.run():
            group = CaptureGroup(tuple(frames[i] for i in indices), basis, preserve)
            result.groups.append(group)
            if on_group_previews is not None and not group.preserve_all:
                on_group_previews(group, [(result.preview_sources[frames[i].path], previews[i]) for i in indices])
        result.review_boundaries.extend(
            (frames[index - 1].path, frames[index].path) for index in segmenter.ambiguous
        )
        result.geometry_calls += segmenter.geometry_calls
        result.reviewed += segmenter.reviewed
        candidates.clear()
        camera, lens = Identity(), Identity()

    emitted = set()
    for capture in ordered_captures:
        native = anchored.get(capture.path)
        if native is not None or capture.timestamp is None:
            flush()
            if native is None:
                result.groups.append(single(capture, "missing_timestamp"))
            elif native.frames[0].path not in emitted:
                result.groups.append(native)
                emitted.add(native.frames[0].path)
            continue
        if candidates:
            previous = candidates[-1][0]
            if (capture.timestamp - previous.timestamp > MAX_INFERRED_GAP_SECONDS
                    or camera.conflicts(capture.camera) or lens.conflicts(capture.lens)
                    or capture.sequence_kind != previous.sequence_kind):
                flush()
        for source in capture.version_paths:
            try:
                preview = preview_factory(source)
            except PREVIEW_ERRORS as error:
                last_error = error
                continue
            result.preview_sources[capture.path] = source
            if source != capture.path:
                result.warnings.append(f"{capture.path.name} 的 RAW 预览读取失败，已用配对的 {source.name}")
            break
        else:
            flush()
            result.groups.append(single(capture, "unreadable"))
            result.warnings.append(f"{capture.path.name} 的图像预览读取失败（{last_error}）")
            continue
        if on_preview_read is not None:
            on_preview_read(source)
        candidates.append((capture, preview))
        camera = camera.merged(capture.camera)
        lens = lens.merged(capture.lens)
    flush()
    return result
