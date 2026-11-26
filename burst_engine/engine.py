"""Local burst segmentation and technical frame ranking."""

from dataclasses import dataclass
from functools import lru_cache
from math import exp, log1p, log2, sqrt
from pathlib import Path
from statistics import median

import cv2
import imagehash
import numpy as np
import rawpy
from PIL import Image, ImageOps


@dataclass(frozen=True)
class Frame:
    path: Path
    timestamp: float | None
    order: int
    camera: str | None = None
    lens: str | None = None
    focal_length: float | None = None
    burst_id: str | None = None


@dataclass(frozen=True)
class Preview:
    gray: np.ndarray
    phash: int


def read_image_variants(path, *long_edges):
    """Decode once, then make each size directly from the original RGB render."""
    if path.suffix.lower() == ".dng" or path.suffix.lower() in (".arw", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2"):
        with rawpy.imread(str(path)) as raw:
            image = Image.fromarray(raw.postprocess(half_size=True, use_camera_wb=True))
    else:
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
    variants = []
    for index, long_edge in enumerate(long_edges, 1):
        scaled = image if index == len(long_edges) else image.copy()
        scaled.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
        variants.append(np.asarray(scaled))
    return tuple(variants)


def read_image(path, long_edge=960):
    return read_image_variants(path, long_edge)[0]


def make_preview(rgb):
    scale = min(1.0, 640 / max(rgb.shape[:2]))
    width, height = round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)
    rgb = cv2.resize(rgb, (max(1, width), max(1, height)), interpolation=cv2.INTER_AREA)
    phash = int(str(imagehash.phash(Image.fromarray(rgb))), 16)
    return Preview(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), phash)


def _conflicts(first, second, key):
    left, right = getattr(first, key), getattr(second, key)
    return left is not None and right is not None and left != right


class Segmenter:
    """Solve a contiguous partition, with frame order used to resolve equal times."""

    def __init__(self, frames, previews, semantic_review=None, review_budget=0):
        self.frames = frames
        self.previews = previews
        self.semantic_review = semantic_review
        self.review_budget = max(0, review_budget)
        self.sift = cv2.SIFT_create(nfeatures=600)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.geometry_calls = 0
        self.reviewed = 0
        self.ambiguous = []
        self._features = lru_cache(maxsize=192)(self._features)
        self._visual = lru_cache(maxsize=8192)(self._visual)

    def _features(self, index):
        return self.sift.detectAndCompute(self.previews[index].gray, None)

    @staticmethod
    def _ratio_matches(matches):
        return {
            row[0].queryIdx: row[0].trainIdx
            for row in matches if len(row) == 2 and row[0].distance < 0.77 * row[1].distance
        }

    def _geometry(self, left, right):
        self.geometry_calls += 1
        kp1, d1 = self._features(left)
        kp2, d2 = self._features(right)
        if d1 is None or d2 is None or min(len(d1), len(d2)) < 16:
            return None
        forward = self._ratio_matches(self.matcher.knnMatch(d1, d2, k=2))
        backward = self._ratio_matches(self.matcher.knnMatch(d2, d1, k=2))
        mutual = [(i, j) for i, j in forward.items() if backward.get(j) == i]
        if len(mutual) < 10:
            return False
        p1 = np.float32([kp1[i].pt for i, _ in mutual])
        p2 = np.float32([kp2[j].pt for _, j in mutual])
        _, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
        if mask is None:
            return False
        keep = mask.ravel() != 0
        if keep.sum() < 10 or keep.mean() < 0.45:
            return False
        coverage = [
            cv2.contourArea(cv2.convexHull(points[keep])) / self.previews[index].gray.size
            for points, index in ((p1, left), (p2, right))
        ]
        return max(coverage) >= 0.12 and min(coverage) >= 0.025

    def _visual(self, left, right):
        distance = (self.previews[left].phash ^ self.previews[right].phash).bit_count()
        f1, f2 = self.frames[left].focal_length, self.frames[right].focal_length
        zoom = bool(f1 and f2 and abs(log2(f2 / f1)) > 0.18)
        if distance <= 9 and not zoom:
            return 2.1
        geometry = self._geometry(left, right)
        if geometry:
            return 1.85
        if geometry is None:
            return 0.4 if distance <= 18 else -0.2
        if distance <= 18 and not zoom:
            return 0.6
        return -2.0 if distance >= 25 else -1.15

    def _edge_scores(self):
        gaps = [None]
        for first, second in zip(self.frames, self.frames[1:]):
            gaps.append(
                second.timestamp - first.timestamp
                if first.timestamp is not None and second.timestamp is not None else None
            )
        visual_edges = [None]
        supported_gaps = [None]
        for index in range(1, len(self.frames)):
            first, second, gap = self.frames[index - 1], self.frames[index], gaps[index]
            visual = None
            if any(_conflicts(first, second, key) for key in ("camera", "lens", "burst_id")):
                pass
            elif first.burst_id and first.burst_id == second.burst_id:
                visual = 8.0
            elif gap is None or gap <= 15:
                visual = self._visual(index - 1, index)
            visual_edges.append(visual)
            if visual is not None and gap is not None and 0 < gap <= 12:
                distance = (self.previews[index - 1].phash ^ self.previews[index].phash).bit_count()
                supported_gaps.append(gap if visual >= 1.85 or distance <= 12 else None)
            else:
                supported_gaps.append(None)
        scores = [None]
        for index, visual in enumerate(visual_edges[1:], 1):
            if visual is None or visual == 8.0:
                scores.append(visual)
                continue
            local = [v for v in supported_gaps[max(1, index - 6):index + 7] if v is not None]
            pace = max(0.06, median(local)) if len(local) >= 3 else 0.18
            gap = gaps[index]
            tempo = 0.0 if gap is None else 0.6 - 1.18 * max(0.0, log2(max(gap, pace) / pace))
            scores.append(visual + max(-5.0, tempo))
        for index in range(1, len(scores) - 1):
            if scores[index] is None or scores[index + 1] is None:
                continue
            if max(scores[index], scores[index + 1]) >= 0.65 or min(scores[index], scores[index + 1]) < -2.0:
                continue
            interval = gaps[index:index + 2]
            if all(gap is not None and 0 <= gap <= 0.65 for gap in interval):
                if self._visual(index - 1, index + 1) > 1.4:
                    scores[index] += 1.55
                    scores[index + 1] += 1.55
        for index, score in enumerate(scores[1:], 1):
            if score is None or abs(score) > 0.38:
                continue
            self.ambiguous.append(index)
            if self.semantic_review and self.reviewed < self.review_budget:
                self.reviewed += 1
                if self.semantic_review(self.frames[max(0, index - 2):index + 2]) == "scene_changed":
                    scores[index] -= 1.0
        return scores

    def _partition(self, start, stop, scores):
        boundaries = [start] + [i for i in range(start + 1, stop) if scores[i] < 1.65] + [stop]
        prefix = {start: 0.0}
        for i in range(start + 1, stop + 1):
            prefix[i] = prefix[i - 1] + (scores[i] if i < stop else 0.0)
        cost = {start: 0.0}
        parents = {}
        for b, end in enumerate(boundaries[1:], 1):
            best = (float("inf"), start)
            candidates = boundaries[max(0, b - 64):b]
            for begin in reversed(candidates):
                value = cost[begin] - (prefix[end - 1] - prefix[begin]) + 0.08
                if value >= best[0]:
                    continue
                if any(len({getattr(frame, key) for frame in self.frames[begin:end]
                            if getattr(frame, key) is not None}) > 1 for key in ("camera", "lens", "burst_id")):
                    continue
                if end - begin >= 3 and b > 1 and self._visual(begin, end - 1) < -1:
                    mid = (begin + end - 1) // 2
                    if min(self._visual(begin, mid), self._visual(mid, end - 1)) <= 0.6:
                        value += 1.8
                if value < best[0]:
                    best = value, begin
            cost[end], parents[end] = best
        groups = []
        while stop > start:
            begin = parents[stop]
            if cost[stop] == float("inf"):
                raise ValueError("conflicting capture metadata inside a forced segment")
            groups.append(list(range(begin, stop)))
            stop = begin
        return groups[::-1]

    def run(self):
        scores = self._edge_scores()
        breaks = []
        identities = {}
        for index, (frame, score) in enumerate(zip(self.frames, scores)):
            known = {key: getattr(frame, key) for key in ("camera", "lens", "burst_id")
                     if getattr(frame, key) is not None}
            # A missing tag must not connect two confirmed physical identities.
            # Native IDs apply when both sides actually supply them.
            if (score is None or score <= -2.6 or any(
                key in identities and identities[key] != value for key, value in known.items()
            )):
                breaks.append(index)
                identities = {}
            identities.update(known)
        groups = []
        for start, stop in zip(breaks, breaks[1:] + [len(self.frames)]):
            groups.extend(self._partition(start, stop, scores))
        return groups


def technical_metrics(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    filtered = cv2.GaussianBlur(gray, (3, 3), 0.55)
    x = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3) / 8
    y = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3) / 8
    energy = (x * x + y * y).mean()
    clipped = float(((gray < 3) | (gray > 252)).mean())
    return {"detail": log1p(sqrt(float(energy))), "clip": clipped}


def contextual_blinks(rows):
    states = [row["blink"] for row in rows]
    if len(rows) < 3 or not all(row["faces"] == 1 and row.get("eye_geometry") for row in rows):
        return states
    geometry = [row["eye_geometry"] for row in rows]
    aperture = np.asarray([eye["aperture"] for eye in geometry])
    # Use the larger openings as a reference only if multiple comparable
    # frames support an open eye. All-closed groups retain the model result.
    n = max(2, (len(rows) + 1) // 2)
    reference = np.median(np.sort(aperture, axis=0)[-n:], axis=0)
    center = np.median([eye["center"] for eye in geometry], axis=0)
    face_scale = median(eye["scale"] for eye in geometry)
    for i, eye in enumerate(geometry):
        if states[i] is True or np.linalg.norm(np.array(eye["center"]) - center) > 0.40 * face_scale:
            continue
        drop = (reference >= 0.19) & (reference - aperture[i] >= 0.06) & (aperture[i] <= 0.71 * reference)
        strength = max(eye["activation"])
        if (drop.all() and strength >= 0.10) or (drop.any() and strength >= 0.38):
            states[i] = True
    return states


def ranking_keys(rows):
    portraits = sum(row["faces"] > 0 for row in rows) * 2 >= len(rows)
    eyes = [log1p(row["eye_detail"]) for row in rows if row["eye_detail"] is not None]
    reference_eye = median(eyes) if eyes else 0.0
    blink_states = contextual_blinks(rows)

    def score(index):
        row = rows[index]
        # Same-scale scores favour detail, with a small penalty for clipped pixels.
        eye = log1p(row["eye_detail"]) if row["eye_detail"] is not None else reference_eye
        detail = row["detail"] if not portraits else 0.55 * row["detail"] + 0.45 * eye
        blink = blink_states[index]
        state = (2 if blink is False else 1 if blink is None else 0) if portraits else 0
        return state, detail - 0.18 * row["clip"]

    return [score(index) for index in range(len(rows))]


def rank_group(rows):
    scores = ranking_keys(rows)
    return sorted(range(len(rows)), key=scores.__getitem__, reverse=True)
