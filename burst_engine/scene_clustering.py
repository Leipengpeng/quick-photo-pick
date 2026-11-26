"""Group similar framings across capture bursts using visual evidence."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain, combinations, product
from pathlib import Path

import cv2
import numpy as np

from engine import Segmenter


_EXHAUSTIVE_LIMIT = 48
_NEIGHBOR_LIMIT = 12
_EXTRA_CHECKS_PER_GROUP = 32
_CONTRAST_CACHE = 128
_RECHECK_MUTUAL_MATCHES = 10
_RECHECK_PHASH_DISTANCE = 24


@dataclass(frozen=True)
class SceneFrame:
    path: Path
    shape: tuple[int, int]
    phash: int
    points: np.ndarray
    responses: np.ndarray
    descriptors: np.ndarray | None
    encoded_gray: bytes | None = None


class SceneFeatureExtractor:
    """Keep compact geometry and a compressed preview for a contrast recheck."""

    def __init__(self, *, cache_contrast=True):
        self.sift = cv2.SIFT_create(600, 3, 0.04, 10, 1.6, descriptorType=cv2.CV_8U)
        self.cache_contrast = cache_contrast

    def __call__(self, path, preview):
        encoded_gray = None
        if self.cache_contrast:
            saved, encoded = cv2.imencode(".jpg", preview.gray, [cv2.IMWRITE_JPEG_QUALITY, 97])
            if saved:
                encoded_gray = encoded.tobytes()
        return self.describe(path, preview.gray, preview.phash, encoded_gray)

    def describe(self, path, gray, phash, encoded_gray=None):
        points, descriptors = self.sift.detectAndCompute(gray, None)
        return SceneFrame(
            path,
            gray.shape,
            phash,
            np.asarray([point.pt for point in points], dtype=np.float32).reshape(-1, 2),
            np.asarray([point.response for point in points], dtype=np.float32),
            descriptors,
            encoded_gray,
        )


def _crop_overlap(transform, source_shape, target_shape):
    def corners(shape):
        height, width = shape
        return np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    source = corners(source_shape)
    # A transform crossing the plane at infinity cannot describe this crop.
    weights = source @ transform[2, :2] + transform[2, 2]
    if weights.min() <= 0 <= weights.max():
        return (0.0, 0.0)
    warped = cv2.perspectiveTransform(source[:, None, :], transform)[:, 0, :]
    if not np.isfinite(warped).all() or not cv2.isContourConvex(warped):
        return (0.0, 0.0)
    area = cv2.contourArea(warped, oriented=True)
    target_area = target_shape[0] * target_shape[1]
    if area <= 0 or not 0.5 <= area / target_area <= 2.0:
        return (0.0, 0.0)
    overlap, _ = cv2.intersectConvexConvex(warped, corners(target_shape))
    return min(1.0, overlap / area), min(1.0, overlap / target_area)


def _matching_evidence(left, right, matcher, tentative_matches=None):
    desc_a, desc_b = left.descriptors, right.descriptors
    if desc_a is None or desc_b is None or min(len(desc_a), len(desc_b)) < 16:
        return None
    # OpenCV's float SIFT matcher is substantially faster than its byte L2
    # path. Keep compact descriptors between comparisons.
    desc_a = desc_a.astype(np.float32)
    desc_b = desc_b.astype(np.float32)
    forward = Segmenter._ratio_matches(matcher.knnMatch(desc_a, desc_b, k=2))
    if len(forward) < 16:
        return None
    reverse = Segmenter._ratio_matches(matcher.knnMatch(desc_b, desc_a, k=2))
    matches = [(a, b) for a, b in forward.items() if reverse.get(b) == a]
    if tentative_matches is not None:
        tentative_matches.append(len(matches))
    if len(matches) < 16:
        return None

    first = left.points[[a for a, _ in matches]]
    second = right.points[[b for _, b in matches]]
    transform, mask = cv2.findHomography(first, second, cv2.RANSAC, 3.0)
    if transform is None or mask is None:
        return None
    keep = mask.ravel() != 0
    inliers = int(keep.sum())
    ratio = inliers / len(matches)
    if inliers < 16 or ratio < 0.55:
        return None

    coverage = [
        cv2.contourArea(cv2.convexHull(points[keep])) / (shape[0] * shape[1])
        for points, shape in ((first, left.shape), (second, right.shape))
    ]
    if min(coverage) < 0.10:
        return None
    overlap = _crop_overlap(transform, left.shape, right.shape)
    if min(overlap) < 0.55:
        return None
    return {
        "phash_distance": (left.phash ^ right.phash).bit_count(),
        "matches": len(matches),
        "inliers": inliers,
        "inlier_ratio": round(ratio, 4),
        "coverage": [round(value, 4) for value in coverage],
        "alignment_overlap": [round(value, 4) for value in overlap],
    }


def _candidate_pairs(grouped):
    if len(grouped) <= _EXHAUSTIVE_LIMIT:
        return list(combinations(grouped, 2))

    # Index shared pHash chunks and approximate RootSIFT words. Local words
    # still retrieve candidates when a crop or lighting change moves pHash.
    projection = np.random.default_rng(634).standard_normal((128, 32), dtype=np.float32)
    bit_weights = np.uint32(1) << np.arange(16, dtype=np.uint32)
    words_by_group = {}
    index = defaultdict(list)
    exact_hashes = defaultdict(list)
    for group, frames in grouped.items():
        words = set()
        for phash in {frame.phash for frame in frames}:
            exact_hashes[phash].append(group)
        for frame in frames:
            phash = frame.phash
            words.update((offset, (phash >> (offset * 12)) & 0xFFF) for offset in range(5))
            if frame.descriptors is None:
                continue
            strongest = np.argsort(-frame.responses, kind="stable")[:96]
            sample = frame.descriptors[strongest].astype(np.float32)
            root = np.sqrt(sample / np.maximum(sample.sum(axis=1, keepdims=True), 1e-6))
            projected = (root @ projection > 0).reshape(-1, 2, 16)
            codes = projected @ bit_weights
            words.update((5 + table, int(value)) for column in codes for table, value in enumerate(column))
        words_by_group[group] = words
        for word in words:
            index[word].append(group)

    # Consecutive exact-hash candidates survive even in very common buckets.
    pairs = {pair for hits in exact_hashes.values() for pair in zip(hits, hits[1:])}
    for group, words in words_by_group.items():
        votes = Counter()
        for word in words:
            hits = index[word]
            if 1 < len(hits) <= 48:
                for other in hits:
                    if other != group:
                        votes[other] += 2 if word[0] < 5 else 1
        best = sorted(votes, key=lambda other: (-votes[other], other))[:_NEIGHBOR_LIMIT]
        pairs.update(tuple(sorted((group, other))) for other in best)
    return sorted(pairs)


def _link_strength(link):
    return min(link["coverage"]), link["inlier_ratio"], link["inliers"]


def cluster_scenes(representatives_by_group, *, contrast_recheck=True):
    """Return scene groups and matched evidence for precomputed burst previews.

    Each value has a small, caller-selected list of SceneFrame values or
    (path, Preview) pairs. Empty entries remain singletons. Every pair within
    a merged cluster must pass geometry, which limits chaining across views.
    """
    describe = SceneFeatureExtractor(cache_contrast=contrast_recheck)
    grouped = {
        group: [frame if isinstance(frame, SceneFrame) else describe(*frame) for frame in frames]
        for group, frames in sorted(representatives_by_group.items())
    }
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    candidates = _candidate_pairs({group: frames for group, frames in grouped.items() if frames})
    checked = {}
    geometry_checks = 0
    contrast_geometry_checks = 0
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @lru_cache(maxsize=_CONTRAST_CACHE)
    def normalized(path, phash, jpeg):
        gray = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        return describe.describe(path, contrast.apply(gray), phash)

    def strongest(pairs, *, use_contrast=False, tentative_matches=None):
        nonlocal geometry_checks, contrast_geometry_checks
        best = None
        for left, right in pairs:
            if use_contrast:
                if left.encoded_gray is None or right.encoded_gray is None:
                    continue
                left = normalized(left.path, left.phash, left.encoded_gray)
                right = normalized(right.path, right.phash, right.encoded_gray)
                contrast_geometry_checks += 1
            else:
                geometry_checks += 1
            evidence = _matching_evidence(left, right, matcher, tentative_matches)
            if evidence is not None and (best is None or _link_strength(evidence) > _link_strength(best)):
                best = evidence | {"paths": [str(left.path), str(right.path)]}
        if use_contrast and best is not None:
            best["matching_pass"] = "contrast_normalized"
        return best

    def validate(a, b):
        left, right = grouped[a], grouped[b]
        tentative_matches = []
        core = list(product(left[:2], right[:2]))
        best = strongest(core, tentative_matches=tentative_matches)
        # Rechecking every unrelated group roughly quadrupled the scene stage.
        # A weak local or global cue can nominate a recheck, never a merge.
        worthwhile = max(tentative_matches, default=0) >= _RECHECK_MUTUAL_MATCHES or any(
            (first.phash ^ second.phash).bit_count() <= _RECHECK_PHASH_DISTANCE for first, second in core
        )
        if best is None and worthwhile:
            best = strongest(chain(product(left[2:], right), product(left[:2], right[2:])))
        if best is None and worthwhile and contrast_recheck:
            best = strongest(product(left, right), use_contrast=True)
        if best is not None:
            best["groups"] = [a, b]
        checked[a, b] = best
        return best

    for a, b in candidates:
        validate(a, b)
    seeds = [pair for pair, evidence in checked.items() if evidence is not None]
    seeds.sort(key=lambda pair: (
        checked[pair].get("matching_pass") != "contrast_normalized", _link_strength(checked[pair])
    ), reverse=True)
    maximum_checks = len(candidates) + len(grouped) * _EXTRA_CHECKS_PER_GROUP

    def can_merge(left, right):
        pairs = [tuple(sorted((a, b))) for a, b in product(sorted(left), sorted(right))]
        if any(pair in checked and checked[pair] is None for pair in pairs):
            return False
        for pair in pairs:
            if pair not in checked and (len(checked) >= maximum_checks or validate(*pair) is None):
                return False
        return True

    members = {group: {group} for group in grouped}
    owner = {group: group for group in grouped}
    for a, b in seeds:
        first, second = owner[a], owner[b]
        if first == second or not can_merge(members[first], members[second]):
            continue
        merged = members.pop(second)
        members[first].update(merged)
        for group in merged:
            owner[group] = first

    ordered = sorted(sorted(groups) for groups in members.values())
    clusters = [{"scene_id": index, "groups": groups} for index, groups in enumerate(ordered, 1)]
    return {
        "scene_clusters": clusters,
        "scene_links": [link for (a, b), link in checked.items() if link is not None and owner[a] == owner[b]],
        "candidate_pairs": len(checked),
        "geometry_checks": geometry_checks,
        "contrast_geometry_checks": contrast_geometry_checks,
    }
