"""Regions and positions for the Mind scene, from the facts' own vectors.

Pure numpy, no I/O, deterministic for a given input (seeded farthest-point
k-means, SVD). The picture it produces:

  1. Every fact is projected to three dimensions by PCA over its vector,
     so distance on screen is distance in meaning.
  2. k-means over those positions finds the regions ("lobes"): k grows
     with the square root of the bank, between 3 and 12.
  3. Each region is named by what distinguishes it: the words of its
     facts' key segments weighted by how particular they are to this
     region (TF-IDF over regions, dates and filler words dropped), so a
     region reads "Pricing · Pro" rather than the title of the document
     most of its facts came from (2026-10-22; the widest segment alone
     named regions "Founder journal 1", "Documentation").
  4. Region centers are pushed apart until no two sit closer than
     `min_separation`, and each fact is drawn toward its region's center
     while keeping its local PCA offset, so regions read as clusters
     without lying about the geometry inside them.

The result scales to a radius of about `radius` units, the scene's unit
(70 since 2026-10-21: at 50 a 365-fact bank's regions overlapped the
core's glow into one bright blob).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

_SEED = 20261021

# Words that name nothing on their own: English filler, document furniture,
# and the key segments the extractor stamps on everything.
_STOP = frozenset("""
a an the and or of to for in on at by with from is are was were be been being as if then than so such any some all
each every this that these those there their they them it its into about over under between per via vs
document documents docs doc documentation passage section notes note item items part parts page pages file files
general misc other overview summary details detail status update updates list entry entries record records log
""".split())
_MONTHS = frozenset("january february march april may june july august september october november december jan feb mar apr jun jul aug sep sept oct nov dec".split())


def _words(segment: str) -> list[str]:
    out = []
    for w in "".join(ch.lower() if ch.isalnum() else " " for ch in segment).split():
        if len(w) < 3 or w in _STOP or w in _MONTHS or w.isdigit():
            continue
        out.append(w)
    return out


def name_regions(keys: list[Optional[str]], labels: np.ndarray, k: int) -> list[str]:
    """One name per region index (empty regions get ""), from the words of
    the keys' segments.

    A key reads general -> specific ("Founder journal|October|Pricing|Pro
    plan"), so the document's own name sits first and the subject last: a
    word weighs by how deep in the key it sits (last segment 1.0, first
    1/segments), summed over the region's facts and divided by the region's
    size, times how few regions carry the word at all (IDF with regions as
    the documents). The top word makes the name; a second joins only when
    it scores at least 40% of the first, and two words from one segment
    keep that segment's order ("Pro plan"). Deterministic: scores are
    rounded, ties fall to the word more facts carry, then alphabetical."""
    import math

    weight: list[dict[str, float]] = [{} for _ in range(k)]
    carried: list[Counter] = [Counter() for _ in range(k)]
    size = [0] * k
    pairs: Counter = Counter()  # (w1, w2) adjacent in one segment, in order
    for i, key in enumerate(keys):
        r = int(labels[i])
        if r < 0 or r >= k:
            continue
        size[r] += 1
        segs = [sg for sg in (key or "").split("|") if sg.strip()]
        depth_of: dict[str, float] = {}
        for s_idx, seg in enumerate(segs):
            ws = _words(seg)
            d = (s_idx + 1) / len(segs)
            for w in ws:
                depth_of[w] = max(depth_of.get(w, 0.0), d)
            for w1, w2 in zip(ws, ws[1:]):
                pairs[(w1, w2)] += 1
        for w, d in depth_of.items():
            weight[r][w] = weight[r].get(w, 0.0) + d
            carried[r][w] += 1
    regions_with = Counter()
    for c in carried:
        for w in c:
            regions_with[w] += 1
    names: list[str] = []
    for r in range(k):
        if not carried[r]:
            names.append("")
            continue
        scored = sorted(
            (
                -round((weight[r][w] / size[r]) * (math.log((k + 1) / (1 + regions_with[w])) + 0.05), 9),
                -carried[r][w],
                w,
            )
            for w in carried[r]
        )
        top = [scored[0][2]]
        if len(scored) > 1 and -scored[1][0] >= 0.4 * -scored[0][0]:
            top.append(scored[1][2])
        if len(top) == 2:
            if pairs[(top[1], top[0])] > pairs[(top[0], top[1])]:
                top.reverse()
            if pairs[(top[0], top[1])]:
                names.append(" ".join(w.capitalize() for w in top))
                continue
        names.append(" · ".join(w.capitalize() for w in top))
    return names


@dataclass
class MindLayout:
    positions: np.ndarray                  # (n, 3) float32
    region_of: np.ndarray                  # (n,) int
    regions: list[dict[str, Any]] = field(default_factory=list)  # {index, name, center, count}


def _kmeans(points: np.ndarray, k: int, iters: int = 20) -> np.ndarray:
    """Deterministic k-means: farthest-point seeding from the point with the
    largest norm, then `iters` Lloyd steps. Returns labels."""
    n = points.shape[0]
    if k <= 1 or n <= k:
        return np.zeros(n, dtype=int) if k <= 1 else np.arange(n) % k
    first = int(np.argmax(np.linalg.norm(points, axis=1)))
    seeds = [first]
    d2 = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(d2))
        seeds.append(nxt)
        d2 = np.minimum(d2, np.sum((points - points[nxt]) ** 2, axis=1))
    centers = points[seeds].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(dist, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = points[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    return labels


def _separate(centers: np.ndarray, min_separation: float, max_scale: float = 3.0) -> np.ndarray:
    """Scale centers out from their mean until the closest pair is at least
    `min_separation` apart (capped), then nudge any pair still too close."""
    if len(centers) < 2:
        return centers
    mean = centers.mean(axis=0)
    rel = centers - mean
    diff = rel[:, None, :] - rel[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    closest = float(dist.min())
    if closest > 0:
        rel = rel * min(max_scale, max(1.0, min_separation / closest))
    for _ in range(8):
        diff = rel[:, None, :] - rel[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        i, j = np.unravel_index(int(np.argmin(dist)), dist.shape)
        if dist[i, j] >= min_separation:
            break
        push = diff[i, j]
        norm = np.linalg.norm(push)
        if norm < 1e-6:
            push = np.array([1.0, 0.0, 0.0]); norm = 1.0
        step = (min_separation - dist[i, j]) / 2.0
        rel[i] += push / norm * step
        rel[j] -= push / norm * step
    return rel + mean


def layout_facts(
    vectors: list[list[float]],
    keys: list[Optional[str]],
    *,
    radius: float = 70.0,
    min_separation: float = 36.0,
    k: Optional[int] = None,
) -> MindLayout:
    """See the module docstring. `vectors` may be ragged or empty; facts
    without a usable vector are placed at the origin in region 0."""
    n = len(vectors)
    if n == 0:
        return MindLayout(np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=int), [])
    dims = Counter(len(v) for v in vectors if v)
    d = dims.most_common(1)[0][0] if dims else 0
    usable = np.array([i for i, v in enumerate(vectors) if v and len(v) == d], dtype=int)
    pos = np.zeros((n, 3), dtype=np.float64)
    if len(usable) >= 2 and d >= 3:
        X = np.asarray([vectors[i] for i in usable], dtype=np.float64)
        X = X - X.mean(axis=0, keepdims=True)
        # PCA by SVD; the sign of each component is pinned so the map is stable.
        _u, s, vt = np.linalg.svd(X, full_matrices=False)
        comps = vt[:3]
        signs = np.sign(comps[:, 0]); signs[signs == 0] = 1.0
        comps = comps * signs[:, None]
        P = X @ comps.T
        if P.shape[1] < 3:
            P = np.pad(P, ((0, 0), (0, 3 - P.shape[1])))
        spread = np.percentile(np.linalg.norm(P, axis=1), 95) or 1.0
        pos[usable] = P / spread * radius
    elif len(usable) == 1:
        pos[usable[0]] = 0.0

    if k is None:
        k = int(np.clip(round(np.sqrt(max(n, 1) / 6.0)), 3, 12))
    k = min(k, max(1, n))
    labels = _kmeans(pos, k) if n > 1 else np.zeros(1, dtype=int)

    centers = np.array([pos[labels == j].mean(axis=0) if np.any(labels == j) else np.zeros(3) for j in range(k)])
    spread_centers = _separate(centers.copy(), min_separation)
    # Keep each fact's local offset, drawn toward its region's (moved) center.
    final = pos.copy()
    for j in range(k):
        m = labels == j
        if not np.any(m):
            continue
        local = pos[m] - centers[j]
        final[m] = spread_centers[j] + local * 0.8

    regions: list[dict[str, Any]] = []
    region_names = name_regions(keys, labels, k)
    for j in range(k):
        m = labels == j
        count = int(m.sum())
        if count == 0:
            continue
        name = region_names[j] or f"Region {j + 1}"
        regions.append({
            "index": j,
            "name": name[:48],
            "center": [float(x) for x in spread_centers[j]],
            "count": count,
        })
    # Regions that share a name get a number so two lobes never read as one.
    seen: Counter = Counter(r["name"] for r in regions)
    nth: Counter = Counter()
    for r in regions:
        if seen[r["name"]] > 1:
            nth[r["name"]] += 1
            r["name"] = f"{r['name']} {nth[r['name']]}"
    return MindLayout(final.astype(np.float32), labels.astype(int), regions)
