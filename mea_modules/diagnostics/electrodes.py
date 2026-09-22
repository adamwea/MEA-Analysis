"""Electrode geometry the diagnostics compute from: pitch, clusters, cluster centres.

The compute side (picking the channels whose traces are cached) needs these, and
must import no drawing code, so they live here; :mod:`.channel_layout` imports
them back for its figures.
"""

import logging

logger = logging.getLogger(__name__)


# Fallback neighbourhood radius when the layout is too sparse to estimate a
# pitch from. 50 um is roughly 3x the MaxOne 17.5 um pitch.
_FALLBACK_EPS_UM = 50.0


_PITCH_TO_EPS = 1.6


def _channel_xy(recording):
    """Return (channel_ids, x, y) or (channel_ids, None, None) if unusable.

    Some recordings carry no probe, and a few carry a malformed one; callers
    need to degrade gracefully rather than crash a whole review run.
    """
    import numpy as np

    channel_ids = list(recording.get_channel_ids())
    if not channel_ids:
        return channel_ids, None, None
    try:
        locations = np.asarray(recording.get_channel_locations(), dtype=float)
    except Exception:
        return channel_ids, None, None
    if locations.ndim != 2 or locations.shape[0] != len(channel_ids) or locations.shape[1] < 2:
        return channel_ids, None, None
    return channel_ids, locations[:, 0], locations[:, 1]


def estimate_electrode_pitch(x, y):
    """Median nearest-neighbour distance across the routed electrodes.

    This is the natural length scale of the layout, and the only sane basis for
    a clustering radius: hardcoding one breaks the moment a different Maxwell
    chip generation is used.
    """
    import numpy as np

    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.size < 2:
        return 0.0

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return 0.0
    return float(np.median(nearest))


def detect_electrode_clusters(x, y, eps=None):
    """Group electrodes into connected components within `eps` micrometres.

    Returns a list of clusters, each a sorted list of positional indices into
    `x`/`y`, largest cluster first. With `eps` None the radius is derived from
    the layout's own pitch (see :func:`estimate_electrode_pitch`).

    **No oversized-cluster warning (Adam, 2026-08-11).** There used to be a
    `max_cluster_size_warn=9` check here that logged whenever a cluster held
    more than nine electrodes, inherited from the previous build. It is gone,
    and should not come back: cluster size is DYNAMIC BY DESIGN. Nine is the
    usual size, the current validation run's MaxWell configuration uses twelve,
    and clusters legitimately deviate by an electrode or two to accommodate
    whatever routing was set in MaxWell's GUI. The check fired on 21 of 21
    segments of the P005843 review run — a warning that always fires carries no
    information and trains readers to ignore the log.
    """
    import numpy as np

    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    n_channels = int(xs.size)
    if n_channels == 0:
        return []

    if eps is None:
        pitch = estimate_electrode_pitch(xs, ys)
        eps = (pitch * _PITCH_TO_EPS) if pitch > 0 else _FALLBACK_EPS_UM

    # Dense adjacency: a recording config tops out around 1k channels, so the
    # n^2 matrix is a few megabytes and far simpler than a spatial index.
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    adj = (dx * dx + dy * dy) <= float(eps) ** 2
    np.fill_diagonal(adj, False)

    visited = np.zeros(n_channels, dtype=bool)
    clusters = []
    for index in range(n_channels):
        if visited[index]:
            continue
        stack = [int(index)]
        visited[index] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(int(current))
            for neighbor in np.flatnonzero(adj[current]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        clusters.append(sorted(component))

    clusters.sort(key=len, reverse=True)
    logger.debug(
        "grouped %d electrodes into %d clusters at eps=%.2f um (largest=%d)",
        n_channels,
        len(clusters),
        float(eps),
        max((len(cluster) for cluster in clusters), default=0),
    )
    return clusters


def cluster_center_channels(recording, channel_ids=None, eps=None):
    """One channel per electrode cluster: the member nearest its centroid.

    The count is DYNAMIC — it is however many clusters the routing produced,
    around thirty on the MaxWell configurations in use, and a caller must never
    assume a number (see :func:`detect_electrode_clusters`).

    `channel_ids` restricts the pool BEFORE clustering, which is the difference
    that matters: clustering the whole array and then filtering would hand back
    centres of clusters that the restricted set does not actually cover. Pass
    the shared electrode set here to get one representative per cluster of the
    electrodes common to every segment.

    Falls back to the (restricted) channel order when the probe carries no
    usable locations, so this never fails on a probe-less recording.
    """
    import numpy as np

    all_ids, xs, ys = _channel_xy(recording)
    if not all_ids:
        return []

    if channel_ids is None:
        pool_ids, pool_x, pool_y = all_ids, xs, ys
    else:
        wanted = {str(cid) for cid in channel_ids}
        keep = [index for index, cid in enumerate(all_ids) if str(cid) in wanted]
        if not keep:
            logger.warning("no requested channel is present in the recording; using all")
            pool_ids, pool_x, pool_y = all_ids, xs, ys
        else:
            pool_ids = [all_ids[index] for index in keep]
            index_array = np.asarray(keep, dtype=int)
            pool_x = None if xs is None else np.asarray(xs, dtype=float)[index_array]
            pool_y = None if ys is None else np.asarray(ys, dtype=float)[index_array]

    if pool_x is None:
        logger.warning("no usable channel locations; falling back to channel order")
        return list(pool_ids)

    clusters = detect_electrode_clusters(pool_x, pool_y, eps=eps)
    if not clusters:
        return list(pool_ids)
    return [pool_ids[_pick_cluster_representative(pool_x, pool_y, cluster)] for cluster in clusters]


def _pick_cluster_representative(x, y, cluster):
    """Positional index of the cluster member closest to the cluster centroid."""
    import numpy as np

    indices = np.asarray(cluster, dtype=int)
    xs = np.asarray(x, dtype=float)[indices]
    ys = np.asarray(y, dtype=float)[indices]
    dx = xs - float(np.mean(xs))
    dy = ys - float(np.mean(ys))
    return int(indices[int(np.argmin(dx * dx + dy * dy))])
