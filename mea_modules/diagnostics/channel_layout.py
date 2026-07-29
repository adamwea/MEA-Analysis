"""Electrode geometry, and the layout review plot built on top of it.

A Maxwell recording config is not a regular grid: the routed electrodes come in
small tight clumps (typically 2-9 electrodes sitting within a pitch or two of
each other) scattered across the array. Most of what the other diagnostics do
starts from that fact — you want *one* channel per clump, not one per electrode
— so the clustering lives here next to the layout plot rather than inside the
trace emitter that consumes it.

Figures are built straight from :class:`matplotlib.figure.Figure` with an Agg
canvas. These emitters run headless inside pipeline workers and in loops over
segments, and pyplot would register every figure in a process-global manager
that only ``plt.close`` clears.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Fallback neighbourhood radius when the layout is too sparse to estimate a
# pitch from. 50 um is roughly 3x the MaxOne 17.5 um pitch.
_FALLBACK_EPS_UM = 50.0
_PITCH_TO_EPS = 1.6

_LAYOUT_FIGSIZE = (7.5, 4.5)
_LAYOUT_DPI = 180


def _new_figure(figsize, dpi):
    """Return a Figure with an Agg canvas already attached.

    Shared by the other diagnostics modules. Attaching the canvas is what makes
    ``fig.savefig`` work without pyplot ever being imported, which keeps these
    functions safe on a headless node and free of global figure state.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    return fig


def _save_and_release(fig, out_path):
    """Write `fig` to `out_path` and drop its artists. Returns the Path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path)
    finally:
        # Called in a loop over segments; releasing artists here keeps peak RSS
        # flat instead of growing with the segment count.
        fig.clear()
    return out_path


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


def detect_electrode_clusters(x, y, eps=None, max_cluster_size_warn=9):
    """Group electrodes into connected components within `eps` micrometres.

    Returns a list of clusters, each a sorted list of positional indices into
    `x`/`y`, largest cluster first. With `eps` None the radius is derived from
    the layout's own pitch (see :func:`estimate_electrode_pitch`).

    A cluster much larger than `max_cluster_size_warn` usually means `eps` has
    swallowed neighbouring clumps, so it is logged rather than silently used.
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
    oversized = [cluster for cluster in clusters if len(cluster) > int(max_cluster_size_warn)]
    if oversized:
        logger.warning(
            "found %d electrode clusters larger than %d (largest=%d) at eps=%.2f um; "
            "eps may be too large for this layout",
            len(oversized),
            int(max_cluster_size_warn),
            max(len(cluster) for cluster in clusters),
            float(eps),
        )
    return clusters


def _pick_cluster_representative(x, y, cluster):
    """Positional index of the cluster member closest to the cluster centroid."""
    import numpy as np

    indices = np.asarray(cluster, dtype=int)
    xs = np.asarray(x, dtype=float)[indices]
    ys = np.asarray(y, dtype=float)[indices]
    dx = xs - float(np.mean(xs))
    dy = ys - float(np.mean(ys))
    return int(indices[int(np.argmin(dx * dx + dy * dy))])


def plot_channel_layout(
    recording,
    out_path,
    title=None,
    highlight_channel_ids=None,
    figsize=_LAYOUT_FIGSIZE,
    dpi=_LAYOUT_DPI,
):
    """Scatter the recording's electrode positions to `out_path`; return the path.

    Every routed electrode is drawn grey; anything in `highlight_channel_ids`
    is drawn red on top. The usual use is to mark the representative channels a
    trace plot was made from, so a reviewer can see at a glance whether they
    sample the whole array or all sit in one corner.

    Reads probe geometry only — no traces.
    """
    import numpy as np

    channel_ids, xs, ys = _channel_xy(recording)
    if not channel_ids:
        raise ValueError("recording has no channels to plot")
    if xs is None:
        raise ValueError("recording has no usable channel locations; attach a probe first")

    highlight = set(map(str, highlight_channel_ids or ()))
    highlighted = np.asarray([str(channel_id) in highlight for channel_id in channel_ids], dtype=bool)

    fig = _new_figure(figsize, dpi)
    ax = fig.subplots()
    ax.scatter(xs[~highlighted], ys[~highlighted], s=4, c="#888888", alpha=0.75, linewidths=0)
    if highlighted.any():
        ax.scatter(xs[highlighted], ys[highlighted], s=10, c="#c0392b", alpha=0.9, linewidths=0)
    ax.set_title(title or "Channel layout")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    # Electrode spacing is isotropic; a stretched aspect makes clumps unreadable.
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    out_path = _save_and_release(fig, out_path)
    logger.info(
        "wrote channel layout: %s (%d channels, %d highlighted)",
        out_path,
        len(channel_ids),
        int(highlighted.sum()),
    )
    return out_path
