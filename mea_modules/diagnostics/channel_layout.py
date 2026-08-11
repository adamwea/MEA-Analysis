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

# Legend/caption defaults. Every figure has to explain its own colours (Adam,
# 2026-08-11): a reader who has never seen this source must be able to decode
# grey-vs-red from the figure alone.
_LEGEND_FONTSIZE = 7
_LEGEND_FRAMEALPHA = 0.85
_CAPTION_FONTSIZE = 6.5
_CAPTION_COLOR = "#444444"
_WRAP_WIDTH = 46


def _wrap_label(text, width=_WRAP_WIDTH):
    """Soft-wrap a legend label so a long one cannot crowd the axes.

    Wrapping rather than truncating: the whole point of these labels is that
    they name the sibling artifact a subset feeds (``traces.png`` and friends),
    and a truncated filename is worse than no label at all.
    """
    import textwrap

    if not text:
        return text
    return "\n".join(textwrap.wrap(str(text), width=int(width)) or [str(text)])


def _legend_dot(color, label, size=6.0):
    """A legend handle that is actually visible.

    Scatter markers on these figures are 2-4 pt so the array reads as a shape;
    reusing those artists as legend handles produces a legend with invisible
    keys, so the handles are built at a legible size instead.
    """
    from matplotlib.lines import Line2D

    return Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=float(size),
        markerfacecolor=color,
        markeredgecolor="none",
        label=_wrap_label(label),
    )


def _legend_line(color, label, lw=1.4, linestyle="-"):
    """A legend handle for a drawn line (threshold rules, segment joins, fits)."""
    from matplotlib.lines import Line2D

    return Line2D([], [], color=color, lw=float(lw), linestyle=linestyle, label=_wrap_label(label))


def _legend_patch(color, label, alpha=1.0):
    """A legend handle for a shaded region (gap bands, reference ranges)."""
    from matplotlib.patches import Patch

    return Patch(facecolor=color, edgecolor="none", alpha=float(alpha), label=_wrap_label(label))


def _fold_caption(parts, width=118):
    """Join caption fragments into wrapped lines, dropping empties.

    Fragments come from :mod:`mea_modules.diagnostics.figure_text`, one per
    encoding the figure actually used, so the caption grows and shrinks with the
    figure rather than always printing the full glossary.
    """
    import textwrap

    text = "  ".join(str(part).strip() for part in parts if part and str(part).strip())
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width=int(width)) or [text])


def _add_caption(fig, text, fontsize=_CAPTION_FONTSIZE):
    """Put an explanatory caption under the axes without overlapping them.

    ``tight_layout`` is re-run with a bottom margin reserved, so the caption
    never lands on data whatever the figure size. Position is fixed, so the
    render stays deterministic.
    """
    if not text:
        fig.tight_layout()
        return
    lines = str(text).count("\n") + 1
    reserved = min(0.32, 0.045 * lines + 0.03)
    fig.tight_layout(rect=(0.0, reserved, 1.0, 1.0))
    fig.text(
        0.01,
        0.012,
        str(text),
        ha="left",
        va="bottom",
        fontsize=float(fontsize),
        color=_CAPTION_COLOR,
    )


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
    highlight_label=None,
    base_label=None,
    caption=None,
    figsize=_LAYOUT_FIGSIZE,
    dpi=_LAYOUT_DPI,
):
    """Scatter the recording's electrode positions to `out_path`; return the path.

    Every routed electrode is drawn grey; anything in `highlight_channel_ids`
    is drawn red on top. The usual use is to mark the representative channels a
    trace plot was made from, so a reviewer can see at a glance whether they
    sample the whole array or all sit in one corner.

    The figure always carries a LEGEND naming both colours (Adam, 2026-08-11) —
    grey and red mean nothing to a reader who has not read this source, and the
    red set in particular is only meaningful once you know *which other figure*
    those channels were drawn into. So `highlight_label` should name the sibling
    artifact by its real emitted filename, e.g.::

        highlight_label="representative channels — traced in traces.png"

    Callers with several downstream artifacts name them all
    (``"... — traced in traces.png, traces_realtime.png, psd.png"``); callers
    whose highlight is a flagged set point at the JSON that enumerates it
    (``"flagged channels — listed in bad_channels.json"``). Labels are wrapped,
    never truncated, so a filename always survives intact.

    `caption` adds a line under the axes for anything a legend key is too short
    to hold (what "representative" means, how the set was ranked).

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
    n_highlighted = int(highlighted.sum())

    fig = _new_figure(figsize, dpi)
    ax = fig.subplots()
    ax.scatter(xs[~highlighted], ys[~highlighted], s=4, c="#888888", alpha=0.75, linewidths=0)
    if highlighted.any():
        ax.scatter(xs[highlighted], ys[highlighted], s=10, c="#c0392b", alpha=0.9, linewidths=0)
    ax.set_title(title or "Channel layout")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    # Electrode spacing is isotropic; a stretched aspect makes clumps unreadable.
    ax.set_aspect("equal", adjustable="box")

    # The legend is not optional: it is the only thing on the figure that says
    # what grey and red are. Counts ride in the labels so the reader never has
    # to open a JSON to learn how big each set is.
    grey_label = base_label or "routed electrodes"
    handles = [
        _legend_dot("#888888", f"{grey_label} (n={len(channel_ids) - n_highlighted})"),
    ]
    if highlighted.any():
        red_label = highlight_label or "highlighted channels"
        handles.append(_legend_dot("#c0392b", f"{red_label} (n={n_highlighted})", size=7.0))
    ax.legend(
        handles=handles,
        loc="best",
        fontsize=_LEGEND_FONTSIZE,
        framealpha=_LEGEND_FRAMEALPHA,
        borderpad=0.5,
        labelspacing=0.7,
    )

    _add_caption(fig, caption)

    out_path = _save_and_release(fig, out_path)
    logger.info(
        "wrote channel layout: %s (%d channels, %d highlighted)",
        out_path,
        len(channel_ids),
        n_highlighted,
    )
    return out_path
