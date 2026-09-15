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
import math
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

# --- bottom matter geometry (Adam, 2026-08-11) ----------------------------
#
# A caption and a figure-level legend are BOTH drawn in the margin under the
# axes, in figure coordinates, so whichever is added second lands on top of the
# first unless one function owns that margin. None did: `_add_caption` reserved
# a band and wrote the caption bottom-LEFT, while `plot_traces` separately
# pinned its legend to the figure's bottom-CENTRE, and on capsule 05's
# traces.png the legend box sat squarely over the caption's third line. Two
# other emitters had already hit the same collision and each carried its own
# private work-around. `_add_caption` is now the single owner of this margin:
# it stacks caption and legend into disjoint horizontal bands and reserves
# exactly the height the two of them measure.
#
# Sizes are in INCHES, not figure fractions. A fixed fraction is wrong at both
# ends — the old 0.045-per-caption-line reserved a cramped strip on a 4.5 in
# layout sheet and over an inch of dead space on a 7.5 in trace stack — and it
# is exactly the kind of tuned constant that silently breaks at the next figure
# size. Inches are what text is measured in, so a band sized in inches holds
# its contents at every figure size.
_BOTTOM_PAD_IN = 0.08  # figure edge -> caption
_BOTTOM_GAP_IN = 0.10  # caption -> legend, and legend -> axes
_CAPTION_LINESPACING = 1.35  # matplotlib's default line multiple, made explicit

# Bottom matter may not eat the figure. A short figure with a long caption — a
# one-row small-multiple sheet is the real case — GROWS to hold its annotation
# rather than letting the annotation squeeze or cover the plot. Shrinking the
# data to fit the fine print is the wrong trade in a review figure.
_MAX_BOTTOM_FRACTION = 0.45

# Width one legend column needs before its wrapped label runs into the next.
# Used to fold a row of keys that would otherwise overrun a narrow sheet.
_LEGEND_COLUMN_WIDTH_IN = 2.6

# `tight_layout` fits each axes' own box into the rect it is given, but an
# artist that OVERHANGS its axes can still land outside — a rotated colour-bar
# label longer than the bar it labels is one. No rect can fix that (raising the
# rect shortens the bar, and the label does not shrink with it), so the result
# is measured and reported rather than chased: an emitter with an overhang has
# a figure to fix, not a margin to widen.
_LAYOUT_TOLERANCE_IN = 0.02


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


def _renderer(fig):
    """The renderer to measure artists against, or None if there is none.

    Every figure built by :func:`_new_figure` carries an Agg canvas, which
    always has one. Returns None rather than raising so a figure built some
    other way still lays out — callers fall back to an estimate.
    """
    canvas = getattr(fig, "canvas", None)
    for getter in (getattr(canvas, "get_renderer", None), getattr(fig, "_get_renderer", None)):
        if getter is None:
            continue
        try:
            renderer = getter()
        except Exception:  # pragma: no cover - exotic/headless canvases
            continue
        if renderer is not None:
            return renderer
    return None


def _artist_height_in(fig, artist, fallback_in=0.0):
    """Height of a drawn `artist` in INCHES, or `fallback_in` if unmeasurable.

    Measuring beats estimating for anything whose size depends on its own
    content: a legend's height is set by its key count, its column count and
    how far each label wrapped, none of which the caller reliably knows.
    """
    renderer = _renderer(fig)
    if renderer is None:
        return float(fallback_in)
    try:
        height_px = float(artist.get_window_extent(renderer).height)
    except Exception:  # pragma: no cover - defensive; artist not yet drawable
        return float(fallback_in)
    dpi = float(fig.dpi) or 1.0
    if not height_px > 0.0:
        return float(fallback_in)
    return height_px / dpi


def _lowest_axes_edge(fig, renderer):
    """Lowest point anything the axes draw reaches, in figure fractions.

    The TIGHT bounding box, so tick labels, axis labels and a colour bar's
    label all count: the reserved band exists to keep the caption and legend
    off ALL of that, not merely off the axes rectangles.
    """
    inverse = fig.transFigure.inverted()
    lowest = None
    for axis in fig.get_axes():
        if not axis.get_visible():
            continue
        try:
            box = axis.get_tightbbox(renderer)
        except Exception:  # pragma: no cover - defensive
            continue
        if box is None:
            continue
        edge = box.transformed(inverse).y0
        lowest = edge if lowest is None else min(lowest, edge)
    return lowest


def _fit_axes_above(fig, reserved):
    """Fit the axes above `reserved`, and report anything that overhangs.

    Deliberately ONE ``tight_layout`` pass. Re-fitting against a measured
    overhang looks like the obvious repair and is a trap: raising the rect
    shortens the axes without shortening the artist that overhangs them, so the
    loop squeezes the plot to a sliver chasing a label that was never going to
    fit. An overhang is a defect in the figure that drew it — say so, and leave
    the layout honest.
    """
    fig.tight_layout(rect=(0.0, reserved, 1.0, 1.0))
    renderer = _renderer(fig)
    if renderer is None:
        return
    lowest = _lowest_axes_edge(fig, renderer)
    tolerance = _LAYOUT_TOLERANCE_IN / (float(fig.get_figheight()) or 1.0)
    if lowest is not None and lowest < reserved - tolerance:
        logger.warning(
            "an axes artist overhangs the reserved bottom margin by %.2f in "
            "(likely a label longer than the axes it belongs to); the caption or "
            "legend may touch it",
            (reserved - lowest) * (float(fig.get_figheight()) or 1.0),
        )


def _legend_columns(fig, n_handles):
    """Key columns that fit across this figure — at least one, never more than
    there are keys. Derived from the figure width so a row of keys cannot run
    off the edge of a narrow sheet."""
    fits = int(float(fig.get_figwidth()) // _LEGEND_COLUMN_WIDTH_IN)
    return max(1, min(int(n_handles), fits or 1))


def _bottom_legend(
    fig,
    handles,
    fontsize=_LEGEND_FONTSIZE,
    framealpha=_LEGEND_FRAMEALPHA,
    ncol=None,
):
    """Create the figure-level legend that :func:`_add_caption` will place.

    For figures where no single axes can hold the key: a stack of trace panels
    or a sheet of small multiples is data edge to edge, and a key describing
    the whole figure does not belong inside one of its panels anyway.

    Created here, POSITIONED by :func:`_add_caption` — only that function knows
    how much room the caption underneath it needs. Callers pass their handles
    to ``_add_caption(..., legend_handles=...)`` rather than calling this.
    """
    handles = [handle for handle in (handles or ()) if handle is not None]
    if not handles:
        return None
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        bbox_transform=fig.transFigure,
        ncol=_legend_columns(fig, len(handles)) if ncol is None else max(1, int(ncol)),
        fontsize=fontsize,
        framealpha=framealpha,
    )
    # `tight_layout` must not try to fit it. Its position is computed against
    # the reserved band below; letting the layout engine react to it as well
    # makes the two fight over the same margin.
    legend.set_in_layout(False)
    return legend


def _estimated_legend_height_in(legend, fontsize):
    """Fallback height for a legend that could not be measured, in inches."""
    handles = list(getattr(legend, "legend_handles", None) or ())
    texts = [text.get_text() for text in legend.get_texts()] or [""]
    columns = max(1, int(getattr(legend, "_ncols", 0) or len(texts)))
    rows = math.ceil(len(handles or texts) / columns)
    tallest = max(str(text).count("\n") + 1 for text in texts)
    return rows * tallest * float(fontsize) * _CAPTION_LINESPACING / 72.0 + 0.10


def _add_caption(
    fig,
    text,
    fontsize=_CAPTION_FONTSIZE,
    legend_handles=None,
    legend_fontsize=_LEGEND_FONTSIZE,
    legend_framealpha=_LEGEND_FRAMEALPHA,
    legend_ncol=None,
):
    """Lay out the figure's bottom matter; return the legend, or None.

    Stacks, from the bottom edge upwards: caption, then the figure legend, then
    the axes ``tight_layout`` is re-run against. Each band is as tall as the
    thing in it actually measures, so the caption and the legend cannot land on
    each other, and neither can land on data — at any figure size, with any
    number of caption lines or legend keys.

    Pass ``legend_handles`` for a figure-wide key; a legend that belongs to one
    axes still goes through ``ax.legend`` and never reaches this margin.
    Positions are computed, not tuned, and depend only on the figure and its
    text, so the render stays deterministic.
    """
    text = "" if text is None else str(text)
    legend = _bottom_legend(
        fig,
        legend_handles,
        fontsize=legend_fontsize,
        framealpha=legend_framealpha,
        ncol=legend_ncol,
    )

    if not text and legend is None:
        fig.tight_layout()
        return None

    caption = None
    caption_height_in = 0.0
    if text:
        # Drawn first at a provisional position so it can be measured, then
        # moved: a multi-line caption's height depends on the font metrics, not
        # just on the line count.
        caption = fig.text(
            0.01,
            0.0,
            text,
            ha="left",
            va="bottom",
            fontsize=float(fontsize),
            color=_CAPTION_COLOR,
        )
        n_lines = text.count("\n") + 1
        caption_height_in = _artist_height_in(
            fig,
            caption,
            fallback_in=n_lines * float(fontsize) * _CAPTION_LINESPACING / 72.0,
        )

    legend_height_in = 0.0
    if legend is not None:
        legend_height_in = _artist_height_in(
            fig,
            legend,
            fallback_in=_estimated_legend_height_in(legend, legend_fontsize),
        )

    cursor_in = _BOTTOM_PAD_IN
    caption_y_in = cursor_in
    if caption is not None:
        cursor_in += caption_height_in + _BOTTOM_GAP_IN
    legend_y_in = cursor_in
    if legend is not None:
        cursor_in += legend_height_in + _BOTTOM_GAP_IN

    # Grow a figure too short to hold its own annotation. Heights above are in
    # inches and text does not rescale with the canvas, so resizing here leaves
    # every measurement valid — only the fractions below are re-derived. Done
    # BEFORE `tight_layout`, so the axes are fitted to the final canvas.
    figure_height_in = float(fig.get_figheight()) or 1.0
    if cursor_in > _MAX_BOTTOM_FRACTION * figure_height_in:
        grown_in = cursor_in / _MAX_BOTTOM_FRACTION
        logger.info(
            "figure bottom matter needs %.2f in of a %.2f in figure; growing it to "
            "%.2f in so the caption and legend cannot cover the plot",
            cursor_in,
            figure_height_in,
            grown_in,
        )
        fig.set_figheight(grown_in)
        figure_height_in = grown_in

    _fit_axes_above(fig, cursor_in / figure_height_in)

    if caption is not None:
        caption.set_position((0.01, caption_y_in / figure_height_in))
    if legend is not None:
        legend.set_bbox_to_anchor(
            (0.5, legend_y_in / figure_height_in), transform=fig.transFigure
        )
    return legend


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


def _save_and_release(fig, out_path, facecolor=None):
    """Write `fig` to `out_path` and drop its artists. Returns the Path.

    `facecolor` (default None) is forwarded to ``savefig`` only when given, so a
    dark-background figure (``style="presentation"`` in the emitters that draw on
    black) writes with its own face colour instead of the rcParam default. None
    keeps the historical call byte-for-byte for every existing white-background
    caller — the diagnostic review-figure family is untouched.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if facecolor is not None:
            fig.savefig(out_path, facecolor=facecolor)
        else:
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
