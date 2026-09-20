"""Shared figure mechanics: legend placement, legend defaults, whitespace.

Three review rulings (2026-09-19) land here so the same behaviour cannot
drift between the concatenated-file figures and the per-segment ones:

1. **A legend goes in a CORNER, never in the middle.** Matplotlib's
   ``loc="best"`` minimises overlap across nine positions, five of which are
   central, and on a dense raster it routinely parks the box across the middle
   of the data. :func:`legend_corner` scores only the four corners and takes
   the emptiest one, so the legend still dodges the data but can only ever land
   where a reader expects to find it.

2. **Legend text is publication shorthand, not a clause.** "one electrode's own
   rate" is a sentence fragment; "per electrode" is a legend entry. Labels live
   in :mod:`.figure_text`; this module only sets the box style.

3. **No unnecessary white space.** :func:`tighten` is the one place the margin
   policy is written down, so a figure that wastes half its canvas is a bug in
   one function rather than in every emitter.

Nothing here decides *what* a figure shows. These are presentation mechanics,
and they are deliberately usable by an analysis repo drawing a
publication-ready version of the same picture — the split between a diagnostic
and a presentation figure is a matter of which options a caller passes, not of
which code it calls (2026-09-19).
"""

import logging

logger = logging.getLogger(__name__)


# --- Legend box style -----------------------------------------------------

LEGEND_FONTSIZE = 7
LEGEND_FRAME_ALPHA = 0.85

# The four corners, in the order ties break. Upper right first: it is where a
# reader looks for a legend by default, so it wins whenever nothing argues
# against it.
_CORNERS = ("upper right", "upper left", "lower right", "lower left")

# The share of the axes each corner box is assumed to cover when scoring how
# crowded that corner is. Generous on purpose — a legend that scores against a
# box smaller than itself picks a corner it then overflows.
_CORNER_WIDTH = 0.38
_CORNER_HEIGHT = 0.34

# Scoring reads artist coordinates, and a threshold raster carries a few
# hundred thousand events. Striding to this many keeps the scan at a fixed
# cost; a deterministic stride (never a random sample) keeps the chosen corner
# reproducible for the same figure.
_MAX_SCORED_POINTS = 20000


def _corner_boxes():
    """(name, x0, x1, y0, y1) per corner, in axes coordinates."""
    left = (0.0, _CORNER_WIDTH)
    right = (1.0 - _CORNER_WIDTH, 1.0)
    lower = (0.0, _CORNER_HEIGHT)
    upper = (1.0 - _CORNER_HEIGHT, 1.0)
    spans = {
        "upper right": (right, upper),
        "upper left": (left, upper),
        "lower right": (right, lower),
        "lower left": (left, lower),
    }
    return [(name, x[0], x[1], y[0], y[1]) for name, (x, y) in spans.items()]


def _renderer(fig):
    """The renderer to measure artists against, or None if there is none.

    Mirrors :func:`mea_modules.diagnostics.channel_layout._renderer` — kept as
    its own copy rather than imported, because `channel_layout` already
    imports FROM this module (:func:`legend_corner`, :func:`tighten`), and
    importing back would be circular. Every figure built on this package's
    shared Agg canvas (:func:`.channel_layout._new_figure`) always has one;
    returns `None` for anything else so a caller degrades rather than raises.
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


def _text_axes_boxes(axis):
    """Each non-empty text artist's rendered footprint on `axis`, in axes
    coordinates, as `(x0, x1, y0, y1)`.

    `_axes_fraction_points` reads `axis.lines` and `axis.collections`, so
    plain text — `ax.text` draws neither — is invisible to it: a comparison
    annotation can sit dead centre in a corner's scoring box and still count
    as empty space (2026-09-19). The anchor point alone would not fix this
    either: a centred, wide label's own anchor can sit outside every corner
    box while the rendered text still reaches into one. Measuring the actual
    rendered box (not just its anchor) is what lets a caller test true overlap
    against a corner, whatever the text's alignment or rotation.

    Needs a renderer; returns `[]` (scores nothing, blocks no corner) when
    none is available, same fallback as :func:`_axes_fraction_points`.
    """
    renderer = _renderer(axis.figure)
    if renderer is None:
        return []

    to_axes = axis.transAxes.inverted()
    boxes = []
    for text in getattr(axis, "texts", ()):
        if not text.get_text():
            continue
        try:
            bbox = text.get_window_extent(renderer)
        except Exception:  # noqa: BLE001 - not yet drawable
            continue
        (x0, y0), (x1, y1) = to_axes.transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
        boxes.append((min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)))
    return boxes


def _boxes_overlap(a, b):
    """Whether axes-fraction boxes `a` and `b` (each `(x0, x1, y0, y1)`) overlap.

    A shared edge is not an overlap — strict inequalities throughout — which
    matches a corner box and a text box that merely touch being fine.
    """
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _axes_fraction_points(axis):
    """Every plotted point on `axis`, in axes coordinates, or None.

    None means "this axis cannot be scored" — an image or a mesh covers the
    whole canvas, so no corner is emptier than any other and the caller should
    fall back to the default corner rather than pretend it measured something.
    """
    import numpy as np

    if getattr(axis, "images", None):
        return None

    chunks = []
    for line in getattr(axis, "lines", ()):
        data = line.get_xydata()
        if data is not None and len(data):
            chunks.append(np.asarray(data, dtype=float))
    for collection in getattr(axis, "collections", ()):
        try:
            offsets = collection.get_offsets()
        except Exception:  # noqa: BLE001 - a mesh/quad collection has no offsets
            return None
        if offsets is not None and len(offsets):
            chunks.append(np.asarray(offsets, dtype=float))
    if not chunks:
        return None

    points = np.vstack(chunks)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return None
    if len(points) > _MAX_SCORED_POINTS:
        points = points[:: max(1, len(points) // _MAX_SCORED_POINTS)]

    to_axes = axis.transData + axis.transAxes.inverted()
    return np.asarray(to_axes.transform(points), dtype=float)


def emptiest_corner(axis, default="upper right"):
    """Name the corner of `axis` holding the fewest plotted points.

    Falls back to `default` when the axis cannot be scored (an image, a mesh,
    or nothing drawn yet) rather than guessing — see
    :func:`_axes_fraction_points`. A corner whose box overlaps a drawn TEXT
    artist (see :func:`_text_axes_boxes`) is never picked over a text-free
    one, whatever its point count: `points` alone cannot see an annotation,
    and a legend parked on top of one is unreadable either way (2026-09-19).
    """
    import numpy as np

    points = _axes_fraction_points(axis)
    if points is None:
        return default

    text_boxes = _text_axes_boxes(axis)

    best_name, best_key = default, None
    for name, x0, x1, y0, y1 in _corner_boxes():
        inside = (
            (points[:, 0] >= x0)
            & (points[:, 0] <= x1)
            & (points[:, 1] >= y0)
            & (points[:, 1] <= y1)
        )
        count = int(np.count_nonzero(inside))
        blocked = any(_boxes_overlap((x0, x1, y0, y1), box) for box in text_boxes)
        # `blocked` dominates the comparison (`False` sorts before `True`), so
        # a text-free corner always beats a texted one regardless of point
        # count; `count` only breaks ties within the same `blocked` state, and
        # strictly-less there keeps `_CORNERS` order as the final tie-break —
        # an empty, text-free figure still lands in the upper right.
        key = (blocked, count)
        if best_key is None or key < best_key:
            best_name, best_key = name, key
    logger.debug(
        "legend corner for %r: %s (blocked=%s, %s points)",
        axis, best_name, best_key[0], best_key[1],
    )
    return best_name


def legend_corner(axis, handles=None, labels=None, loc=None, **kwargs):
    """Draw `axis`'s legend in the emptiest corner. Returns the Legend.

    `loc` given pins the corner and skips the scoring, which is what a caller
    does when the figure's own geometry already decides the answer (a timeline
    whose bars always start at the left, say). Anything else is forwarded to
    ``axis.legend``, so a caller can still set `ncol`, `title` or `markerscale`.

    Scoring happens against what is ON the axis at call time, so call this
    AFTER the data is drawn — a legend placed first scores an empty canvas and
    always lands in the upper right.
    """
    corner = loc if loc is not None else emptiest_corner(axis)
    options = {
        "loc": corner,
        "fontsize": LEGEND_FONTSIZE,
        "framealpha": LEGEND_FRAME_ALPHA,
    }
    options.update(kwargs)
    if handles is not None and labels is not None:
        return axis.legend(handles, labels, **options)
    if handles is not None:
        return axis.legend(handles=handles, **options)
    return axis.legend(**options)


# --- White space ----------------------------------------------------------


def tighten(fig, rect=None, pad=0.6):
    """Fit `fig`'s axes to its canvas with a thin, uniform margin.

    One place, one margin policy. `pad` is in font-size units, as
    ``tight_layout`` takes it, and 0.6 is deliberately tighter than
    matplotlib's 1.08 default: these figures are read at a glance and a wide
    border costs pixels that could have been data (2026-09-19).

    `rect` reserves a strip — pass one only when something is drawn outside the
    axes that ``tight_layout`` cannot see, such as a figure-level legend.
    """
    try:
        if rect is None:
            fig.tight_layout(pad=pad)
        else:
            fig.tight_layout(rect=rect, pad=pad)
    except Exception as exc:  # noqa: BLE001 - layout is cosmetic, never fatal
        logger.debug("tight_layout skipped: %s", exc)
    return fig
