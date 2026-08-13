"""Whole-chip activity map — the amplitude-weighted firing-rate field.

The per-unit "global firing rate" map (capsule 19) draws one marker per unit at
that unit's template extremum. Adam's review of it (2026-08-12) asked for the
other thing: *information at all channels, not at unit locations only*. This
module builds that — a value for **every electrode in the dense union**, so the
whole chip carries a number and empty regions read as genuinely quiet rather
than merely un-plotted.

The quantity (Adam's choice) is **template-projected activity**::

    activity[e] = sum over units u of ( rate_u * ptp_u[e] )

where ``rate_u`` is unit *u*'s firing rate (spikes / recorded second) and
``ptp_u[e]`` is the peak-to-peak amplitude of *u*'s DENSE template on electrode
*e* (over the covered channels of 18's curated bundle). Units are microvolt-
hertz (µV·Hz): an amplitude-weighted firing-rate field that projects all
reconstructed spiking onto the array. A loud, fast unit lights up its whole
footprint; a quiet or small one barely registers; a channel no unit's footprint
reaches sits at zero.

Two public pieces, deliberately split so the science is testable without a
figure:

* :func:`template_projected_activity` — templates + weights + rates -> the
  per-channel field. Pure numpy, chunked over units, NaN-safe.
* :func:`plot_whole_chip_activity` — the field + electrode positions -> one
  figure. Presentation-clean by default (short title, compact colorbar, no
  caption or legend box), with a ``style="diagnostic"`` knob for the verbose
  variant. Renders as a gridded image when the array is a regular grid (a
  MaxWell config is 17.5 µm on both axes), falling back to a square-marker
  scatter otherwise. The colour scale goes log when the dynamic range demands
  it (``scale="auto"``), which an amplitude-weighted rate field almost always
  does.

Figures are built straight from :class:`matplotlib.figure.Figure` on an Agg
canvas — no pyplot, so this is safe on a headless node and leaves no global
figure state behind. Output is deterministic: a fixed input renders identical
PNG bytes.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# The figure this module emits, named once so the capsule and its tests agree.
WHOLE_CHIP_ACTIVITY_FILENAME = "whole_chip_activity_map.png"

# `scale="auto"` switches to a log colour scale when the ratio of the largest
# to the smallest positive activity exceeds this. Matches capsule 19's rate-map
# threshold so the two figures make the same call on the same data.
DEFAULT_LOG_RATIO = 20.0

# A MaxWell array is near-square; a near-square sheet keeps the equal-aspect
# axes from leaving big empty bands.
_PRESENTATION_FIGSIZE = (8.0, 8.0)
_DEFAULT_DPI = 200

# Grid inference tolerance: a position may deviate from an exact pitch multiple
# by this fraction of the pitch before the layout is judged irregular and the
# renderer falls back to scatter.
_GRID_SNAP_TOL = 0.15
# Refuse to build a grid whose cell count balloons past this multiple of the
# channel count — a degenerate pitch estimate must not allocate a huge array.
_GRID_MAX_FILL_INVERSE = 6.0


def template_projected_activity(templates, weight, rates, *, chunk=32):
    """Per-channel amplitude-weighted firing-rate field, in µV·Hz.

    ``activity[e] = sum_u rate_u * ptp_u[e]``, where ``ptp_u[e]`` is the
    peak-to-peak amplitude over samples of unit *u*'s dense template on
    electrode *e*, taken only where the unit actually contributes
    (``weight > 0``); everywhere else the unit adds nothing.

    Parameters
    ----------
    templates : array-like, shape (n_units, n_channels, n_samples)
        The curated dense bundle's templates (18 dense_merge_apply's
        ``merged_templates.npy`` orientation). May be a memmap and may carry
        NaNs on uncovered channels; both are handled.
    weight : array-like, shape (n_units, n_channels)
        Contributing weight per unit and channel; ``> 0`` marks a covered
        channel (``contributing_weight.npy``).
    rates : array-like, shape (n_units,)
        Firing rate per unit in hertz, aligned to the template unit axis. NaN
        or non-finite entries contribute zero (a unit with no spike count).
    chunk : int
        Units processed per block, to bound peak memory on a 13k-channel dense
        bundle. Does not affect the result.

    Returns
    -------
    numpy.ndarray, shape (n_channels,), dtype float64
        The activity field. Deterministic for a given input.
    """
    import numpy as np

    weight = np.asarray(weight)
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D (n_units, n_channels); got {weight.shape}")
    n_units, n_channels = int(weight.shape[0]), int(weight.shape[1])

    t_shape = getattr(templates, "shape", None)
    if t_shape is None or len(t_shape) != 3:
        raise ValueError(
            "templates must be 3-D (n_units, n_channels, n_samples); "
            f"got shape {t_shape}"
        )
    if t_shape[0] != n_units or t_shape[1] != n_channels:
        raise ValueError(
            "templates/weight shape mismatch: templates is "
            f"{tuple(t_shape)}, weight is {weight.shape} — expected the first "
            "two axes to be (n_units, n_channels)"
        )

    rates = np.nan_to_num(
        np.asarray(rates, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    ).reshape(-1)
    if rates.shape[0] != n_units:
        raise ValueError(
            f"rates has {rates.shape[0]} entries but there are {n_units} units"
        )

    activity = np.zeros(n_channels, dtype=float)
    step = max(1, int(chunk))
    for start in range(0, n_units, step):
        stop = min(start + step, n_units)
        block = np.nan_to_num(
            np.asarray(templates[start:stop], dtype=float), nan=0.0
        )
        ptp = np.ptp(block, axis=2)  # (block_units, n_channels)
        covered = np.asarray(weight[start:stop]) > 0
        ptp = np.where(covered, ptp, 0.0)
        activity += (ptp * rates[start:stop][:, None]).sum(axis=0)
    return activity


def _infer_grid(locations):
    """Snap electrode positions to a regular grid, or return None.

    A MaxWell recording config sits on a 17.5 µm lattice, so most electrodes
    map cleanly onto integer (row, col) cells and the field can be drawn as a
    crisp image with one pixel per electrode. Anything that does not snap
    cleanly — an irregular layout, a degenerate pitch estimate — returns None,
    and the caller scatters instead.
    """
    import numpy as np

    xs = np.asarray(locations[:, 0], dtype=float)
    ys = np.asarray(locations[:, 1], dtype=float)
    if xs.size == 0:
        return None

    def _pitch(values):
        uniq = np.unique(np.round(values, 3))
        if uniq.size < 2:
            return None, uniq
        diffs = np.diff(uniq)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return None, uniq
        return float(np.median(diffs)), uniq

    pitch_x, _ = _pitch(xs)
    pitch_y, _ = _pitch(ys)
    if not pitch_x or not pitch_y:
        return None

    x0, y0 = float(xs.min()), float(ys.min())
    col_f = (xs - x0) / pitch_x
    row_f = (ys - y0) / pitch_y
    cols = np.round(col_f).astype(int)
    rows = np.round(row_f).astype(int)
    # Every electrode must land within tolerance of a lattice node.
    if np.max(np.abs(col_f - cols)) > _GRID_SNAP_TOL:
        return None
    if np.max(np.abs(row_f - rows)) > _GRID_SNAP_TOL:
        return None

    n_cols = int(cols.max()) + 1
    n_rows = int(rows.max()) + 1
    if n_rows * n_cols > _GRID_MAX_FILL_INVERSE * xs.size:
        return None
    # No two electrodes may claim the same cell (a real grid never does).
    flat = rows.astype(np.int64) * n_cols + cols.astype(np.int64)
    if np.unique(flat).size != flat.size:
        return None

    return {
        "rows": rows,
        "cols": cols,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
        "x0": x0,
        "y0": y0,
        "fill_fraction": float(xs.size) / float(n_rows * n_cols),
    }


def _new_figure(figsize, dpi):
    """A Figure with an Agg canvas attached — savefig without pyplot."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    return fig


def _resolve_cmap(name):
    """A copy of colormap `name` whose "bad" (masked) colour is its low end.

    Absent grid cells (electrodes that do not exist) are masked; painting them
    the colormap's lowest colour makes the few interior holes vanish into the
    quiet background rather than punching bright specks through the field.
    """
    import matplotlib

    base = matplotlib.colormaps[name]
    return base.with_extremes(bad=base(0.0))


def plot_whole_chip_activity(
    activity,
    locations,
    out_path,
    *,
    style="presentation",
    cmap="inferno",
    scale="auto",
    render="auto",
    log_ratio=DEFAULT_LOG_RATIO,
    dpi=_DEFAULT_DPI,
    marker_scale=1.0,
    title=None,
    caption=None,
    colorbar_label=None,
    units_label="µV·Hz",
    figsize=None,
    background=None,
    svg_path=None,
):
    """Draw the whole-chip activity field to `out_path`; return a manifest dict.

    Every electrode is coloured by ``activity`` on true-micrometre axes with an
    equal aspect, so the array is never stretched. A regular grid is drawn as a
    gridded image (one pixel per electrode); an irregular one is scattered as
    square markers. The colour scale is log when the positive dynamic range
    exceeds ``log_ratio`` (``scale="auto"``) — an amplitude-weighted rate field
    typically spans two orders of magnitude.

    Parameters
    ----------
    activity : array-like, shape (n_channels,)
        The per-channel field, e.g. from :func:`template_projected_activity`.
    locations : array-like, shape (n_channels, 2)
        Electrode x/y positions in micrometres, aligned to ``activity``.
    out_path : path-like
        Where to write the PNG.
    style : {"presentation", "diagnostic"}
        ``presentation`` (default) is poster-clean: short title, compact
        colorbar label, no caption or legend. ``diagnostic`` adds a wrapped
        caption line (pass ``caption``) for a review context.
    cmap, scale, render : str
        Colormap name; ``auto``/``log``/``linear``; ``auto``/``image``/
        ``scatter``. ``auto`` render uses an image when the grid is regular.
    log_ratio : float
        Max-over-min-positive ratio above which ``scale="auto"`` picks log.
    dpi : float
        Figure DPI — raise for poster renders.
    marker_scale : float
        Scatter marker-size multiplier (ignored in image render).
    title, caption, colorbar_label, units_label : str
        Text overrides. ``units_label`` names the activity unit (µV·Hz).
    figsize : tuple or None
        Overrides the near-square default.
    background : str or None
        Figure/axes face colour. ``None`` (default) ties it to ``style``: black
        for ``"presentation"`` (matching the circle reconstruction plot — Adam,
        2026-08-12: "dark backgrounds") and white for ``"diagnostic"``. An
        explicit colour always wins.
    svg_path : path-like or None
        If given, also write a vector SVG (for poster use).

    Returns
    -------
    dict
        JSON-serializable manifest: files, chosen render/scale, colormap,
        activity dynamic range, grid descriptor, and the render params.
    """
    import numpy as np
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Rectangle

    activity = np.asarray(activity, dtype=float).reshape(-1)
    locations = np.asarray(locations, dtype=float)[:, :2]
    if activity.shape[0] != locations.shape[0]:
        raise ValueError(
            f"activity has {activity.shape[0]} channels but locations has "
            f"{locations.shape[0]}"
        )
    if activity.size == 0:
        raise ValueError("no channels to plot")

    positive = activity[activity > 0]
    ratio = float(positive.max() / positive.min()) if positive.size else 1.0
    use_log = (scale == "log") or (
        scale == "auto" and positive.size > 0 and ratio > log_ratio
    )

    if use_log:
        vmin = float(positive.min()) if positive.size else 1e-9
        vmax = float(activity.max())
        if not vmax > vmin:
            vmax = vmin * 10.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
        display = np.clip(activity, vmin, None)
    else:
        vmax = float(activity.max()) if activity.max() > 0 else 1.0
        norm = Normalize(vmin=0.0, vmax=vmax)
        display = activity

    base_cmap = _resolve_cmap(cmap)

    grid = _infer_grid(locations)
    if render == "image":
        use_image = grid is not None
        if grid is None:
            logger.warning("render='image' requested but the layout is not a "
                           "regular grid; scattering instead")
    elif render == "scatter":
        use_image = False
    else:  # auto
        use_image = grid is not None

    # Background follows style unless the caller overrides it: presentation is a
    # black canvas to match the circle reconstruction plot (Adam, 2026-08-12);
    # diagnostic stays white. text_color tracks it so every mark reads.
    if background is None:
        background = "black" if style == "presentation" else "#ffffff"
    dark = str(background).strip().lower() in {"black", "k", "#000", "#000000"}
    text_color = "white" if dark else "black"

    fig = _new_figure(figsize or _PRESENTATION_FIGSIZE, dpi)
    fig.set_facecolor(background)
    ax = fig.subplots()
    ax.set_facecolor(background)

    x = locations[:, 0]
    y = locations[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    if use_image:
        gx, gy = grid["pitch_x"], grid["pitch_y"]
        field = np.full((grid["n_rows"], grid["n_cols"]), np.nan)
        field[grid["rows"], grid["cols"]] = display
        masked = np.ma.masked_invalid(field)
        extent = (xmin - gx / 2.0, xmax + gx / 2.0, ymin - gy / 2.0, ymax + gy / 2.0)
        mappable = ax.imshow(
            masked, origin="lower", extent=extent, aspect="equal",
            cmap=base_cmap, norm=norm, interpolation="nearest",
        )
        render_used = "image"
    else:
        span = max(xmax - xmin, ymax - ymin, 1.0)
        # Square markers; the scatter path is the fallback for irregular
        # layouts, so a fixed readable size beats trying to tile perfectly.
        s = 6.0 * float(marker_scale)
        order = np.argsort(activity, kind="stable")  # loudest drawn last
        mappable = ax.scatter(
            x[order], y[order], c=display[order], cmap=base_cmap, norm=norm,
            marker="s", s=s, linewidths=0, rasterized=True,
        )
        pad = 0.02 * span
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal", adjustable="box")
        render_used = "scatter"

    # Chip outline: the electrode bounding box, so an empty margin reads as
    # off-array rather than cropped.
    pad = 0.015 * max(xmax - xmin, ymax - ymin, 1.0)
    ax.add_patch(Rectangle(
        (xmin - pad, ymin - pad), (xmax - xmin) + 2 * pad, (ymax - ymin) + 2 * pad,
        fill=False, edgecolor=("0.6" if dark else "0.4"), linestyle="--",
        linewidth=1.0, zorder=5,
    ))

    ax.set_xlabel("x (µm)", color=text_color)
    ax.set_ylabel("y (µm)", color=text_color)
    if dark:
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)

    if title is None:
        title = ("Whole-chip activity" if style == "presentation"
                 else "Whole-chip template-projected activity (rate × template amplitude)")
    ax.set_title(title, fontsize=(14 if style == "presentation" else 12), color=text_color)

    if colorbar_label is None:
        colorbar_label = f"activity ({units_label})" + (" — log scale" if use_log else "")
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=11, color=text_color)
    if dark:
        cbar.ax.tick_params(colors=text_color)
        cbar.outline.set_edgecolor(text_color)

    if style == "diagnostic" and caption:
        import textwrap

        fig.tight_layout(rect=(0.0, 0.075, 1.0, 1.0))
        fig.text(
            0.5, 0.01, "\n".join(textwrap.wrap(str(caption), width=150)),
            ha="center", va="bottom", fontsize=8, color="#444444",
        )
    else:
        fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    files = {"png": str(out_path), "svg": None}
    if svg_path is not None:
        svg_path = Path(svg_path)
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(svg_path, facecolor=fig.get_facecolor())
        files["svg"] = str(svg_path)
    fig.clear()

    manifest = {
        "files": files,
        "render": render_used,
        "scale": "log" if use_log else "linear",
        "style": style,
        "cmap": str(cmap),
        "units_label": str(units_label),
        "n_channels": int(activity.size),
        "n_channels_positive": int(positive.size),
        "activity_uV_Hz": {
            "min": float(activity.min()),
            "median": float(np.median(activity)),
            "p95": float(np.percentile(activity, 95)),
            "max": float(activity.max()),
            "max_over_min_positive": ratio if positive.size else None,
        },
        "grid": None if grid is None else {
            "n_rows": int(grid["n_rows"]),
            "n_cols": int(grid["n_cols"]),
            "pitch_x_um": float(grid["pitch_x"]),
            "pitch_y_um": float(grid["pitch_y"]),
            "fill_fraction": float(grid["fill_fraction"]),
        },
        "params": {
            "dpi": float(dpi),
            "marker_scale": float(marker_scale),
            "log_ratio": float(log_ratio),
        },
    }
    logger.info(
        "wrote whole-chip activity map: %s (%s render, %s scale, %d channels, "
        "%.1f-%.1f µV·Hz)",
        out_path, render_used, manifest["scale"], activity.size,
        manifest["activity_uV_Hz"]["min"], manifest["activity_uV_Hz"]["max"],
    )
    return manifest
