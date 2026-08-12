"""Where the recomputed units sit on the array — the standard figure for the
pipeline's `recompute_unit_locations` capsule (stage 20).

One emitter: :func:`plot_unit_locations`. It scatters every unit's recomputed
location over the routed-electrode geometry, so a reviewer can see at a glance
where the sorted population actually lives on the chip — and, because BOTH
estimation methods the capsule emits are drawn together, where the two methods
disagree (which is itself the capsule's own headline caveat: a location whose
extremum sits on thinly-covered electrodes stands on less evidence, and the
two estimates drift apart exactly there).

Overlay, not one-plot-per-method (deliberate): the two methods estimate the
SAME quantity per unit, so their disagreement is the diagnostic signal — a
thin joining line per unit makes that distance directly visible, where two
side-by-side plots would make it a memory exercise. A caller that wants one
method alone passes ``methods=("monopolar",)`` or ``("com",)``.

Pure plotting, matching this package's house rules: takes arrays the caller
already holds, writes one figure to an explicit path, returns that path.
No file reading, no argparse, no directory invention. Figures are built on an
Agg canvas without pyplot (:func:`.channel_layout._new_figure`), so the render
is deterministic and safe on a headless node; the output format follows the
path's extension (``.png`` normally; ``.svg`` renders vector output for print).

Every encoding is explained on the figure itself (Adam, 2026-08-11): the
legend names what each marker is and which estimation method produced it with
unit counts, the caption expands CoM and says in plain language how each
method works and why any units are missing. A reader who has never seen this
source must be able to decode the figure unaided.
"""

import logging
from pathlib import Path

from .channel_layout import (
    _add_caption,
    _fold_caption,
    _legend_dot,
    _legend_line,
    _new_figure,
    _save_and_release,
    _wrap_label,
)
from .figure_text import acronym_note

logger = logging.getLogger(__name__)

#: The filename the pipeline's `recompute_unit_locations` capsule emits.
UNIT_LOCATIONS_PLOT_FILENAME = "unit_locations.png"

# Matched to channel_layout's layout figure, so every spatial figure a capsule
# emits reads as one family rather than five house styles. DPI is a knob on
# the emitter because this figure doubles as a deliverable: 180 is right for a
# review PNG rendered per well on every run; a print/poster render passes
# dpi=600 (7.5 in -> 4500 px across) or an .svg path instead.
DEFAULT_UNIT_LOCATIONS_FIGSIZE = (7.5, 6.0)
DEFAULT_UNIT_LOCATIONS_DPI = 180

#: Filled-marker area (matplotlib points^2) for a unit-location dot. The
#: electrode backdrop stays at the fixed small size the other layout figures
#: use (channel_layout's s=4) — units are the subject, electrodes the context.
DEFAULT_UNIT_MARKER_SIZE = 26.0
DEFAULT_UNIT_MARKER_ALPHA = 0.85

# Open CoM rings are drawn a little larger than the filled triangulation dots
# so an agreeing pair reads as a ring around a dot, not one occluding blob.
_OPEN_MARKER_SCALE = 1.9

_ELECTRODE_COLOR = "#888888"
_MONOPOLAR_COLOR = "#c0392b"
_COM_COLOR = "#2980b9"
_CONNECTOR_COLOR = "#999999"

_METHOD_NAMES = ("monopolar", "com")


def _finite_xy_mask(array):
    """Boolean mask of rows whose first two columns are both finite."""
    import numpy as np

    if array is None:
        return None
    return np.isfinite(array[:, 0]) & np.isfinite(array[:, 1])


def _as_locations_2d(name, array, n_units=None):
    """Validate an (n, >=2) float array; return it as float, or raise."""
    import numpy as np

    array = np.asarray(array, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"{name} must be (n, >=2); got {array.shape}")
    if n_units is not None and array.shape[0] != n_units:
        raise ValueError(
            f"{name} holds {array.shape[0]} row(s) for {n_units} unit(s) -- "
            "method arrays must be row-aligned to the same unit axis"
        )
    return array[:, :2]


def _missing_reasons_fragment(
    n_units,
    n_missing_monopolar,
    n_zero_coverage,
    n_low_local_coverage,
    n_monopolar_failed,
):
    """One plain-language sentence on why units lack a triangulation fit.

    Built only from the counts the caller actually supplied — a reason with an
    unknown count is left unsaid rather than guessed. Reasons with a count of
    zero are dropped: "0 fits failed" is noise on a figure.
    """
    if n_missing_monopolar <= 0:
        return ""
    reasons = []
    if n_zero_coverage:
        reasons.append(
            f"{n_zero_coverage} had no measured signal on any electrode "
            "(nothing to locate — shown by neither method)"
        )
    if n_low_local_coverage:
        reasons.append(
            f"{n_low_local_coverage} had too few measured electrodes near the "
            "signal peak for the 4-parameter fit (their centre of mass is "
            "still shown)"
        )
    if n_monopolar_failed:
        reasons.append(f"{n_monopolar_failed} fit(s) did not converge")
    sentence = (
        f"{n_missing_monopolar} of {n_units} units have no triangulation fit"
    )
    if reasons:
        sentence += ": " + "; ".join(reasons)
    return sentence + "."


def plot_unit_locations(
    channel_locations,
    out_path,
    monopolar=None,
    com=None,
    title=None,
    methods=_METHOD_NAMES,
    marker_size=DEFAULT_UNIT_MARKER_SIZE,
    alpha=DEFAULT_UNIT_MARKER_ALPHA,
    connect_methods=True,
    n_zero_coverage=None,
    n_low_local_coverage=None,
    n_monopolar_failed=None,
    caption=None,
    figsize=DEFAULT_UNIT_LOCATIONS_FIGSIZE,
    dpi=DEFAULT_UNIT_LOCATIONS_DPI,
    invert_y_axis=False,
):
    """Scatter recomputed unit locations over the array geometry; return the path.

    Parameters
    ----------
    channel_locations : (n_channels, >=2) array
        Electrode positions in micrometres — the grey backdrop that gives the
        unit positions their spatial context (the union channel set the
        locations were computed against, e.g. ``channel_locations_xy.npy``).
    out_path : path-like
        Where the figure is written. Extension selects the format (``.png``
        raster at `dpi`; ``.svg`` vector for print).
    monopolar : (n_units, >=2) array, optional
        Per-unit monopolar-triangulation estimates; columns 0/1 are x/y in µm
        (extra columns — z, alpha — are ignored here). NaN rows are units that
        could not be fit; they are counted and explained, never dropped
        silently.
    com : (n_units, 2) array, optional
        Per-unit centre-of-mass estimates, same alignment and NaN convention.
    title : str, optional
        Axes title. Defaults to ``"Recomputed unit locations"``.
    methods : sequence of {"monopolar", "com"}
        Which estimate layers to draw. Both by default — their disagreement is
        part of what the figure is for (see the module docstring).
    marker_size : float
        Filled-marker area (points^2) for a unit dot; the open CoM ring is
        drawn ~2x that area so an agreeing pair reads as a ring around a dot.
    alpha : float
        Opacity of the unit markers (the electrode backdrop has its own fixed
        style, shared with the other layout figures).
    connect_methods : bool
        Draw a thin line joining the two estimates of each unit when both
        methods are plotted — the per-unit disagreement, made visible.
    n_zero_coverage, n_low_local_coverage, n_monopolar_failed : int, optional
        Why-units-are-missing counts, straight from the capsule's own
        manifest. Reported in the caption when given; a reason whose count is
        unknown is left unsaid rather than guessed.
    caption : str, optional
        Extra caption fragment, appended after the standard wording.
    figsize, dpi : figure geometry knobs
        `dpi` is the print-resolution knob: 180 (default) for the per-run
        review PNG, 600 for a poster-grade render.
    invert_y_axis : bool
        False by default, matching :func:`.channel_layout.plot_channel_layout`
        (the array geometry as recorded); pass True for the row-0-at-top MEA
        convention some reconstruction figures use.

    Returns the written path. Raises ``ValueError`` on misaligned inputs or if
    no method layer was requested/available at all.
    """
    import numpy as np

    methods = tuple(str(m).strip().lower() for m in (methods or ()))
    unknown = [m for m in methods if m not in _METHOD_NAMES]
    if unknown:
        raise ValueError(f"unknown method(s) {unknown}; valid: {_METHOD_NAMES}")

    channel_locations = _as_locations_2d("channel_locations", channel_locations)

    n_units = None
    if monopolar is not None:
        monopolar = _as_locations_2d("monopolar", monopolar)
        n_units = monopolar.shape[0]
    if com is not None:
        com = _as_locations_2d("com", com, n_units=n_units)
        n_units = com.shape[0] if n_units is None else n_units

    draw_monopolar = "monopolar" in methods and monopolar is not None
    draw_com = "com" in methods and com is not None
    if not draw_monopolar and not draw_com:
        raise ValueError(
            "nothing to plot: pass monopolar= and/or com= arrays and request "
            "their method(s) via methods="
        )
    n_units = int(n_units or 0)

    mono_mask = _finite_xy_mask(monopolar) if draw_monopolar else None
    com_mask = _finite_xy_mask(com) if draw_com else None
    n_mono = int(mono_mask.sum()) if mono_mask is not None else 0
    n_com = int(com_mask.sum()) if com_mask is not None else 0

    fig = _new_figure(figsize, dpi)
    ax = fig.subplots()

    # The context layer: every electrode of the union channel set, in the same
    # fixed grey the other layout figures use — the units are the subject.
    ax.scatter(
        channel_locations[:, 0], channel_locations[:, 1],
        s=4, c=_ELECTRODE_COLOR, alpha=0.75, linewidths=0, zorder=1,
    )

    handles = [
        _legend_dot(
            _ELECTRODE_COLOR,
            f"recording electrodes on the array (n={channel_locations.shape[0]})",
        ),
    ]

    # Connectors under the markers: one thin line per unit both methods
    # located, so disagreement reads as distance without hiding either marker.
    n_connected = 0
    if connect_methods and draw_monopolar and draw_com:
        from matplotlib.collections import LineCollection

        both = mono_mask & com_mask
        n_connected = int(both.sum())
        if n_connected:
            segments = np.stack(
                [monopolar[both], com[both]], axis=1,
            )  # (n, 2 points, xy)
            ax.add_collection(LineCollection(
                segments, colors=_CONNECTOR_COLOR, linewidths=0.8, alpha=0.8,
                zorder=2,
            ))
            handles.append(_legend_line(
                _CONNECTOR_COLOR,
                "line joining a unit's two estimates — longer line = the "
                "methods disagree more there",
                lw=0.8,
            ))

    if draw_com and n_com:
        ax.scatter(
            com[com_mask, 0], com[com_mask, 1],
            s=float(marker_size) * _OPEN_MARKER_SCALE,
            facecolors="none", edgecolors=_COM_COLOR,
            linewidths=1.1, alpha=float(alpha), zorder=3,
        )
        from matplotlib.lines import Line2D

        handles.append(Line2D(
            [], [], linestyle="none", marker="o", markersize=7.0,
            markerfacecolor="none", markeredgecolor=_COM_COLOR,
            markeredgewidth=1.2,
            label=_wrap_label(
                f"unit location, centre of mass (CoM) estimate "
                f"(n={n_com} of {n_units} units)"
            ),
        ))

    if draw_monopolar and n_mono:
        ax.scatter(
            monopolar[mono_mask, 0], monopolar[mono_mask, 1],
            s=float(marker_size), c=_MONOPOLAR_COLOR,
            linewidths=0, alpha=float(alpha), zorder=4,
        )
        handles.append(_legend_dot(
            _MONOPOLAR_COLOR,
            f"unit location, monopolar-triangulation fit "
            f"(n={n_mono} of {n_units} units)",
            size=7.0,
        ))

    ax.set_title(title or "Recomputed unit locations")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    # Electrode spacing is isotropic; a stretched aspect turns a real distance
    # between two estimates into a lie.
    ax.set_aspect("equal", adjustable="box")
    if invert_y_axis:
        ax.invert_yaxis()

    caption_parts = [
        "Each marker is one sorted unit's estimated position on the array, "
        "recomputed from its stitched full-array footprint (the per-electrode "
        "size of its average spike waveform), using only electrodes that "
        "actually measured the unit.",
    ]
    if draw_monopolar:
        caption_parts.append(
            "The triangulation fit places a point source above the array so "
            "that its predicted signal fall-off best matches the measured "
            "sizes near the peak electrode."
        )
    if draw_com:
        caption_parts.append(acronym_note("CoM") + ".")
    if draw_monopolar and n_units:
        caption_parts.append(_missing_reasons_fragment(
            n_units,
            n_units - n_mono,
            n_zero_coverage,
            n_low_local_coverage,
            n_monopolar_failed,
        ))
    if draw_com and n_units and n_com < n_units and not draw_monopolar:
        caption_parts.append(
            f"{n_units - n_com} of {n_units} units have no estimate — no "
            "measured signal on any electrode."
        )
    if caption:
        caption_parts.append(caption)

    _add_caption(fig, _fold_caption(caption_parts), legend_handles=handles)

    out_path = _save_and_release(fig, out_path)
    logger.info(
        "wrote unit locations plot: %s (%d unit(s): %d triangulated, %d CoM, "
        "%d joined; %d electrodes)",
        out_path, n_units, n_mono, n_com, n_connected,
        channel_locations.shape[0],
    )
    return out_path


__all__ = [
    "plot_unit_locations",
    "UNIT_LOCATIONS_PLOT_FILENAME",
    "DEFAULT_UNIT_LOCATIONS_FIGSIZE",
    "DEFAULT_UNIT_LOCATIONS_DPI",
    "DEFAULT_UNIT_MARKER_SIZE",
    "DEFAULT_UNIT_MARKER_ALPHA",
]
