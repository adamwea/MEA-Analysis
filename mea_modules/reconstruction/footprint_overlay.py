"""Reconstruction overlay toolkit — the tracked axon arbor drawn ON TOP of the
amplitude/latency circle footprint (:mod:`mea_modules.reconstruction.plots`).

Per-branch or per-segment (graded) conduction-velocity colouring, soma / branch
node markers (a plain diamond or a face-aligned directional polygon), a
full-height velocity colourbar, and a full-array locator inset. The high-level
:func:`render_footprint_reconstruction_panel` composes the base footprint
(``plots._render_footprint_core``, arbor-framed, with the solid amplitude scale
disc + µV reference) with this overlay into one annotated candidate panel.

**Provenance.** Migrated 2026-08-25 from the ``roy-grant-figures`` figures repo
(the functional plotting belongs in ``mea_modules``, imported by a thin
figures repo — not hand-rolled there) and DECOUPLED from that repo's ``Unit``
dataclass and ``config`` module: every function takes plain numpy arrays and
explicit style params, defaulting to the module constants below. The figures
repo now loads a unit's portable stage-34 arrays and calls these — it holds no
plotting implementation of its own.

Nothing here reads a well, a run directory, a manifest, or a ``gtr`` — the arbor
is drawn straight from exported branch polylines (µm) + per-branch/per-vertex
metadata, so a caller with the portable ``reconstruction_geometry.npz`` arrays
needs no heavy tracker object. Headless (Agg) — imported only in a figure-render
context, never at ``mea_modules.reconstruction`` import time.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

# --- style defaults (were the figures repo's config.*) -------------------------
# Okabe-Ito qualitative palette (colour-blind safe, survives grayscale).
OKABE_ITO = {
    "blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
    "orange": "#E69F00", "sky": "#56B4E9", "yellow": "#F0E442",
    "purple": "#CC79A7", "grey": "#BBBBBB", "black": "#222222",
}
VELOCITY_CMAP = "plasma"          # axon colour = conduction velocity
SOMA_COLOR = "#D55E00"            # vermillion — the soma / AIS electrode
BRANCHPOINT_COLOR = "#56B4E9"     # sky blue — branch-point (junction) electrodes
NODE_RADIUS_PITCH = 0.75          # directional-node polygon radius, in electrode pitches
# Colour branches by velocity only when it varies enough to earn a scale;
# otherwise by identity (a full-height colourbar over a <1% range is false
# precision). Rule: n_branches >= 2 AND (vmax-vmin)/mean > this fraction.
VELOCITY_INFORMATIVE_FRACTION = 0.15
GRADED_CLIP_PCT = (5.0, 95.0)     # clip per-segment velocities to this percentile window


# --- velocity encoding decision -----------------------------------------------
def velocity_is_informative(velocities, *, min_branches: int = 2,
                            frac: float | None = None) -> bool:
    """True when branch velocity varies enough to earn a colour scale."""
    if velocities is None:
        return False
    v = np.asarray(velocities, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < min_branches:
        return False
    mean = float(np.mean(v))
    if mean == 0:
        return False
    frac = VELOCITY_INFORMATIVE_FRACTION if frac is None else frac
    return (float(v.max()) - float(v.min())) / abs(mean) > frac


def _finite_norm(velocities) -> Normalize:
    v = np.asarray(velocities, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return Normalize(0.0, 1.0)
    if np.ptp(v) == 0:
        return Normalize(float(v[0]) - 1.0, float(v[0]) + 1.0)
    return Normalize(float(v.min()), float(v.max()))


def halo_color(text_color: str) -> str:
    """Line-halo / marker-edge colour contrasting the background: white on a dark
    ground, black on a light one (a white halo is invisible on a white panel)."""
    return "white" if str(text_color).strip().lower() in {"white", "#fff", "#ffffff"} else "black"


def fmt_velocity_range(velocities) -> str:
    """Human velocity range; collapses a degenerate min==max to "≈V (const)"."""
    if velocities is None:
        return "—"
    v = np.asarray(velocities, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "—"
    lo, hi = float(v.min()), float(v.max())
    return f"≈{lo:.0f} (const)" if abs(hi - lo) < 0.5 else f"{lo:.0f}–{hi:.0f}"


def rows_for_indices(channel_indices, wanted) -> np.ndarray:
    """Row positions (into a covered-channel array) for the wanted channel ids."""
    lookup = {int(v): i for i, v in enumerate(np.asarray(channel_indices).tolist())}
    return np.asarray([lookup[int(w)] for w in wanted if int(w) in lookup], dtype=int)


# --- geometry helpers (graded velocity + directional node markers) -------------
def _pitch_um(locations) -> float:
    loc = np.asarray(locations, float)
    ux, uy = np.unique(loc[:, 0]), np.unique(loc[:, 1])
    dx = float(np.median(np.diff(ux))) if ux.size > 1 else 20.0
    dy = float(np.median(np.diff(uy))) if uy.size > 1 else 20.0
    p = min(dx, dy)
    return p if p > 0 else 20.0


def channel_latency_ms(template, fs) -> np.ndarray:
    """Per-row peak latency (ms) from the template — each channel's own peak
    |amplitude| sample time. Indexed by row, matching ``channel_indices``."""
    t = np.abs(np.asarray(template, float))
    return np.argmax(t, axis=1) / float(fs) * 1e3


def segment_velocities(polylines, channels_per_branch, channel_indices, template, fs):
    """Per-branch (segments, velocities): each electrode-to-electrode segment's
    LOCAL conduction velocity = Δdistance / Δpeak-time (µm/ms). Returns
    (list of (segments (m,2,2), vels (m,)), flat_finite_vels). Per-electrode times
    are sample-quantized, so |Δt| is floored at half a sample to avoid div-by-0."""
    lat = channel_latency_ms(template, fs)
    lookup = {int(v): i for i, v in enumerate(np.asarray(channel_indices).tolist())}
    min_dt = 0.5 / float(fs) * 1e3
    per_branch, flat = [], []
    for pts, chans in zip(polylines, channels_per_branch):
        pts = np.asarray(pts, float)
        if pts.shape[0] < 2:
            per_branch.append((np.empty((0, 2, 2)), np.empty(0)))
            continue
        rows = [lookup.get(int(c)) for c in np.asarray(chans).tolist()]
        times = np.array([lat[r] if r is not None else np.nan for r in rows])
        # Per-segment distances via the canonical primitive (single-definition
        # guard). pts.shape[0] >= 2 here (guarded above), so this equals the
        # inline norm(diff) it replaces. Function-local import avoids any
        # reconstruction<->morphometrics package-init cycle.
        from ..morphometrics import segment_lengths_um
        dd = segment_lengths_um(pts)
        dt = np.abs(np.diff(times))
        dt = np.where(np.isfinite(dt) & (dt >= min_dt), dt, min_dt)
        v = dd / dt
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        per_branch.append((segs, v))
        flat.append(v[np.isfinite(v)])
    return per_branch, (np.concatenate(flat) if flat else np.array([]))


def _cluster_dirs(vecs, tol_deg: float = 18.0):
    """Unit directions with near-duplicates merged (a junction's distinct arms)."""
    out = []
    for v in vecs:
        v = np.asarray(v, float)
        nrm = float(np.hypot(v[0], v[1]))
        if nrm < 1e-9:
            continue
        u = v / nrm
        if all(np.degrees(np.arccos(np.clip(float(u @ o), -1, 1))) > tol_deg for o in out):
            out.append(u)
    return out


def axon_nodes(polylines, channels_per_branch, init_channel):
    """Detect junction electrodes and the axon directions leaving each. A node =
    an electrode shared by >=2 branches (or the soma). Its "arms" are the
    directions to adjacent vertices, clustered. Returns list of
    {channel, xy, dirs, is_soma}."""
    from collections import defaultdict
    if not polylines or channels_per_branch is None or len(channels_per_branch) == 0:
        return []
    arms, xy_of, in_branches = defaultdict(list), {}, defaultdict(set)
    for bi, (pts, cs) in enumerate(zip(polylines, channels_per_branch)):
        pts = np.asarray(pts, float)
        cs = [int(x) for x in np.asarray(cs).tolist()]
        for i, c in enumerate(cs):
            in_branches[c].add(bi)
            xy_of[c] = pts[i]
            if i + 1 < len(pts):
                arms[c].append(pts[i + 1] - pts[i])
            if i - 1 >= 0:
                arms[c].append(pts[i - 1] - pts[i])
    nodes = []
    for c, brs in in_branches.items():
        is_soma = init_channel is not None and c == init_channel
        if not (len(brs) >= 2 or is_soma):
            continue
        dirs = _cluster_dirs(arms[c])
        if len(dirs) >= 2 or is_soma:
            nodes.append({"channel": c, "xy": xy_of[c], "dirs": dirs, "is_soma": is_soma})
    return nodes


def directional_polygon(center, arm_dirs, radius: float):
    """A closed polygon whose flat faces sit perpendicular to each axon leaving
    the node. k>=3 arms -> one face per arm (bisector-vertex construction);
    k==2 -> a 4-gon with a flat face per arm. None if k<2."""
    c = np.asarray(center, float)
    angs = sorted(float(np.arctan2(d[1], d[0])) for d in arm_dirs
                  if float(np.hypot(d[0], d[1])) > 1e-9)
    k = len(angs)
    if k < 2:
        return None
    if k == 2:
        verts = []
        for a in angs:
            d = np.array([np.cos(a), np.sin(a)])
            perp = np.array([-d[1], d[0]])
            fc = c + radius * d
            w = radius * 0.6
            verts += [fc - w * perp, fc + w * perp]
        return np.array(verts)
    verts = []
    ang = np.array(angs)
    for i in range(k):
        a1 = ang[i]
        a2 = ang[(i + 1) % k] + (2 * np.pi if i == k - 1 else 0.0)
        mid = (a1 + a2) / 2.0
        verts.append(c + radius * np.array([np.cos(mid), np.sin(mid)]))
    return np.array(verts)


def draw_axon_nodes(ax, *, polylines, channels_per_branch, init_channel, init_xy,
                    locations, halo, directional: bool = False,
                    mark_branch_points: bool = False, radius_um: float | None = None,
                    soma_color: str = SOMA_COLOR, branchpoint_color: str = BRANCHPOINT_COLOR,
                    node_radius_pitch: float = NODE_RADIUS_PITCH):
    """Draw the soma / AIS electrode (uniquely coloured) and, optionally,
    branch-point electrodes. `directional` swaps the marker for a polygon aligned
    to the axon directions leaving the node; else a plain diamond at the soma."""
    from matplotlib.patches import Polygon
    if not directional:
        if init_xy is not None:
            ax.plot([init_xy[0]], [init_xy[1]], marker="D", ls="", markersize=10.0,
                    markerfacecolor=soma_color, markeredgecolor=halo,
                    markeredgewidth=1.0, zorder=12)
        return
    r = radius_um if radius_um is not None else node_radius_pitch * _pitch_um(locations)
    nodes = axon_nodes(polylines, channels_per_branch, init_channel)
    drew_soma = False
    for nd in nodes:
        is_soma = nd["is_soma"]
        if not is_soma and not mark_branch_points:
            continue
        color = soma_color if is_soma else branchpoint_color
        poly = directional_polygon(nd["xy"], nd["dirs"], r if is_soma else 0.8 * r)
        z = 13 if is_soma else 12
        if poly is not None and len(poly) >= 3:
            ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor=halo,
                                 linewidth=1.0, zorder=z, alpha=0.95))
        else:  # too few arms to orient — fall back to a marker
            ax.plot([nd["xy"][0]], [nd["xy"][1]], marker="D" if is_soma else "o", ls="",
                    markersize=9.0 if is_soma else 6.0, markerfacecolor=color,
                    markeredgecolor=halo, markeredgewidth=0.9, zorder=z)
        drew_soma = drew_soma or is_soma
    if not drew_soma and init_xy is not None:  # soma not in node list — still mark it
        ax.plot([init_xy[0]], [init_xy[1]], marker="D", ls="", markersize=10.0,
                markerfacecolor=soma_color, markeredgecolor=halo, markeredgewidth=1.0, zorder=13)


# --- the axon overlay ----------------------------------------------------------
def overlay_reconstruction(ax, *, polylines, text_color: str, branch_velocity=None,
                           channels_per_branch=None, channel_indices=None, template=None,
                           fs=None, init_xy=None, init_channel=None, locations=None,
                           shared_norm: Normalize | None = None, velocity_cmap: str | None = None,
                           fixed_color: str | None = None,
                           linewidth: float = 2.6, draw_vertices: bool = True,
                           graded: bool = False, node_style: str = "diamond",
                           mark_branch_points: bool = False,
                           soma_color: str = SOMA_COLOR, branchpoint_color: str = BRANCHPOINT_COLOR,
                           node_radius_pitch: float = NODE_RADIUS_PITCH):
    """Overlay the reconstructed arbor on an existing footprint axes.

    `graded=False`: colour each whole branch by its velocity when that varies,
    else by branch identity (Okabe-Ito). `graded=True`: colour each
    electrode-to-electrode SEGMENT by its local velocity, so velocity varies down
    the axon (needs `channels_per_branch` + `channel_indices` + `template` + `fs`).
    `fixed_color` (added 2026-08-26): draw the WHOLE arbor in one given colour,
    overriding velocity/identity/graded — the multi-unit overlay case (F2P1: N of
    our units at one MaxLive cluster, each a distinct colour on a shared
    footprint), where a per-unit solid colour is what distinguishes the units. The
    soma diamond follows the same colour. Everything else (halo, vertices, node
    markers) is unchanged, so this stays the P2 line vocabulary, not a new style.
    Every line gets a background-contrast halo. The soma is drawn by
    :func:`draw_axon_nodes` — a diamond, or (`node_style="directional"`) a polygon
    whose faces align to the axons; `mark_branch_points` also marks junctions.
    Returns a dict describing what was drawn (`colored_by`, `mappable`, `norm`,
    `n`, `graded`)."""
    velocity_cmap = velocity_cmap or VELOCITY_CMAP
    halo = halo_color(text_color)
    polylines = list(polylines) if polylines is not None else []
    n = len(polylines)
    directional = str(node_style).strip().lower() == "directional"
    node_kw = dict(polylines=polylines, channels_per_branch=channels_per_branch,
                   init_channel=init_channel, init_xy=init_xy, locations=locations, halo=halo,
                   directional=directional, mark_branch_points=mark_branch_points,
                   soma_color=(fixed_color if fixed_color is not None else soma_color),
                   branchpoint_color=branchpoint_color,
                   node_radius_pitch=node_radius_pitch)
    if n == 0:
        draw_axon_nodes(ax, **node_kw)
        return {"colored_by": None, "mappable": None, "norm": None, "n": 0, "graded": graded}

    result: dict = {"n": n, "graded": graded}

    if fixed_color is not None:
        for pl in polylines:
            pl = np.asarray(pl, dtype=float)
            line, = ax.plot(pl[:, 0], pl[:, 1], color=fixed_color, linewidth=linewidth,
                            solid_capstyle="round", zorder=10, alpha=0.97)
            line.set_path_effects([pe.Stroke(linewidth=linewidth + 1.9, foreground=halo),
                                   pe.Normal()])
        result.update(colored_by="fixed", mappable=None, norm=None)
    elif graded and channels_per_branch is not None and len(channels_per_branch):
        per_branch, allv = segment_velocities(polylines, channels_per_branch,
                                              channel_indices, template, fs)
        finite = allv[np.isfinite(allv)]
        if shared_norm is not None:
            norm = shared_norm
        elif finite.size:
            lo, hi = np.percentile(finite, GRADED_CLIP_PCT)
            norm = Normalize(float(lo), float(hi)) if hi > lo else _finite_norm(finite)
        else:
            norm = Normalize(0.0, 1.0)
        segs = [s for (sg, _v) in per_branch for s in sg]
        vels = np.concatenate([v for (_sg, v) in per_branch]) if per_branch else np.array([])
        lc = LineCollection(segs, cmap=velocity_cmap, norm=norm, linewidths=linewidth,
                            capstyle="round", joinstyle="round", zorder=10, alpha=0.97)
        lc.set_array(vels)
        lc.set_path_effects([pe.Stroke(linewidth=linewidth + 1.9, foreground=halo), pe.Normal()])
        ax.add_collection(lc)
        result.update(colored_by="graded", mappable=lc, norm=norm)
    else:
        by_velocity = velocity_is_informative(branch_velocity)
        result["colored_by"] = "velocity" if by_velocity else "identity"
        if by_velocity:
            norm = shared_norm or _finite_norm(branch_velocity)
            lc = LineCollection(polylines, cmap=velocity_cmap, norm=norm, linewidths=linewidth,
                                capstyle="round", joinstyle="round", zorder=10, alpha=0.97)
            lc.set_array(np.asarray(branch_velocity, dtype=float)[:n])
            lc.set_path_effects([pe.Stroke(linewidth=linewidth + 1.9, foreground=halo), pe.Normal()])
            ax.add_collection(lc)
            result.update(mappable=lc, norm=norm)
        else:
            palette = [OKABE_ITO[k] for k in
                       ("blue", "vermillion", "green", "orange", "sky", "purple", "yellow")]
            for i, pl in enumerate(polylines):
                pl = np.asarray(pl, dtype=float)
                line, = ax.plot(pl[:, 0], pl[:, 1], color=palette[i % len(palette)],
                                linewidth=linewidth, solid_capstyle="round", zorder=10, alpha=0.97)
                line.set_path_effects([pe.Stroke(linewidth=linewidth + 1.9, foreground=halo),
                                       pe.Normal()])
            result.update(mappable=None, norm=None)

    if draw_vertices:
        verts = np.vstack([np.asarray(p, dtype=float) for p in polylines])
        ax.scatter(verts[:, 0], verts[:, 1], s=9, facecolor=halo,
                   edgecolor=OKABE_ITO["black"] if halo == "white" else "white",
                   linewidths=0.35, zorder=11)

    draw_axon_nodes(ax, **node_kw)
    return result


# --- footprint corner inset + colourbar (drawn INTO the footprint axes) --------
def emptiest_corner_loc(polylines, zoom_bbox, size=(0.24, 0.27), pad=0.015):
    """Pick the corner of the framed footprint with the least branch mass, so the
    locator inset never lands on the reconstructed arbor. Pipeline chrome occupies
    the top-left scale circle and bottom-right scale bar, so prefer bottom-left /
    top-right; break ties by branch density."""
    w, h = size
    corners = {  # name -> (loc rect, its centre in axes-fraction)
        "bl": ((pad, pad, w, h), (pad + w / 2, pad + h / 2)),
        "tr": ((1 - pad - w, 1 - pad - h, w, h), (1 - pad - w / 2, 1 - pad - h / 2)),
        "br": ((1 - pad - w, pad, w, h), (1 - pad - w / 2, pad + h / 2)),
        "tl": ((pad, 1 - pad - h, w, h), (pad + w / 2, 1 - pad - h / 2)),
    }
    if zoom_bbox is None or not polylines:
        return corners["bl"][0]
    x0, x1, y0, y1 = (float(v) for v in zoom_bbox)
    xlo, xhi, ylo, yhi = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
    verts = np.vstack([np.asarray(p, float) for p in polylines])
    # to axes fraction; note the footprint y-axis is inverted (row 0 at top)
    fx = np.clip((verts[:, 0] - xlo) / max(xhi - xlo, 1e-9), 0, 1)
    fy = np.clip(1 - (verts[:, 1] - ylo) / max(yhi - ylo, 1e-9), 0, 1)
    prefer = {"bl": 0.0, "tr": 0.0, "br": 0.35, "tl": 0.35}  # nudge away from core chrome corners
    best, best_score = "bl", 1e9
    for name, (rect, (cx, cy)) in corners.items():
        near = np.mean((np.abs(fx - cx) < w) & (np.abs(fy - cy) < h))  # branch fraction in the corner box
        score = near + prefer[name]
        if score < best_score:
            best, best_score = name, score
    return corners[best][0]


def add_locator_inset(ax, *, polylines, locations, zoom_bbox, text_color: str, init_xy=None,
                      loc=None, show_focused_axon: bool = True, other_arbors=None,
                      soma_color: str = SOMA_COLOR):
    """Full-array mini-map showing where the arbor sits, with the crop rectangle,
    auto-placed in the emptiest corner so it never occludes the reconstruction.
    Draws the focused unit's own reconstructed axon; when `other_arbors` is given
    (a list of per-unit polyline-lists — every OTHER reconstructed unit on the
    chip), it draws those faintly too, in the background-contrast colour, with the
    focused unit highlighted. `other_arbors` is loaded by the caller (data
    loading is not this module's job)."""
    loc = loc or emptiest_corner_loc(polylines, zoom_bbox)
    halo = halo_color(text_color)
    iax = ax.inset_axes(loc)
    loc_xy = np.asarray(locations, float)
    # fainter electrode backdrop when other axons share the minimap, so the axons win
    dot_alpha = 0.45 if other_arbors else 1.0
    iax.scatter(loc_xy[:, 0], loc_xy[:, 1], s=0.35, c="#8a8a8a", edgecolor="none",
                alpha=dot_alpha, zorder=1)
    iax.set_aspect("equal")
    iax.invert_yaxis()

    if other_arbors:
        # unfocused axons in the contrast colour (light on dark / dark on light) so
        # they read against the grey electrode dots — grey-on-grey was invisible.
        segs = [np.asarray(p, float) for polys in other_arbors for p in polys
                if np.asarray(p).shape[0] >= 2]
        oc = LineCollection(segs, colors=halo, linewidths=0.6, alpha=0.6, zorder=2)
        iax.add_collection(oc)
        iax.set_title(f"chip — {len(other_arbors) + 1} units", color=text_color, fontsize=6.5, pad=2)
    else:
        iax.set_title("full array", color=text_color, fontsize=6.5, pad=2)

    if show_focused_axon and polylines:
        fc = LineCollection([np.asarray(p, float) for p in polylines if np.asarray(p).shape[0] >= 2],
                            colors=soma_color, linewidths=1.1, alpha=0.95, zorder=4)
        iax.add_collection(fc)

    if zoom_bbox is not None:
        x0, x1, y0, y1 = (float(v) for v in zoom_bbox)
        iax.add_patch(plt.Rectangle((min(x0, x1), min(y0, y1)), abs(x1 - x0), abs(y1 - y0),
                                    fill=False, edgecolor=halo, linewidth=1.2, zorder=5))
    if init_xy is not None:
        iax.plot([init_xy[0]], [init_xy[1]], marker="D", ls="", markersize=4.0,
                 markerfacecolor=soma_color, markeredgecolor=halo, markeredgewidth=0.4, zorder=6)
    iax.set_xticks([]); iax.set_yticks([])
    iax.patch.set_alpha(0.15)
    for s in iax.spines.values():
        s.set_color(text_color); s.set_linewidth(0.6)
    return iax


def add_velocity_colorbar(fig, ax, mappable, text_color: str, *, label=None):
    """Velocity colourbar on the LEFT, FULL plotting-area height and vertically
    centered — mirrors the pipeline core's latency colourbar on the right."""
    cax = ax.inset_axes([-0.14, 0.0, 0.025, 1.0])
    cb = fig.colorbar(mappable, cax=cax)
    cax.yaxis.set_ticks_position("left"); cax.yaxis.set_label_position("left")
    cb.set_label(label or "conduction velocity (µm/ms)", color=text_color, fontsize=8)
    cb.ax.tick_params(colors=text_color, labelsize=7)
    cb.outline.set_edgecolor(text_color)
    return cb


# --- arbor zoom bbox -----------------------------------------------------------
def arbor_bbox(polylines, init_xy=None, locations=None, unit_id=None, include_init: bool = True):
    """The reconstruction's spatial bounding box (xmin, xmax, ymin, ymax) in µm,
    via the pipeline's own :func:`overlay.arbor_bbox_from_record`. None when there
    is no drawable branch (caller then frames the full array)."""
    from . import overlay
    init = tuple(float(v) for v in init_xy) if init_xy is not None else None
    record = {
        "unit_id": unit_id,
        "branch_paths": list(polylines) if polylines else [],
        "init_xy": init,
        "n_branches_total": len(polylines) if polylines else 0,
        "locations": locations,
    }
    return overlay.arbor_bbox_from_record(record, include_init=include_init)


# --- the high-level annotated panel --------------------------------------------
def render_footprint_reconstruction_panel(
    template, locations, fs, *, polylines, unit_id=None,
    branch_velocity=None, channels_per_branch=None, channel_indices=None,
    init_xy=None, init_channel=None, zoom_bbox=None,
    background: str = "black", style: str = "presentation", substrate_cmap: str = "viridis_r",
    graded_velocity: bool = False, directional_nodes: bool = False,
    mark_branch_points: bool = False, minimap: bool = True, other_arbors=None,
    add_title: bool = True, dpi: float | None = None, figsize=(8.4, 7.2), out_path=None,
):
    """One annotated candidate panel: the amplitude/latency circle footprint
    (:func:`plots._render_footprint_core`, arbor-framed, with the solid amplitude
    scale disc + explicit µV reference) + the reconstructed axon arbor overlaid
    (per-branch or graded velocity), soma / branch node markers, a full-height
    velocity colourbar and a full-array locator inset.

    Saves to `out_path` and returns `(out_path, info)` when `out_path` is given;
    otherwise returns `(fig, info)` for the caller to compose or save. `info` is
    the :func:`overlay_reconstruction` result dict."""
    from . import plots

    if zoom_bbox is None and polylines is not None and len(polylines):
        zoom_bbox = arbor_bbox(polylines, init_xy, locations, unit_id=unit_id)

    fig, ax, *_rest = plots._render_footprint_core(
        template, locations, fs,
        dpi=plots.DEFAULT_FOOTPRINT_DPI, figsize=tuple(figsize), invert_y_axis=True,
        cmap_name=substrate_cmap,
        max_radius_pitch_fraction=plots.DEFAULT_MAX_RADIUS_PITCH_FRACTION,
        min_radius_max_fraction=plots.DEFAULT_MIN_RADIUS_MAX_FRACTION,
        radius_scaling=plots.DEFAULT_RADIUS_SCALING,
        background=background, style=style, zoom_bbox=zoom_bbox,
        scale_circle_value=True,  # the grant panel shows the explicit max µV
    )
    text_color = _rest[-1]

    info = overlay_reconstruction(
        ax, polylines=polylines, text_color=text_color, branch_velocity=branch_velocity,
        channels_per_branch=channels_per_branch, channel_indices=channel_indices,
        template=template, fs=fs, init_xy=init_xy, init_channel=init_channel,
        locations=locations, graded=graded_velocity,
        node_style="directional" if directional_nodes else "diamond",
        mark_branch_points=mark_branch_points)

    if info.get("mappable") is not None:
        label = ("local conduction velocity (µm/ms)" if info.get("colored_by") == "graded"
                 else None)
        add_velocity_colorbar(fig, ax, info["mappable"], text_color, label=label)

    if minimap:
        add_locator_inset(ax, polylines=polylines, locations=locations, zoom_bbox=zoom_bbox,
                          text_color=text_color, init_xy=init_xy, other_arbors=other_arbors)

    if add_title:
        by = info.get("colored_by")
        note = (f"{info['n']} branch{'es' if info['n'] != 1 else ''}"
                + (" · graded velocity" if by == "graded"
                   else " · velocity-coloured" if by == "velocity"
                   else " · branch-coloured (v≈const)" if by == "identity" else ""))
        ax.set_title(f"Unit {unit_id} — axon over footprint · {note}",
                     color=text_color, fontsize=11)

    if out_path is not None:
        fig.savefig(out_path, dpi=dpi or plots.DEFAULT_FOOTPRINT_DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_path, info
    return fig, info
