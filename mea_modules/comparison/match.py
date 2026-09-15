"""Match our reconstructed units to MaxLive's tracked neurons by electrode cluster.

MaxLive tracks (at most) **one neuron per fixed-electrode cluster**; our sorter can
resolve several units at the same cluster. So the fair unit of comparison is the
cluster: each MaxLive neuron (anchored at its init-site electrode) defines a
cluster, and we gather **all our units whose init/extremum electrode falls within
that cluster** — a 1↔(few) mapping.

Two MaxWell coordinate frames *should* already agree (both are electrode µm), but
MaxLive occasionally reflects y relative to the pipeline; :func:`choose_orientation`
tests identity vs a y-flip on the soma + arbor point clouds and picks whichever
co-locates better (report it per dataset — do not assume).

numpy only. Inputs are the plain dicts the readers hand back:
  ml_somata : {neuron_id: {"x": µm, "y": µm, "electrodeNo": int|None}}   (maxlive.neuron_somata)
  pl_somata : {unit_id: (x, y) or np.ndarray shape (2,)}                 (PipelineWell.somata)
"""
from __future__ import annotations

import numpy as np

DEFAULT_CLUSTER_RADIUS_UM = 100.0    # a pipeline unit within this of a MaxLive init site
                                     # is "at the same electrode cluster" (2026-08-27:
                                     # 50 -> 100 µm). Units in overlapping radii go to the
                                     # NEARER soma (the argmin in match_by_cluster).


def _xy_array(somata: dict, keys) -> np.ndarray:
    out = []
    for k in keys:
        v = somata[k]
        if isinstance(v, dict):
            out.append([v["x"], v["y"]])
        else:
            out.append([float(v[0]), float(v[1])])
    return np.asarray(out, float)


def _med_nn(A: np.ndarray, B: np.ndarray):
    """(median nearest-neighbour distance A->B, fraction of A within 35 µm)."""
    if len(A) == 0 or len(B) == 0:
        return np.inf, 0.0
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    nn = d.min(1)
    return float(np.median(nn)), float((nn < 35).mean())


def choose_orientation(ml_somata: dict, pl_somata: dict,
                       ml_arbors: dict | None = None, pl_arbors: dict | None = None):
    """Return (flip: bool, C: float, report: dict). Flipped y' = C - y.

    Tests identity vs y-reflection on soma clouds (and arbor clouds if given), and
    picks whichever co-locates MaxLive onto pipeline better.
    """
    ml_xy = _xy_array(ml_somata, list(ml_somata))
    pl_xy = _xy_array(pl_somata, list(pl_somata))
    yall = np.r_[ml_xy[:, 1], pl_xy[:, 1]]
    C = float(yall.min() + yall.max())

    def clouds(arbors):
        if not arbors:
            return None
        pts = []
        for v in arbors.values():
            branches = v.values() if isinstance(v, dict) else v   # {branch: xy} or [xy, ...]
            for b in branches:
                b = np.asarray(b, float)
                if b.ndim == 2 and len(b):
                    pts.append(b)
        return np.vstack(pts) if pts else None

    ml_pts, pl_pts = clouds(ml_arbors), clouds(pl_arbors)
    if ml_pts is not None and len(ml_pts) > 4000:
        ml_pts = ml_pts[np.random.RandomState(0).choice(len(ml_pts), 4000, replace=False)]
    if pl_pts is not None and len(pl_pts) > 8000:
        pl_pts = pl_pts[np.random.RandomState(1).choice(len(pl_pts), 8000, replace=False)]

    def score(flip):
        m = ml_xy.copy()
        if flip:
            m[:, 1] = C - m[:, 1]
        s_med, s_frac = _med_nn(m, pl_xy)
        a_med = np.nan
        if ml_pts is not None and pl_pts is not None:
            a = ml_pts.copy()
            if flip:
                a[:, 1] = C - a[:, 1]
            a_med, _ = _med_nn(a, pl_pts)
        return s_med, s_frac, a_med

    ident = score(False)
    flipd = score(True)
    id_key = ident[0] + (ident[2] if np.isfinite(ident[2]) else 0.0)
    fl_key = flipd[0] + (flipd[2] if np.isfinite(flipd[2]) else 0.0)
    flip = fl_key < id_key
    report = dict(C=C, chosen=("y-flip" if flip else "identity"),
                  identity=dict(soma_med_nn=ident[0], soma_frac_lt35=ident[1], arbor_med_nn=ident[2]),
                  yflip=dict(soma_med_nn=flipd[0], soma_frac_lt35=flipd[1], arbor_med_nn=flipd[2]))
    return flip, C, report


def apply_flip(xy, flip: bool, C: float) -> np.ndarray:
    xy = np.asarray(xy, float).copy()
    if flip and xy.size:
        xy[..., 1] = C - xy[..., 1]
    return xy


def match_by_cluster(ml_somata: dict, pl_somata: dict, flip: bool = False, C: float = 0.0,
                     radius: float = DEFAULT_CLUSTER_RADIUS_UM):
    """Assign each pipeline unit to the nearest MaxLive cluster within `radius`.

    Returns (clusters, unmatched):
      clusters  : {ml_neuron_id: [(unit_id, dist_um), ...] sorted by distance}
      unmatched : [unit_id, ...] with no MaxLive cluster within radius
    Pipeline somata are flipped into MaxLive's frame before matching.
    """
    ml_ids = list(ml_somata)
    ml_xy = _xy_array(ml_somata, ml_ids)
    clusters = {i: [] for i in ml_ids}
    unmatched = []
    # `flip` was chosen (in choose_orientation) as "reflect MaxLive's y onto the
    # pipeline frame". Apply it once to the MaxLive cloud; pipeline stays fixed.
    ml_xy_f = ml_xy.copy()
    if flip:
        ml_xy_f[:, 1] = C - ml_xy_f[:, 1]
    for uid, xy in pl_somata.items():
        p = np.asarray(xy, float)
        d = np.sqrt(((ml_xy_f - p) ** 2).sum(1))
        j = int(np.argmin(d))
        if d[j] <= radius:
            clusters[ml_ids[j]].append((int(uid), float(d[j])))
        else:
            unmatched.append(int(uid))
    for k in clusters:
        clusters[k].sort(key=lambda t: t[1])
    return clusters, unmatched


def hero_candidates(clusters: dict, ml_well, pl_well, min_units: int = 1, max_units: int = 4,
                    top: int = 8):
    """Rank MaxLive neurons that make good F2 hero candidates.

    A good hero: a clean 1↔(few) match (few, close pipeline units), a rich MaxLive
    arbor (branches + length), and good velocity-fit on the matched pipeline units.
    Returns a list of dicts sorted best-first.
    """
    ml_som = ml_well.somata()
    ml_nm = ml_well.neuron_metrics
    nm_ids = np.asarray(ml_nm["neuron"], float).astype(int)
    def nm(col, nid):
        if col not in ml_nm:
            return np.nan
        idx = np.where(nm_ids == nid)[0]
        return float(np.asarray(ml_nm[col], float)[idx[0]]) if len(idx) else np.nan

    rows = []
    for nid, matched in clusters.items():
        n_u = len(matched)
        if n_u < min_units or n_u > max_units:
            continue
        total_len = nm("totalAxonLen", nid)
        vel = nm("neuronConductionVel", nid)
        n_br_ml = _ml_branch_count(ml_well, nid)
        # pipeline side: matched units' arbor richness + velocity-fit r2
        pl_r2, pl_len, pl_nbr = [], [], []
        for uid, dist in matched:
            u = pl_well.units.get(uid)
            if u is None:
                continue
            pl_len.append(u.total_length_um())
            pl_nbr.append(u.n_branches)
            for b in u.branches:
                if b.get("r2") is not None:
                    pl_r2.append(float(b["r2"]))
        mean_r2 = float(np.nanmean(pl_r2)) if pl_r2 else np.nan
        closeness = float(np.mean([d for _, d in matched])) if matched else np.nan
        # composite score: reward length + branches + r2, penalise many/distant units
        score = (np.nan_to_num(total_len) / 1000.0
                 + 1.5 * np.nan_to_num(n_br_ml)
                 + 8.0 * np.nan_to_num(mean_r2)
                 - 1.0 * (n_u - 1)
                 - 0.02 * np.nan_to_num(closeness))
        rows.append(dict(ml_neuron=int(nid), n_pipeline_units=n_u,
                         unit_ids=[u for u, _ in matched],
                         ml_total_axon_len_um=total_len, ml_conduction_vel=vel,
                         ml_branches=n_br_ml, pipeline_mean_r2=mean_r2,
                         mean_dist_um=closeness, soma_xy=ml_som[nid], score=float(score)))
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top]


def shared_recon_frame(ml_well, pl_well, nid: int, matched_uids, margin: float = 90.0):
    """The single (x0, x1, y0, y1) µm frame that contains BOTH reconstructions at a
    cluster — MaxLive neuron ``nid``'s arbor + soma AND our matched units' arbors +
    init sites. Both F2P1 (our units) and F2P2 (MaxLive) render with THIS frame so
    the two axon-reconstruction plots share an identical plotting area
    (2026-08-27). Deterministic given the same ``matched_uids``."""
    pts = []
    for b in ml_well.arbors().get(nid, {}).values():
        b = np.asarray(b, float)
        if b.ndim == 2 and len(b):
            pts.append(b)
    som = ml_well.somata().get(nid)
    if som is not None:
        pts.append(np.asarray([[som["x"], som["y"]]], float))
    for uid in matched_uids:
        u = pl_well.units.get(int(uid))
        if u is None:
            continue
        for b in u.arbor():
            b = np.asarray(b, float)
            if b.ndim == 2 and len(b):
                pts.append(b)
        pts.append(np.asarray([u.init_xy], float))
    P = np.vstack(pts)
    return (float(P[:, 0].min() - margin), float(P[:, 0].max() + margin),
            float(P[:, 1].min() - margin), float(P[:, 1].max() + margin))


def _ml_branch_count(ml_well, nid: int) -> int:
    ti = ml_well.tracking_info
    if not ti or "neuron" not in ti:
        return 0
    neuron = np.asarray(ti["neuron"], float).astype(int)
    branch = np.asarray(ti["branch"], float).astype(int)
    return int(np.unique(branch[neuron == nid]).size)
