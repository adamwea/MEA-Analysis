"""Pure-numpy tests for the stitch diagnostics wiring + cross-segment modules.

No SpikeInterface, no matplotlib rendering assertions (figures are smoke-
tested to a real PNG path only) — these guard the ARITHMETIC seams: the
transpose/zero-fill re-wire, routing reconstruction, the retention readers,
backbone agreement, peak consistency, and the missed-merge candidate rule.

The synthetic well: 3 segments over 6 union channels ('0'..'5'), 4 units,
20 samples, fs 10 kHz, ms_before 0.5 (nbefore = 5).

    segment 0 routes channels 0,1,2,3   segment 1 routes 0,1,2,4
    segment 2 routes 0,1,2,5            => backbone = {0,1,2}

    unit u0: fires in every segment, peak fixed on channel 1
    unit u1: fires in segments 0+1 only, peak MOVES (ch 0 in seg 0, ch 2 in seg 1)
    unit u2: fires in segment 0 only  \\  same footprint shape on the backbone,
    unit u3: fires in segment 2 only  /  disjoint activity => missed-merge pair
"""

import json
from pathlib import Path

import numpy as np
import pytest

from mea_modules.diagnostics import stitch_consistency, stitch_wiring

FS = 10_000.0
MS_BEFORE = 0.5
N_SAMPLES = 20
NBEFORE = int(MS_BEFORE * FS / 1000.0)  # 5

UNION_CHANNELS = ["0", "1", "2", "3", "4", "5"]
SEGMENT_CHANNELS = [
    ["0", "1", "2", "3"],
    ["0", "1", "2", "4"],
    ["0", "1", "2", "5"],
]
POSITIONS = np.array(
    [[0.0, 0.0], [20.0, 0.0], [40.0, 0.0], [0.0, 20.0], [20.0, 20.0], [40.0, 20.0]]
)
UNIT_IDS = ["u0", "u1", "u2", "u3"]

# spike counts per (unit, segment)
ACTIVITY = np.array([
    [50, 60, 70],   # u0 everywhere
    [40, 40, 0],    # u1 in segments 0 and 1
    [30, 0, 0],     # u2 in segment 0 only
    [0, 0, 30],     # u3 in segment 2 only
])


def _wave(peak_uv):
    """A toy spike: flat baseline, a trough of `peak_uv` after NBEFORE."""
    w = np.zeros(N_SAMPLES)
    w[NBEFORE + 3] = -abs(peak_uv)
    return w


def _segment_template(unit, seg):
    """(n_ch_seg, N_SAMPLES) for one unit in one segment; NaN row when inactive."""
    channels = SEGMENT_CHANNELS[seg]
    out = np.zeros((len(channels), N_SAMPLES))
    if ACTIVITY[unit, seg] == 0:
        out[:] = np.nan
        return out

    def put(channel_id, amp):
        if channel_id in channels:
            out[channels.index(channel_id)] = _wave(amp)

    if unit == 0:
        put("1", 100.0)
        put("0", 40.0)
        put("2", 40.0)
    elif unit == 1:
        # the moving peak: ch 0 loudest in segment 0, ch 2 loudest in segment 1
        if seg == 0:
            put("0", 80.0)
            put("2", 20.0)
        else:
            put("0", 20.0)
            put("2", 80.0)
    elif unit == 2:
        put("1", 60.0)
        put("0", 30.0)
        put("3", 25.0)
    elif unit == 3:
        put("1", 60.0)
        put("0", 30.0)
        put("5", 25.0)
    return out


def _build_stitch_dir(root: Path, averaging_method="spike_count", retention=True):
    """Write a miniature capsule-10 output (+ retention) the modules can read."""
    well = root / "well000"
    well.mkdir(parents=True, exist_ok=True)

    n_units, n_channels = len(UNIT_IDS), len(UNION_CHANNELS)
    templates = np.full((n_units, n_channels, N_SAMPLES), np.nan)
    weight = np.zeros((n_units, n_channels))

    for u in range(n_units):
        acc_w = np.zeros(n_channels)
        acc_t = np.zeros((n_channels, N_SAMPLES))
        for s in range(3):
            if ACTIVITY[u, s] == 0:
                continue
            seg_t = _segment_template(u, s)
            w = 1.0 if averaging_method == "uniform" else float(ACTIVITY[u, s])
            for local, channel_id in enumerate(SEGMENT_CHANNELS[s]):
                c = UNION_CHANNELS.index(channel_id)
                acc_t[c] += seg_t[local] * w
                acc_w[c] += w
        covered = acc_w > 0
        templates[u, covered, :] = acc_t[covered] / acc_w[covered, None]
        weight[u] = np.where(covered, acc_w, 0.0)

    np.save(well / "merged_templates.npy", templates)
    np.save(well / "contributing_weight.npy", weight)
    np.save(well / "channel_locations_xy.npy", POSITIONS)
    (well / "channel_ids.json").write_text(json.dumps(UNION_CHANNELS))
    (well / "unit_ids.json").write_text(json.dumps(UNIT_IDS))
    (well / "merge_manifest.json").write_text(json.dumps({
        "version": 2, "well": "well000", "n_segments": 3,
        "n_units": n_units, "n_channels_union": n_channels,
        "averaging_method": averaging_method, "weighting": averaging_method,
        "ms_before": MS_BEFORE, "ms_after": 1.5,
        "sampling_frequency_hz": FS,
        "segment_contributions_dir": "segment_contributions" if retention else None,
    }))

    if retention:
        ret = well / "segment_contributions"
        index = []
        for s in range(3):
            seg_dir = ret / f"segment_{s:03d}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            seg_templates = np.stack(
                [_segment_template(u, s) for u in range(n_units)]
            ).astype(np.float32)
            np.save(seg_dir / "templates.npy", seg_templates)
            np.save(seg_dir / "spike_counts.npy", ACTIVITY[:, s].astype(np.int64))
            np.save(seg_dir / "locations_xy.npy", np.array(
                [POSITIONS[UNION_CHANNELS.index(c)] for c in SEGMENT_CHANNELS[s]]
            ))
            (seg_dir / "channel_ids.json").write_text(json.dumps(SEGMENT_CHANNELS[s]))
            index.append({
                "index": s, "dirname": seg_dir.name, "analyzer_dir": f"fake/{s}",
                "n_channels": len(SEGMENT_CHANNELS[s]),
                "n_units_contributing": int((ACTIVITY[:, s] > 0).sum()),
            })
        (ret / "unit_ids.json").write_text(json.dumps(UNIT_IDS))
        (ret / "segments_index.json").write_text(json.dumps(index))
    return well


@pytest.fixture()
def stitch_well(tmp_path):
    return _build_stitch_dir(tmp_path / "stitch")


# --------------------------------------------------------------------------- #
# stitch_wiring
# --------------------------------------------------------------------------- #


def test_nbefore_from_manifest(stitch_well):
    manifest = json.loads((stitch_well / "merge_manifest.json").read_text())
    assert stitch_wiring.nbefore_from_manifest(manifest) == NBEFORE


def test_sensitivity_templates_transposes_and_zero_fills(stitch_well):
    templates = np.load(stitch_well / "merged_templates.npy")
    weight = np.load(stitch_well / "contributing_weight.npy")
    sens = stitch_wiring.sensitivity_templates(templates, weight)

    assert sens.shape == (len(UNIT_IDS), N_SAMPLES, len(UNION_CHANNELS))
    assert not np.isnan(sens).any()
    # u2 never covered channels 4 ('4') and 5 ('5') -> zero-filled
    assert np.all(sens[2, :, 4] == 0.0)
    assert np.all(sens[2, :, 5] == 0.0)
    # a covered value survives the transpose in the right place:
    # u0 peak on union channel 1, at sample NBEFORE+3
    assert sens[0, NBEFORE + 3, 1] == pytest.approx(-100.0)


def test_sensitivity_templates_rejects_mask_mismatch(stitch_well):
    templates = np.load(stitch_well / "merged_templates.npy")
    weight = np.load(stitch_well / "contributing_weight.npy").copy()
    weight[2, 5] = 7.0  # claims coverage where the template says NaN
    with pytest.raises(ValueError, match="disagree"):
        stitch_wiring.sensitivity_templates(templates, weight)


def test_routing_from_retention_is_exact(stitch_well):
    routing, index = stitch_wiring.routing_from_retention(
        stitch_well / "segment_contributions", UNION_CHANNELS,
    )
    assert routing.shape == (3, 6)
    expected_counts = np.array([3, 3, 3, 1, 1, 1])  # backbone 0,1,2; tiles 3,4,5
    assert np.array_equal(routing.sum(axis=0), expected_counts)
    assert routing[1, UNION_CHANNELS.index("4")]  # segment 1 routed channel '4'
    assert not routing[1, UNION_CHANNELS.index("5")]
    assert [e["dirname"] for e in index] == [f"segment_{s:03d}" for s in range(3)]


def test_routing_surrogate_matches_uniform_column_sums(tmp_path):
    well = _build_stitch_dir(tmp_path / "uni", averaging_method="uniform",
                             retention=False)
    weight = np.load(well / "contributing_weight.npy")
    routing = stitch_wiring.routing_surrogate_from_weight(weight, 3)
    # u0 fires everywhere, so its uniform weights recover the true counts
    assert np.array_equal(routing.sum(axis=0), np.array([3, 3, 3, 1, 1, 1]))


def test_resolve_routing_precedence(tmp_path, stitch_well):
    manifest = json.loads((stitch_well / "merge_manifest.json").read_text())
    weight = np.load(stitch_well / "contributing_weight.npy")
    _routing, source = stitch_wiring.resolve_routing(stitch_well, manifest, weight)
    assert source == "retention"

    uni = _build_stitch_dir(tmp_path / "uni2", averaging_method="uniform",
                            retention=False)
    uni_manifest = json.loads((uni / "merge_manifest.json").read_text())
    uni_weight = np.load(uni / "contributing_weight.npy")
    _routing, source = stitch_wiring.resolve_routing(uni, uni_manifest, uni_weight)
    assert source == "uniform-weight surrogate"

    spk = _build_stitch_dir(tmp_path / "spk", averaging_method="spike_count",
                            retention=False)
    spk_manifest = json.loads((spk / "merge_manifest.json").read_text())
    spk_weight = np.load(spk / "contributing_weight.npy")
    routing, source = stitch_wiring.resolve_routing(spk, spk_manifest, spk_weight)
    assert routing is None and source == "unavailable"


def test_nan_coverage_summary_counts(stitch_well):
    weight = np.load(stitch_well / "contributing_weight.npy")
    summary = stitch_wiring.nan_coverage_summary(weight, unit_ids=UNIT_IDS,
                                                 n_segments=3)
    # u0: 6 covered channels; u1: 5 (0,1,2,3,4); u2: 4 (0,1,2,3); u3: 4 (0,1,2,5)
    assert summary["covered_cells"] == 6 + 5 + 4 + 4
    assert summary["n_zero_coverage_units"] == 0
    assert summary["per_unit_covered_channels"]["max"] == 6
    expected_nan = 1.0 - (19 / 24)
    assert summary["nan_fraction_unit_channel_cells"] == pytest.approx(expected_nan)


def test_compare_stitches_flags_only_multi_coverage(tmp_path):
    a = _build_stitch_dir(tmp_path / "a", averaging_method="uniform")
    b = _build_stitch_dir(tmp_path / "b", averaging_method="spike_count")
    from mea_modules.templates import load_well_inputs

    ia, ib = load_well_inputs(a), load_well_inputs(b)
    summary, arrays = stitch_wiring.compare_stitches(ia, ib, ("uniform", "spike_count"))
    # single-coverage cells must agree exactly regardless of weighting
    assert summary["single_coverage_max_abs_diff_uv"] == pytest.approx(0.0)
    # u1's backbone cells (covered by 2 segments with UNEQUAL waves) must differ:
    # uniform mean of 80/20 vs spike-count mean of equal counts -> equal here...
    # counts are 40/40 so spike_count == uniform for u1; u0 has unequal counts
    # (50/60/70) but IDENTICAL per-segment waves, so it agrees too. Perturb b:
    ib["templates"][0, 1, NBEFORE + 3] += 5.0
    summary2, _ = stitch_wiring.compare_stitches(ia, ib, ("uniform", "perturbed"))
    assert summary2["multi_coverage_abs_diff_uv"]["max"] == pytest.approx(5.0)
    assert summary2["multi_coverage_cells_changed_gt_0p01uv"] == 1


# --------------------------------------------------------------------------- #
# stitch_consistency
# --------------------------------------------------------------------------- #


def test_backbone_and_activity_readers(stitch_well):
    ret = stitch_well / "segment_contributions"
    index, unit_ids = stitch_consistency.load_retention_index(ret)
    assert unit_ids == UNIT_IDS
    assert stitch_consistency.backbone_channel_ids(ret, index) == ["0", "1", "2"]
    activity = stitch_consistency.segment_activity(ret, index)
    assert np.array_equal(activity, ACTIVITY)


def test_backbone_agreement_perfect_for_consistent_unit(stitch_well):
    templates = np.load(stitch_well / "merged_templates.npy")
    agreement = stitch_consistency.backbone_agreement(
        stitch_well / "segment_contributions", templates, UNION_CHANNELS,
    )
    r, active = agreement["r"], agreement["active"]
    assert active.shape == (4, 3)
    assert np.array_equal(active, ACTIVITY > 0)
    # u0's per-segment backbone waves are identical to the stitched average
    assert np.all(r[0, :] > 0.999)
    # u1's two segments disagree (peak swaps 0 <-> 2), so agreement with the
    # average is strictly lower than a consistent unit's
    assert np.all(r[1, [0, 1]] < 0.999)
    assert np.isnan(r[1, 2])  # inactive segment stays NaN
    # u2 fired once: its single segment IS the stitched value on the backbone
    assert r[2, 0] > 0.999


def test_peak_consistency_flags_the_moving_unit(stitch_well):
    result = stitch_consistency.peak_consistency(
        stitch_well / "segment_contributions",
    )
    # u0's backbone peak is channel '1' in every segment
    assert result["n_distinct_peaks"][0] == 1
    assert result["max_spread_um"][0] == 0.0
    # u1's backbone peak moves '0' -> '2' = 40 um apart
    assert result["n_distinct_peaks"][1] == 2
    assert result["max_spread_um"][1] == pytest.approx(40.0)
    # single-segment units cannot move
    assert result["n_distinct_peaks"][2] == 1
    assert result["n_active_segments"][3] == 1


def test_missed_merge_candidates_finds_the_planted_pair(stitch_well):
    templates = np.load(stitch_well / "merged_templates.npy")
    weight = np.load(stitch_well / "contributing_weight.npy")
    candidates = stitch_consistency.missed_merge_candidates(
        templates, weight, POSITIONS, ACTIVITY, unit_ids=UNIT_IDS,
        peak_distance_um=150.0, min_footprint_cosine=0.8,
        max_activity_jaccard=0.5, min_active_segments=1,
        min_shared_channels=3,  # the toy union is only 6 channels wide
    )
    pairs = {(c["unit_a"], c["unit_b"]) for c in candidates}
    assert ("u2", "u3") in pairs  # same backbone footprint, disjoint segments
    for c in candidates:
        if (c["unit_a"], c["unit_b"]) == ("u2", "u3"):
            assert c["activity_jaccard"] == 0.0
            assert c["footprint_cosine"] > 0.9


def test_plots_smoke(tmp_path, stitch_well):
    """Every emitter writes a real PNG without raising."""
    templates = np.load(stitch_well / "merged_templates.npy")
    weight = np.load(stitch_well / "contributing_weight.npy")
    ret = stitch_well / "segment_contributions"

    out = stitch_wiring.plot_nan_coverage_summary(
        weight, POSITIONS, tmp_path / "nan.png")
    assert Path(out).stat().st_size > 0

    agreement = stitch_consistency.backbone_agreement(ret, templates, UNION_CHANNELS)
    out = stitch_consistency.plot_backbone_agreement(agreement, tmp_path / "agree.png")
    assert Path(out).stat().st_size > 0

    result = stitch_consistency.peak_consistency(ret)
    out = stitch_consistency.plot_peak_consistency(
        result, tmp_path / "peaks.png", unit_ids=UNIT_IDS, spread_flag_um=10.0)
    assert Path(out).stat().st_size > 0
    out = stitch_consistency.plot_unit_peak_map(
        result, 1, tmp_path / "peak_map.png", unit_id="u1")
    assert Path(out).stat().st_size > 0

    candidates = stitch_consistency.missed_merge_candidates(
        templates, weight, POSITIONS, ACTIVITY, unit_ids=UNIT_IDS,
        min_shared_channels=3)
    out = stitch_consistency.plot_missed_merge_candidates(
        candidates, templates, weight, POSITIONS, UNIT_IDS, tmp_path / "mm.png")
    if candidates:
        assert Path(out).stat().st_size > 0
