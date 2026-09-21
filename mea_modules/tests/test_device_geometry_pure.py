"""Synthetic-grid tests for mea_modules.io.device — measure, don't assume.

The load-bearing property: electrode pitch is MEASURED from the routed
coordinates (Adam's 2026-08-11 ruling), and the two pitches the module
reports mean different things —

* the effective pitch (nearest-neighbour mode, per-axis modal spacing) tracks
  the ROUTED subset: an every-other-electrode config on a 17.5 µm array
  honestly measures 35 µm, which is what downstream cluster/eps logic needs;
* the physical grid pitch, derived from the electrode-id↔coordinate relation,
  recovers 17.5 µm even under that sparse routing, because ids count the
  electrodes routing skipped.

Plus the h5 survey end-to-end on a synthetic Maxwell-shaped file: device
facts extracted with provenance, ADC bit depth derived from lsb×gain, absent
facts recorded as absent, and the pitch sanity check flagging a
non-integer-multiple (checkerboard/diagonal) configuration.
"""

import json

import numpy as np
import pytest

from mea_modules.io import measure_electrode_geometry, survey_well_device

PITCH = 17.5
NCOLS = 220  # Maxwell numbers electrodes row-major over the full array


def _grid(cols, rows, keep=None):
    """Electrode ids + coordinates for a window of the full array.

    `keep(col, row) -> bool` selects the routed subset; ids follow Maxwell's
    row-major full-array numbering, so skipped electrodes leave id gaps.
    """
    ids, xs, ys = [], [], []
    for row in rows:
        for col in cols:
            if keep is not None and not keep(col, row):
                continue
            ids.append(row * NCOLS + col)
            xs.append(col * PITCH)
            ys.append(row * PITCH)
    return np.asarray(ids), np.asarray(xs, float), np.asarray(ys, float)


def test_dense_grid_measures_known_pitch():
    ids, xs, ys = _grid(range(30, 50), range(10, 22))
    geo = measure_electrode_geometry(xs, ys, electrode_ids=ids)

    assert geo["n_electrodes"] == 20 * 12
    assert geo["pitch"]["nn_modal_um"] == pytest.approx(PITCH)
    assert geo["pitch"]["nn_modal_fraction"] == pytest.approx(1.0)
    assert geo["pitch"]["x_pitch_um"] == pytest.approx(PITCH)
    assert geo["pitch"]["y_pitch_um"] == pytest.approx(PITCH)
    assert geo["grid"]["pitch_x_um"] == pytest.approx(PITCH)
    assert geo["grid"]["pitch_y_um"] == pytest.approx(PITCH)
    assert geo["grid"]["row_stride_ids"] == NCOLS
    assert geo["x_extent_um"] == pytest.approx(19 * PITCH)
    assert geo["y_extent_um"] == pytest.approx(11 * PITCH)


def test_every_other_electrode_routing_reports_both_pitches():
    # Every other column AND row routed: the effective pitch doubles; the
    # physical x pitch is still recoverable from the id gaps. The y-side
    # derivation degrades to the effective spacing (nobody routed the odd
    # rows, so the gcd cannot see them) — documented, and honest.
    ids, xs, ys = _grid(
        range(0, 40), range(0, 20),
        keep=lambda col, row: col % 2 == 0 and row % 2 == 0,
    )
    geo = measure_electrode_geometry(xs, ys, electrode_ids=ids)

    assert geo["pitch"]["nn_modal_um"] == pytest.approx(2 * PITCH)  # 35 µm effective
    assert geo["pitch"]["x_pitch_um"] == pytest.approx(2 * PITCH)
    assert geo["pitch"]["y_pitch_um"] == pytest.approx(2 * PITCH)
    assert geo["grid"]["pitch_x_um"] == pytest.approx(PITCH)  # physical recovered
    assert geo["grid"]["pitch_y_um"] == pytest.approx(2 * PITCH)
    assert geo["grid"]["row_stride_ids"] == 2 * NCOLS


def test_partial_random_routing_recovers_physical_grid():
    rng = np.random.default_rng(20260811)
    routed = rng.random((40, 24)) < 0.45  # ~45% of a 40x24 window

    ids, xs, ys = _grid(
        range(40), range(24), keep=lambda col, row: routed[col, row]
    )
    geo = measure_electrode_geometry(xs, ys, electrode_ids=ids)

    # Sparse+irregular routing: the physical grid still comes out exactly.
    assert geo["grid"]["pitch_x_um"] == pytest.approx(PITCH)
    assert geo["grid"]["pitch_y_um"] == pytest.approx(PITCH)
    assert geo["grid"]["row_stride_ids"] == NCOLS
    # The effective modal NN distance is a grid distance — an integer
    # multiple of the pitch on at least one axis (or a diagonal, which this
    # dense-enough draw does not produce).
    ratio = geo["pitch"]["nn_modal_um"] / PITCH
    assert abs(ratio - round(ratio)) < 0.05


def test_degenerate_inputs_do_not_crash():
    empty = measure_electrode_geometry([], [])
    assert empty == {"n_electrodes": 0, "units": "um"}

    single = measure_electrode_geometry([17.5], [35.0])
    assert single["n_electrodes"] == 1
    assert "pitch" not in single

    # Coordinates without ids: geometry yes, grid no.
    _, xs, ys = _grid(range(5), range(5))
    geo = measure_electrode_geometry(xs, ys)
    assert geo["pitch"]["nn_modal_um"] == pytest.approx(PITCH)
    assert "grid" not in geo


# --------------------------------------------------------------------------- #
# survey_well_device on a synthetic Maxwell-shaped h5
# --------------------------------------------------------------------------- #


def _write_synthetic_h5(path, keep=None, neighbors="12"):
    import h5py

    ids, xs, ys = _grid(range(20), range(12), keep=keep)
    mapping = np.zeros(
        ids.size,
        dtype=[("channel", "<i4"), ("electrode", "<i4"), ("x", "<f8"), ("y", "<f8")],
    )
    mapping["channel"] = np.arange(ids.size)
    mapping["electrode"] = ids
    mapping["x"] = xs
    mapping["y"] = ys

    blob = json.dumps(
        {
            "electrodes": {"0": {"fixed": [], "scan": [ids.tolist()]}},
            "properties": {"neighbors": neighbors, "scanning_density": "Full"},
        }
    )

    with h5py.File(path, "w") as h5:
        h5.create_dataset("version", data=[b"20190530"])
        h5.create_dataset("mxw_version", data=[b"25.1.8.2"])
        h5.create_dataset("hdf_version", data=[b"1.14.5"])
        h5.create_dataset("wellplate/id", data=[b"T00001"])
        h5.create_dataset("wellplate/version", data=[b"MaxOne Single Well MEA"])
        h5.create_dataset("wellplate/variant", data=[b"0"])
        h5.create_dataset("wellplate/well000/name", data=[b"1"])
        h5.create_dataset("wellplate/well000/group_name", data=[b"Default Group"])
        h5.create_dataset("assay/run_id", data=[b"000031"])
        h5.create_dataset("assay/script_id", data=[b"AxonRecord_v1.0"])
        h5.create_dataset("assay/inputs/record_time", data=[b"300"])
        h5.create_dataset("assay/inputs/electrodes", data=[blob.encode()])
        temperature = np.zeros(
            3, dtype=[(" Time", "<f8"), ("wellplate_temperature", "<f8")]
        )
        temperature["wellplate_temperature"] = [36.4, 36.5, 36.6]
        h5.create_dataset("environment/temperature", data=temperature)

        rec = "wells/well000/rec0000"
        h5.create_dataset(f"{rec}/settings/gain", data=[512.0])
        h5.create_dataset(f"{rec}/settings/lsb", data=[6.29425039733178e-06])
        h5.create_dataset(f"{rec}/settings/sampling", data=[20000.0])
        h5.create_dataset(f"{rec}/settings/hpf", data=[1.0])
        h5.create_dataset(f"{rec}/settings/spike_threshold", data=[5.5])
        h5.create_dataset(f"{rec}/settings/mapping", data=mapping)
        h5.create_dataset(
            f"{rec}/groups/routed/raw", data=np.zeros((4, 8), dtype=np.uint16)
        )


def test_survey_extracts_device_facts_with_provenance(tmp_path):
    h5_path = tmp_path / "data.raw.h5"
    _write_synthetic_h5(h5_path)

    survey = survey_well_device(h5_path, "well000")
    device = survey["device"]

    assert device["model"] == "MaxOne Single Well MEA"
    assert device["family"] == "MaxOne"
    assert device["wells_in_model"] == 1
    assert device["plate_id"] == "T00001"
    assert device["mxw_version"] == "25.1.8.2"
    assert device["hdf_version"] == "1.14.5"
    assert device["format_version"] == 20190530
    assert device["assay"]["run_id"] == "000031"
    assert device["assay"]["record_time_s"] == 300
    # The GUI cluster size whose assumption prompted this module:
    assert device["assay"]["properties"]["neighbors"] == "12"
    assert device["well_info"]["group_name"] == "Default Group"
    assert device["environment"]["temperature"]["rows"] == 3
    assert device["environment"]["temperature"]["wellplate_temperature_c"]["max"] == 36.6
    assert device["settings"]["gain"] == [512.0]
    assert device["settings"]["sampling_hz"] == [20000.0]
    assert device["adc"]["raw_dtype"] == "uint16"
    assert device["adc"]["bits_derived"] == 10  # from lsb x gain, not assumed

    geometry = device["geometry"]
    assert geometry["grid"]["pitch_x_um"] == pytest.approx(PITCH)
    assert geometry["grid"]["pitch_y_um"] == pytest.approx(PITCH)
    assert geometry["nn_modal_um"] == [pytest.approx(PITCH)]
    assert geometry["pitch_sanity"]["checked"] is True
    assert geometry["pitch_sanity"]["all_integer_multiples"] is True

    # Provenance for every read fact; absent facts recorded, never invented.
    assert device["provenance"]["model"] == "wellplate/version"
    assert device["provenance"]["plate_id"] == "wellplate/id"
    assert "assay.properties" in device["provenance"]
    assert isinstance(device["absent"], list)

    segment = survey["per_segment"]["rec0000"]
    assert segment["n_routed"] == 20 * 12
    assert segment["pitch"]["nn_modal_um"] == pytest.approx(PITCH)
    assert segment["settings"]["gain"] == 512.0
    assert segment["raw_dtype"] == "uint16"

    # JSON-serializable end to end (the ingest manifest embeds this verbatim).
    json.dumps(survey)


def test_survey_flags_diagonal_pitch_oddity(tmp_path):
    # Checkerboard routing: nearest neighbours sit on the diagonal at
    # sqrt(2) x pitch — not an integer multiple; flagged, never fatal.
    h5_path = tmp_path / "data.raw.h5"
    _write_synthetic_h5(h5_path, keep=lambda col, row: (col + row) % 2 == 0)

    survey = survey_well_device(h5_path, "well000")
    sanity = survey["device"]["geometry"]["pitch_sanity"]

    assert sanity["checked"] is True
    assert sanity["all_integer_multiples"] is False
    assert sanity["oddities"][0]["ratio_to_physical"] == pytest.approx(1.414, abs=0.01)
    assert any("pitch oddity" in w for w in survey["device"]["warnings"])


def test_survey_records_absent_facts_on_bare_file(tmp_path):
    import h5py

    h5_path = tmp_path / "bare.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("mxw_version", data=[b"24.2.4"])

    survey = survey_well_device(h5_path, "well000")
    device = survey["device"]

    assert device["mxw_version"] == "24.2.4"
    assert "model" not in device
    assert "model" in device["absent"]
    assert "family" in device["absent"]
    assert survey["per_segment"] == {}
    assert any("per-recording survey unavailable" in w for w in device["warnings"])
