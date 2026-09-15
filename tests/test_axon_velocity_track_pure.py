"""`track_unit_axon`'s typed no-axon outcome and `save_no_axon`'s on-disk shape.

`axon_velocity` reports "this unit yields no axon" by raising a plain
``Exception`` with one of three messages. The wrapper must surface exactly
those as `NoAxonFound` and let every other exception through unchanged, so a
caller can count a no-axon unit as a result rather than a failure.
"""
import inspect
import json
import re

import numpy as np
import pytest

axon_velocity = pytest.importorskip("axon_velocity")

from mea_modules.reconstruction import (  # noqa: E402
    GTR_FILENAME,
    SUMMARY_FILENAME,
    NoAxonFound,
    save_no_axon,
    track_unit_axon,
)
from mea_modules.reconstruction.axon_velocity_track import _NO_AXON_MESSAGES  # noqa: E402

FS = 20000.0


def _inputs():
    return np.zeros((5, 10)), np.zeros((5, 2))


@pytest.mark.parametrize("message", sorted(_NO_AXON_MESSAGES))
def test_no_axon_messages_become_typed_outcome(monkeypatch, message):
    def fake_tracker(*_args, **_kwargs):
        raise Exception(message)

    monkeypatch.setattr(axon_velocity, "compute_graph_propagation_velocity", fake_tracker)
    template, locations = _inputs()
    with pytest.raises(NoAxonFound) as info:
        track_unit_axon(template, locations, FS)
    assert str(info.value) == message
    assert str(info.value.__cause__) == message  # chained to the library's own exception


def test_no_axon_messages_are_the_installed_library_literals():
    """The routing matches message text verbatim, so the installed
    axon_velocity must still raise exactly these strings; a repunctuated
    message upstream would silently turn every no-axon unit into a failure."""
    from axon_velocity import tracking_classes

    source = inspect.getsource(tracking_classes)
    literals = set(re.findall(r'raise Exception\("([^"]+)"\)', source))
    missing = sorted(_NO_AXON_MESSAGES - literals)
    assert not missing, f"installed axon_velocity no longer raises verbatim: {missing}"


def test_other_errors_propagate_unchanged(monkeypatch):
    def fake_tracker(*_args, **_kwargs):
        raise ValueError("a real error, not a no-axon outcome")

    monkeypatch.setattr(axon_velocity, "compute_graph_propagation_velocity", fake_tracker)
    template, locations = _inputs()
    with pytest.raises(ValueError):
        track_unit_axon(template, locations, FS)


def test_no_axon_is_a_runtime_error_subclass():
    # a caller catching RuntimeError still sees it; a caller catching
    # NoAxonFound first can single it out
    assert issubclass(NoAxonFound, RuntimeError)


def test_save_no_axon_writes_summary_only(tmp_path):
    out_dir = tmp_path / "unit0194"
    summary = save_no_axon(out_dir, unit_id="194", reason="No branches left after cleaning")

    assert (out_dir / SUMMARY_FILENAME).is_file()
    assert not (out_dir / GTR_FILENAME).exists()
    assert not (out_dir / (SUMMARY_FILENAME + ".tmp")).exists()  # the atomic write left no temp file
    assert json.loads((out_dir / SUMMARY_FILENAME).read_text()) == summary
    assert summary["status"] == "no_axon"
    assert summary["reason"] == "No branches left after cleaning"
    assert summary["n_branches"] == 0
    assert summary["branches"] == []
    assert summary["selected_channels"] == []
    # nothing invented for values the tracker never produced
    assert summary["n_selected_channels"] is None
    assert summary["init_channel"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
