"""The cache and its readable JSONs appear whole, or not at all.

A plot suite reads an entity while the capsule beside it is still writing, and a
segment capsule fans out: two writers can reach one well. A half-written JSON is
not an error a reader notices -- it is a number nobody computed.
"""
import json
import os
import threading
from pathlib import Path

import pytest

from mea_modules.diagnostics import atomic





def test_a_reader_sees_the_old_file_or_the_new_one(tmp_path):
    path = tmp_path / "diagnostics_cache.json"
    atomic.write_json(path, {"version": 1})
    atomic.write_json(path, {"version": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["diagnostics_cache.json"]


def test_a_body_that_raises_leaves_neither_the_file_nor_its_scratch(tmp_path):
    path = tmp_path / "diagnostics_cache.npz"

    def explode(tmp):
        Path(tmp).write_bytes(b"half an archive")
        raise RuntimeError("the arrays did not fit")

    with pytest.raises(RuntimeError, match="did not fit"):
        atomic.write_with(path, explode)
    assert not path.exists() and list(tmp_path.iterdir()) == []


def test_the_temp_file_is_the_writer_s_own(tmp_path):
    seen = {}
    real = os.replace  # captured before the patch: the spy must not call itself

    def spy(source, target):
        seen["tmp"] = Path(source).name
        real(source, target)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(atomic.os, "replace", spy)
        atomic.write_text(tmp_path / "qc_report.json", "{}")
    assert seen["tmp"].startswith(f".qc_report.json.{os.getpid()}.")
    # and the thread, so two threads of one process never share a scratch file
    assert seen["tmp"].endswith(f".{threading.get_ident()}.tmp")


def test_every_file_a_suite_reads_is_replaced_exactly_once(tmp_path, monkeypatch):
    """Behaviour, not a grep: each file a capsule leaves for a suite -- the
    record, the arrays, the readable JSONs, the timing -- lands by one replace
    of a complete temporary, so no reader can meet a half-written one."""
    from mea_modules.diagnostics.readable import write_segment_reports
    from mea_modules.diagnostics.timing import write_timing
    from test_segment_reports import _cache  # the real producer's own fixture

    replaced, real = [], os.replace

    def spy(source, target):
        replaced.append(Path(target).name)
        real(source, target)

    monkeypatch.setattr(atomic.os, "replace", spy)
    _cache(tmp_path)
    write_segment_reports(tmp_path, capsule="preprocess_segment", well="well000",
                          rec="rec0000")
    write_timing(tmp_path, capsule="preprocess_segment", entity={"well": "well000"},
                 steps={"collect": {"seconds": 1.0, "started": 1.0, "ended": 2.0}})

    assert "diagnostics_cache.json" in replaced and "diagnostics_cache.npz" in replaced
    assert "timing.json" in replaced
    assert any(name.endswith(".json") and name.startswith("qc_report") for name in replaced)
    # nothing was written twice, and no temporary survives
    assert len(replaced) == len(set(replaced))
    folder = next(p for p in tmp_path.rglob("diagnostics") if p.is_dir())
    assert not [p for p in folder.iterdir() if p.name.startswith(".")]
