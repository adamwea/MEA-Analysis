"""The cache and its readable JSONs appear whole, or not at all.

A plot suite reads an entity while the capsule beside it is still writing, and a
segment capsule fans out: two writers can reach one well. A half-written JSON is
not an error a reader notices -- it is a number nobody computed.
"""
import json
import os
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
    assert seen["tmp"] == f".qc_report.json.{os.getpid()}.tmp"


def test_the_cache_and_its_readable_json_go_through_it(tmp_path):
    """Not a style point: these are the two files a suite reads."""
    from mea_modules.diagnostics import cache as cache_mod
    from mea_modules.diagnostics import readable

    source = (Path(cache_mod.__file__).read_text(encoding="utf-8")
              + Path(readable.__file__).read_text(encoding="utf-8"))
    assert "(out / RECORD_NAME).write_text(" not in source
    assert "path.write_text(json.dumps(" not in source
