"""Locate the Maxwell HDF5 compression plugin and register it with libhdf5.

Maxwell writes its raw files with a proprietary compression filter. libhdf5
loads that filter from whatever directory ``HDF5_PLUGIN_PATH`` points at, so the
directory has to be set *before* the file is opened or the read fails with an
opaque "unable to open" error.

This package is lab-shared, so it never hardcodes where the plugin lives — the
caller supplies candidate directories. MEA-recon-pipeline vendors the plugin at
``vendor/maxwell_hdf5_plugin/<platform>/`` and passes that in; other callers can
pass their own, or rely on ``HDF5_PLUGIN_PATH`` already being exported.

Setting HDF5_PLUGIN_PATH to a directory that already holds the library also
stops SpikeInterface/neo from trying to download its own copy at import time.
"""

import os
import platform
from pathlib import Path

# libhdf5 filter plugin filename, per platform.
PLUGIN_LIB_NAMES = {
    "Linux": "libcompression.so",
    "Darwin": "libcompression.dylib",
    "Windows": "compression.dll",
}


def plugin_lib_name(system=None):
    """Return the plugin filename for this platform (default: Linux's)."""
    return PLUGIN_LIB_NAMES.get(system or platform.system(), "libcompression.so")


def is_plugin_dir(path):
    """True if `path` is a directory holding this platform's plugin library."""
    if not path:
        return False
    return (Path(path).expanduser() / plugin_lib_name()).is_file()


def find_plugin_dir(candidates):
    """Return the first candidate directory that actually holds the plugin.

    Candidates may be None or missing; they are skipped. Returns None if none
    of them qualify.
    """
    for candidate in candidates:
        if candidate and is_plugin_dir(candidate):
            return Path(candidate).expanduser().resolve()
    return None


def set_plugin_path(hdf5_plugin_path=None, extra_candidates=()):
    """Point HDF5_PLUGIN_PATH at the Maxwell plugin; return the dir used.

    Resolution order: the explicit argument, then ``$HDF5_PLUGIN_PATH``, then
    any `extra_candidates` the caller offers as fallbacks. Returns None when
    nothing usable is found — that is not fatal here, because a caller may have
    the filter registered by other means; the failure surfaces at file open.

    Raises FileNotFoundError when `hdf5_plugin_path` is given explicitly but
    holds no plugin, so a typo fails loudly and early rather than as an opaque
    HDF5 read error later.
    """
    if hdf5_plugin_path:
        plugin_dir = Path(hdf5_plugin_path).expanduser().resolve()
        if not is_plugin_dir(plugin_dir):
            raise FileNotFoundError(
                f"no {plugin_lib_name()} in HDF5 plugin dir: {plugin_dir}. "
                "Supply a directory holding the Maxwell HDF5 compression plugin."
            )
    else:
        plugin_dir = find_plugin_dir([os.environ.get("HDF5_PLUGIN_PATH"), *extra_candidates])

    if plugin_dir is None:
        return None

    os.environ["HDF5_PLUGIN_PATH"] = str(plugin_dir)
    return plugin_dir
