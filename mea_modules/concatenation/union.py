"""A recording over the union electrode set, without writing a byte.

Concatenation narrowed the array to the electrodes routed in *every* segment —
353 of 13,384 on the reference scan — because that intersection is the only
channel set that sits on one continuous timeline. The sort happened there. To
put the sorted units back on the whole array, something has to present the
UNION of every segment's electrodes as one recording, since SpikeInterface's
``create_sorting_analyzer`` reaches into ``recording.get_probes()`` and
``recording.has_scaleable_traces()`` and will not accept ``recording=None``.

:func:`union_recording` builds that lazily. Each segment is padded from its own
electrode set up to the union — real traces where the electrode was routed,
zeros where it was not — and the padded segments are concatenated. Nothing is
read and nothing is written until somebody asks for traces.

**Zeros are absence, not silence.** A segment that never routed an electrode
contributes zeros there, so a mean computed straight off this recording is
scaled down by the fraction of segments that missed the electrode. That is why
templates for this path are merged per segment by `kssynth` — which knows which
segments actually reached each channel — and then injected into the analyzer,
rather than recomputed from these traces. Do not call
``analyzer.compute("templates")`` on a union recording and expect the amplitudes
to mean anything.

SpikeInterface ships ``preprocessing.zero_channel_pad``, which does exactly the
per-segment padding, and this module uses it when it works. Through 0.103.2 its
explicit-``channel_mapping`` path is broken twice over: the mapping is validated
and then never assigned, so construction raises ``AttributeError``; and the
segment passes the mapping as the *parent's* ``channel_indices``, which is out
of range for any mapping reaching past the parent's channel count — i.e. for
every mapping worth passing. Both are fixed in
``adamwea/spikeinterface@fix/zero-channel-pad-explicit-mapping`` and are on
their way upstream.

:func:`_padder` probes for the fix once and falls back to the local
:class:`UnionChannelRecording` when the installed SpikeInterface still has the
bug, so this works on the pinned 0.103.2 today and drops onto upstream's
implementation the moment the fix ships. The fallback is behaviour-identical;
the probe exists so nobody has to remember which SI they are running.
"""

import logging

import numpy as np
from spikeinterface.core import BaseRecording, BaseRecordingSegment

logger = logging.getLogger(__name__)


class UnionChannelRecording(BaseRecording):
    """One segment presented on the global channel set, zero elsewhere.

    `channel_mapping[i]` is the global index of this segment's local channel
    `i`. Global channels named by nobody read as zeros.
    """

    def __init__(self, recording, n_global_channels, channel_mapping, channel_ids=None):
        mapping = np.asarray(channel_mapping, dtype=np.int64)
        n_local = int(recording.get_num_channels())
        if mapping.size != n_local:
            raise ValueError(
                f"channel_mapping has {mapping.size} entries for a recording with "
                f"{n_local} channels; every local channel needs a global slot"
            )
        if mapping.size and (mapping.min() < 0 or mapping.max() >= int(n_global_channels)):
            raise ValueError(
                f"channel_mapping targets global index {mapping.min()}..{mapping.max()} "
                f"outside the grid of {n_global_channels} channels"
            )
        duplicates = mapping.size - np.unique(mapping).size
        if duplicates:
            # Two local channels landing on one global slot means the grid
            # tolerance merged electrodes that this segment holds apart; the
            # second write would silently win.
            raise ValueError(
                f"channel_mapping sends {duplicates} local channel(s) to a global "
                "slot already taken; the channel grid tolerance is too loose"
            )

        # Channel ids default to global grid position, but a caller who knows
        # the electrode numbers should pass them: on Maxwell the channel id IS
        # the electrode id, and losing it here means every footprint downstream
        # is labelled by grid slot instead of by electrode.
        if channel_ids is None:
            ids = np.arange(int(n_global_channels))
        else:
            ids = np.asarray(list(channel_ids))
            if ids.size != int(n_global_channels):
                raise ValueError(
                    f"channel_ids has {ids.size} entries for a grid of "
                    f"{n_global_channels} channels"
                )

        BaseRecording.__init__(
            self,
            recording.get_sampling_frequency(),
            ids,
            recording.get_dtype(),
        )
        self._parent = recording
        self.n_global_channels = int(n_global_channels)
        self.channel_mapping = mapping

        for segment in recording._recording_segments:
            # Sampling frequency comes from the RECORDING, not the segment:
            # SpikeInterface stopped exposing it on every segment type, and
            # reading it off the segment breaks on newer releases.
            self.add_recording_segment(
                _UnionChannelSegment(
                    segment,
                    self.n_global_channels,
                    mapping,
                    recording.get_dtype(),
                    recording.get_sampling_frequency(),
                )
            )

        self._kwargs = {
            "recording": recording,
            "n_global_channels": int(n_global_channels),
            "channel_mapping": mapping.tolist(),
            "channel_ids": None if channel_ids is None else ids.tolist(),
        }


class _UnionChannelSegment(BaseRecordingSegment):
    """Widen one segment's traces to the global channel count."""

    def __init__(self, parent_segment, n_global_channels, channel_mapping, dtype, sampling_frequency):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._parent_segment = parent_segment
        self._n_global = int(n_global_channels)
        self._mapping = channel_mapping
        self._dtype = dtype

    def get_num_samples(self):
        return self._parent_segment.get_num_samples()

    def get_traces(self, start_frame, end_frame, channel_indices):
        # Resolve which global channels were asked for, then pull ONLY the
        # local channels backing them. Reading the parent's full width and
        # discarding most of it is what makes a 13k-channel grid unusable.
        wanted = np.arange(self._n_global)[channel_indices] if channel_indices is not None \
            else np.arange(self._n_global)
        wanted = np.atleast_1d(wanted)

        # global index -> local index, -1 where this segment has nothing.
        inverse = np.full(self._n_global, -1, dtype=np.int64)
        inverse[self._mapping] = np.arange(self._mapping.size, dtype=np.int64)
        local = inverse[wanted]
        present = local >= 0

        n_samples = int(end_frame) - int(start_frame)
        out = np.zeros((n_samples, wanted.size), dtype=self._dtype)
        if not present.any():
            return out

        traces = self._parent_segment.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_indices=local[present],
        )
        out[:, present] = np.asarray(traces, dtype=self._dtype)
        return out


def build_channel_mapping(positions, grid_positions, tolerance_um=1.0):
    """Global index for each of `positions`, matched by xy within tolerance.

    Raises rather than guessing when an electrode has no match: an unmatched
    channel silently dropped from the grid is a hole in the recovered footprint
    that nothing downstream would report.
    """
    positions = np.asarray(positions, dtype=float)[:, :2]
    grid_positions = np.asarray(grid_positions, dtype=float)[:, :2]
    if grid_positions.size == 0:
        raise ValueError("the channel grid is empty; nothing to map onto")

    mapping = np.empty(positions.shape[0], dtype=np.int64)
    worst = 0.0
    for index, point in enumerate(positions):
        distances = np.sum((grid_positions - point) ** 2, axis=1)
        nearest = int(np.argmin(distances))
        distance = float(np.sqrt(distances[nearest]))
        if distance > float(tolerance_um):
            raise ValueError(
                f"electrode at {tuple(point)} has no grid position within "
                f"{tolerance_um} um (nearest is {distance:.3f} um away); the "
                "grid was not built from these segments"
            )
        mapping[index] = nearest
        worst = max(worst, distance)

    logger.debug("mapped %d electrodes onto the grid (worst offset %.4f um)", mapping.size, worst)
    return mapping


def _upstream_padding_works():
    """True when the installed SpikeInterface honours an explicit mapping.

    Probed rather than version-gated: the fix may arrive in a patch release, a
    fork install, or a nightly, and a version comparison would get all three
    wrong. The probe is a two-channel recording padded to four with a mapping
    that reaches past the parent's channel count — the exact shape that fails
    on a broken build — and it costs microseconds, once.
    """
    try:
        from spikeinterface.core import NumpyRecording
        from spikeinterface.preprocessing import zero_channel_pad
    except Exception:
        return False

    try:
        probe = NumpyRecording(
            traces_list=[np.arange(8, dtype="float32").reshape(4, 2)],
            sampling_frequency=1000.0,
        )
        # `copy_metadata(only_main=True)` inside zero_channel_pad reads a "name"
        # annotation that a bare NumpyRecording does not carry; without it the
        # probe fails for a reason unrelated to what it is testing.
        probe.annotate(name="padding_probe")
        padded = zero_channel_pad(probe, num_channels=4, channel_mapping=[2, 3])
        traces = padded.get_traces()
    except Exception as exc:
        logger.debug("spikeinterface zero_channel_pad is unusable here (%s)", exc)
        return False

    ok = (
        traces.shape == (4, 4)
        and np.allclose(traces[:, 2:], probe.get_traces())
        and np.allclose(traces[:, :2], 0.0)
    )
    if not ok:
        logger.debug("spikeinterface zero_channel_pad placed channels wrongly")
    return bool(ok)


_UPSTREAM_PADDING = None


def _padder(recording, n_global, mapping, channel_ids):
    """One padded segment, from upstream when it works and locally when not."""
    global _UPSTREAM_PADDING
    if _UPSTREAM_PADDING is None:
        _UPSTREAM_PADDING = _upstream_padding_works()
        logger.info(
            "channel padding: using %s",
            "spikeinterface.preprocessing.zero_channel_pad"
            if _UPSTREAM_PADDING
            else "the local UnionChannelRecording (installed SpikeInterface "
            "mishandles an explicit channel_mapping)",
        )

    if not _UPSTREAM_PADDING:
        return UnionChannelRecording(recording, n_global, mapping, channel_ids=channel_ids)

    from spikeinterface.preprocessing import zero_channel_pad

    padded = zero_channel_pad(recording, num_channels=int(n_global), channel_mapping=list(mapping))
    if channel_ids is not None:
        # zero_channel_pad numbers channels by grid position; on Maxwell the id
        # is the electrode, and losing it makes every footprint downstream
        # unreadable.
        padded = padded.rename_channels(list(channel_ids))
    return padded


def union_recording(recordings, grid_positions, tolerance_um=1.0, channel_ids=None):
    """Concatenate `recordings` onto the global grid; return the lazy recording.

    Segments keep their own electrode sets; each is widened to `grid_positions`
    and they are joined in the order given — which must be the order the
    concatenated timeline used, or the spike frames will not land in the right
    segment.

    `channel_ids` names the grid's channels (electrode ids on Maxwell). Left
    None they are numbered by grid position, which loses electrode identity —
    pass them whenever the caller knows them.
    """
    from .concat import concatenate_segments

    grid_positions = np.asarray(grid_positions, dtype=float)
    n_global = int(grid_positions.shape[0])

    padded, covered = [], np.zeros(n_global, dtype=bool)
    for recording in recordings:
        mapping = build_channel_mapping(
            recording.get_channel_locations(), grid_positions, tolerance_um=tolerance_um
        )
        covered[mapping] = True
        padded.append(_padder(recording, n_global, mapping, channel_ids))

    if not covered.all():
        # The grid claims electrodes no segment provides. Not fatal — those
        # channels simply read zero everywhere — but it means the grid did not
        # come from these recordings, which is worth saying.
        logger.warning(
            "%d of %d grid channels are not routed by any segment and will read "
            "as zeros throughout", int((~covered).sum()), n_global,
        )

    joined = concatenate_segments(padded)
    logger.info(
        "union recording: %d segment(s) -> %d channels x %d samples (lazy)",
        len(padded), joined.get_num_channels(), joined.get_num_samples(),
    )
    return joined


__all__ = ["UnionChannelRecording", "build_channel_mapping", "union_recording"]
