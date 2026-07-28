from dataclasses import dataclass


@dataclass
class QCConfig:
    highpass_hz: float = 300.0
    mad_threshold: float = 5.0
    activity_min_rate_hz: float = 0.1
    dead_well_frac: float = 0.5
