from dataclasses import dataclass


@dataclass
class QCInputs:
    data_h5: str
    well: str | None = None
