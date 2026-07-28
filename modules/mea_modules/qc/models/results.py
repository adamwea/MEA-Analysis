import json
from dataclasses import asdict, dataclass


@dataclass
class QCResult:
    passed: bool
    metrics: dict
    per_well: dict

    def to_json(self) -> str:
        """TODO: confirm serialization shape once QC science is implemented."""
        return json.dumps(asdict(self))
