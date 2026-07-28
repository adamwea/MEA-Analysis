from .config import QCConfig
from .models.inputs import QCInputs
from .models.results import QCResult


def run_qc(inputs: QCInputs, config: QCConfig) -> QCResult:
    """MODULE 1 — orchestrates load->highpass->MAD->activity/dead-well QC.

    TODO: implement via runner.
    """
    raise NotImplementedError
