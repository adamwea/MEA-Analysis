import argparse
import json
import os

from .api import run_qc
from .config import QCConfig
from .models.inputs import QCInputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MEA QC (Module 1)")
    parser.add_argument("--data-h5", required=True, help="Path to data.raw.h5")
    parser.add_argument("--output-dir", required=True, help="Directory to write QC result json")
    parser.add_argument("--well", default=None, help="Optional well id")
    parser.add_argument("--highpass-hz", type=float, default=QCConfig.highpass_hz)
    parser.add_argument("--mad-threshold", type=float, default=QCConfig.mad_threshold)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    inputs = QCInputs(data_h5=args.data_h5, well=args.well)
    config = QCConfig(highpass_hz=args.highpass_hz, mad_threshold=args.mad_threshold)

    result = run_qc(inputs, config)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "qc_result.json")
    with open(out_path, "w") as f:
        f.write(result.to_json())


if __name__ == "__main__":
    main()
