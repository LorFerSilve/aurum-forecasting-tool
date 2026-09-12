"""Import a locally downloaded, development-bounded FRED DFII10 CSV for Phase 10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gold_forecasting.phase10.contracts import load_source
from gold_forecasting.phase10.rate_fred import FRED_RATE_FILENAME, import_fred_dfii10


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build authenticated Phase-10 rate bundles from local FRED DFII10 data."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/phase10/rate") / FRED_RATE_FILENAME,
        help=f"Local FRED CSV, exactly named {FRED_RATE_FILENAME}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/context/phase10/rate"),
        help="New output directory for annual authenticated rate bundles.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("configs/phase10_rate_exploratory.yaml"),
        help="Frozen modeled-latency source contract.",
    )
    args = parser.parse_args()
    result = import_fred_dfii10(args.csv, args.output, load_source(args.source))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
