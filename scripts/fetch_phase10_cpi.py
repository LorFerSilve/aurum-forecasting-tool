"""Acquire the fixed official BLS CPI-U flat file for Phase 10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gold_forecasting.phase10.cpi_acquisition import download_bls_cpi
from gold_forecasting.phase10.cpi_bls import BLS_CPI_FILENAME


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely download the official Phase-10 BLS CPI source."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/phase10/cpi") / BLS_CPI_FILENAME,
        help=f"Destination file, which must be named exactly {BLS_CPI_FILENAME}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing regular destination file.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Positive HTTP timeout in seconds.",
    )
    args = parser.parse_args()
    result = download_bls_cpi(
        args.destination,
        overwrite=args.overwrite,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
