"""Recompute the persisted Phase-9 cumulative five-step path-return MAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gold_forecasting.phase9.audit import audit_phase9_path_return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/phase9_path_return_audit.json"),
    )
    args = parser.parse_args()
    result = audit_phase9_path_return(
        args.run_directory,
        report_path=args.report,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
