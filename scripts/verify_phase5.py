"""Rebuild phase-5 data twice and optionally verify the complete unchanged MVP model.

Run from the repository root with the pinned Python environment. Each successful
check is persisted under reports; original immutable raw archives are reused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from gold_forecasting.artifacts import DatasetManifest, sha256_file, write_json_atomic
from gold_forecasting.data_pipeline import update_mvp_data, validate_existing_mvp_data
from gold_forecasting.pipeline import run_mvp_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-mvp", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/phase5.yaml"
    mvp_config = root / "configs/mvp.yaml"
    baseline_paths = [
        root / "data/raw/histdata/XAUUSD/1min/manifest.json",
        *(
            root / f"data/curated/histdata/XAU_USD/{tf}/manifest.json"
            for tf in ("1min", "3min", "15min")
        ),
    ]
    baseline_hashes = {str(path): sha256_file(path) for path in baseline_paths}
    print("Building phase-5 data via incremental updater (first pass)", flush=True)
    started = perf_counter()
    first = update_mvp_data(config)
    first_seconds = perf_counter() - started
    completion_path = first.curated_manifest_paths["1min"].parent.parent / "phase5_build.json"
    first_completion = completion_path.read_bytes()
    print(f"First pass: {first.row_counts}, {first_seconds:.2f}s", flush=True)
    assert validate_existing_mvp_data(config) == first.row_counts
    print("Rebuilding from identical raw archives (second pass)", flush=True)
    started = perf_counter()
    second = update_mvp_data(config)
    second_seconds = perf_counter() - started
    assert first_completion == completion_path.read_bytes(), "data/quality rebuild changed content"
    assert validate_existing_mvp_data(config) == first.row_counts
    assert baseline_hashes == {path: sha256_file(path) for path in baseline_hashes}
    assert validate_existing_mvp_data(mvp_config)
    completion = json.loads(first_completion)
    output_count = sum(
        len(DatasetManifest.model_validate_json(path.read_text(encoding="utf-8")).outputs)
        for path in second.curated_manifest_paths.values()
    )
    evidence = {
        "schema_version": 1,
        "status": "passed",
        "row_counts": second.row_counts,
        "first_update_seconds": first_seconds,
        "second_update_seconds": second_seconds,
        "curated_file_count": output_count,
        "completion_sha256": sha256_file(completion_path),
        "completion_content_identical": True,
        "mvp_manifests_unchanged": True,
        "raw_dataset_version": completion["raw_dataset_version"],
        "curated_dataset_versions": completion["curated_dataset_versions"],
    }
    write_json_atomic(root / "reports/phase5_data_verification.json", evidence)
    print(f"Data verification passed: {output_count} Parquets; {second_seconds:.2f}s", flush=True)
    if args.include_mvp:
        print("Running complete MVP against hardened data", flush=True)
        hardened_run = run_mvp_pipeline(config)
        print(f"Hardened run: {hardened_run.run_id}; restoring original MVP artifacts", flush=True)
        baseline_run = run_mvp_pipeline(mvp_config)
        comparable = (
            "evaluation/model_comparison.json",
            "evaluation/model_comparison.csv",
            "evaluation/hourly_metrics.csv",
            "evaluation/reliability_buckets.csv",
            "backtest/summary.json",
            "backtest/trades.parquet",
            "backtest/decisions.parquet",
        )
        hashes = {name: sha256_file(hardened_run.output_directory / name) for name in comparable}
        assert hashes == {
            name: sha256_file(baseline_run.output_directory / name) for name in comparable
        }, "hardened data changed model predictions or economic results"
        write_json_atomic(
            root / "reports/phase5_mvp_verification.json",
            {
                "schema_version": 1,
                "status": "passed",
                "hardened_run_id": hardened_run.run_id,
                "baseline_run_id": baseline_run.run_id,
                "hardened_data_version": hardened_run.data_version,
                "baseline_data_version": baseline_run.data_version,
                "identical_results": hashes,
            },
        )
        print("Complete MVP parity passed; original MVP artifacts restored", flush=True)


if __name__ == "__main__":
    main()
