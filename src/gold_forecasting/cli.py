"""Command-line entrypoint for the research pipeline."""

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Leakage-aware XAU/USD forecasting research pipeline.")
config_app = typer.Typer(help="Validate versioned project configuration.")
data_app = typer.Typer(help="Import and validate market data.")
dataset_app = typer.Typer(help="Build leakage-safe model datasets.")
mvp_app = typer.Typer(help="Run the vertical research MVP.")
benchmark_app = typer.Typer(help="Run the guarded phase-6 multi-horizon benchmark.")
phase7_app = typer.Typer(help="Run phase-7 richer price-only feature ablations.")
phase8_app = typer.Typer(help="Run the guarded phase-8 multi-timeframe neural challenger.")
phase9_app = typer.Typer(help="Develop and verify the phase-9 future-path challenger.")
phase10_app = typer.Typer(help="Build and inspect point-in-time external context research.")

app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(dataset_app, name="dataset")
app.add_typer(mvp_app, name="mvp")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(phase7_app, name="phase7")
app.add_typer(phase8_app, name="phase8")
app.add_typer(phase9_app, name="phase9")
app.add_typer(phase10_app, name="phase10")

ConfigOption = Annotated[
    Path,
    typer.Option(exists=True, dir_okay=False),
]
DryRunOption = Annotated[
    bool,
    typer.Option(help="Validate and record the empty phase-1 flow."),
]
YearsOption = Annotated[
    str | None,
    typer.Option(help="Optional comma-separated development years, for example 2023,2024."),
]


@phase10_app.command("silver-import")
def import_phase10_silver_context(
    archive_directory: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Directory containing annual HISTDATA_COM_ASCII_XAGUSD_M1_<year>.zip files.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New destination for authenticated annual silver context bundles."),
    ] = Path("data/context/phase10/silver"),
    source: ConfigOption = Path("configs/phase10_silver_exploratory.yaml"),
    years: YearsOption = None,
) -> None:
    """Convert local XAGUSD archives into modeled-latency Phase-10 context bundles."""
    _import_phase10_context(archive_directory, output, source, years, symbol="XAGUSD")


@phase10_app.command("dollar-import")
def import_phase10_dollar_context(
    archive_directory: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Directory containing annual HISTDATA_COM_ASCII_EURUSD_M1_<year>.zip files.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(help="New destination for authenticated annual dollar context bundles."),
    ] = Path("data/context/phase10/dollar"),
    source: ConfigOption = Path("configs/phase10_dollar_exploratory.yaml"),
    years: YearsOption = None,
) -> None:
    """Import local EURUSD bid closes for the exploratory inverse-EURUSD dollar proxy."""
    _import_phase10_context(archive_directory, output, source, years, symbol="EURUSD")


def _import_phase10_context(
    archive_directory: Path,
    output: Path,
    source: Path,
    years: str | None,
    *,
    symbol: str,
) -> None:
    from gold_forecasting.phase10.contracts import load_source
    from gold_forecasting.phase10.histdata_context import import_histdata_context_archives

    selected_years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024)
    if years is not None:
        try:
            selected_years = tuple(int(item.strip()) for item in years.split(",") if item.strip())
        except ValueError as exc:
            raise typer.BadParameter("years must be comma-separated integers") from exc
    result = import_histdata_context_archives(
        archive_directory,
        output,
        load_source(source),
        symbol=symbol,
        years=selected_years,
    )
    typer.echo(json.dumps(result, indent=2))


@phase10_app.command("dry-run")
def dry_run_phase10_context(
    output: Annotated[
        Path, typer.Option(help="New destination for a synthetic context integration run.")
    ] = Path("reports/phase10_dry_run"),
) -> None:
    """Exercise ingestion, context joins, features and an isolated synthetic ablation."""
    from gold_forecasting.phase10.dry_run import run_context_dry_run

    typer.echo(json.dumps(run_context_dry_run(output), indent=2))


@phase10_app.command("validate")
def validate_phase10_context(
    run_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Verify synthetic or exploratory run integrity and its declared research scope."""
    from gold_forecasting.phase10.artifacts import verify_phase10_run
    from gold_forecasting.phase10.dry_run import verify_context_dry_run

    if (run_directory / "run.json").exists():
        result = verify_phase10_run(run_directory)
    else:
        result = verify_context_dry_run(run_directory)
    typer.echo(json.dumps(result, indent=2))


@phase10_app.command("preflight")
def preflight_phase10_context(
    source: ConfigOption = Path("configs/phase10_silver.yaml"),
    bundle: Annotated[
        Path | None, typer.Option(help="Optional observation bundle manifest.")
    ] = None,
    config: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="Real-data ablation config.")
    ] = None,
    report: Annotated[Path | None, typer.Option(help="Optional real-data JSON report.")] = None,
) -> None:
    """Report source-readiness blockers; optional disabled sources are never loaded."""
    from gold_forecasting.phase10.preflight import inspect_context_source

    if config is not None:
        from gold_forecasting.phase10.config import load_phase10_config
        from gold_forecasting.phase10.real_preflight import run_phase10_preflight

        if bundle is not None or source != Path("configs/phase10_silver.yaml"):
            raise typer.BadParameter(
                "--config uses its own source/bundle; do not mix source options"
            )
        protocol: str | None = None
        try:
            protocol = load_phase10_config(config).protocol_version
            result = run_phase10_preflight(config, report_path=report)
        except (OSError, ValueError) as exc:
            typer.echo(
                json.dumps(
                    {
                        "protocol": protocol,
                        "status": "blocked",
                        "exploratory_ablation_ready": False,
                        "holdout_opened": False,
                        "blockers": [str(exc)],
                    },
                    indent=2,
                )
            )
            raise typer.Exit(code=1) from exc
    else:
        if report is not None:
            raise typer.BadParameter("--report requires a real-data --config")
        result = inspect_context_source(source, bundle)
    typer.echo(json.dumps(result, indent=2))
    if result["status"] == "blocked":
        raise typer.Exit(code=1)


@phase10_app.command("run")
def run_phase10_silver(
    config: ConfigOption = Path("configs/phase10_silver_ablation.yaml"),
) -> None:
    """Run the configured exploratory source ablation after all local integrity gates pass."""
    from gold_forecasting.phase10.pipeline import run_phase10

    typer.echo(str(run_phase10(config)))


@phase9_app.command("dry-run")
def dry_run_phase9_research(
    output: Annotated[
        Path,
        typer.Option(help="New destination for the isolated synthetic phase-9 integration run."),
    ] = Path("reports/phase9_dry_run"),
) -> None:
    """Exercise phase-9 end to end without opening a formal benchmark."""

    from gold_forecasting.phase9.dry_run import run_phase9_synthetic_dry_run

    result = run_phase9_synthetic_dry_run(output)
    typer.echo(json.dumps(result, indent=2))


@phase9_app.command("validate")
def validate_phase9_research(
    run_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Verify a persisted phase-9 dry-run or future formal run."""

    from gold_forecasting.phase9.artifacts import verify_phase9

    typer.echo(json.dumps(verify_phase9(run_directory), indent=2))


@phase9_app.command("preflight")
def preflight_phase9_research(
    config: ConfigOption = Path("configs/phase9.yaml"),
    report: Annotated[
        Path,
        typer.Option(help="JSON evidence report under reports/; no formal benchmark is started."),
    ] = Path("reports/phase9_preflight.json"),
) -> None:
    """Run guarded real-data checks without opening the formal Phase-9 benchmark."""

    from gold_forecasting.phase9.preflight import run_phase9_preflight

    result = run_phase9_preflight(
        config,
        report_path=report,
    )
    typer.echo(
        f"Phase 9 preflight {result['status']}: "
        f"rows={result['dataset']['common_eligible']}, report={report}"
    )


@phase9_app.command("run")
def run_phase9_research(
    config: ConfigOption = Path("configs/phase9.yaml"),
) -> None:
    """Run the canonical Phase-9 benchmark behind its embedded real-data preflight."""

    from gold_forecasting.phase9.pipeline import run_phase9

    output = run_phase9(config)
    typer.echo(f"Completed phase 9: {output}")


@phase8_app.command("run")
def run_phase8_research(config: ConfigOption = Path("configs/phase8.yaml")) -> None:
    """Run the compact neural challenger against the frozen phase-7 reference."""
    from gold_forecasting.phase8.pipeline import run_phase8

    output = run_phase8(config)
    typer.echo(f"Completed phase 8: {output}")


@phase8_app.command("validate")
def validate_phase8_research(
    run_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Verify a completed phase-8 run and all persisted artifact hashes."""
    from gold_forecasting.phase8.pipeline import verify_phase8

    typer.echo(json.dumps(verify_phase8(run_directory), indent=2))


@phase7_app.command("run")
def run_phase7_research(config: ConfigOption = Path("configs/phase7.yaml")) -> None:
    """Run guarded richer-feature ablations on the phase-6 evaluation contract."""
    from gold_forecasting.phase7.pipeline import run_phase7

    output = run_phase7(config)
    typer.echo(f"Completed phase 7: {output}")


@phase7_app.command("validate")
def validate_phase7_research(
    run_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Verify a completed phase-7 run and all persisted artifact hashes."""
    from gold_forecasting.phase7.pipeline import verify_phase7

    typer.echo(json.dumps(verify_phase7(run_directory), indent=2))


@benchmark_app.command("run")
def run_phase6_benchmark(config: ConfigOption = Path("configs/benchmark.yaml")) -> None:
    """Tune only on inner history, then evaluate frozen outer policies."""
    from gold_forecasting.benchmark.pipeline import run_benchmark

    output = run_benchmark(config)
    typer.echo(f"Completed benchmark: {output}")


@benchmark_app.command("validate")
def validate_phase6_benchmark(
    run_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Verify completion status and every persisted artifact hash."""
    from gold_forecasting.benchmark.pipeline import verify_benchmark

    typer.echo(json.dumps(verify_benchmark(run_directory), indent=2))


@benchmark_app.command("reproduce")
def reproduce_phase6_benchmark(
    run_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    report: Annotated[
        Path | None,
        typer.Option(help="New JSON report outside the original run directory.", dir_okay=False),
    ] = None,
) -> None:
    """Refit frozen final candidates and verify prediction and inner-policy parity."""
    from gold_forecasting.artifacts import write_json_atomic
    from gold_forecasting.benchmark.reproduce import reproduce_benchmark

    original = run_directory.resolve(strict=True)
    destination = (
        report
        if report is not None
        else original.parent.parent / f"phase6_reproduction_{original.name}.json"
    ).resolve()
    if destination.is_relative_to(original):
        raise typer.BadParameter(
            "report must be outside the original run directory", param_hint="report"
        )
    if destination.exists():
        raise typer.BadParameter("report already exists; choose a new path", param_hint="report")
    try:
        result = reproduce_benchmark(original)
        # Refitting can take minutes; recheck before writing if another run created the report.
        if destination.exists():
            raise FileExistsError(f"report appeared during reproduction: {destination}")
        write_json_atomic(destination, result)
    except (ValueError, OSError) as exc:
        typer.echo(f"Reproduction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Reproduced {result['run_id']}: "
        f"final fits={result['final_fit_parity_count']}, "
        f"inner policies={result['inner_policy_parity_count']}, "
        f"verified files={result['verified_files']}"
    )
    typer.echo(f"Reproduction report: {destination}")


@config_app.command("validate")
def validate_config(
    config: ConfigOption = Path("configs/mvp.yaml"),
) -> None:
    """Validate the root config and all referenced component configs."""
    from gold_forecasting.config import load_project_config

    loaded = load_project_config(config)
    typer.echo(
        f"Valid configuration: {loaded.instrument.instrument.id}, "
        f"protocol {loaded.root.protocol_version}"
    )


@data_app.command("import")
def import_data(
    config: ConfigOption = Path("configs/mvp.yaml"),
    years: YearsOption = None,
) -> None:
    """Download, curate, resample, and manifest the public MVP data."""
    from gold_forecasting.data_pipeline import build_mvp_data

    selected_years = None
    if years is not None:
        try:
            selected_years = tuple(int(item.strip()) for item in years.split(",") if item.strip())
        except ValueError as exc:
            raise typer.BadParameter("years must be comma-separated integers") from exc
    result = build_mvp_data(config, years=selected_years)
    counts = ", ".join(f"{key}={value}" for key, value in result.row_counts.items())
    typer.echo(f"Built {result.dataset_version}: {counts}")
    typer.echo(f"Coverage report: {result.coverage_report_path}")
    if result.quality_report_path is not None:
        typer.echo(f"Daily quality report: {result.quality_report_path}")


@data_app.command("update")
def update_data(
    config: ConfigOption = Path("configs/phase5.yaml"),
) -> None:
    """Resume immutable acquisitions and rebuild the configured development history."""
    from gold_forecasting.data_pipeline import update_mvp_data

    result = update_mvp_data(config)
    summary = ", ".join(f"{key}={value}" for key, value in result.row_counts.items())
    typer.echo(f"Updated {result.dataset_version}: {summary}")
    typer.echo(f"Coverage report: {result.coverage_report_path}")


@data_app.command("validate")
def validate_data(
    config: ConfigOption = Path("configs/mvp.yaml"),
) -> None:
    """Validate curated files, hashes, schemas, and candle invariants."""
    from gold_forecasting.data_pipeline import validate_existing_mvp_data

    counts = validate_existing_mvp_data(config)
    summary = ", ".join(f"{key}={value}" for key, value in counts.items())
    typer.echo(f"Valid curated data: {summary}")


@dataset_app.command("build")
def build_dataset(
    config: ConfigOption = Path("configs/mvp.yaml"),
) -> None:
    """Build features, labels, splits, and the train-only preprocessor."""
    from gold_forecasting.datasets import build_mvp_dataset

    result = build_mvp_dataset(config)
    counts = ", ".join(f"{key}={value}" for key, value in result.row_counts.items())
    typer.echo(f"Built {result.dataset_version}: {counts}, features={result.feature_count}")
    typer.echo(f"Model table: {result.model_table_path}")


@dataset_app.command("validate")
def validate_dataset(
    config: ConfigOption = Path("configs/mvp.yaml"),
) -> None:
    """Validate the versioned model table and train-only preprocessor."""
    from gold_forecasting.datasets import validate_mvp_dataset

    counts = validate_mvp_dataset(config)
    summary = ", ".join(f"{key}={value}" for key, value in counts.items())
    typer.echo(f"Valid MVP model table: {summary}")


@app.command("train")
def train(
    config: ConfigOption = Path("configs/mvp.yaml"),
) -> None:
    """Select and persist the train-only MVP logistic model."""
    from gold_forecasting.training import train_mvp_model

    result = train_mvp_model(config)
    typer.echo(
        f"Trained {result.bundle.model_version}: "
        f"train={result.train_rows}, validation={result.validation_rows}"
    )


@app.command("evaluate")
def evaluate(
    config: ConfigOption = Path("configs/mvp.yaml"),
) -> None:
    """Run the complete MVP and write the comparable evaluation report."""
    from gold_forecasting.pipeline import run_mvp_pipeline

    result = run_mvp_pipeline(config)
    typer.echo(f"Evaluation: {result.output_directory / 'evaluation' / 'model_comparison.md'}")


@app.command("backtest")
def backtest(
    config: ConfigOption = Path("configs/mvp.yaml"),
) -> None:
    """Run the complete MVP and write the costs-aware backtest report."""
    from gold_forecasting.pipeline import run_mvp_pipeline

    result = run_mvp_pipeline(config)
    typer.echo(f"Backtest: {result.output_directory / 'backtest' / 'summary.md'}")


@app.command("predict")
def predict(
    config: ConfigOption = Path("configs/mvp.yaml"),
) -> None:
    """Generate one held-out example prediction from the latest saved model."""
    from gold_forecasting.config import load_project_config
    from gold_forecasting.datasets import load_mvp_model_table
    from gold_forecasting.inference import PredictionRecord
    from gold_forecasting.training import load_latest_mvp_model

    loaded = load_project_config(config)
    model = load_latest_mvp_model(config)
    table, _ = load_mvp_model_table(config)
    test = table.loc[table["split"].eq("test")]
    if test.empty:
        raise typer.BadParameter("model table contains no test rows")
    row = test.iloc[[-1]]
    batch = model.predict(row)
    probabilities = batch.probabilities[0]
    prediction = PredictionRecord(
        prediction_time_utc=row.iloc[0]["prediction_time_utc"],
        instrument=loaded.instrument.instrument.id,
        horizon_minutes=loaded.labels.prediction.primary_horizon_minutes,
        predicted_class=batch.predicted_class[0],
        p_down=float(probabilities[0]),
        p_neutral=float(probabilities[1]),
        p_up=float(probabilities[2]),
        model_version=model.model_version,
        data_version=model.data_version,
        calibration_status="preliminary",
    )
    typer.echo(json.dumps(prediction.model_dump(mode="json"), indent=2))


@mvp_app.command("run")
def run_mvp(
    config: ConfigOption = Path("configs/mvp.yaml"),
    dry_run: DryRunOption = False,
) -> None:
    """Run the complete MVP, or its phase-1 dry-run skeleton."""
    from gold_forecasting.pipeline import run_mvp_pipeline

    result = run_mvp_pipeline(config_path=config, dry_run=dry_run)
    typer.echo(f"Run {result.run_id}: {result.status} ({result.output_directory})")
