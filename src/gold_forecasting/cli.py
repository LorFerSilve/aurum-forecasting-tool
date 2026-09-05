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

app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(dataset_app, name="dataset")
app.add_typer(mvp_app, name="mvp")

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
    typer.echo(
        f"Built {result.dataset_version}: {counts}, features={result.feature_count}"
    )
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
