"""Structural guards for the phase-8 benchmark preflight."""

from __future__ import annotations

import numpy as np
import pytest

from gold_forecasting.phase8.pipeline import (
    _locked_runtime_versions,
    _validate_locked_runtime_dependencies,
    _validate_phase7_baseline_metrics,
)


def _valid_baseline() -> dict[str, object]:
    return {
        "macro_f1": {"mean": 0.4, "worst": 0.3},
        "brier": {"mean": 0.6},
        "log_loss": {"mean": 1.0},
        "return_mae_bps": {"mean": 12.0},
        "net_bps": {"worst": 0.0},
    }


def test_phase7_baseline_metric_contract_accepts_finite_reference() -> None:
    _validate_phase7_baseline_metrics(
        _valid_baseline(),
        horizon=15,
    )


@pytest.mark.parametrize(
    ("metric", "field", "value"),
    [
        ("macro_f1", "mean", np.nan),
        ("macro_f1", "worst", np.inf),
        ("brier", "mean", None),
        ("log_loss", "mean", "1.0"),
        ("return_mae_bps", "mean", True),
        ("net_bps", "worst", np.nan),
    ],
)
def test_phase7_baseline_metric_contract_rejects_invalid_values(
    metric: str,
    field: str,
    value: object,
) -> None:
    baseline = _valid_baseline()
    section = dict(baseline[metric])  # type: ignore[arg-type]
    section[field] = value
    baseline[metric] = section

    with pytest.raises(ValueError, match=metric):
        _validate_phase7_baseline_metrics(
            baseline,
            horizon=30,
        )


def test_phase7_baseline_metric_contract_rejects_missing_section() -> None:
    baseline = _valid_baseline()
    del baseline["brier"]

    with pytest.raises(ValueError, match="brier"):
        _validate_phase7_baseline_metrics(
            baseline,
            horizon=60,
        )


def _write_runtime_lock(tmp_path) -> dict[str, str]:
    expected = {
        "numpy": "2.4.6",
        "pandas": "2.3.3",
        "scipy": "1.17.1",
        "scikit-learn": "1.9.0",
        "torch": "2.11.0",
        "pyarrow": "23.0.1",
    }
    (tmp_path / "requirements.lock").write_text(
        "\n".join(
            f"{name}=={version}"
            for name, version in expected.items()
        )
        + "\n",
        encoding="utf-8",
    )
    return expected


def test_locked_runtime_versions_reads_required_pins(tmp_path) -> None:
    expected = _write_runtime_lock(tmp_path)

    assert _locked_runtime_versions(tmp_path) == expected


def test_locked_runtime_dependencies_rejects_stale_venv(
    tmp_path,
    monkeypatch,
) -> None:
    expected = _write_runtime_lock(tmp_path)

    def installed(name: str) -> str:
        if name == "torch":
            return "2.10.0+cu128"
        return expected[name]

    monkeypatch.setattr(
        "gold_forecasting.phase8.pipeline.importlib.metadata.version",
        installed,
    )

    with pytest.raises(RuntimeError, match=r"torch: installed 2\.10\.0"):
        _validate_locked_runtime_dependencies(tmp_path)


def test_locked_runtime_dependencies_accepts_local_cuda_build_tag(
    tmp_path,
    monkeypatch,
) -> None:
    expected = _write_runtime_lock(tmp_path)

    def installed(name: str) -> str:
        if name == "torch":
            return "2.11.0+cu128"
        return expected[name]

    monkeypatch.setattr(
        "gold_forecasting.phase8.pipeline.importlib.metadata.version",
        installed,
    )

    actual = _validate_locked_runtime_dependencies(tmp_path)
    assert actual["torch"] == "2.11.0+cu128"
