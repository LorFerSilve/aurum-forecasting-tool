"""Source readiness inventory; metadata assumptions never imply promotion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gold_forecasting.phase10.bundle import load_context_data
from gold_forecasting.phase10.contracts import load_source


def inspect_context_source(
    source_path: str | Path,
    bundle_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect strict-PIT and exploratory readiness without crossing either gate."""
    source = load_source(source_path)
    operational_blockers: list[str] = []
    formal_blockers: list[str] = []
    if not source.enabled:
        operational_blockers.append("source_disabled")
    if bundle_path is None:
        operational_blockers.append("no_observation_bundle")
    if source.availability_basis != "provider_timestamp":
        formal_blockers.append("historical_release_evidence_unproven")

    observations = None
    if source.enabled and bundle_path is not None:
        observations = load_context_data(bundle_path, source)

    if operational_blockers:
        status = "blocked"
    elif formal_blockers:
        status = "ready_for_exploratory_ablation"
    else:
        status = "ready_for_source_review"

    observation_span = None
    if observations is not None and not observations.empty:
        observation_span = {
            "start_utc": observations["observed_at_utc"].min().isoformat(),
            "end_utc": observations["observed_at_utc"].max().isoformat(),
        }

    return {
        "source_id": source.source_id,
        "status": status,
        "availability_basis": source.availability_basis,
        "enabled": source.enabled,
        "blockers": [*operational_blockers, *formal_blockers],
        "operational_blockers": operational_blockers,
        "formal_blockers": formal_blockers,
        "bundle_loaded": observations is not None,
        "observation_rows": len(observations) if observations is not None else None,
        "observation_span": observation_span,
        "exploratory_ablation_ready": (
            observations is not None
            and not operational_blockers
            and source.availability_basis in {"modeled_latency", "provider_timestamp"}
        ),
        "strict_pit_source_ready": (
            observations is not None
            and not operational_blockers
            and not formal_blockers
        ),
        "formal_benchmark_ready": False,
        "remaining_gate": (
            "freeze common-fold ablation protocol"
            if observations is not None and not formal_blockers
            else (
                "exploratory modeled-latency ablation only; obtain provider release evidence "
                "before any strict point-in-time promotion"
                if observations is not None and formal_blockers
                else "source evidence review and observation bundle"
            )
        ),
        "fallback": source.missing_policy,
        "holdout_opened": False,
        "champion_changed": False,
    }


__all__ = ["inspect_context_source"]
