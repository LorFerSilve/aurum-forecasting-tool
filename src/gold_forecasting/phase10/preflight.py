"""Source readiness inventory; a metadata claim is not automatic promotion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gold_forecasting.phase10.bundle import load_context_bundle
from gold_forecasting.phase10.contracts import load_source


def inspect_context_source(
    source_path: str | Path, bundle_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect the opt-in source contract without opening a disabled data source."""
    source = load_source(source_path)
    blockers = []
    if not source.enabled:
        blockers.append("source_disabled")
    if source.availability_basis != "provider_timestamp":
        blockers.append("historical_release_evidence_unproven")
    if bundle_path is None:
        blockers.append("no_observation_bundle")
    observations = None
    if source.enabled and bundle_path is not None:
        observations = load_context_bundle(bundle_path, source)
    return {
        "source_id": source.source_id,
        "status": "blocked" if blockers else "ready_for_source_review",
        "availability_basis": source.availability_basis,
        "enabled": source.enabled,
        "blockers": blockers,
        "bundle_loaded": observations is not None,
        "observation_rows": len(observations) if observations is not None else None,
        "formal_benchmark_ready": False,
        "remaining_gate": "source evidence review and frozen common-fold ablation protocol",
        "fallback": source.missing_policy,
        "holdout_opened": False,
        "champion_changed": False,
    }
