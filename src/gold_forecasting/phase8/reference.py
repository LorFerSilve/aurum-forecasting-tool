"""Read the frozen phase-7 reference artifacts needed by phase 8."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from gold_forecasting.artifacts import content_version, sha256_file
from gold_forecasting.config import _load_yaml_mapping


class Phase8ReferenceError(ValueError):
    """Raised when the frozen v0.2 reference cannot be proven intact."""


class ResearchChampion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: str = Field(min_length=1)
    family: str = Field(min_length=1)


class Phase7ChampionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    release: str
    protocol: str
    run_id: str
    code_version: str
    scope: str
    holdout_opened: bool
    trading_champion: None = None
    research_champions: dict[int, ResearchChampion]
    economic_promotion_candidates: int
    note: str


@dataclass(frozen=True, slots=True)
class Phase7Reference:
    root: Path
    champions: Phase7ChampionConfig
    summary: dict[str, Any]
    completion: dict[str, Any]

    def champion(self, horizon: int) -> ResearchChampion:
        try:
            return self.champions.research_champions[horizon]
        except KeyError as exc:
            raise Phase8ReferenceError(
                f"no frozen phase-7 champion for horizon {horizon}"
            ) from exc

    def aggregate_metrics(self, horizon: int) -> dict[str, Any]:
        champion = self.champion(horizon)
        path = (
            self.root
            / f"horizon_{horizon}"
            / champion.variant
            / "variant_summary.json"
        )
        payload = _verified_json(path, self.root, self.completion)
        models = payload.get("models")
        if not isinstance(models, dict):
            raise Phase8ReferenceError("phase-7 variant summary has no model mapping")
        metrics = models.get(champion.family)
        if not isinstance(metrics, dict):
            raise Phase8ReferenceError(
                f"phase-7 variant summary has no metrics for {champion.family}"
            )
        return cast(dict[str, Any], metrics)

    def test_digest(self, horizon: int, fold_name: str) -> str:
        path = (
            self.root
            / f"horizon_{horizon}"
            / "mvp"
            / fold_name
            / "fold_summary.json"
        )
        payload = _verified_json(path, self.root, self.completion)
        return str(payload["split"]["test_digest"])


def _completion_entry(
    root: Path,
    completion: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    files = completion.get("files")
    if not isinstance(files, list):
        raise Phase8ReferenceError("phase-7 completion manifest has no file list")
    for entry in files:
        if not isinstance(entry, dict):
            raise Phase8ReferenceError("phase-7 completion manifest has an invalid file entry")
        if entry.get("path") == relative:
            return cast(dict[str, Any], entry)
    raise Phase8ReferenceError(
        f"phase-7 completion manifest does not cover {relative}"
    )


def _verified_json(
    path: Path,
    root: Path,
    completion: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise Phase8ReferenceError(
            f"required phase-7 reference artifact is missing: {path}"
        )
    entry = _completion_entry(root, completion, path)
    if (
        entry.get("sha256") != sha256_file(path)
        or entry.get("size_bytes") != path.stat().st_size
    ):
        raise Phase8ReferenceError(
            f"phase-7 reference artifact digest mismatch: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase8ReferenceError(
            f"invalid JSON reference artifact: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise Phase8ReferenceError(
            f"reference artifact must contain a JSON object: {path}"
        )
    return payload


def load_phase7_reference(
    root: Path,
    reference_directory: str,
    champion_config: Path,
) -> Phase7Reference:
    reference = (root / reference_directory).resolve(strict=True)
    expected_root = (root / "reports" / "phase7_runs").resolve()
    if not reference.is_relative_to(expected_root):
        raise Phase8ReferenceError(
            "phase-7 reference escapes the expected run root"
        )
    champions = Phase7ChampionConfig.model_validate(
        _load_yaml_mapping(champion_config)
    )
    if champions.protocol != "phase7-v2" or champions.release != "v0.2.0":
        raise Phase8ReferenceError(
            "phase 8 requires the frozen phase7-v2 / v0.2.0 reference"
        )
    if champions.holdout_opened or champions.trading_champion is not None:
        raise Phase8ReferenceError(
            "phase-7 reference unexpectedly opened holdout or trading champion"
        )
    if reference.name != champions.run_id:
        raise Phase8ReferenceError(
            "phase-7 reference run_id differs from champion config"
        )
    try:
        run = json.loads(
            (reference / "run.json").read_text(encoding="utf-8")
        )
        completion = json.loads(
            (reference / "completion.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase8ReferenceError(
            "cannot read phase-7 run/completion manifests"
        ) from exc
    if (
        run.get("status") != "succeeded"
        or run.get("run_id") != champions.run_id
    ):
        raise Phase8ReferenceError(
            "phase-7 reference run is not the frozen successful run"
        )
    if completion.get("version") != content_version(
        {"files": completion.get("files", [])}
    ):
        raise Phase8ReferenceError(
            "phase-7 completion manifest version mismatch"
        )
    if run.get("code_version") != champions.code_version:
        raise Phase8ReferenceError(
            "phase-7 reference code identity differs from champion config"
        )
    summary = _verified_json(
        reference / "summary.json",
        reference,
        completion,
    )
    if (
        summary.get("protocol") != "phase7-v2"
        or summary.get("holdout_opened") is not False
    ):
        raise Phase8ReferenceError(
            "phase-7 summary violates the frozen reference contract"
        )
    return Phase7Reference(
        reference,
        champions,
        summary,
        completion,
    )


__all__ = [
    "Phase7ChampionConfig",
    "Phase7Reference",
    "Phase8ReferenceError",
    "ResearchChampion",
    "load_phase7_reference",
]
