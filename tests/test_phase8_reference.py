"""Reference-artifact contracts for the frozen phase-7 benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gold_forecasting.artifacts import (
    content_version,
    file_digest,
)
from gold_forecasting.phase8.reference import (
    Phase7ChampionConfig,
    Phase7Reference,
    Phase8ReferenceError,
)


def _reference_with_digest(
    tmp_path: Path,
    digest: object,
) -> Phase7Reference:
    root = tmp_path / "phase7-run"
    fold = root / "horizon_3" / "mvp" / "test_2022"
    fold.mkdir(parents=True)
    summary = fold / "fold_summary.json"
    summary.write_text(
        json.dumps(
            {
                "split": {
                    "test_digest": digest,
                }
            }
        ),
        encoding="utf-8",
    )
    files = [
        file_digest(
            summary,
            relative_to=root,
        ).model_dump(mode="json")
    ]
    completion = {
        "files": files,
        "version": content_version(
            {"files": files}
        ),
    }
    champions = Phase7ChampionConfig.model_validate(
        {
            "schema_version": 1,
            "release": "v0.2.0",
            "protocol": "phase7-v2",
            "run_id": "phase7-run",
            "code_version": "abc123",
            "scope": "research_fallback_only",
            "holdout_opened": False,
            "trading_champion": None,
            "research_champions": {
                3: {
                    "variant": "mvp",
                    "family": "logistic",
                }
            },
            "economic_promotion_candidates": 0,
            "note": "test",
        }
    )
    return Phase7Reference(
        root=root,
        champions=champions,
        summary={},
        completion=completion,
    )


def test_phase7_test_digest_accepts_repository_sample_digest_contract(
    tmp_path: Path,
) -> None:
    digest = "sha256:" + "a" * 64
    reference = _reference_with_digest(
        tmp_path,
        digest,
    )

    assert reference.test_digest(
        3,
        "test_2022",
    ) == digest


@pytest.mark.parametrize(
    "digest",
    [
        "a" * 64,
        "sha256:" + "a" * 63,
        "sha256:" + "g" * 64,
        "",
        None,
    ],
)
def test_phase7_test_digest_rejects_invalid_formats(
    tmp_path: Path,
    digest: object,
) -> None:
    reference = _reference_with_digest(
        tmp_path,
        digest,
    )

    with pytest.raises(
        Phase8ReferenceError,
        match="test digest",
    ):
        reference.test_digest(
            3,
            "test_2022",
        )
