"""Small, durable local registry for research runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from gold_forecasting.runtime import format_utc, utc_now

_MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_NAME = "run.json"
_CONFIG_SNAPSHOT_NAME = "config.yaml"
_RUNNING_STATUS = "running"
_STATUS_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Immutable view of one registered run."""

    run_id: str
    status: str
    started_at: str
    finished_at: str | None
    output_directory: Path
    manifest_path: Path
    config_path: Path
    config_snapshot_path: Path
    data_version: str
    code_version: str
    dry_run: bool


def get_git_code_version(root: Path | str | None = None) -> str:
    """Return the current Git revision, including dirty state, when available.

    A missing Git executable, a non-repository path, or an unborn repository is
    represented explicitly instead of preventing a research run from starting.
    """

    working_directory = Path.cwd() if root is None else Path(root)
    command = ["git", "-C", str(working_directory), "rev-parse", "HEAD"]
    try:
        revision_result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"

    revision = revision_result.stdout.strip()
    if revision_result.returncode != 0 or not revision:
        return "uncommitted"

    try:
        status_result = subprocess.run(
            ["git", "-C", str(working_directory), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return revision

    if status_result.returncode == 0 and status_result.stdout:
        return f"{revision}+dirty"
    return revision


class RunRegistry:
    """Create and finish append-only run directories with atomic manifests."""

    def __init__(
        self,
        output_root: Path | str,
        config_path: Path | str,
        data_version: str,
        *,
        code_root: Path | str | None = None,
    ) -> None:
        if not data_version.strip():
            raise ValueError("data_version must not be empty")

        self.output_root = Path(output_root).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.data_version = data_version
        self.code_root = (
            self.config_path.parent if code_root is None else Path(code_root).expanduser().resolve()
        )
        self._lock = threading.Lock()

    def start_run(self, *, dry_run: bool = False) -> RunRecord:
        """Create a new run directory, config snapshot, and running manifest."""

        if not self.config_path.is_file():
            raise FileNotFoundError(f"config file does not exist: {self.config_path}")

        self.output_root.mkdir(parents=True, exist_ok=True)
        output_directory, run_id = self._reserve_output_directory()
        manifest_path = output_directory / _MANIFEST_NAME
        config_snapshot_path = output_directory / _CONFIG_SNAPSHOT_NAME

        try:
            config_bytes = self.config_path.read_bytes()
            _atomic_write_bytes(config_snapshot_path, config_bytes)
            config_hash = hashlib.sha256(config_bytes).hexdigest()
            started_at = format_utc(utc_now())
            code_version = get_git_code_version(self.code_root)
            manifest: dict[str, object] = {
                "schema_version": _MANIFEST_SCHEMA_VERSION,
                "run_id": run_id,
                "status": _RUNNING_STATUS,
                "dry_run": dry_run,
                "started_at": started_at,
                "finished_at": None,
                "config_path": str(self.config_path),
                "config_snapshot_path": str(config_snapshot_path),
                "config_sha256": config_hash,
                "code_version": code_version,
                "data_version": self.data_version,
                "output_directory": str(output_directory),
                "error": None,
                "metadata": {},
            }
            _atomic_write_json(manifest_path, manifest)
        except Exception:
            # Do not recursively delete an incompletely created run.  Its unique
            # directory is intentionally retained as audit evidence, while the
            # absence of run.json makes clear that registration never completed.
            raise

        return RunRecord(
            run_id=run_id,
            status=_RUNNING_STATUS,
            started_at=started_at,
            finished_at=None,
            output_directory=output_directory,
            manifest_path=manifest_path,
            config_path=self.config_path,
            config_snapshot_path=config_snapshot_path,
            data_version=self.data_version,
            code_version=code_version,
            dry_run=dry_run,
        )

    def finish_run(
        self,
        record: RunRecord,
        *,
        status: str = "succeeded",
        error: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> RunRecord:
        """Atomically transition a running manifest to a terminal status."""

        if status == _RUNNING_STATUS or _STATUS_PATTERN.fullmatch(status) is None:
            raise ValueError(
                "terminal status must start with a lowercase letter and contain only "
                "lowercase letters, digits, underscores, or hyphens"
            )
        self._validate_record_location(record)

        with self._lock:
            manifest = _read_manifest(record.manifest_path)
            if manifest.get("run_id") != record.run_id:
                raise ValueError("manifest run_id does not match the RunRecord")
            current_status = manifest.get("status")
            if current_status != _RUNNING_STATUS:
                raise RuntimeError(f"run {record.run_id} is already {current_status!r}")

            finished_at = format_utc(utc_now())
            manifest["status"] = status
            manifest["finished_at"] = finished_at
            manifest["error"] = error
            if metadata is not None:
                manifest["metadata"] = dict(metadata)
            _atomic_write_json(record.manifest_path, manifest)

        return replace(record, status=status, finished_at=finished_at)

    def _reserve_output_directory(self) -> tuple[Path, str]:
        for _ in range(100):
            run_id = _new_run_id()
            output_directory = self.output_root / run_id
            try:
                output_directory.mkdir()
            except FileExistsError:
                continue
            return output_directory, run_id
        raise RuntimeError("could not allocate a unique run ID")

    def _validate_record_location(self, record: RunRecord) -> None:
        output_directory = record.output_directory.resolve()
        if output_directory.parent != self.output_root:
            raise ValueError("RunRecord does not belong to this registry")
        if output_directory.name != record.run_id:
            raise ValueError("RunRecord output directory does not match its run_id")
        if record.manifest_path.resolve() != output_directory / _MANIFEST_NAME:
            raise ValueError("RunRecord manifest path is invalid")


def _new_run_id() -> str:
    timestamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    _atomic_write_bytes(path, f"{serialized}\n".encode())


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"run manifest is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"run manifest must contain a JSON object: {path}")
    return payload
