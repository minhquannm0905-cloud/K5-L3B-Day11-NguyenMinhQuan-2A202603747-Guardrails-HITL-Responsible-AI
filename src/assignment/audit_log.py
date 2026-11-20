"""
Assignment 11 — Audit Log.
Records interactions for review and exports JSON under outputs/.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/audit_log.json."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-independent audit logger."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(
        self,
        *,
        user_id: str,
        text: str,
        request_id: str | None = None,
    ):
        """Store an input and start time until its output is recorded."""
        key = request_id or f"{user_id}:{len(self._open)}"
        self._open[key] = {
            "request_id": key,
            "user_id": user_id,
            "input": text,
            "started_at": time.monotonic(),
            "timestamp": utc_now_iso(),
        }
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Append one complete, reviewable interaction to the audit log."""
        key = request_id or f"{user_id}:unknown"
        opened = self._open.pop(key, None)

        latency_ms = None
        if opened is not None:
            latency_ms = round((time.monotonic() - opened["started_at"]) * 1000, 2)

        entry = {
            "request_id": key,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": opened["input"] if opened else None,
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": latency_ms,
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs as a JSON array under outputs/ by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()