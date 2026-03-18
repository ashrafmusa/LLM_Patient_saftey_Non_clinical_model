"""Structured audit logging for assessment calls."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Dict

from .config import settings


def write_assessment_audit(entry: Dict) -> Dict:
    log_dir = Path(settings.AUDIT_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "audit_id": str(uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **entry,
    }

    log_path = log_dir / "assess_log.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    return {
        "audit_id": payload["audit_id"],
        "audit_log_path": str(log_path),
    }
