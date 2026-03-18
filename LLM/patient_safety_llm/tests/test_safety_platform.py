import json
from pathlib import Path

from fastapi.testclient import TestClient

from src import app as app_module
from src.audit_logging import write_assessment_audit
from src.config import settings
from src.retrieval import retrieve_relevant_context


def test_retrieve_relevant_context_returns_matching_source(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "iso.md").write_text(
        "ISO 15189 requires traceability, quality management, and corrective action for laboratory risks.",
        encoding="utf-8",
    )
    (docs_dir / "other.md").write_text(
        "Generic project planning document with unrelated content.",
        encoding="utf-8",
    )

    monkeypatch.setattr(settings, "RETRIEVAL_ENABLED", True)
    result = retrieve_relevant_context("laboratory quality traceability", docs_dir=docs_dir, max_chunks=2)

    assert result["retrieval_enabled"] is True
    assert result["sources"]
    assert result["sources"][0]["path"].endswith("iso.md")
    assert "traceability" in result["context"].lower()


def test_write_assessment_audit_writes_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "AUDIT_LOG_DIR", str(tmp_path / "audit_logs"))

    meta = write_assessment_audit({"input_text": "test", "guardrail_triggered": False})

    log_path = Path(meta["audit_log_path"])
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["input_text"] == "test"
    assert "audit_id" in payload


def test_assess_endpoint_returns_retrieval_and_audit(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "retrieve_relevant_context",
        lambda text: {"context": "Source: iso.md\nTraceability requirement.", "sources": [{"path": "iso.md", "score": 1.0}]},
    )
    monkeypatch.setattr(
        app_module,
        "classify_with_llm",
        lambda text, strategy='safety_focused': {
            "risk_level": "high",
            "raw_response": '{"risk_level":"high"}',
            "guardrail_triggered": True,
            "guardrail_reasons": ["high_harm_signal"],
        },
    )
    monkeypatch.setattr(
        app_module,
        "assess_risk",
        lambda text: {"risk_level": "medium", "score": 0.7, "model_based": True},
    )
    monkeypatch.setattr(
        app_module,
        "write_assessment_audit",
        lambda entry: {"audit_id": "audit-1", "audit_log_path": "reports/audit_logs/assess_log.jsonl"},
    )

    client = TestClient(app_module.app)
    response = client.post("/assess", json={"text": "Medication error with severe harm"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_level"] == "high"
    assert payload["details"]["retrieval"]["sources"]
    assert payload["details"]["audit"]["audit_id"] == "audit-1"
    assert payload["details"]["llm_assessment"]["guardrail_triggered"] is True
