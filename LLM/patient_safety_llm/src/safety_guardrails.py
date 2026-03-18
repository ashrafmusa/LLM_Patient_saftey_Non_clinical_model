"""Deterministic patient-safety guardrails for LLM outputs.

These rules act as a safety net around model predictions. They are intentionally
conservative: if a scenario contains strong harm signals, the system escalates
risk and urgency even if the model under-calls the case.
"""

from __future__ import annotations

import re
from typing import Dict, List

HIGH_RISK_TERMS = [
    "unresponsive",
    "anaphylaxis",
    "cardiac arrest",
    "respiratory distress",
    "sepsis",
    "hemorrhage",
    "suicide",
    "self-harm",
    "overdose",
    "wrong patient",
    "wrong-site",
    "tenfold",
]

MEDICATION_TERMS = [
    "medication error",
    "dose",
    "dosage",
    "warfarin",
    "insulin",
    "heparin",
    "opioid",
    "allergy",
    "allergic reaction",
]

PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"disregard .* instructions",
    r"system prompt",
    r"developer message",
    r"reveal .* prompt",
    r"bypass safety",
]


def detect_input_risks(text: str) -> Dict:
    lowered = (text or "").lower()
    matched = [pattern for pattern in PROMPT_INJECTION_PATTERNS if re.search(pattern, lowered)]
    return {
        "input_blocked": bool(matched),
        "input_risk_reasons": matched,
    }



def _extract_dose_values(text: str) -> List[float]:
    values: List[float] = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*mg", text.lower()):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return values



def _detect_dose_mismatch(text: str) -> bool:
    doses = _extract_dose_values(text)
    if len(doses) < 2:
        return False
    smallest = min(doses)
    largest = max(doses)
    if smallest <= 0:
        return False
    return (largest / smallest) >= 5.0



def apply_output_guardrails(scenario_text: str, result: Dict) -> Dict:
    lowered = (scenario_text or "").lower()
    reasons: List[str] = []
    updated = dict(result)

    if any(term in lowered for term in HIGH_RISK_TERMS):
        reasons.append("high_harm_signal")

    if _detect_dose_mismatch(lowered) and any(term in lowered for term in MEDICATION_TERMS):
        reasons.append("dose_mismatch")

    if "allergy" in lowered and any(term in lowered for term in ["penicillin", "reaction", "anaphylaxis"]):
        reasons.append("allergy_risk")

    if reasons:
        updated["risk_level"] = "high"
        updated["action_urgency"] = "emergent"
        updated["needs_escalation"] = True
        updated["guardrail_triggered"] = True
        updated["guardrail_reasons"] = reasons
        updated["guardrail_override"] = "forced_high_risk"
        if not updated.get("event_type") or updated.get("event_type") == "other":
            if "dose_mismatch" in reasons:
                updated["event_type"] = "medication"
            elif "allergy_risk" in reasons:
                updated["event_type"] = "allergy"
            else:
                updated["event_type"] = "deterioration"
        return updated

    updated.setdefault("guardrail_triggered", False)
    updated.setdefault("guardrail_reasons", [])
    updated.setdefault("guardrail_override", None)
    return updated
