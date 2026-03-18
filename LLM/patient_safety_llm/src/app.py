from fastapi import FastAPI
from pydantic import BaseModel
from .risk_assessment import assess_risk
from .llm_evaluation import classify_with_llm
from .retrieval import retrieve_relevant_context
from .audit_logging import write_assessment_audit

app = FastAPI(title="Patient Safety LLM Study API")


class AssessmentRequest(BaseModel):
    text: str


class AssessmentResponse(BaseModel):
    risk_level: str
    details: dict


@app.post('/assess', response_model=AssessmentResponse)
def assess(req: AssessmentRequest):
    retrieval = retrieve_relevant_context(req.text)
    llm_result = classify_with_llm(req.text, strategy='safety_focused')
    baseline_result = assess_risk(req.text)

    final_risk_level = llm_result.get('risk_level') or baseline_result.get('risk_level', 'unknown')

    audit_meta = write_assessment_audit(
        {
            'input_text': req.text,
            'retrieved_context': retrieval.get('context', ''),
            'retrieval_sources': retrieval.get('sources', []),
            'llm_raw_output': llm_result.get('raw_response', ''),
            'llm_assessment': llm_result,
            'baseline_assessment': baseline_result,
            'guardrail_triggered': llm_result.get('guardrail_triggered', False),
            'guardrail_reasons': llm_result.get('guardrail_reasons', []),
        }
    )

    details = {
        "input_text": req.text,
        "retrieval": retrieval,
        "llm_assessment": llm_result,
        "baseline_assessment": baseline_result,
        "audit": audit_meta,
    }

    return AssessmentResponse(risk_level=final_risk_level, details=details)
