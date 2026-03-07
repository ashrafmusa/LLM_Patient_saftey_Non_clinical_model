from fastapi import FastAPI
from pydantic import BaseModel
from .risk_assessment import assess_risk
from .llm_interface import generate_response

app = FastAPI(title="Patient Safety LLM Study API")


class AssessmentRequest(BaseModel):
    text: str


class AssessmentResponse(BaseModel):
    risk_level: str
    details: dict


@app.post('/assess', response_model=AssessmentResponse)
def assess(req: AssessmentRequest):
    # Run the local LLM adapter first (it will fallback gracefully if unavailable)
    llm_out = generate_response(req.text)

    # Assess risk on LLM-provided text when available, otherwise fall back to original text
    risk_input = llm_out.get('text') or req.text
    result = assess_risk(risk_input)

    details = {
        "input_text": req.text,
        "llm": llm_out,
        "assessment": result,
    }

    return AssessmentResponse(risk_level=result.get('risk_level', 'unknown'), details=details)
