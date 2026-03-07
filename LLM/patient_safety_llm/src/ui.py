import streamlit as st

from .deid import deidentify_text
from .llm_interface import generate_response
from .risk_assessment import assess_risk
from .explain import explain_text

st.set_page_config(page_title="Patient Safety LLM — Demo", layout="centered")

st.title("Patient Safety & Quality — LLM Assessment Demo")

st.markdown("Enter clinical text below to see de-identified input, the LLM response, and risk stratification.")

text = st.text_area("Clinical text", height=200)
extra_names = st.text_input("Extra names to redact (semicolon-separated)")

if st.button("Assess"):
    if not text.strip():
        st.warning("Please provide clinical text to assess.")
    else:
        names = [n.strip() for n in extra_names.split(";") if n.strip()]
        deid = deidentify_text(text, extra_names=names)
        st.subheader("De-identified Input")
        st.code(deid["text"], language="text")

        st.subheader("LLM Response")
        with st.spinner("Generating response..."):
            llm_out = generate_response(deid["text"])  # send de-id text to LLM adapter
        st.write(llm_out.get("text"))
        st.json(llm_out.get("meta", {}))

        st.subheader("Risk Assessment")
        risk_input = llm_out.get("text") or deid["text"]
        result = assess_risk(risk_input)
        st.metric(label="Risk Level", value=result.get("risk_level", "unknown"))
        st.json(result)

        st.subheader("Redaction Summary")
        st.json(deid["found"])

        st.subheader("Model Explanation")
        with st.spinner("Computing explanation..."):
            expl = explain_text(risk_input, top_k=10)

        if not expl.get("available"):
            st.info("No model-based explanation available: " + str(expl.get("reason", "")))
        else:
            exps = expl.get("explanations") or []
            if exps:
                st.table(exps)
                # show a simple bar chart of contributions
                try:
                    import pandas as _pd
                    df = _pd.DataFrame(exps)
                    df = df.set_index('feature')['contribution']
                    st.bar_chart(df)
                except Exception:
                    pass
            else:
                st.write(expl)