"""Streamlit operator UI for the custom patient safety LLM API.

Run:
    streamlit run src/ui_custom_llm.py
"""

from __future__ import annotations

import os
from typing import Any, Dict

import streamlit as st
import httpx


DEFAULT_API_URL = os.getenv("CUSTOM_LLM_API_URL", "http://127.0.0.1:8010")


@st.cache_data(show_spinner=False)
def check_health(api_url: str) -> Dict[str, Any]:
    try:
        r = httpx.get(f"{api_url.rstrip('/')}/health", timeout=10)
        return {"ok": r.status_code == 200, "status_code": r.status_code, "body": r.json()}
    except Exception as exc:
        return {"ok": False, "status_code": None, "body": {"error": str(exc)}}


def run_predict(api_url: str, text: str, max_new_tokens: int) -> Dict[str, Any]:
    r = httpx.post(
        f"{api_url.rstrip('/')}/predict",
        json={"text": text, "max_new_tokens": max_new_tokens},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    st.set_page_config(page_title="Custom Patient Safety LLM UI", layout="wide")
    st.title("Custom Patient Safety LLM - Operator UI")
    st.caption("Use this UI to query the local custom LLM API and review risk outputs quickly.")

    with st.sidebar:
        st.header("Connection")
        api_url = st.text_input("API Base URL", value=DEFAULT_API_URL)
        max_new_tokens = st.slider("Max New Tokens", min_value=16, max_value=512, value=128, step=8)
        if st.button("Check API Health"):
            health = check_health(api_url)
            if health["ok"]:
                st.success(f"API healthy (HTTP {health['status_code']})")
            else:
                st.error(f"API unavailable: {health['body']}")

    sample = "Wrong-patient procedure started before timeout confirmation."
    text = st.text_area("Clinical Scenario", value=sample, height=180)

    col1, col2 = st.columns([1, 3])
    with col1:
        run_btn = st.button("Run Prediction", type="primary")

    if run_btn:
        if not text.strip():
            st.warning("Enter clinical scenario text before running prediction.")
            return

        with st.spinner("Calling custom LLM API..."):
            try:
                result = run_predict(api_url, text.strip(), max_new_tokens=max_new_tokens)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

        st.subheader("Result")
        st.metric("Risk Level", result.get("risk_level", "unknown"))
        st.write(result.get("reasoning", ""))

        with st.expander("Raw Response"):
            st.code(result.get("raw_response", ""), language="text")

        with st.expander("Full JSON"):
            st.json(result)


if __name__ == "__main__":
    main()
