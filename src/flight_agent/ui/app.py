"""Streamlit chat UI for the flight ops agent."""

from __future__ import annotations

import streamlit as st

from flight_agent.agent.graph import ask_agent
from flight_agent.serve import services

st.set_page_config(
    page_title="Flight Delay Ops Agent",
    page_icon="✈️",
    layout="centered",
)

st.title("Flight Delay Ops Agent")
st.caption(
    "Ask about delay risk on LAX / JFK / ORD / DEN / ATL / IAD / DFW routes. "
    "Answers use the trained model + curated lake tools."
)

with st.sidebar:
    st.header("Model status")
    try:
        services.load_model()
        st.success("Model loaded")
    except FileNotFoundError as exc:
        st.error(str(exc))
    metrics = services.load_metrics()
    if metrics.get("overall"):
        o = metrics["overall"]
        st.metric("ROC-AUC", f"{o.get('roc_auc', 0):.3f}")
        st.metric("F1", f"{o.get('f1', 0):.3f}")
    st.markdown(
        """
**Example questions**
- Will Delta from Atlanta to Los Angeles on a Monday morning be delayed?
- What's the historical delay rate O'Hare→JFK on American?
- How's the weather at Dulles today at 10pm?
- How congested is Dallas Fort Worth at 5pm?
- How good is the delay model?
"""
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about a flight delay…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Consulting tools…"):
            answer = ask_agent(prompt)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
