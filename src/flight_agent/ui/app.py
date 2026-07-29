"""Streamlit chat UI for the flight ops agent."""

from __future__ import annotations

import streamlit as st

# Must be the first Streamlit command.
st.set_page_config(
    page_title="Flight Delay Ops Agent",
    page_icon="✈️",
    layout="centered",
)

from flight_agent.config import get_settings  # noqa: E402
from flight_agent.runtime_secrets import apply_runtime_secrets  # noqa: E402

apply_runtime_secrets()
get_settings.cache_clear()

from flight_agent.agent.graph import ask_agent  # noqa: E402
from flight_agent.serve import services  # noqa: E402


@st.cache_resource(show_spinner=False)
def _warm_runtime() -> str:
    """Load model + open light R2 warehouse once per server process."""
    try:
        services.load_model()
    except Exception as exc:  # noqa: BLE001
        return f"model:{exc}"
    try:
        from flight_agent.ingest.warehouse import warehouse_available, warehouse_connection

        if warehouse_available():
            with warehouse_connection(read_only=True, light=True) as con:
                con.execute("select 1").fetchone()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"warehouse:{exc}"


warm_status = _warm_runtime()

st.title("Flight Delay Ops Agent")
st.caption(
    "Ask about delay risk on LAX / JFK / ORD / DEN / ATL / IAD / DFW routes. "
    "Answers use the trained model + curated lake tools (R2) via Groq."
)

with st.sidebar:
    st.header("Model status")
    if warm_status == "ok" or warm_status.startswith("warehouse:"):
        try:
            services.load_model()
            st.success("Model loaded")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
    else:
        st.error(warm_status)
    metrics = services.load_metrics()
    if metrics.get("overall"):
        o = metrics["overall"]
        st.metric("ROC-AUC", f"{o.get('roc_auc', 0):.3f}")
        st.metric("F1", f"{o.get('f1', 0):.3f}")
    st.markdown(
        """
**Example questions**
- Will DL from ATL to LAX on a Monday morning be delayed?
- What's the historical delay rate ORD→JFK on United?
- How congested is DFW at 5pm (taxi / NAS)?
- How often is IAD→DFW delayed in winter?
- How good is the delay model?
"""
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# 1) Paint history first so the page never goes blank while waiting.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about a flight delay…")

# 2) On submit: store user turn and rerun so the message is visible before work starts.
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.awaiting_reply = True
    st.rerun()

# 3) Answer exactly once after the user message is on screen.
if st.session_state.get("awaiting_reply"):
    st.session_state.awaiting_reply = False
    question = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        with st.spinner("Consulting model + lake tools…"):
            try:
                answer = ask_agent(question)
            except Exception as exc:  # noqa: BLE001
                answer = (
                    "Sorry — something went wrong. The page stayed up so you can retry.\n\n"
                    f"`{type(exc).__name__}: {exc}`"
                )
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
