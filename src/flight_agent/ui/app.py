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


def _messages() -> list:
    """Always return a real list in session_state (never use missing attrs)."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    return st.session_state["messages"]


@st.cache_resource(show_spinner="Preparing model + local lake cache…")
def _warm_runtime() -> str:
    """Load model + sync small serve marts once per server process."""
    try:
        services.load_model()
    except Exception as exc:  # noqa: BLE001
        return f"model:{exc}"
    try:
        from flight_agent.ingest.serve_cache import ensure_serve_cache, serve_cache_dir
        from flight_agent.ingest.warehouse import (
            reset_serve_connection,
            warehouse_available,
            warehouse_connection,
        )

        ensure_serve_cache()
        reset_serve_connection()
        if warehouse_available():
            with warehouse_connection(read_only=True, light=True) as con:
                con.execute("select 1").fetchone()
        return f"ok:{serve_cache_dir()}"
    except Exception as exc:  # noqa: BLE001
        return f"warehouse:{exc}"


warm_status = _warm_runtime()

st.title("Flight Delay Ops Agent")
st.caption(
    "Ask about delay risk on LAX / JFK / ORD / DEN / ATL / IAD / DFW routes. "
    "Answers use the trained model + local serve-cache marts + Groq."
)

with st.sidebar:
    st.header("Model status")
    if warm_status.startswith("ok") or warm_status.startswith("warehouse:"):
        try:
            services.load_model()
            st.success("Model loaded")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
    else:
        st.error(warm_status)
    if warm_status.startswith("ok:"):
        st.caption(f"Serve cache: `{warm_status.split('ok:', 1)[1]}`")
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

msgs = _messages()
if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = None

# Paint history first.
for msg in msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about a flight delay…")

# Submit → store user turn + pending flag → rerun so the user bubble is on screen.
if prompt:
    msgs.append({"role": "user", "content": prompt})
    st.session_state["pending_question"] = prompt
    st.rerun()

# Answer only while a pending question exists; clear it AFTER we finish.
pending = st.session_state.get("pending_question")
if pending:
    with st.chat_message("assistant"):
        box = st.empty()
        box.info("Consulting model + lake tools…")
        try:
            answer = ask_agent(pending)
        except Exception as exc:  # noqa: BLE001
            answer = (
                "Sorry — something went wrong. You can retry the question.\n\n"
                f"`{type(exc).__name__}: {exc}`"
            )
        box.markdown(answer)

    # Re-bind messages after the long call (session can be flaky across waits).
    history = _messages()
    history.append({"role": "assistant", "content": answer})
    st.session_state["messages"] = history
    st.session_state["pending_question"] = None
