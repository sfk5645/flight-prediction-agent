"""LangGraph + Ollama ops agent for flight delay questions."""

from __future__ import annotations

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from flight_agent.agent import tools as tool_fns
from flight_agent.codes import find_airports_in_text, find_carriers_in_text
from flight_agent.config import get_settings

SYSTEM_PROMPT = """You are an aviation operations assistant for US hub flights
(LAX, JFK, ORD, DEN, ATL, IAD, DFW). You help travelers and ops staff understand delay risk.

Rules:
- Use tools for facts (predictions, route stats, weather, congestion, model metrics). Do not invent numbers.
- Users may say full airline/airport names (United, Dulles, O'Hare). Tools accept names or IATA codes.
- For weather dates use YYYY-MM-DD, "today", or "yesterday". Weather is DAILY (not hourly) and is refreshed more often than BTS flight labels; if the exact day is missing the tool falls back to the latest lake day.
- If a tool returns n_flights=0 or a note, explain that clearly. Do NOT tell the user to run `flight train` unless the model/metrics tool says the model is missing.
- Explain results in plain English: probability, congestion/weather drivers, practical advice.
- Stay on aviation delay / schedule topics. Politely refuse unrelated requests.
- fl_dow uses DuckDB convention: 0=Sunday … 6=Saturday.
"""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _build_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=tool_fns.tool_predict_delay,
            name="predict_delay",
            description=(
                "Predict arrival delay ≥15 min probability. "
                "Args: op_unique_carrier (code or name e.g. UA/United), "
                "origin/dest (IATA or name e.g. IAD/Dulles), "
                "fl_month (1-12), fl_dow (0=Sun..6=Sat), crs_dep_hour (0-23), "
                "optional distance/weather."
            ),
        ),
        StructuredTool.from_function(
            func=tool_fns.tool_route_stats,
            name="route_stats",
            description=(
                "Historical delay rate for origin→dest. "
                "origin/dest and optional carrier accept IATA codes or full names "
                "(United, American, Dulles, O'Hare)."
            ),
        ),
        StructuredTool.from_function(
            func=tool_fns.tool_weather,
            name="weather",
            description=(
                "Daily weather features for an airport (IATA or name). "
                "Optional date: YYYY-MM-DD, today, or yesterday. "
                "Weather is not hourly — do not pass clock times as the date."
            ),
        ),
        StructuredTool.from_function(
            func=tool_fns.tool_airport_congestion,
            name="airport_congestion",
            description=(
                "Historical congestion for an airport (IATA or name) at a clock hour (0-23): "
                "avg taxi-out/in, NAS/carrier/weather/late-aircraft delay minutes, "
                "operation counts, delay rate."
            ),
        ),
        StructuredTool.from_function(
            func=tool_fns.tool_model_metrics,
            name="model_metrics",
            description="Offline evaluation metrics for the trained delay model.",
        ),
    ]


def build_agent():
    settings = get_settings()
    tools = _build_tools()
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
    ).bind_tools(tools)

    def chatbot(state: AgentState):
        return {"messages": [llm.invoke(state["messages"])]}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", chatbot)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def _looks_like_llm_down(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = (
        "connection refused",
        "connect error",
        "connection error",
        "failed to establish",
        "ollama",
        "httpx",
        "timed out",
        "timeout",
        "name or service not known",
    )
    return any(n in msg for n in needles)


def ask_agent(question: str) -> str:
    """One-shot Q&A. Falls back to a deterministic tool summary if Ollama is down."""
    try:
        agent = build_agent()
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=question),
                ]
            }
        )
        messages = result["messages"]
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                return str(msg.content)
        return str(messages[-1].content)
    except Exception as exc:  # noqa: BLE001
        if _looks_like_llm_down(exc):
            return _fallback_answer(question, error=str(exc))
        return (
            f"I hit an error while answering: {exc}\n\n"
            "Try rephrasing (e.g. weather dates as YYYY-MM-DD or 'today'), "
            "or ask about route delay stats / congestion."
        )


def _fallback_answer(question: str, error: str) -> str:
    """Heuristic fallback so demos work without Ollama."""
    hubs = find_airports_in_text(question)
    carriers = find_carriers_in_text(question)
    origin = hubs[0] if hubs else "LAX"
    dest = hubs[1] if len(hubs) > 1 else ("JFK" if origin != "JFK" else "DFW")
    carrier = carriers[0] if carriers else "DL"

    # Weather-ish questions: answer with weather tool only
    q_lower = question.lower()
    if "weather" in q_lower or "forecast" in q_lower or "temperature" in q_lower:
        wx = tool_fns.tool_weather(origin, "today" if "today" in q_lower else None)
        return (
            f"(LLM unavailable: {error})\n\n"
            f"Weather lookup for {origin}:\n{wx}\n"
            "Start Ollama (`ollama serve` + `ollama pull llama3.2:3b`) for full chat reasoning."
        )

    stats = tool_fns.tool_route_stats(origin, dest, carrier)
    pred = tool_fns.tool_predict_delay(
        op_unique_carrier=carrier,
        origin=origin,
        dest=dest,
        fl_month=6,
        fl_dow=1,
        crs_dep_hour=8,
    )
    return (
        f"(LLM unavailable: {error})\n\n"
        f"Fallback tool summary for {carrier} {origin}→{dest}:\n"
        f"- route_stats: {stats}\n"
        f"- predict_delay: {pred}\n"
        "Start Ollama (`ollama serve` + `ollama pull llama3.2:3b`) for full chat reasoning."
    )
