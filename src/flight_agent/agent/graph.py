"""LangGraph + Groq ops agent for flight delay questions."""

from __future__ import annotations

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from flight_agent.agent import tools as tool_fns
from flight_agent.config import get_settings

SYSTEM_PROMPT = """You are an aviation operations assistant for US hub flights
(LAX, JFK, ORD, DEN, ATL, IAD, DFW). You help travelers and ops staff understand delay risk.

Rules:
- Use tools for facts (predictions, route stats, weather, congestion, model metrics). Do not invent numbers.
- Explain results in plain English: probability, congestion/weather drivers, practical advice.
- Stay on aviation delay / schedule topics. Politely refuse unrelated requests.
- If the model is missing, tell the user to run `flight train`.
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
                "Args: op_unique_carrier, origin, dest, fl_month (1-12), "
                "fl_dow (0=Sun..6=Sat), crs_dep_hour (0-23), optional distance/weather."
            ),
        ),
        StructuredTool.from_function(
            func=tool_fns.tool_route_stats,
            name="route_stats",
            description="Historical delay rate for origin→dest, optional carrier.",
        ),
        StructuredTool.from_function(
            func=tool_fns.tool_weather,
            name="weather",
            description="Weather features for an airport IATA code, optional date YYYY-MM-DD.",
        ),
        StructuredTool.from_function(
            func=tool_fns.tool_airport_congestion,
            name="airport_congestion",
            description=(
                "Historical congestion for an airport at a clock hour (0-23): "
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
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env (https://console.groq.com/keys)."
        )
    tools = _build_tools()
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
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


def ask_agent(question: str) -> str:
    """One-shot Q&A. Falls back to a deterministic tool summary if Groq is unavailable."""
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
        return _fallback_answer(question, error=str(exc))


def _fallback_answer(question: str, error: str) -> str:
    """Heuristic fallback so demos work without Groq."""
    q = question.upper()
    from flight_agent.config import load_project_config

    hubs = list(load_project_config()["hubs"])
    found = [h for h in hubs if h in q]
    origin = found[0] if found else "LAX"
    dest = found[1] if len(found) > 1 else ("JFK" if origin != "JFK" else "DFW")
    carrier = next((c for c in ["DL", "AA", "UA", "B6", "WN", "AS"] if c in q), "DL")

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
        f"(Groq unavailable: {error})\n\n"
        f"Fallback tool summary for {carrier} {origin}→{dest}:\n"
        f"- route_stats: {stats}\n"
        f"- predict_delay: {pred}\n"
        "Set GROQ_API_KEY in .env (https://console.groq.com/keys) for full chat reasoning."
    )
