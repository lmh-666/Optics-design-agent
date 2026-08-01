from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.executor import update_design_state_from_message
from agent.planner import classify_intent
from agent.response_builder import build_agent_response
from agent.state_manager import DEFAULT_DESIGN_STATE, state_manager
from agent.tools import ToolRegistry, tool_registry


class OpticalGraphState(TypedDict, total=False):
    session_id: str
    user_message: str
    top_k: int
    intent: str
    design_state: Dict[str, Any]
    called_tools: List[str]
    tool_outputs: Dict[str, Any]
    answer: str
    error: Optional[str]


def _ensure_graph_state(state: OpticalGraphState) -> OpticalGraphState:
    state.setdefault("called_tools", [])
    state.setdefault("tool_outputs", {})
    state.setdefault("top_k", 9)
    return state


def _record_tool(state: OpticalGraphState, name: str, output: Any) -> None:
    state.setdefault("called_tools", []).append(name)
    state.setdefault("tool_outputs", {})[name] = output


def _set_error(state: OpticalGraphState, step: str, error: str) -> None:
    state["error"] = error
    _record_tool(state, step, {"error": error})


def _public_design_state(design_state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "raw_requirement": design_state.get("raw_requirement"),
        "parsed_requirement": design_state.get("parsed_requirement"),
        "user_preferences": design_state.get("user_preferences"),
        "iteration": design_state.get("iteration"),
    }


class LangGraphOpticalDesignAgent:
    """LangGraph implementation of the optical design workflow agent."""

    def __init__(self, registry: ToolRegistry = tool_registry):
        self.registry = registry
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(OpticalGraphState)

        workflow.add_node("load_state", self._load_state)
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("parse_requirement", self._parse_requirement)
        workflow.add_node("update_constraints", self._update_constraints)
        workflow.add_node("retrieve_candidates", self._retrieve_candidates)
        workflow.add_node("rerank_candidates", self._rerank_candidates)
        workflow.add_node("run_raytrace", self._run_raytrace)
        workflow.add_node("explain_recommendation", self._explain_recommendation)
        workflow.add_node("save_state", self._save_state)

        workflow.add_edge(START, "load_state")
        workflow.add_edge("load_state", "classify_intent")
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_after_intent,
            {
                "parse_requirement": "parse_requirement",
                "update_constraints": "update_constraints",
                "retrieve_candidates": "retrieve_candidates",
                "run_raytrace": "run_raytrace",
                "explain_recommendation": "explain_recommendation",
            },
        )

        workflow.add_edge("parse_requirement", "retrieve_candidates")
        workflow.add_edge("update_constraints", "retrieve_candidates")
        workflow.add_edge("retrieve_candidates", "rerank_candidates")
        workflow.add_edge("rerank_candidates", "run_raytrace")
        workflow.add_edge("run_raytrace", "explain_recommendation")
        workflow.add_edge("explain_recommendation", "save_state")
        workflow.add_edge("save_state", END)

        return workflow.compile()

    def _load_state(self, state: OpticalGraphState) -> OpticalGraphState:
        state = _ensure_graph_state(state)
        session_id = state["session_id"]
        design_state = deepcopy(state_manager.get_state(session_id))
        state["design_state"] = design_state
        _record_tool(state, "load_state", {"session_id": session_id})
        return state

    def _classify_intent(self, state: OpticalGraphState) -> OpticalGraphState:
        design_state = state["design_state"]
        user_message = state["user_message"]

        state_manager.append_history(design_state, role="user", content=user_message)
        intent = classify_intent(user_message=user_message, design_state=design_state)

        state["intent"] = intent
        _record_tool(state, "classify_intent", {"intent": intent})
        return state

    def _route_after_intent(self, state: OpticalGraphState) -> str:
        intent = state.get("intent")
        if intent == "new_design_task":
            return "parse_requirement"
        if intent == "modify_constraint":
            return "update_constraints"
        if intent == "retrieve_again":
            return "retrieve_candidates"
        if intent == "run_evaluation":
            return "run_raytrace"
        return "explain_recommendation"

    def _parse_requirement(self, state: OpticalGraphState) -> OpticalGraphState:
        result = self.registry.call("parse_requirement", user_text=state["user_message"])
        if not result.success:
            _set_error(state, "parse_requirement", result.error)
            return state

        design_state = state["design_state"]
        design_state["raw_requirement"] = state["user_message"]
        design_state["parsed_requirement"] = result.data
        _record_tool(state, "parse_requirement", result.data)
        return state

    def _update_constraints(self, state: OpticalGraphState) -> OpticalGraphState:
        design_state = update_design_state_from_message(
            user_message=state["user_message"],
            design_state=state["design_state"],
        )
        state["design_state"] = design_state
        _record_tool(
            state,
            "update_constraints",
            {
                "updated_requirement": design_state.get("parsed_requirement"),
                "user_preferences": design_state.get("user_preferences"),
            },
        )
        return state

    def _retrieve_candidates(self, state: OpticalGraphState) -> OpticalGraphState:
        if state.get("error"):
            return state

        design_state = state["design_state"]
        parsed = design_state.get("parsed_requirement") or {}
        raw_text = design_state.get("raw_requirement") or state["user_message"]
        result = self.registry.call(
            "retrieve_candidates",
            parsed_requirement=parsed,
            raw_text=raw_text,
            top_k=max(int(state.get("top_k") or 9), 9),
        )
        if not result.success:
            _set_error(state, "retrieve_candidates", result.error)
            return state

        design_state["current_candidates"] = result.data or []
        _record_tool(
            state,
            "retrieve_candidates",
            {
                "candidate_count": len(design_state["current_candidates"]),
                "candidates": design_state["current_candidates"],
            },
        )
        return state

    def _rerank_candidates(self, state: OpticalGraphState) -> OpticalGraphState:
        if state.get("error"):
            return state

        design_state = state["design_state"]
        result = self.registry.call(
            "rerank_candidates",
            parsed_requirement=design_state.get("parsed_requirement") or {},
            candidates=design_state.get("current_candidates") or [],
        )
        if not result.success:
            _set_error(state, "rerank_candidates", result.error)
            return state

        design_state["last_rerank_result"] = result.data
        if isinstance(result.data, dict) and isinstance(result.data.get("top_candidates"), list):
            design_state["current_candidates"] = result.data["top_candidates"]
        _record_tool(state, "rerank_candidates", result.data)
        return state

    def _run_raytrace(self, state: OpticalGraphState) -> OpticalGraphState:
        if state.get("error"):
            return state

        design_state = state["design_state"]
        result = self.registry.call(
            "run_raytrace",
            candidates=design_state.get("current_candidates") or [],
        )
        if not result.success:
            _set_error(state, "run_raytrace", result.error)
            return state

        design_state["last_raytrace_result"] = result.data
        if isinstance(result.data, dict) and isinstance(result.data.get("raytrace_reranked_candidates"), list):
            design_state["current_candidates"] = result.data["raytrace_reranked_candidates"]
        _record_tool(state, "run_raytrace", result.data)
        return state

    def _explain_recommendation(self, state: OpticalGraphState) -> OpticalGraphState:
        design_state = state["design_state"]

        if state.get("error"):
            explanation = f"LangGraph 工作流执行到 {state['called_tools'][-1]} 时失败：{state['error']}"
        else:
            result = self.registry.call(
                "explain_recommendation",
                parsed_requirement=design_state.get("parsed_requirement") or {},
                candidates=design_state.get("current_candidates") or [],
                raytrace_result=design_state.get("last_raytrace_result"),
            )
            if result.success:
                explanation = result.data
            else:
                explanation = f"候选处理完成，但推荐解释生成失败：{result.error}"

        _record_tool(state, "explain_recommendation", explanation)
        state["answer"] = build_agent_response(
            intent=state.get("intent") or "unknown",
            design_state=design_state,
            called_tools=state.get("called_tools") or [],
            tool_outputs=state.get("tool_outputs") or {},
        )
        return state

    def _save_state(self, state: OpticalGraphState) -> OpticalGraphState:
        design_state = state["design_state"]
        design_state["iteration"] = int(design_state.get("iteration") or 0) + 1
        state_manager.append_history(
            design_state,
            role="assistant",
            content=state.get("answer") or "",
        )
        state_manager.save_state(state["session_id"], design_state)
        _record_tool(
            state,
            "save_state",
            {
                "session_id": state["session_id"],
                "iteration": design_state["iteration"],
            },
        )
        return state

    def step(self, session_id: str, user_message: str, top_k: int = 9) -> Dict[str, Any]:
        initial_state: OpticalGraphState = {
            "session_id": session_id,
            "user_message": user_message,
            "top_k": top_k,
            "called_tools": [],
            "tool_outputs": {},
            "design_state": deepcopy(DEFAULT_DESIGN_STATE),
        }
        final_state = self.graph.invoke(initial_state)
        design_state = final_state.get("design_state") or {}

        return {
            "session_id": session_id,
            "framework": "langgraph",
            "intent": final_state.get("intent"),
            "called_tools": final_state.get("called_tools") or [],
            "design_state": _public_design_state(design_state),
            "candidates": design_state.get("current_candidates") or [],
            "last_rerank_result": design_state.get("last_rerank_result"),
            "last_raytrace_result": design_state.get("last_raytrace_result"),
            "tool_outputs": final_state.get("tool_outputs") or {},
            "answer": final_state.get("answer"),
            "error": final_state.get("error"),
        }


langgraph_optical_agent = LangGraphOpticalDesignAgent()

