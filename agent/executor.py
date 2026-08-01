# agent/executor.py

from typing import Any, Dict

from agent.planner import build_plan
from agent.tools import tool_registry


def update_design_state_from_message(user_message: str, design_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    多轮对话中，根据用户补充输入修改当前 design_state。
    """

    text = user_message.strip()
    parsed_requirement = design_state.get("parsed_requirement") or {}

    # 1. 总长更短
    if any(k in text for k in ["短一点", "更短", "太长", "紧凑", "总长短", "总长再短"]):
        design_state["user_preferences"]["prefer_compact"] = True

        parsed_requirement.setdefault("total_length", {})
        parsed_requirement["total_length"]["target"] = parsed_requirement["total_length"].get("target")
        parsed_requirement["total_length"]["preference"] = "smaller"
        parsed_requirement["total_length"]["constraint_type"] = "soft"

    # 2. F 数更小
    if any(k in text for k in ["F数更小", "F 数更小", "光圈更大", "低照度", "更亮"]):
        design_state["user_preferences"]["prefer_low_f_number"] = True

        parsed_requirement.setdefault("f_number", {})
        parsed_requirement["f_number"]["target"] = parsed_requirement["f_number"].get("target")
        parsed_requirement["f_number"]["preference"] = "smaller"
        parsed_requirement["f_number"]["constraint_type"] = "soft"

    # 3. 视场更大
    if any(k in text for k in ["视场更大", "广角一点", "角度更大", "视场角更大"]):
        design_state["user_preferences"]["prefer_large_fov"] = True

        parsed_requirement.setdefault("fov", {})
        parsed_requirement["fov"]["target"] = parsed_requirement["fov"].get("target")
        parsed_requirement["fov"]["preference"] = "larger"
        parsed_requirement["fov"]["constraint_type"] = "soft"

    # 4. 结构更简单
    if any(k in text for k in ["简单一点", "元件少", "镜片少", "结构简单"]):
        design_state["user_preferences"]["prefer_simple_structure"] = True

        parsed_requirement.setdefault("element_count", {})
        parsed_requirement["element_count"]["target"] = parsed_requirement["element_count"].get("target")
        parsed_requirement["element_count"]["preference"] = "smaller"
        parsed_requirement["element_count"]["constraint_type"] = "soft"

    design_state["parsed_requirement"] = parsed_requirement
    return design_state


def execute_plan(
    intent: str,
    user_message: str,
    design_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    根据 planner 给出的 intent 执行对应工具链。
    """

    plan = build_plan(intent)
    called_tools = []
    tool_outputs = {}

    for step in plan:

        if step == "update_design_state":
            design_state = update_design_state_from_message(
                user_message=user_message,
                design_state=design_state,
            )
            called_tools.append(step)
            tool_outputs[step] = {
                "updated_requirement": design_state.get("parsed_requirement"),
                "user_preferences": design_state.get("user_preferences"),
            }
            continue

        if step == "parse_requirement":
            result = tool_registry.call(
                "parse_requirement",
                user_text=user_message,
            )
            called_tools.append(step)

            if not result.success:
                tool_outputs[step] = {"error": result.error}
                break

            design_state["raw_requirement"] = user_message
            design_state["parsed_requirement"] = result.data
            tool_outputs[step] = result.data
            continue

        if step == "retrieve_candidates":
            result = tool_registry.call(
                "retrieve_candidates",
                parsed_requirement=design_state.get("parsed_requirement") or {},
                raw_text=design_state.get("raw_requirement") or user_message,
                top_k=20,
            )
            called_tools.append(step)

            if not result.success:
                tool_outputs[step] = {"error": result.error}
                break

            design_state["current_candidates"] = result.data or []
            tool_outputs[step] = {
                "candidate_count": len(design_state["current_candidates"]),
                "candidates": design_state["current_candidates"],
            }
            continue

        if step == "rerank_candidates":
            result = tool_registry.call(
                "rerank_candidates",
                parsed_requirement=design_state.get("parsed_requirement") or {},
                candidates=design_state.get("current_candidates") or [],
            )
            called_tools.append(step)

            if not result.success:
                tool_outputs[step] = {"error": result.error}
                break

            design_state["last_rerank_result"] = result.data

            if isinstance(result.data, dict):
                top_candidates = result.data.get("top_candidates")
                if top_candidates is not None:
                    design_state["current_candidates"] = top_candidates

            tool_outputs[step] = result.data
            continue

        if step == "run_raytrace":
            result = tool_registry.call(
                "run_raytrace",
                candidates=design_state.get("current_candidates") or [],
            )
            called_tools.append(step)

            if not result.success:
                tool_outputs[step] = {"error": result.error}
                break

            design_state["last_raytrace_result"] = result.data

            if isinstance(result.data, dict):
                rr = result.data.get("raytrace_reranked_candidates")
                if isinstance(rr, list):
                    design_state["current_candidates"] = rr

            tool_outputs[step] = result.data
            continue

        if step == "extrapolate_structure":
            result = tool_registry.call(
                "extrapolate_structure",
                candidates=design_state.get("current_candidates") or [],
            )
            called_tools.append(step)

            if not result.success:
                tool_outputs[step] = {"error": result.error}
                break

            tool_outputs[step] = result.data
            continue

        if step == "explain_recommendation":
            result = tool_registry.call(
                "explain_recommendation",
                parsed_requirement=design_state.get("parsed_requirement") or {},
                candidates=design_state.get("current_candidates") or [],
                raytrace_result=design_state.get("last_raytrace_result"),
            )
            called_tools.append(step)

            if not result.success:
                tool_outputs[step] = {"error": result.error}
                break

            tool_outputs[step] = result.data
            continue

    design_state["iteration"] = int(design_state.get("iteration") or 0) + 1

    return {
        "design_state": design_state,
        "called_tools": called_tools,
        "tool_outputs": tool_outputs,
    }