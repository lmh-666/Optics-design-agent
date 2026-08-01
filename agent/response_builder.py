# agent/response_builder.py

from typing import Any, Dict, List


def build_agent_response(
    intent: str,
    design_state: Dict[str, Any],
    called_tools: List[str],
    tool_outputs: Dict[str, Any],
) -> str:
    candidates = design_state.get("current_candidates") or []
    parsed = design_state.get("parsed_requirement") or {}

    explanation = tool_outputs.get("explain_recommendation")

    if isinstance(explanation, str):
        base_explanation = explanation
    else:
        base_explanation = "系统已完成本轮处理。"

    if intent == "new_design_task":
        prefix = (
            "已识别为新的光学镜头设计任务。\n"
            f"本轮调用工具：{', '.join(called_tools)}。\n"
            f"当前候选结构数量：{len(candidates)}。\n"
        )
        return prefix + "\n" + base_explanation

    if intent == "modify_constraint":
        prefix = (
            "已识别为对上一轮设计需求的约束修改。\n"
            f"本轮调用工具：{', '.join(called_tools)}。\n"
            "系统已基于当前 design_state 更新约束，并重新处理候选结构。\n"
        )
        return prefix + "\n" + base_explanation

    if intent == "retrieve_again":
        prefix = (
            "已识别为重新检索候选结构。\n"
            f"本轮调用工具：{', '.join(called_tools)}。\n"
            f"当前候选结构数量：{len(candidates)}。\n"
        )
        return prefix + "\n" + base_explanation

    if intent == "run_evaluation":
        prefix = (
            "已识别为重新进行光线追迹/评价。\n"
            f"本轮调用工具：{', '.join(called_tools)}。\n"
        )
        return prefix + "\n" + base_explanation

    if intent == "explain_candidate":
        prefix = (
            "已识别为解释当前候选结构推荐理由。\n"
            f"本轮调用工具：{', '.join(called_tools)}。\n"
        )
        return prefix + "\n" + base_explanation

    return (
        f"已完成本轮 Agent 处理。\n"
        f"识别意图：{intent}\n"
        f"调用工具：{', '.join(called_tools)}\n\n"
        f"{base_explanation}"
    )