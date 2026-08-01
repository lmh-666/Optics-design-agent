# agent/planner.py

from typing import Dict, Any, List


def classify_intent(user_message: str, design_state: Dict[str, Any]) -> str:
    """
    判断用户当前输入属于哪类任务。
    第一版先用规则判断，稳定、可控、方便调试。
    """

    text = user_message.strip()

    has_existing_task = design_state.get("parsed_requirement") is not None

    # 1. 明确新建设计任务
    if any(k in text for k in ["设计", "镜头", "光学系统", "F数", "视场角", "焦距", "CCTV", "车载", "手机"]):
        if not has_existing_task:
            return "new_design_task"

        if any(k in text for k in ["重新设计", "新的需求", "换个需求", "重新开始"]):
            return "new_design_task"

    # 2. 修改约束
    if any(k in text for k in ["短一点", "更短", "太长", "紧凑", "总长短", "总长再短"]):
        return "modify_constraint"

    if any(k in text for k in ["F数更小", "F 数更小", "光圈更大", "低照度", "更亮"]):
        return "modify_constraint"

    if any(k in text for k in ["视场更大", "广角一点", "角度更大", "视场角更大"]):
        return "modify_constraint"

    if any(k in text for k in ["简单一点", "元件少", "镜片少", "结构简单"]):
        return "modify_constraint"

    # 3. 重新检索
    if any(k in text for k in ["重新检索", "换一批", "重新推荐", "还有别的吗", "再找几个"]):
        return "retrieve_again"

    # 4. 重新评价
    if any(k in text for k in ["ray tracing", "ray spread", "光线追迹", "重新评价", "仿真", "跑一下评价"]):
        return "run_evaluation"

    # 5. 解释候选
    if any(k in text for k in ["为什么", "解释", "原因", "推荐理由", "第一个怎么样", "第二个怎么样", "第2个"]):
        return "explain_candidate"

    # 6. 外推生成
    if any(k in text for k in ["外推", "生成新结构", "数据库没有", "更接近目标", "新结构"]):
        return "extrapolate_structure"

    # 如果已经有任务，用户模糊追问，默认解释当前结果
    if has_existing_task:
        return "general_followup"

    return "new_design_task"


def build_plan(intent: str) -> List[str]:
    """
    根据意图生成工具调用链。
    """

    if intent == "new_design_task":
        return [
            "parse_requirement",
            "retrieve_candidates",
            "rerank_candidates",
            "run_raytrace",
            "explain_recommendation",
        ]

    if intent == "modify_constraint":
        return [
            "update_design_state",
            "retrieve_candidates",
            "rerank_candidates",
            "explain_recommendation",
        ]

    if intent == "retrieve_again":
        return [
            "retrieve_candidates",
            "rerank_candidates",
            "explain_recommendation",
        ]

    if intent == "run_evaluation":
        return [
            "run_raytrace",
            "explain_recommendation",
        ]

    if intent == "explain_candidate":
        return [
            "explain_recommendation",
        ]

    if intent == "extrapolate_structure":
        return [
            "extrapolate_structure",
            "run_raytrace",
            "explain_recommendation",
        ]

    return [
        "explain_recommendation",
    ]