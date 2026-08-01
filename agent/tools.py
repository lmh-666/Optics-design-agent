# agent/tools.py

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    message: str = ""
    error: str = ""


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, func: Callable, description: str) -> None:
        self.tools[name] = {
            "func": func,
            "description": description,
        }

    def call(self, name: str, **kwargs) -> ToolResult:
        if name not in self.tools:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {name}",
            )

        try:
            result = self.tools[name]["func"](**kwargs)
            return ToolResult(
                success=True,
                data=result,
                message=f"Tool {name} executed successfully.",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
            )


tool_registry = ToolRegistry()


# =========================
# 1. 需求解析工具
# =========================

def _extract_number_after_keywords(text: str, keywords: List[str]) -> Optional[float]:
    for kw in keywords:
        pattern = rf"{kw}\s*[为是=:：]?\s*([0-9]+(?:\.[0-9]+)?)"
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def parse_requirement_tool(user_text: str) -> Dict[str, Any]:
    """
    工具1：需求解析。
    第一版使用规则解析兜底。
    后续你可以在这里接 OpticsGPT / DeepSeek / Qwen 的 LLM JSON 解析结果。
    """

    text = user_text.strip()

    scene = None
    if any(k in text for k in ["车载", "vehicle", "automotive"]):
        scene = "vehicle"
    elif any(k in text for k in ["CCTV", "监控"]):
        scene = "CCTV"
    elif any(k in text for k in ["手机", "mobile"]):
        scene = "mobile"

    f_number = _extract_number_after_keywords(
        text,
        ["F数", "F 数", "F#", "F", "f_number", "光圈"],
    )

    fov = _extract_number_after_keywords(
        text,
        ["视场角", "视场", "FOV", "fov", "角度"],
    )

    focal_length = _extract_number_after_keywords(
        text,
        ["焦距", "focal_length", "EFL", "efl"],
    )

    total_length = _extract_number_after_keywords(
        text,
        ["总长", "TTL", "ttl", "total_length"],
    )

    element_count = _extract_number_after_keywords(
        text,
        ["元件数量", "镜片数量", "片数", "element_count"],
    )

    parsed = {
        "raw_text": user_text,
        "scene": scene,
        "f_number": {
            "target": f_number,
            "preference": "smaller" if any(k in text for k in ["F数小", "F数更小", "低照度", "光圈更大"]) else None,
            "constraint_type": "soft" if f_number is not None else None,
        },
        "fov": {
            "target": fov,
            "preference": "greater_than" if any(k in text for k in ["大于", "超过", "更大", "广角"]) else None,
            "constraint_type": "hard" if fov is not None and any(k in text for k in ["大于", "超过", ">"]) else "soft" if fov is not None else None,
        },
        "focal_length": {
            "target": focal_length,
            "preference": None,
            "constraint_type": "soft" if focal_length is not None else None,
        },
        "total_length": {
            "target": total_length,
            "preference": "smaller" if any(k in text for k in ["短", "紧凑", "小型化"]) else None,
            "constraint_type": "soft" if total_length is not None or any(k in text for k in ["短", "紧凑", "小型化"]) else None,
        },
        "element_count": {
            "target": element_count,
            "preference": "smaller" if any(k in text for k in ["元件少", "镜片少", "结构简单"]) else None,
            "constraint_type": "soft" if element_count is not None else None,
        },
        "distortion": {
            "target": None,
            "preference": "controlled" if any(k in text for k in ["畸变", "失真", "车载"]) else None,
            "constraint_type": "soft" if any(k in text for k in ["畸变", "失真", "车载"]) else None,
        },
    }

    parsed = apply_scene_prior(parsed)
    parsed = validate_requirement(parsed)

    return parsed


def apply_scene_prior(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    轻量领域先验约束。
    """

    scene = req.get("scene")

    if scene == "vehicle":
        req["fov"]["preference"] = req["fov"].get("preference") or "larger"
        req["distortion"]["preference"] = req["distortion"].get("preference") or "controlled"
        req["distortion"]["constraint_type"] = req["distortion"].get("constraint_type") or "soft"

    if scene == "CCTV":
        req["fov"]["preference"] = req["fov"].get("preference") or "larger"
        req["f_number"]["preference"] = req["f_number"].get("preference") or "smaller"

    if scene == "mobile":
        req["total_length"]["preference"] = req["total_length"].get("preference") or "smaller"
        req["element_count"]["preference"] = req["element_count"].get("preference") or "smaller"

    return req


def validate_requirement(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    参数合法性校验。
    """

    fnum = req.get("f_number", {}).get("target")
    if fnum is not None and fnum <= 0:
        req["f_number"]["target"] = None

    fov = req.get("fov", {}).get("target")
    if fov is not None and (fov <= 0 or fov > 180):
        req["fov"]["target"] = None

    ttl = req.get("total_length", {}).get("target")
    if ttl is not None and ttl <= 0:
        req["total_length"]["target"] = None

    return req


# =========================
# 2. 现有光学代码适配层
# =========================

def _try_call(func: Callable, call_variants: List[Dict[str, Any]]) -> Any:
    """
    兼容不同函数签名。
    你现有项目函数签名可能变化过，所以这里逐个尝试。
    """

    last_error = None

    for variant in call_variants:
        args = variant.get("args", [])
        kwargs = variant.get("kwargs", {})

        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"All call variants failed. Last error: {last_error}")


def retrieve_candidates_tool(
    parsed_requirement: Dict[str, Any],
    raw_text: Optional[str] = None,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    工具2：候选结构检索。
    优先调用你现有 scripts.step10_top9_raytrace_rerank.build_step5_top9。
    """

    user_text = raw_text or parsed_requirement.get("raw_text", "")

    try:
        from scripts.step10_top9_raytrace_rerank import build_step5_top9
    except Exception as e:
        raise RuntimeError(
            "无法导入 build_step5_top9，请确认 scripts/step10_top9_raytrace_rerank.py 存在。"
            f" 原始错误：{e}"
        )

    result = _try_call(
        build_step5_top9,
        [
            {"args": [user_text], "kwargs": {"top_k": top_k}},
            {"args": [user_text], "kwargs": {}},
            {"args": [], "kwargs": {"raw_text": user_text, "top_k": top_k}},
            {"args": [], "kwargs": {"requirement": parsed_requirement, "top_k": top_k}},
            {"args": [], "kwargs": {"parsed_requirement": parsed_requirement, "top_k": top_k}},
        ],
    )

    # 兼容不同返回格式
    if isinstance(result, dict):
        if "candidates" in result:
            return result["candidates"]
        if "top_candidates" in result:
            return result["top_candidates"]
        if "top9" in result:
            return result["top9"]
        return [result]

    if isinstance(result, list):
        return result

    return []


def rerank_candidates_tool(
    parsed_requirement: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    工具3：候选结构重排序。
    优先调用 design_result_optimizer.optimize_hybrid_result_after_scale。
    """

    try:
        from design_result_optimizer import (
            optimize_hybrid_result_after_scale,
            build_optimized_recommendation,
        )
    except Exception:
        # 如果当前项目暂时没有这个文件，就返回原候选
        return {
            "top_candidates": candidates[:9],
            "candidate_roles": [],
            "recommendation": None,
            "ranking_reason": "未导入 design_result_optimizer，暂时使用检索顺序作为排序结果。",
        }

    hybrid_result = _try_call(
        optimize_hybrid_result_after_scale,
        [
            {
                "args": [],
                "kwargs": {
                    "requirement": parsed_requirement,
                    "candidates": candidates,
                },
            },
            {
                "args": [],
                "kwargs": {
                    "parsed_requirement": parsed_requirement,
                    "candidates": candidates,
                },
            },
            {
                "args": [parsed_requirement, candidates],
                "kwargs": {},
            },
        ],
    )

    try:
        recommendation = _try_call(
            build_optimized_recommendation,
            [
                {
                    "args": [],
                    "kwargs": {
                        "requirement": parsed_requirement,
                        "hybrid_result": hybrid_result,
                    },
                },
                {
                    "args": [],
                    "kwargs": {
                        "parsed_requirement": parsed_requirement,
                        "hybrid_result": hybrid_result,
                    },
                },
                {
                    "args": [parsed_requirement, hybrid_result],
                    "kwargs": {},
                },
            ],
        )
    except Exception:
        recommendation = None

    top_candidates = candidates[:9]

    if isinstance(hybrid_result, dict):
        top_candidates = (
            hybrid_result.get("top_candidates")
            or hybrid_result.get("candidates")
            or hybrid_result.get("reranked_candidates")
            or candidates[:9]
        )

    return {
        "top_candidates": top_candidates,
        "candidate_roles": hybrid_result.get("candidate_roles") if isinstance(hybrid_result, dict) else None,
        "hybrid_result": hybrid_result,
        "recommendation": recommendation,
    }


def run_raytrace_tool(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    工具4：ray tracing 评价。
    调用你现有 add_raytrace_eval_to_top9 和 rerank_top9_by_raytrace。
    """

    if not candidates:
        return {
            "evaluated_candidates": [],
            "raytrace_summary": "当前没有候选结构，无法执行 ray tracing。",
        }

    try:
        from scripts.step10_top9_raytrace_rerank import (
            add_raytrace_eval_to_top9,
            rerank_top9_by_raytrace,
        )
    except Exception as e:
        raise RuntimeError(
            "无法导入 ray tracing 相关函数，请确认 scripts/step10_top9_raytrace_rerank.py。"
            f" 原始错误：{e}"
        )

    evaluated = _try_call(
        add_raytrace_eval_to_top9,
        [
            {"args": [candidates[:9]], "kwargs": {}},
            {"args": [], "kwargs": {"top9": candidates[:9]}},
            {"args": [], "kwargs": {"candidates": candidates[:9]}},
        ],
    )

    reranked = _try_call(
        rerank_top9_by_raytrace,
        [
            {"args": [evaluated], "kwargs": {}},
            {"args": [], "kwargs": {"top9": evaluated}},
            {"args": [], "kwargs": {"candidates": evaluated}},
        ],
    )

    return {
        "evaluated_candidates": evaluated,
        "raytrace_reranked_candidates": reranked,
        "raytrace_summary": "已完成 Top-9 候选结构 ray tracing / ray spread 评价。",
    }


def explain_recommendation_tool(
    parsed_requirement: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    raytrace_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    工具5：推荐解释。
    第一版先用规则生成，避免大模型乱编。
    后续可以把这里换成 LLM，根据真实 candidates 和 raytrace_result 生成解释。
    """

    scene = parsed_requirement.get("scene")
    fnum = parsed_requirement.get("f_number", {}).get("target")
    fov = parsed_requirement.get("fov", {}).get("target")
    ttl_pref = parsed_requirement.get("total_length", {}).get("preference")

    n = len(candidates or [])

    parts = []
    parts.append(f"系统已根据当前需求完成候选结构处理，共保留 {n} 个候选结构。")

    if scene:
        parts.append(f"当前识别的应用场景为 {scene}。")

    if fnum is not None:
        parts.append(f"F 数目标约为 {fnum}，排序时会优先考虑 F 数接近或更小的结构。")

    if fov is not None:
        parts.append(f"视场角目标约为 {fov} 度，排序时会优先考虑视场角满足或接近目标的结构。")

    if ttl_pref == "smaller":
        parts.append("用户存在短总长/紧凑化偏好，因此重排序时会提高总长约束权重。")

    if raytrace_result:
        parts.append("当前已加入 ray tracing / ray spread 评价，用于进一步判断候选结构的光线传播稳定性。")
    else:
        parts.append("当前尚未或未成功获得 ray tracing 评价结果，推荐主要依据参数匹配和结构规则。")

    parts.append("该 Agent 的推荐结果不是由大模型直接生成镜头结构，而是基于专利数据库检索、领域约束、多目标排序和光线追迹评价得到。")

    return "\n".join(parts)


def extrapolate_structure_tool(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    工具6：结构外推。
    第一版先占位，后续可以接你的尺度变换、曲率缩放、空气间隔调整。
    """

    return {
        "status": "not_implemented",
        "message": "当前版本暂未启用结构外推。后续可基于当前 Top 候选进行尺度缩放、总长压缩和局部参数扰动。",
        "base_candidates": candidates[:3] if candidates else [],
    }


# =========================
# 3. 注册工具
# =========================

tool_registry.register(
    name="parse_requirement",
    func=parse_requirement_tool,
    description="将用户自然语言光学设计需求解析为结构化 JSON。",
)

tool_registry.register(
    name="retrieve_candidates",
    func=retrieve_candidates_tool,
    description="从光学专利数据库中检索候选镜头结构。",
)

tool_registry.register(
    name="rerank_candidates",
    func=rerank_candidates_tool,
    description="对候选结构进行尺度归一化、多目标优化和重排序。",
)

tool_registry.register(
    name="run_raytrace",
    func=run_raytrace_tool,
    description="对候选结构执行 ray tracing / ray spread 评价。",
)

tool_registry.register(
    name="explain_recommendation",
    func=explain_recommendation_tool,
    description="根据结构化需求、候选结构和评价结果生成推荐解释。",
)

tool_registry.register(
    name="extrapolate_structure",
    func=extrapolate_structure_tool,
    description="基于已有候选结构进行初始结构外推生成。",
)