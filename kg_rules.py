from typing import Dict, Any, List, Optional, Tuple


# ============================================================
# 轻量光学知识图谱：
# scene -> lens_type -> constraints -> explanation
# ============================================================

LIGHTWEIGHT_OPTICS_KG = {
    "phone_wide_angle": {
        "name": "手机广角镜头",
        "lens_type": "wide_angle",
        "fov_range": [80, 120],
        "preferred_f_number_range": [1.4, 2.4],
        "preferred_ttl": "short",
        "preferred_element_count": "few",
        "avoid": [
            "narrow_angle",
            "large_f_number",
            "very_long_ttl"
        ],
        "design_focus": [
            "紧凑结构",
            "较大视场",
            "较小F数",
            "较低结构复杂度"
        ],
        "default_constraints": {
            "fov": {
                "target": 90.0,
                "preference": "wide",
                "constraint_type": "soft"
            },
            "f_number": {
                "target": 2.0,
                "preference": "small",
                "constraint_type": "soft"
            },
            "total_length": {
                "target": None,
                "preference": "short",
                "constraint_type": "soft"
            },
            "element_count": {
                "target": None,
                "preference": "few",
                "constraint_type": "soft"
            }
        },
        "explanation": "手机广角镜头通常要求较大视场、较小F数和较短总长，因此应优先筛选广角结构，并避免窄视场或总长过大的候选。"
    },

    "drone_wide_angle": {
        "name": "无人机广角镜头",
        "lens_type": "wide_angle",
        "fov_range": [100, 140],
        "preferred_f_number_range": [2.0, 3.5],
        "preferred_ttl": "short",
        "preferred_element_count": "few",
        "avoid": [
            "narrow_angle",
            "very_long_ttl",
            "large_f_number"
        ],
        "design_focus": [
            "大视场",
            "轻量化",
            "短总长",
            "结构简单"
        ],
        "default_constraints": {
            "fov": {
                "target": 120.0,
                "preference": "wide",
                "constraint_type": "soft"
            },
            "f_number": {
                "target": 2.8,
                "preference": "small",
                "constraint_type": "soft"
            },
            "total_length": {
                "target": None,
                "preference": "short",
                "constraint_type": "soft"
            },
            "element_count": {
                "target": None,
                "preference": "few",
                "constraint_type": "soft"
            }
        },
        "explanation": "无人机镜头通常重视大视场、轻量化和结构紧凑性，因此应优先选择广角且总长较短的候选结构。"
    },

    "car_wide_angle": {
        "name": "车载广角镜头",
        "lens_type": "wide_angle",
        "fov_range": [100, 160],
        "preferred_f_number_range": [1.8, 2.8],
        "preferred_ttl": "medium",
        "preferred_element_count": None,
        "avoid": [
            "narrow_angle",
            "large_f_number"
        ],
        "design_focus": [
            "大视场",
            "低畸变",
            "环境稳定性",
            "夜间成像能力"
        ],
        "default_constraints": {
            "fov": {
                "target": 120.0,
                "preference": "wide",
                "constraint_type": "soft"
            },
            "f_number": {
                "target": 2.8,
                "preference": "small",
                "constraint_type": "soft"
            },
            "distortion": {
                "target": None,
                "preference": "low",
                "constraint_type": "soft"
            }
        },
        "explanation": "车载镜头通常强调大视场和低畸变，同时需要较好的低照度表现，因此候选结构应优先满足广角和较小F数要求。"
    },

    "indoor_wide_angle": {
        "name": "室内广角镜头",
        "lens_type": "wide_angle",
        "fov_range": [80, 120],
        "preferred_f_number_range": [1.8, 2.8],
        "preferred_ttl": "medium",
        "preferred_element_count": None,
        "avoid": [
            "narrow_angle",
            "large_f_number"
        ],
        "design_focus": [
            "广角覆盖",
            "较好通光能力",
            "中等结构复杂度"
        ],
        "default_constraints": {
            "fov": {
                "target": 90.0,
                "preference": "wide",
                "constraint_type": "soft"
            },
            "f_number": {
                "target": 2.0,
                "preference": "small",
                "constraint_type": "soft"
            }
        },
        "explanation": "室内镜头通常需要较大视场和较好通光能力，因此应避免视场过窄或F数过大的候选。"
    },

    "security": {
        "name": "安防监控镜头",
        "lens_type": "normal_or_wide_angle",
        "fov_range": [60, 120],
        "preferred_f_number_range": [1.6, 2.8],
        "preferred_ttl": "medium",
        "preferred_element_count": None,
        "avoid": [
            "very_large_f_number"
        ],
        "design_focus": [
            "低照度表现",
            "成像稳定性",
            "成本控制"
        ],
        "default_constraints": {
            "f_number": {
                "target": 2.0,
                "preference": "small",
                "constraint_type": "soft"
            },
            "low_light_performance": {
                "target": None,
                "preference": "good",
                "constraint_type": "soft"
            }
        },
        "explanation": "安防监控镜头通常重视低照度成像和稳定性，因此应优先避免F数过大的暗光结构。"
    }
}


# ============================================================
# 轻量图谱工具函数
# ============================================================

def _ensure_field(parsed_result: Dict[str, Any], field: str) -> None:
    if field not in parsed_result or not isinstance(parsed_result.get(field), dict):
        parsed_result[field] = {
            "target": None,
            "preference": None,
            "constraint_type": None
        }


def get_scene_kg(scene: Optional[str]) -> Optional[Dict[str, Any]]:
    if scene is None:
        return None
    return LIGHTWEIGHT_OPTICS_KG.get(scene)


def get_scene_lens_type(scene: Optional[str]) -> Optional[str]:
    kg = get_scene_kg(scene)
    if not kg:
        return None
    return kg.get("lens_type")


def enhance_requirement_with_kg(
    user_text: str,
    parsed_result: Dict[str, Any],
    notes: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    """
    用轻量知识图谱增强 parsed_result。

    作用：
    1. 根据 application_scene 自动补全缺失约束；
    2. 给 parsed_result 增加 _kg 字段，记录图谱约束；
    3. 返回 notes，方便接口输出解释。
    """
    if notes is None:
        notes = []

    scene = parsed_result.get("application_scene")
    kg = get_scene_kg(scene)

    if kg is None:
        return parsed_result, notes, {
            "kg_used": False,
            "scene": scene,
            "message": "未匹配到场景知识图谱"
        }

    default_constraints = kg.get("default_constraints", {})

    for field, constraint in default_constraints.items():
        _ensure_field(parsed_result, field)

        cur_target = parsed_result[field].get("target")
        cur_pref = parsed_result[field].get("preference")

        # 只有用户没有明确给出时，图谱才补全
        if cur_target is None and cur_pref is None:
            parsed_result[field] = constraint.copy()
            notes.append(f"知识图谱补全：{scene} -> {field}")

    parsed_result["_kg"] = {
        "kg_used": True,
        "scene": scene,
        "scene_name": kg.get("name"),
        "lens_type": kg.get("lens_type"),
        "fov_range": kg.get("fov_range"),
        "preferred_f_number_range": kg.get("preferred_f_number_range"),
        "preferred_ttl": kg.get("preferred_ttl"),
        "preferred_element_count": kg.get("preferred_element_count"),
        "avoid": kg.get("avoid"),
        "design_focus": kg.get("design_focus"),
        "explanation": kg.get("explanation")
    }

    kg_info = parsed_result["_kg"]

    return parsed_result, notes, kg_info


def check_candidate_against_kg(
    candidate: Dict[str, Any],
    parsed_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    判断候选镜头与知识图谱约束是否一致。
    不直接否决，只给出解释和风险。
    """
    kg_info = parsed_result.get("_kg") or {}
    if not kg_info.get("kg_used"):
        return {
            "kg_checked": False,
            "kg_match_score": None,
            "kg_risks": [],
            "kg_reason": "未使用知识图谱约束"
        }

    specs = candidate.get("key_specs") or {}
    f_number = specs.get("f_number")
    full_fov = specs.get("full_fov")
    ttl = specs.get("ttl_real_mm") or specs.get("total_length")
    lens_type = candidate.get("lens_type")

    risks = []
    score_parts = []

    # lens_type
    target_type = kg_info.get("lens_type")
    if target_type == "wide_angle":
        if lens_type == "wide_angle":
            score_parts.append(1.0)
        else:
            score_parts.append(0.0)
            risks.append("候选结构类型不是广角结构")

    # FOV range
    fov_range = kg_info.get("fov_range")
    if fov_range and full_fov is not None:
        low, high = fov_range
        if low <= full_fov <= high:
            score_parts.append(1.0)
        elif full_fov < low:
            score_parts.append(0.4)
            risks.append(f"候选全视场 {full_fov:.1f}° 低于该场景推荐范围 {low}-{high}°")
        else:
            score_parts.append(0.7)
            risks.append(f"候选全视场 {full_fov:.1f}° 高于该场景推荐范围 {low}-{high}°，可能接近鱼眼结构")

    # F-number range
    fnum_range = kg_info.get("preferred_f_number_range")
    if fnum_range and f_number is not None:
        low, high = fnum_range
        if low <= f_number <= high:
            score_parts.append(1.0)
        elif f_number > high:
            score_parts.append(0.4)
            risks.append(f"候选F数 {f_number:.3f} 偏大，通光能力可能不足")
        else:
            score_parts.append(0.8)
            risks.append(f"候选F数 {f_number:.3f} 小于推荐范围，可能增加结构复杂度")

    # TTL
    preferred_ttl = kg_info.get("preferred_ttl")
    if preferred_ttl == "short" and ttl is not None:
        if ttl <= 10:
            score_parts.append(1.0)
        elif ttl <= 30:
            score_parts.append(0.6)
            risks.append(f"候选总长约 {ttl:.3f}mm，紧凑性一般")
        else:
            score_parts.append(0.2)
            risks.append(f"候选总长约 {ttl:.3f}mm，可能不适合紧凑场景")

    if score_parts:
        kg_match_score = round(sum(score_parts) / len(score_parts), 4)
    else:
        kg_match_score = None

    if not risks:
        reason = "候选与当前场景知识约束基本一致"
    else:
        reason = "；".join(risks)

    return {
        "kg_checked": True,
        "kg_match_score": kg_match_score,
        "kg_risks": risks,
        "kg_reason": reason
    }


def enrich_candidates_with_kg(
    candidates: List[Dict[str, Any]],
    parsed_result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    给 top_k 候选添加 kg_match_score / kg_risks。
    """
    enriched = []
    for cand in candidates:
        cand = dict(cand)
        kg_eval = check_candidate_against_kg(cand, parsed_result)
        cand["kg_evaluation"] = kg_eval

        # 可选：把 kg_match_score 加入总分展示，不强行改 rerank
        if kg_eval.get("kg_match_score") is not None:
            cand["kg_match_score"] = kg_eval["kg_match_score"]
        else:
            cand["kg_match_score"] = None

        enriched.append(cand)

    return enriched


def build_kg_explanation(parsed_result: Dict[str, Any], best_candidate: Optional[Dict[str, Any]]) -> str:
    kg_info = parsed_result.get("_kg") or {}

    if not kg_info.get("kg_used"):
        return "当前未使用场景知识图谱约束。"

    scene_name = kg_info.get("scene_name")
    lens_type = kg_info.get("lens_type")
    fov_range = kg_info.get("fov_range")
    fnum_range = kg_info.get("preferred_f_number_range")
    design_focus = kg_info.get("design_focus") or []

    base = (
        f"知识图谱判断当前需求属于“{scene_name}”，"
        f"对应推荐结构类型为 {lens_type}，"
        f"建议视场范围约为 {fov_range}°，"
        f"建议F数范围约为 {fnum_range}。"
    )

    if design_focus:
        base += " 设计重点包括：" + "、".join(design_focus) + "。"

    if best_candidate is not None:
        kg_eval = best_candidate.get("kg_evaluation") or {}
        if kg_eval.get("kg_checked"):
            base += " 当前候选的图谱一致性判断为：" + kg_eval.get("kg_reason", "")

    return base