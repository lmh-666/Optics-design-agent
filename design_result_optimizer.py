# design_result_optimizer.py

import math
import re
from typing import Any, Dict, List, Optional, Tuple


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def _ensure_notes(notes):
    if notes is None:
        return []
    return notes


# ============================================================
# 1. generic 场景兜底归一化
# ============================================================

def normalize_scene_for_generic_wide(
    user_text: str,
    parsed_result: Dict[str, Any],
    completion_notes: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], List[str]]:
    """
    修复没有明确应用场景时的归一化问题。

    例如：
    F数2.0，视场角120度，孔径4mm，要求大视场和短总长

    不应该把 application_scene 识别成整句文本，
    而应该归一化为 generic_wide_angle。
    """
    completion_notes = _ensure_notes(completion_notes)
    text = user_text or ""

    scene = parsed_result.get("application_scene")

    fov_target = _to_float(_safe_get(parsed_result, ["fov", "target"]))
    has_wide_word = any(k in text for k in ["广角", "大视场", "视场大", "视野大", "超广角", "wide"])
    has_compact_word = any(k in text for k in ["短", "薄", "紧凑", "轻巧", "总长"])

    scene_invalid = False

    if scene is None:
        scene_invalid = True
    elif isinstance(scene, str):
        s = scene.strip()
        if s == "":
            scene_invalid = True
        # 如果模型把整句原文塞进 scene，也认为无效
        if len(s) > 20 and ("F数" in s or "视场" in s or "孔径" in s):
            scene_invalid = True
        if s in ["generic", "通用", "unknown", "未知"]:
            scene_invalid = True

    if scene_invalid:
        if has_wide_word or (fov_target is not None and fov_target >= 80):
            parsed_result["application_scene"] = "generic_wide_angle"
            completion_notes.append("场景兜底归一化：识别为 generic_wide_angle")
        elif has_compact_word:
            parsed_result["application_scene"] = "generic_compact"
            completion_notes.append("场景兜底归一化：识别为 generic_compact")
        else:
            parsed_result["application_scene"] = "generic"
            completion_notes.append("场景兜底归一化：识别为 generic")

    return parsed_result, completion_notes


# ============================================================
# 2. 场景真实尺寸约束
# ============================================================

def get_scene_policy(scene: Optional[str], user_text: str = "") -> Dict[str, Any]:
    """
    真实尺寸约束。
    注意：这是工程筛选规则，不是最终光学设计标准。
    """
    scene = scene or ""
    text = user_text or ""

    is_phone = "phone" in scene or "手机" in text
    is_drone = "drone" in scene or "无人机" in text
    is_vehicle = "vehicle" in scene or "车载" in text or "汽车" in text
    is_security = "security" in scene or "安防" in text or "监控" in text or "CCTV" in text
    is_indoor = "indoor" in scene or "室内" in text

    if is_phone:
        return {
            "scene_group": "phone",
            "ttl_ideal": 6.0,
            "ttl_good": 10.0,
            "ttl_acceptable": 18.0,
            "ttl_risk": 35.0,
            "ttl_hard_reject": 45.0,
            "scale_ideal_min": 0.5,
            "scale_ideal_max": 2.0,
            "scale_warn_high": 4.0,
            "scale_risk_high": 6.0,
            "scale_warn_low": 0.25,
            "scale_risk_low": 0.12,
        }

    if is_drone:
        return {
            "scene_group": "drone",
            "ttl_ideal": 10.0,
            "ttl_good": 18.0,
            "ttl_acceptable": 30.0,
            "ttl_risk": 45.0,
            "ttl_hard_reject": 65.0,
            "scale_ideal_min": 0.4,
            "scale_ideal_max": 2.5,
            "scale_warn_high": 5.0,
            "scale_risk_high": 8.0,
            "scale_warn_low": 0.2,
            "scale_risk_low": 0.1,
        }

    if is_vehicle:
        return {
            "scene_group": "vehicle",
            "ttl_ideal": 20.0,
            "ttl_good": 35.0,
            "ttl_acceptable": 60.0,
            "ttl_risk": 90.0,
            "ttl_hard_reject": 130.0,
            "scale_ideal_min": 0.35,
            "scale_ideal_max": 3.0,
            "scale_warn_high": 6.0,
            "scale_risk_high": 10.0,
            "scale_warn_low": 0.15,
            "scale_risk_low": 0.08,
        }

    if is_security or is_indoor:
        return {
            "scene_group": "security_or_indoor",
            "ttl_ideal": 15.0,
            "ttl_good": 25.0,
            "ttl_acceptable": 45.0,
            "ttl_risk": 75.0,
            "ttl_hard_reject": 110.0,
            "scale_ideal_min": 0.35,
            "scale_ideal_max": 3.0,
            "scale_warn_high": 6.0,
            "scale_risk_high": 10.0,
            "scale_warn_low": 0.15,
            "scale_risk_low": 0.08,
        }

    return {
        "scene_group": "generic",
        "ttl_ideal": 15.0,
        "ttl_good": 25.0,
        "ttl_acceptable": 45.0,
        "ttl_risk": 80.0,
        "ttl_hard_reject": 120.0,
        "scale_ideal_min": 0.35,
        "scale_ideal_max": 3.0,
        "scale_warn_high": 6.0,
        "scale_risk_high": 10.0,
        "scale_warn_low": 0.15,
        "scale_risk_low": 0.08,
    }


# ============================================================
# 3. 评分函数
# ============================================================

def score_error_pct(error_pct_abs: Optional[float]) -> float:
    if error_pct_abs is None:
        return 0.5
    e = abs(error_pct_abs)
    if e <= 5:
        return 1.0
    if e <= 10:
        return 0.85
    if e <= 20:
        return 0.65
    if e <= 35:
        return 0.35
    return 0.1


def score_fov_error(error_deg_abs: Optional[float]) -> float:
    if error_deg_abs is None:
        return 0.5
    e = abs(error_deg_abs)
    if e <= 3:
        return 1.0
    if e <= 8:
        return 0.85
    if e <= 15:
        return 0.65
    if e <= 25:
        return 0.4
    return 0.15


def score_ttl(ttl_real: Optional[float], policy: Dict[str, Any]) -> float:
    if ttl_real is None:
        return 0.5

    if ttl_real <= policy["ttl_ideal"]:
        return 1.0
    if ttl_real <= policy["ttl_good"]:
        return 0.85
    if ttl_real <= policy["ttl_acceptable"]:
        return 0.65
    if ttl_real <= policy["ttl_risk"]:
        return 0.25
    return 0.05


def score_scale(scale_factor: Optional[float], policy: Dict[str, Any]) -> float:
    if scale_factor is None:
        return 0.5

    s = abs(scale_factor)

    if policy["scale_ideal_min"] <= s <= policy["scale_ideal_max"]:
        return 1.0

    if 0.25 <= s <= policy["scale_warn_high"]:
        return 0.55

    if policy["scale_risk_low"] <= s <= policy["scale_risk_high"]:
        return 0.25

    return 0.05


def pct_error(value: Optional[float], target: Optional[float]) -> Optional[float]:
    value = _to_float(value)
    target = _to_float(target)

    if value is None or target is None or target == 0:
        return None

    return (value - target) / target * 100.0


def abs_error(value: Optional[float], target: Optional[float]) -> Optional[float]:
    value = _to_float(value)
    target = _to_float(target)

    if value is None or target is None:
        return None

    return value - target


def get_targets(parsed_result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "target_f_number": _to_float(_safe_get(parsed_result, ["f_number", "target"])),
        "target_full_fov": _to_float(_safe_get(parsed_result, ["fov", "target"])),
        "target_aperture": _to_float(_safe_get(parsed_result, ["aperture", "target"])),
        "target_focal_length": _to_float(_safe_get(parsed_result, ["focal_length", "target"])),
    }


# ============================================================
# 4. 单候选风险评价
# ============================================================

def evaluate_candidate_real_constraints(
    candidate: Dict[str, Any],
    parsed_result: Dict[str, Any],
    user_text: str
) -> Dict[str, Any]:
    specs = candidate.get("key_specs") or {}
    lens_data = candidate.get("lens_data") or {}
    scores = candidate.get("scores") or {}

    scene = parsed_result.get("application_scene")
    policy = get_scene_policy(scene, user_text)
    targets = get_targets(parsed_result)

    cand_fnum = _to_float(specs.get("f_number"))
    cand_fov = _to_float(specs.get("full_fov"))
    cand_aperture = _to_float(specs.get("aperture_real_mm") or lens_data.get("aperture_real_mm"))
    cand_ttl_real = _to_float(specs.get("ttl_real_mm") or lens_data.get("ttl_real_mm"))
    cand_scale = _to_float(specs.get("scale_factor") or lens_data.get("scale_factor"))

    target_fnum = targets["target_f_number"]
    target_fov = targets["target_full_fov"]
    target_aperture = targets["target_aperture"]

    fnum_err_pct = pct_error(cand_fnum, target_fnum)
    fov_err_deg = abs_error(cand_fov, target_fov)
    aperture_err_pct = pct_error(cand_aperture, target_aperture)

    fnum_score = score_error_pct(fnum_err_pct)
    fov_score = score_fov_error(fov_err_deg)
    aperture_score = score_error_pct(aperture_err_pct)
    ttl_score = score_ttl(cand_ttl_real, policy)
    scale_score = score_scale(cand_scale, policy)

    original_score = _to_float(candidate.get("score") or scores.get("final_score")) or 0.0
    raytrace_score = _to_float(scores.get("raytrace_score")) or 0.5
    kg_score = _to_float(candidate.get("kg_match_score")) or 0.7

    optimized_score = (
        0.18 * original_score +
        0.18 * raytrace_score +
        0.17 * fov_score +
        0.15 * fnum_score +
        0.16 * ttl_score +
        0.08 * scale_score +
        0.05 * aperture_score +
        0.03 * kg_score
    )

    risks = []
    hard_risks = []

    if cand_ttl_real is not None:
        if cand_ttl_real > policy["ttl_hard_reject"]:
            hard_risks.append(f"真实总长 {cand_ttl_real:.3f}mm 超过 {policy['scene_group']} 场景硬风险阈值")
        elif cand_ttl_real > policy["ttl_risk"]:
            risks.append(f"真实总长 {cand_ttl_real:.3f}mm 偏长")
        elif cand_ttl_real > policy["ttl_acceptable"]:
            risks.append(f"真实总长 {cand_ttl_real:.3f}mm 紧凑性一般")

    if cand_scale is not None:
        if cand_scale >= policy["scale_risk_high"] or cand_scale <= policy["scale_risk_low"]:
            hard_risks.append(f"scale_factor={cand_scale:.3f} 过大/过小，结构缩放风险高")
        elif cand_scale >= policy["scale_warn_high"] or cand_scale <= policy["scale_warn_low"]:
            risks.append(f"scale_factor={cand_scale:.3f} 偏大/偏小，需要检查可缩放性")

    if fov_err_deg is not None:
        if abs(fov_err_deg) > 30:
            hard_risks.append(f"视场角偏差 {fov_err_deg:.1f}° 过大")
        elif abs(fov_err_deg) > 15:
            risks.append(f"视场角偏差 {fov_err_deg:.1f}°，需要重点优化视场")

    if fnum_err_pct is not None:
        if abs(fnum_err_pct) > 35:
            hard_risks.append(f"F数偏差 {fnum_err_pct:.1f}% 过大")
        elif abs(fnum_err_pct) > 15:
            risks.append(f"F数偏差 {fnum_err_pct:.1f}%，通光能力需要优化")

    if aperture_err_pct is not None:
        if abs(aperture_err_pct) > 35:
            hard_risks.append(f"孔径偏差 {aperture_err_pct:.1f}% 过大")
        elif abs(aperture_err_pct) > 15:
            risks.append(f"孔径偏差 {aperture_err_pct:.1f}%，真实口径需要校正")

    # 硬风险封顶，避免特别长的手机镜头继续排在最前
    if hard_risks:
        optimized_score = min(optimized_score, 0.58)

    # 手机等紧凑场景下，TTL极长要额外封顶
    if policy["scene_group"] == "phone" and cand_ttl_real is not None and cand_ttl_real > 35:
        optimized_score = min(optimized_score, 0.55)

    return {
        "scene_group": policy["scene_group"],
        "policy": policy,

        "target_f_number": target_fnum,
        "target_full_fov": target_fov,
        "target_aperture": target_aperture,
        "target_focal_length": targets["target_focal_length"],

        "candidate_f_number": cand_fnum,
        "candidate_full_fov": cand_fov,
        "candidate_aperture_real_mm": cand_aperture,
        "candidate_ttl_real_mm": cand_ttl_real,
        "candidate_scale_factor": cand_scale,

        "f_number_error_pct": round(fnum_err_pct, 4) if fnum_err_pct is not None else None,
        "fov_error_deg": round(fov_err_deg, 4) if fov_err_deg is not None else None,
        "aperture_error_pct": round(aperture_err_pct, 4) if aperture_err_pct is not None else None,

        "fnum_match_score": round(fnum_score, 4),
        "fov_match_score": round(fov_score, 4),
        "aperture_match_score": round(aperture_score, 4),
        "ttl_real_score": round(ttl_score, 4),
        "scale_factor_score": round(scale_score, 4),

        "original_score": round(original_score, 4),
        "raytrace_score": round(raytrace_score, 4),
        "kg_score": round(kg_score, 4),
        "optimized_score": round(optimized_score, 4),

        "risks": risks,
        "hard_risks": hard_risks,
    }


# ============================================================
# 5. 多角色候选：结构最接近 / 尺寸最紧凑 / 综合折中
# ============================================================

def _candidate_brief(candidate: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidate:
        return None

    specs = candidate.get("key_specs") or {}
    real_eval = candidate.get("real_constraint_evaluation") or {}

    return {
        "lens_id": candidate.get("lens_id"),
        "score": candidate.get("score"),
        "optimized_score": real_eval.get("optimized_score"),
        "f_number": specs.get("f_number"),
        "full_fov": specs.get("full_fov"),
        "ttl_real_mm": specs.get("ttl_real_mm"),
        "scale_factor": specs.get("scale_factor"),
        "ray_spread": specs.get("ray_spread"),
        "raytrace_status": specs.get("raytrace_status"),
        "risks": real_eval.get("risks"),
        "hard_risks": real_eval.get("hard_risks"),
    }


def select_candidate_roles(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return {
            "balanced_best": None,
            "structure_best": None,
            "compact_best": None,
        }

    def structure_score(c):
        e = c.get("real_constraint_evaluation") or {}
        return (
            0.35 * (e.get("fov_match_score") or 0) +
            0.30 * (e.get("fnum_match_score") or 0) +
            0.20 * (e.get("scale_factor_score") or 0) +
            0.15 * (e.get("raytrace_score") or 0)
        )

    def compact_score(c):
        e = c.get("real_constraint_evaluation") or {}
        return (
            0.35 * (e.get("ttl_real_score") or 0) +
            0.20 * (e.get("raytrace_score") or 0) +
            0.15 * (e.get("fov_match_score") or 0) +
            0.15 * (e.get("fnum_match_score") or 0) +
            0.15 * (e.get("scale_factor_score") or 0)
        )

    balanced_best = candidates[0]
    structure_best = sorted(candidates, key=structure_score, reverse=True)[0]
    compact_best = sorted(candidates, key=compact_score, reverse=True)[0]

    return {
        "balanced_best": _candidate_brief(balanced_best),
        "structure_best": _candidate_brief(structure_best),
        "compact_best": _candidate_brief(compact_best),
    }


# ============================================================
# 6. 重新排序 + feasibility 重判定
# ============================================================

def optimize_hybrid_result_after_scale(
    hybrid_result: Dict[str, Any],
    parsed_result: Dict[str, Any],
    user_text: str
) -> Dict[str, Any]:
    if not isinstance(hybrid_result, dict):
        return hybrid_result

    top_k = hybrid_result.get("top_k") or []
    if not top_k:
        return hybrid_result

    optimized_candidates = []

    for cand in top_k:
        cand = dict(cand)
        scores = dict(cand.get("scores") or {})

        real_eval = evaluate_candidate_real_constraints(
            cand,
            parsed_result,
            user_text
        )

        scores["pre_optimized_score"] = cand.get("score")
        scores["real_constraint_score"] = real_eval["optimized_score"]

        cand["scores"] = scores
        cand["real_constraint_evaluation"] = real_eval
        cand["score"] = real_eval["optimized_score"]

        optimized_candidates.append(cand)

    optimized_candidates = sorted(
        optimized_candidates,
        key=lambda x: _to_float(x.get("score")) or 0,
        reverse=True
    )

    hybrid_result["top_k"] = optimized_candidates
    hybrid_result["candidate_roles"] = select_candidate_roles(optimized_candidates)

    # 更新 feasibility
    best = optimized_candidates[0]
    best_eval = best.get("real_constraint_evaluation") or {}

    optimized_score = _to_float(best_eval.get("optimized_score")) or 0
    hard_risks = best_eval.get("hard_risks") or []
    risks = best_eval.get("risks") or []

    fov_err = abs(_to_float(best_eval.get("fov_error_deg")) or 0)
    fnum_err = abs(_to_float(best_eval.get("f_number_error_pct")) or 0)
    scale_score = _to_float(best_eval.get("scale_factor_score")) or 0
    ttl_score = _to_float(best_eval.get("ttl_real_score")) or 0

    if hard_risks:
        feasibility_label = "hard_to_meet"
        reason = "当前最佳候选存在硬风险，只能作为参考结构，不能直接作为可用初始方案。"
        adjustments = hard_risks + risks
    elif optimized_score >= 0.82 and fov_err <= 8 and fnum_err <= 10 and scale_score >= 0.55 and ttl_score >= 0.65:
        feasibility_label = "direct_match"
        reason = "该候选在F数、视场角、真实尺寸和光线追迹评价上均较接近，可作为较优初始结构。"
        adjustments = risks
    elif optimized_score >= 0.60:
        feasibility_label = "adjustable"
        reason = "该候选具备作为初始结构的潜力，但仍需要后续参数微调或物理优化。"
        adjustments = risks
    else:
        feasibility_label = "hard_to_meet"
        reason = "当前候选与需求差距较大，建议重新检索或补充更多结构库。"
        adjustments = risks

    hybrid_result["feasibility_result"] = {
        "feasibility": feasibility_label,
        "base_lens": best.get("lens_id"),
        "score": best.get("score"),
        "adjustments": list(dict.fromkeys(adjustments)),
        "reason": reason,
        "base_lens_specs": best.get("key_specs"),
        "real_constraint_evaluation": best_eval,
    }

    return hybrid_result


# ============================================================
# 7. 更严谨推荐说明
# ============================================================

def _fmt_float(x, nd=3):
    v = _to_float(x)
    if v is None:
        return "未知"
    return f"{v:.{nd}f}"


def build_optimized_recommendation(
    hybrid_result: Dict[str, Any],
    parsed_result: Dict[str, Any]
) -> str:
    top_k = hybrid_result.get("top_k") or []
    if not top_k:
        return "当前没有检索到合适的候选镜头方案。"

    feasibility = hybrid_result.get("feasibility_result") or {}
    roles = hybrid_result.get("candidate_roles") or {}

    best = top_k[0]
    best_specs = best.get("key_specs") or {}
    best_eval = best.get("real_constraint_evaluation") or {}

    best_id = best.get("lens_id")
    feasibility_label = feasibility.get("feasibility")

    parts = []

    if feasibility_label == "direct_match":
        parts.append(f"综合推荐 {best_id} 作为初始结构。")
    elif feasibility_label == "adjustable":
        parts.append(f"建议以 {best_id} 作为可微调初始结构。")
    else:
        parts.append(f"当前没有直接满足需求的方案，{best_id} 仅建议作为参考候选。")

    parts.append(
        f"该方案F数约为{_fmt_float(best_specs.get('f_number'))}，"
        f"全视场约为{_fmt_float(best_specs.get('full_fov'), 1)}°，"
        f"尺度适配后总长约为{_fmt_float(best_specs.get('ttl_real_mm'))}mm，"
        f"scale_factor约为{_fmt_float(best_specs.get('scale_factor'))}，"
        f"ray_spread约为{_fmt_float(best_specs.get('ray_spread'), 4)}。"
    )

    risks = feasibility.get("adjustments") or []
    if risks:
        parts.append("主要风险：" + "；".join(risks) + "。")

    structure_best = roles.get("structure_best")
    compact_best = roles.get("compact_best")
    balanced_best = roles.get("balanced_best")

    if structure_best and compact_best:
        sid = structure_best.get("lens_id")
        cid = compact_best.get("lens_id")
        bid = balanced_best.get("lens_id") if balanced_best else best_id

        if sid != cid:
            parts.append(
                f"从候选角色看，结构参数更接近的是 {sid}，"
                f"真实尺寸更紧凑的是 {cid}，"
                f"综合折中排序最高的是 {bid}。"
            )

    parts.append("该结果属于初始结构推荐与筛选结果，仍需后续完整光学仿真和专家复核。")

    return "".join(parts)