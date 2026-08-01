# aperture_scale_utils.py

import re
import math
from typing import Any, Dict, List, Optional, Tuple


def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str):
            x = x.strip()
            if x == "" or x.lower() in ["none", "null", "nan"]:
                return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _ensure_optical_field(parsed_result: Dict[str, Any], field: str) -> None:
    if field not in parsed_result or not isinstance(parsed_result.get(field), dict):
        parsed_result[field] = {
            "target": None,
            "preference": None,
            "constraint_type": None
        }


def parse_aperture_from_text(text: str) -> Optional[float]:
    """
    从中文/英文输入中解析孔径/入瞳直径/EPD。

    支持：
    - 孔径4mm
    - 孔径 4 mm
    - 入瞳直径4mm
    - 入口瞳孔直径 4mm
    - EPD 4mm
    - aperture 4mm
    - 口径4毫米
    """
    if not text:
        return None

    patterns = [
        r"(?:孔径|口径|入瞳直径|入口瞳孔直径|入射瞳孔直径|EPD|epd|aperture|Aperture)\s*(?:为|是|=|约|大约|左右|:|：)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:mm|毫米)?",
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:mm|毫米)\s*(?:孔径|口径|入瞳|EPD|epd)",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            return _to_float(m.group(1))

    return None


def enhance_requirement_with_aperture_scale(
    user_text: str,
    parsed_result: Dict[str, Any],
    notes: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    """
    增强 parsed_result：
    1. 从原始文本中解析 aperture / EPD；
    2. 如果已知 F# 和 aperture，则反推 target_focal_length = F# × EPD；
    3. 将反推焦距写入 parsed_result["focal_length"]["target"]，
       使后续已有 scale 逻辑自动生效。
    """
    if notes is None:
        notes = []

    _ensure_optical_field(parsed_result, "f_number")
    _ensure_optical_field(parsed_result, "focal_length")

    aperture_target = parse_aperture_from_text(user_text)

    if "aperture" not in parsed_result or not isinstance(parsed_result.get("aperture"), dict):
        parsed_result["aperture"] = {
            "target": None,
            "preference": None,
            "constraint_type": None,
            "type": "entrance_pupil_diameter"
        }

    if aperture_target is not None:
        parsed_result["aperture"]["target"] = aperture_target
        parsed_result["aperture"]["type"] = "entrance_pupil_diameter"
        parsed_result["aperture"]["constraint_type"] = "hard"
        notes.append(f"识别到孔径/EPD={aperture_target}mm")

    f_number_target = _to_float((parsed_result.get("f_number") or {}).get("target"))
    focal_target = _to_float((parsed_result.get("focal_length") or {}).get("target"))

    derived_focal = None
    scale_status = "not_available"
    scale_note = "未提供孔径/EPD或F数，无法反推目标焦距"

    if aperture_target is not None and f_number_target is not None:
        derived_focal = round(f_number_target * aperture_target, 6)

        if focal_target is None:
            parsed_result["focal_length"]["target"] = derived_focal
            parsed_result["focal_length"]["constraint_type"] = "derived"
            parsed_result["focal_length"]["source"] = "derived_from_f_number_and_aperture"
            notes.append(
                f"根据 F#={f_number_target} 与 EPD={aperture_target}mm 反推目标焦距={derived_focal}mm"
            )
            scale_status = "derived_focal_length"
            scale_note = (
                f"已由 F#×EPD 反推目标焦距：{f_number_target}×{aperture_target}={derived_focal}mm，"
                "后续候选将按该焦距进行尺度适配"
            )
        else:
            scale_status = "explicit_focal_length_exists"
            scale_note = (
                f"用户已提供焦距={focal_target}mm，同时提供 EPD={aperture_target}mm；"
                "优先使用显式焦距进行尺度适配"
            )

    scale_info = {
        "aperture_used": aperture_target is not None,
        "aperture_target_mm": aperture_target,
        "f_number_target": f_number_target,
        "derived_focal_length_mm": derived_focal,
        "scale_status": scale_status,
        "scale_note": scale_note
    }

    parsed_result["_scale_info"] = scale_info

    return parsed_result, notes, scale_info


def get_target_focal_from_parsed(parsed_result: Dict[str, Any]) -> Optional[float]:
    return _to_float((parsed_result.get("focal_length") or {}).get("target"))


def get_target_aperture_from_parsed(parsed_result: Dict[str, Any]) -> Optional[float]:
    return _to_float((parsed_result.get("aperture") or {}).get("target"))


def recompute_candidate_scale(
    candidate: Dict[str, Any],
    target_focal_length: Optional[float],
    target_aperture: Optional[float]
) -> Dict[str, Any]:
    """
    对单个候选重新计算真实尺度字段。
    如果已有 pipeline 已经 scale，也不会破坏；如果没 scale，则补上。
    """
    cand = dict(candidate)
    specs = dict(cand.get("key_specs") or {})
    lens_data = dict(cand.get("lens_data") or {})

    base_focal = _to_float(
        specs.get("focal_length")
        or lens_data.get("focal_length")
        or lens_data.get("focal_length_real_mm")
    )

    base_ttl = _to_float(
        specs.get("total_length")
        or lens_data.get("total_length")
        or lens_data.get("ttl")
    )

    base_aperture = _to_float(
        lens_data.get("aperture_norm")
        or lens_data.get("aperture_real_mm")
    )

    old_scale = _to_float(
        specs.get("scale_factor")
        or lens_data.get("scale_factor")
    )

    scale_factor = old_scale
    scale_basis = None

    if target_focal_length is not None and base_focal is not None and base_focal > 0:
        scale_factor = target_focal_length / base_focal
        scale_basis = "target_focal_length"
    elif target_aperture is not None and base_aperture is not None and base_aperture > 0:
        scale_factor = target_aperture / base_aperture
        scale_basis = "target_aperture"
    else:
        scale_factor = None
        scale_basis = None

    if scale_factor is not None:
        focal_real = base_focal * scale_factor if base_focal is not None else None
        ttl_real = base_ttl * scale_factor if base_ttl is not None else None
        aperture_real = base_aperture * scale_factor if base_aperture is not None else target_aperture

        specs["scale_factor"] = round(scale_factor, 6)
        specs["focal_length_real_mm"] = round(focal_real, 6) if focal_real is not None else None
        specs["ttl_real_mm"] = round(ttl_real, 6) if ttl_real is not None else None
        specs["aperture_real_mm"] = round(aperture_real, 6) if aperture_real is not None else None

        lens_data["scale_factor"] = specs["scale_factor"]
        lens_data["focal_length_real_mm"] = specs["focal_length_real_mm"]
        lens_data["ttl_real_mm"] = specs["ttl_real_mm"]
        lens_data["aperture_real_mm"] = specs["aperture_real_mm"]
        lens_data["scale_status"] = "applied"
        lens_data["scale_basis"] = scale_basis
        lens_data["scale_note"] = (
            f"已按 {scale_basis} 进行尺度适配，scale_factor={specs['scale_factor']}"
        )

        cand["scale_note"] = lens_data["scale_note"]

    else:
        lens_data["scale_status"] = lens_data.get("scale_status") or "missing_scale_target"
        lens_data["scale_note"] = lens_data.get("scale_note") or "未提供焦距或孔径，无法进行真实尺度适配"
        cand["scale_note"] = lens_data["scale_note"]

    cand["key_specs"] = specs
    cand["lens_data"] = lens_data

    return cand


def apply_scale_to_candidate_list(
    candidates: List[Dict[str, Any]],
    parsed_result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    target_focal = get_target_focal_from_parsed(parsed_result)
    target_aperture = get_target_aperture_from_parsed(parsed_result)

    return [
        recompute_candidate_scale(c, target_focal, target_aperture)
        for c in candidates
    ]


def apply_scale_to_hybrid_result(
    hybrid_result: Dict[str, Any],
    parsed_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    对 hybrid_result 中 top_k 和 buckets 补充真实尺度。
    """
    if not isinstance(hybrid_result, dict):
        return hybrid_result

    if isinstance(hybrid_result.get("top_k"), list):
        hybrid_result["top_k"] = apply_scale_to_candidate_list(
            hybrid_result["top_k"],
            parsed_result
        )

    buckets = hybrid_result.get("buckets")
    if isinstance(buckets, dict):
        for bucket_name, items in list(buckets.items()):
            if isinstance(items, list):
                buckets[bucket_name] = apply_scale_to_candidate_list(
                    items,
                    parsed_result
                )
        hybrid_result["buckets"] = buckets

    return hybrid_result


def tighten_feasibility_by_recall_and_scale(
    hybrid_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    根据测试结果收紧 feasibility：
    - strict 可以 direct_match；
    - relaxed_15_12 可以 direct_match 或 adjustable；
    - relaxed_25_18 / relaxed_40_25 默认不能 direct_match；
    - 如果 KG 有风险，也不能 direct_match；
    - 如果目标明确但 scale_factor 为空，也不能 direct_match。
    """
    if not isinstance(hybrid_result, dict):
        return hybrid_result

    feasibility = hybrid_result.get("feasibility_result") or {}
    if not feasibility:
        return hybrid_result

    if feasibility.get("feasibility") != "direct_match":
        return hybrid_result

    recall_mode = hybrid_result.get("recall_mode") or ""
    top_k = hybrid_result.get("top_k") or []
    best = top_k[0] if top_k else {}
    specs = best.get("key_specs") or {}
    kg_eval = best.get("kg_evaluation") or {}

    downgrade_reasons = []

    if "relaxed_25_18" in recall_mode or "relaxed_40_25" in recall_mode:
        downgrade_reasons.append(f"召回模式为 {recall_mode}，属于较大范围放宽召回")

    kg_risks = kg_eval.get("kg_risks") or []
    if kg_risks:
        downgrade_reasons.extend(kg_risks)

    scale_factor = specs.get("scale_factor")
    scale_status = ((best.get("lens_data") or {}).get("scale_status"))
    if scale_status in ["missing_target_focal_length", "missing_scale_target"]:
        downgrade_reasons.append("候选尚未完成真实尺度适配")

    if downgrade_reasons:
        feasibility["feasibility"] = "adjustable"
        feasibility["adjustments"] = list(dict.fromkeys(downgrade_reasons))
        feasibility["reason"] = (
            "该候选具备作为初始结构的潜力，但当前结果来自放宽召回、知识约束风险或真实尺度尚未完全确定，"
            "因此建议作为可微调方案，而不是直接匹配方案。"
        )
        hybrid_result["feasibility_result"] = feasibility

    return hybrid_result


def build_scale_aware_recommendation(hybrid_result: Dict[str, Any]) -> str:
    """
    替代原 build_hybrid_recommendation 的更严谨推荐文字：
    - scale_factor 存在时，才说“尺度适配后总长”；
    - scale_factor 为空时，说“当前结构总长”；
    - feasibility 为 adjustable 时，明确是基础方案继续微调。
    """
    top_k = hybrid_result.get("top_k") or []
    feasibility = hybrid_result.get("feasibility_result") or {}

    if not top_k:
        return "当前没有检索到合适的候选镜头方案。"

    best = top_k[0]
    specs = best.get("key_specs") or {}
    lens_data = best.get("lens_data") or {}

    lens_id = best.get("lens_id")
    f_number = _to_float(specs.get("f_number"))
    full_fov = _to_float(specs.get("full_fov"))
    ttl_real = _to_float(specs.get("ttl_real_mm"))
    ttl_raw = _to_float(specs.get("total_length"))
    ray_spread = _to_float(specs.get("ray_spread"))
    scale_factor = _to_float(specs.get("scale_factor") or lens_data.get("scale_factor"))

    feasibility_label = feasibility.get("feasibility")
    adjustments = feasibility.get("adjustments") or []

    if feasibility_label == "direct_match":
        prefix = f"推荐方案：{lens_id}。"
    elif feasibility_label == "adjustable":
        prefix = f"建议以 {lens_id} 作为基础方案继续微调。"
    else:
        prefix = f"当前未找到可直接满足需求的方案，{lens_id} 可作为参考候选。"

    parts = [prefix]

    spec_text = []
    if f_number is not None:
        spec_text.append(f"F数约为{f_number:.3f}")
    if full_fov is not None:
        spec_text.append(f"全视场约为{full_fov:.1f}°")

    if scale_factor is not None and ttl_real is not None:
        spec_text.append(f"尺度适配后总长约为{ttl_real:.3f}mm")
    elif ttl_raw is not None:
        spec_text.append(f"当前结构总长约为{ttl_raw:.3f}mm，尚未完成真实尺度适配")

    if ray_spread is not None:
        spec_text.append(f"ray_spread约为{ray_spread:.4f}")

    if spec_text:
        parts.append("该方案" + "，".join(spec_text) + "。")

    if adjustments:
        parts.append("主要调整方向：" + "；".join(adjustments) + "。")
    elif feasibility_label == "direct_match":
        parts.append("该候选在F数、视场角和当前评价指标上匹配度较高，可作为起始结构。")
    else:
        parts.append("建议后续继续检查F数、视场角、总长和像质指标。")

    parts.append("该结果属于候选结构推荐，仍需后续完整光学仿真验证。")

    return "".join(parts)