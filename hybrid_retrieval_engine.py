from __future__ import annotations

import os
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from realtime_raytrace_engine import add_realtime_raytrace_to_topk


# ============================================================
# Hybrid retrieval engine
# 目标：把结构化 recall + scale adaptation + Top9 多样性选择 + raytrace 重排
# 接入现有 FastAPI app.py。
# ============================================================


RAYTRACE_RESULT_PATH = os.getenv("RAYTRACE_RESULT_PATH", "data/eval/step10_top9_raytrace_rerank.xlsx")
LAYOUT_STATIC_PREFIX = os.getenv("LAYOUT_STATIC_PREFIX", "/static/lens_layouts")


# ============================================================
# 基础工具
# ============================================================
def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    v = _safe_float(x, None)
    if v is None:
        return default
    return int(round(v))


def _get_field(parsed_result: Dict[str, Any], name: str) -> Dict[str, Any]:
    v = parsed_result.get(name)
    return v if isinstance(v, dict) else {"target": None, "preference": None, "constraint_type": None}


def _get_target(parsed_result: Dict[str, Any], name: str) -> Optional[float]:
    return _safe_float(_get_field(parsed_result, name).get("target"))


def _get_pref(parsed_result: Dict[str, Any], name: str) -> Optional[str]:
    v = _get_field(parsed_result, name).get("preference")
    if v is None:
        return None
    return str(v).strip().lower()


def _normalize_pref(pref: Optional[str], field_name: Optional[str] = None) -> Optional[str]:
    if pref is None:
        return None
    p = str(pref).strip().lower()

    if field_name == "f_number":
        if p in ["small", "小", "小一点", "bright", "大光圈", "低照度好", "亮", "亮一点"]:
            return "small"
    if field_name == "fov":
        if p in ["wide", "large", "大", "大一点", "广角", "大视场", "视野大", "视场角大"]:
            return "wide"
    if field_name == "total_length":
        if p in ["short", "small", "短", "短一点", "轻巧", "轻便", "紧凑", "便携", "薄", "薄一点"]:
            return "short"
    if field_name == "element_count":
        if p in ["few", "少", "少一点", "镜片少", "片数少"]:
            return "few"
    if field_name == "distortion":
        if p in ["low", "低", "低畸变", "畸变小"]:
            return "low"
    if field_name == "low_light_performance":
        if p in ["good", "bright", "excellent", "亮", "低照度好", "暗光好"]:
            return "good"

    return p


def _normalize_scene(scene: Any) -> Optional[str]:
    if scene is None:
        return None
    s = str(scene).strip().lower()
    if not s:
        return None
    if any(k in s for k in ["手机", "phone", "mobile"]):
        return "phone_wide_angle"
    if any(k in s for k in ["无人机", "drone"]):
        return "drone_wide_angle"
    if any(k in s for k in ["车载", "汽车", "vehicle", "car"]):
        return "car_wide_angle"
    if any(k in s for k in ["室内", "indoor"]):
        return "indoor_wide_angle"
    if any(k in s for k in ["安防", "监控", "cctv", "security"]):
        return "security"
    return s


def _relative_error(value: Optional[float], target: Optional[float]) -> Optional[float]:
    if value is None or target is None or target == 0:
        return None
    return abs(value - target) / abs(target)


# ============================================================
# 需求规格转换
# ============================================================
def build_normalized_requirement(parsed_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    把当前 app.py 的 parsed_result 转成适合结构化 recall / rank 的 normalized_req。
    注意：光学结构匹配以 F数 + 半视场为核心；焦距进入 scale 阶段。
    """
    fnum_target = _get_target(parsed_result, "f_number")
    fnum_pref = _normalize_pref(_get_pref(parsed_result, "f_number"), "f_number")

    fov_target_full = _get_target(parsed_result, "fov")
    fov_pref = _normalize_pref(_get_pref(parsed_result, "fov"), "fov")

    focal_target = _get_target(parsed_result, "focal_length")
    ttl_target = _get_target(parsed_result, "total_length")
    ttl_pref = _normalize_pref(_get_pref(parsed_result, "total_length"), "total_length")
    elem_pref = _normalize_pref(_get_pref(parsed_result, "element_count"), "element_count")
    dist_pref = _normalize_pref(_get_pref(parsed_result, "distortion"), "distortion")
    low_pref = _normalize_pref(_get_pref(parsed_result, "low_light_performance"), "low_light_performance")

    scene = _normalize_scene(parsed_result.get("application_scene"))

    # 场景先验：只在缺核心参数时补，不覆盖用户明确给出的目标
    scene_prior = {
        "phone_wide_angle": {"full_fov": 90.0, "f_number": 2.0, "ttl_pref": "short"},
        "drone_wide_angle": {"full_fov": 120.0, "f_number": 2.8, "ttl_pref": "short"},
        "car_wide_angle": {"full_fov": 120.0, "f_number": 2.8},
        "indoor_wide_angle": {"full_fov": 90.0, "f_number": 2.0},
        "security": {"full_fov": 70.0, "f_number": 2.0},
    }

    prior = scene_prior.get(scene, {})
    auto_filled: List[str] = []

    if fov_target_full is None and fov_pref is None and "full_fov" in prior:
        fov_target_full = prior["full_fov"]
        fov_pref = "wide" if fov_target_full >= 90 else None
        auto_filled.append("fov_by_scene_prior")

    if fnum_target is None and fnum_pref is None and "f_number" in prior:
        fnum_target = prior["f_number"]
        fnum_pref = "small"
        auto_filled.append("f_number_by_scene_prior")

    if ttl_target is None and ttl_pref is None and prior.get("ttl_pref"):
        ttl_pref = prior["ttl_pref"]
        auto_filled.append("total_length_by_scene_prior")

    # 如果只有 wide，没有数值，给一个合理默认值，避免只有偏好导致误召回过宽。
    if fov_target_full is None and fov_pref == "wide":
        fov_target_full = 100.0
        auto_filled.append("fov_default_for_wide")

    # 如果只有 small，没有数值，给一个弱目标，避免选到 F8 这种暗镜头。
    if fnum_target is None and fnum_pref == "small":
        fnum_target = 2.8
        auto_filled.append("f_number_default_for_small")

    half_fov_target = fov_target_full / 2.0 if fov_target_full is not None else None

    return {
        "application_scene": scene,
        "f_number": {"target": fnum_target, "preference": fnum_pref},
        "half_fov_deg": {"target": half_fov_target, "source_full_fov": fov_target_full, "preference": fov_pref},
        "focal_length": {"target": focal_target},
        "total_length": {"target": ttl_target, "preference": ttl_pref},
        "element_count": {"preference": elem_pref},
        "distortion": {"preference": dist_pref},
        "low_light_performance": {"preference": low_pref},
        "auto_filled": auto_filled,
    }


# ============================================================
# 数据表转换
# ============================================================
def lens_database_to_df(lens_database: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(lens_database).copy()
    if df.empty:
        return df

    # 兼容当前 app.py 的字段名与旧工程字段名
    rename_map = {
        "total_length": "ttl",
        "full_fov": "full_fov",
        "half_fov": "half_fov",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "lens_id" not in df.columns:
        for c in ["文件名", "file_name", "name"]:
            if c in df.columns:
                df["lens_id"] = df[c]
                break

    if "half_fov" not in df.columns and "full_fov" in df.columns:
        df["half_fov"] = pd.to_numeric(df["full_fov"], errors="coerce") / 2.0
    if "full_fov" not in df.columns and "half_fov" in df.columns:
        df["full_fov"] = pd.to_numeric(df["half_fov"], errors="coerce") * 2.0

    numeric_cols = ["f_number", "half_fov", "full_fov", "ttl", "focal_length", "element_count", "distortion"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 保留原始行，方便输出 lens_data
    return df


def basic_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    required = ["lens_id", "f_number", "half_fov", "full_fov", "ttl", "focal_length"]
    for c in required:
        if c not in df.columns:
            df[c] = None
    out = df[
        (df["f_number"].notna()) & (df["f_number"] > 0) & (df["f_number"] <= 50) &
        (df["half_fov"].notna()) & (df["half_fov"] > 0) & (df["half_fov"] <= 180) &
        (df["ttl"].notna()) & (df["ttl"] > 0) & (df["ttl"] <= 1000) &
        (df["focal_length"].notna()) & (df["focal_length"] > 0) & (df["focal_length"] <= 1000)
    ].copy()
    return out.reset_index(drop=True)


# ============================================================
# 镜头类型与结构过滤
# ============================================================
def infer_lens_type_from_requirement(normalized_req: Dict[str, Any]) -> Optional[str]:
    scene = normalized_req.get("application_scene")
    half_fov = normalized_req.get("half_fov_deg", {}).get("target")
    pref = normalized_req.get("half_fov_deg", {}).get("preference")

    if scene in ["phone_wide_angle", "drone_wide_angle", "car_wide_angle", "indoor_wide_angle"]:
        return "wide_angle"
    if pref == "wide":
        return "wide_angle"
    if half_fov is not None:
        full = half_fov * 2
        if full >= 90:
            return "wide_angle"
        if full <= 30:
            return "narrow_angle"
        return "normal_angle"
    return None


def infer_lens_type_from_lens(row: pd.Series) -> Optional[str]:
    full_fov = _safe_float(row.get("full_fov"))
    if full_fov is None:
        return None
    if full_fov >= 90:
        return "wide_angle"
    if full_fov <= 30:
        return "narrow_angle"
    return "normal_angle"


def structure_filter(df: pd.DataFrame, normalized_req: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    if df.empty:
        return df.copy(), ["候选库为空"]

    out = df.copy()
    req_type = infer_lens_type_from_requirement(normalized_req)
    fnum_target = normalized_req.get("f_number", {}).get("target")
    fnum_pref = normalized_req.get("f_number", {}).get("preference")

    out["lens_type"] = out.apply(infer_lens_type_from_lens, axis=1)

    if req_type == "wide_angle":
        before = len(out)
        out = out[out["lens_type"] == "wide_angle"].copy()
        notes.append(f"结构过滤：广角需求，仅保留 full_fov>=90° 候选，{before}->{len(out)}")

    if req_type == "narrow_angle":
        before = len(out)
        out = out[out["lens_type"] == "narrow_angle"].copy()
        notes.append(f"结构过滤：窄视场需求，仅保留 full_fov<=30° 候选，{before}->{len(out)}")

    # 小 F 数 / 场景广角时，过滤极暗镜头，避免 FOV 大但 F8 的鱼眼被选中
    if fnum_pref == "small" or fnum_target is not None:
        before = len(out)
        max_f = max(4.0, (fnum_target or 2.8) * 1.8)
        out = out[out["f_number"] <= max_f].copy()
        notes.append(f"F数结构过滤：剔除 F数>{max_f:.2f} 的暗镜头，{before}->{len(out)}")

    return out.reset_index(drop=True), notes


# ============================================================
# Recall：STRICT + RELAXED
# ============================================================
def _range_filter(df: pd.DataFrame, col: str, target: Optional[float], tol: float) -> pd.DataFrame:
    if target is None or col not in df.columns:
        return df.copy()
    lower = target * (1 - tol)
    upper = target * (1 + tol)
    return df[(df[col] >= lower) & (df[col] <= upper)].copy()


def structured_recall(df: pd.DataFrame, normalized_req: Dict[str, Any], min_count: int = 12) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    notes: List[str] = []
    if df.empty:
        return df.copy(), {"recall_mode": "empty", "recall_notes": ["结构过滤后无候选"]}

    f_target = normalized_req.get("f_number", {}).get("target")
    hfov_target = normalized_req.get("half_fov_deg", {}).get("target")

    # 如果核心目标缺失，就不做严格数值召回，直接交给 ranker。
    if f_target is None and hfov_target is None:
        notes.append("缺少 F数和视场角目标，跳过 strict/relaxed 数值召回，直接进入排序")
        return df.copy(), {"recall_mode": "no_core_target", "recall_notes": notes}

    strict = df.copy()
    strict = _range_filter(strict, "f_number", f_target, 0.10)
    strict = _range_filter(strict, "half_fov", hfov_target, 0.10)

    if len(strict) >= min_count:
        notes.append(f"STRICT recall：F数±10%，半视场±10%，候选数 {len(strict)}")
        return strict.reset_index(drop=True), {"recall_mode": "strict", "recall_notes": notes}

    notes.append(f"STRICT 数量不足（{len(strict)}<{min_count}），启用渐进式 RELAXED recall")

    levels = [
        ("relaxed_15_12", 0.15, 0.12),
        ("relaxed_25_18", 0.25, 0.18),
        ("relaxed_40_25", 0.40, 0.25),
    ]

    best = strict.copy()
    mode = "strict"
    for name, f_tol, fov_tol in levels:
        cur = df.copy()
        cur = _range_filter(cur, "f_number", f_target, f_tol)
        cur = _range_filter(cur, "half_fov", hfov_target, fov_tol)
        notes.append(f"{name}：F数±{int(f_tol*100)}%，半视场±{int(fov_tol*100)}%，候选数 {len(cur)}")
        best = cur.copy()
        mode = f"strict+{name}"
        if len(cur) >= min_count:
            break

    # 如果放宽后仍为空，至少保留结构过滤后的 df，避免无结果；后面由分数与 feasibility 控制。
    if best.empty:
        notes.append("渐进式召回仍为空，回退到结构过滤候选，并由排序阶段决定是否可用")
        best = df.copy()
        mode = "fallback_after_empty_relaxed"

    return best.reset_index(drop=True), {"recall_mode": mode, "recall_notes": notes}


# ============================================================
# Scale adaptation
# ============================================================
def apply_scale_adaptation(df: pd.DataFrame, normalized_req: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    focal_target = normalized_req.get("focal_length", {}).get("target")

    out["aperture_norm"] = out["focal_length"] / out["f_number"]
    out["scale_factor"] = None
    out["focal_length_real_mm"] = None
    out["ttl_real_mm"] = None
    out["aperture_real_mm"] = None
    out["scale_status"] = "not_applied"
    out["scale_note"] = ""

    if focal_target is None:
        out["scale_status"] = "missing_target_focal_length"
        out["focal_length_real_mm"] = out["focal_length"]
        out["ttl_real_mm"] = out["ttl"]
        out["aperture_real_mm"] = out["aperture_norm"]
        out["scale_note"] = "用户未提供目标焦距，保留库内归一化/原始尺寸"
        return out

    valid = out["focal_length"].notna() & (out["focal_length"] > 0)
    out.loc[valid, "scale_factor"] = focal_target / out.loc[valid, "focal_length"]
    out.loc[valid, "focal_length_real_mm"] = out.loc[valid, "focal_length"] * out.loc[valid, "scale_factor"]
    out.loc[valid, "ttl_real_mm"] = out.loc[valid, "ttl"] * out.loc[valid, "scale_factor"]
    out.loc[valid, "aperture_real_mm"] = out.loc[valid, "aperture_norm"] * out.loc[valid, "scale_factor"]
    out.loc[valid, "scale_status"] = "applied_from_focal_length"
    out.loc[valid, "scale_note"] = "已根据目标焦距进行整体尺度缩放"

    out.loc[~valid, "scale_status"] = "invalid_focal_length"
    out.loc[~valid, "scale_note"] = "焦距非法，无法执行尺度缩放"

    return out


# ============================================================
# Ranker
# ============================================================
def _score_fov(half_fov: Optional[float], target: Optional[float]) -> float:
    err = _relative_error(half_fov, target)
    if err is None:
        return 0.5
    if err <= 0.08:
        return 1.0
    if err <= 0.18:
        return 0.75
    if err <= 0.30:
        return 0.45
    return 0.15


def _score_fnum(fnum: Optional[float], target: Optional[float]) -> float:
    err = _relative_error(fnum, target)
    if err is None:
        return 0.5
    if err <= 0.10:
        return 1.0
    if err <= 0.25:
        return 0.75
    if err <= 0.45:
        return 0.45
    return 0.15


def _score_scale(scale: Optional[float]) -> float:
    scale = _safe_float(scale)
    if scale is None:
        return 0.6
    if scale < 0.25 or scale > 4.0:
        return 0.05
    diff = abs(scale - 1.0)
    if diff <= 0.2:
        return 1.0
    if diff <= 0.5:
        return 0.8
    if diff <= 1.0:
        return 0.55
    return 0.25


def _score_ttl(ttl_real: Optional[float], ttl_pref: Optional[str]) -> float:
    ttl_real = _safe_float(ttl_real)
    if ttl_real is None:
        return 0.5
    if ttl_pref == "short":
        if ttl_real <= 8:
            return 1.0
        if ttl_real <= 20:
            return 0.7
        if ttl_real <= 50:
            return 0.35
        return 0.1
    # 未要求短时，只做正常范围鼓励
    if ttl_real <= 50:
        return 0.8
    return 0.4


def _score_element_count(x: Any) -> float:
    n = _safe_int(x)
    if n is None:
        return 0.5
    if n <= 6:
        return 1.0
    if n <= 10:
        return 0.75
    if n <= 14:
        return 0.45
    return 0.2


def rank_candidates(df: pd.DataFrame, normalized_req: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    target_f = normalized_req.get("f_number", {}).get("target")
    target_hfov = normalized_req.get("half_fov_deg", {}).get("target")
    ttl_pref = normalized_req.get("total_length", {}).get("preference")

    rows = []
    for _, row in out.iterrows():
        fov_score = _score_fov(_safe_float(row.get("half_fov")), target_hfov)
        fnum_score = _score_fnum(_safe_float(row.get("f_number")), target_f)
        scale_score = _score_scale(row.get("scale_factor"))
        ttl_score = _score_ttl(row.get("ttl_real_mm"), ttl_pref)
        element_score = _score_element_count(row.get("element_count"))

        # 核心：FOV + F数决定结构；scale/TTL/element 决定后续可优化性
        closeness_score = 0.55 * fov_score + 0.45 * fnum_score
        optimizability_score = 0.45 * scale_score + 0.35 * ttl_score + 0.20 * element_score
        final_score = 0.65 * closeness_score + 0.35 * optimizability_score

        can_be_initial = (fov_score >= 0.45 and fnum_score >= 0.45 and scale_score >= 0.25)
        if closeness_score >= 0.85 and optimizability_score >= 0.65:
            level = "direct"
        elif closeness_score >= 0.70 and optimizability_score >= 0.45:
            level = "mild"
        elif closeness_score >= 0.50 and optimizability_score >= 0.25:
            level = "moderate"
        else:
            level = "reject"
            can_be_initial = False

        r = row.to_dict()
        r.update({
            "fov_score": round(fov_score, 6),
            "fnum_score": round(fnum_score, 6),
            "scale_score": round(scale_score, 6),
            "ttl_score": round(ttl_score, 6),
            "element_score": round(element_score, 6),
            "closeness_score": round(closeness_score, 6),
            "optimizability_score": round(optimizability_score, 6),
            "final_score": round(final_score, 6),
            "adjustment_level": level,
            "can_be_initial_structure": bool(can_be_initial),
        })
        rows.append(r)

    ranked = pd.DataFrame(rows)
    priority = {"direct": 0, "mild": 1, "moderate": 2, "reject": 3}
    ranked["adjustment_priority"] = ranked["adjustment_level"].map(priority).fillna(9)
    ranked = ranked.sort_values(
        by=["adjustment_priority", "can_be_initial_structure", "final_score"],
        ascending=[True, False, False]
    ).reset_index(drop=True)
    return ranked


# ============================================================
# Diversity selector：Top9 三桶
# ============================================================
def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "lens_id" not in df.columns:
        return df.copy()
    return df.drop_duplicates("lens_id", keep="first").reset_index(drop=True)


def select_diverse_top9(ranked: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if ranked.empty:
        empty = ranked.copy()
        return {"top_close": empty, "top_optimizable": empty, "top_diverse": empty, "top9": empty}

    base = _dedup(ranked)

    top_close = base[
        (base["can_be_initial_structure"] == True) &
        (base["adjustment_level"].isin(["direct", "mild"])) &
        (base["fov_score"] >= 0.75) &
        (base["fnum_score"] >= 0.75)
    ].head(3).copy()

    rest = base[~base["lens_id"].astype(str).isin(set(top_close.get("lens_id", [] ).astype(str) if not top_close.empty else []))].copy()

    top_optimizable = rest[
        (rest["adjustment_level"].isin(["mild", "moderate"])) &
        (rest["scale_score"] >= 0.25)
    ].sort_values(by=["optimizability_score", "final_score"], ascending=[False, False]).head(3).copy()

    used = set(top_close.get("lens_id", pd.Series(dtype=str)).astype(str)) | set(top_optimizable.get("lens_id", pd.Series(dtype=str)).astype(str))
    rest2 = base[~base["lens_id"].astype(str).isin(used)].copy()

    # 多样性：按 element_count / half_fov 拉开，同时不完全牺牲 final_score
    selected = []
    if not rest2.empty:
        rest2 = rest2.sort_values("final_score", ascending=False).reset_index(drop=True)
        selected.append(rest2.iloc[0])
        while len(selected) < min(3, len(rest2)):
            best_i = None
            best_obj = None
            for i, row in rest2.iterrows():
                if any(str(row.get("lens_id")) == str(s.get("lens_id")) for s in selected):
                    continue
                penalty = 0.0
                for s in selected:
                    h1, h2 = _safe_float(s.get("half_fov")), _safe_float(row.get("half_fov"))
                    e1, e2 = _safe_float(s.get("element_count")), _safe_float(row.get("element_count"))
                    if h1 is not None and h2 is not None:
                        penalty += max(0, 1 - abs(h1 - h2) / 15.0) * 0.08
                    if e1 is not None and e2 is not None:
                        penalty += max(0, 1 - abs(e1 - e2) / 5.0) * 0.05
                obj = -_safe_float(row.get("final_score"), 0.0) + penalty
                if best_obj is None or obj < best_obj:
                    best_obj = obj
                    best_i = i
            if best_i is None:
                break
            selected.append(rest2.loc[best_i])
    top_diverse = pd.DataFrame(selected).reset_index(drop=True) if selected else rest2.head(0).copy()

    top9 = pd.concat([top_close, top_optimizable, top_diverse], ignore_index=True)
    top9 = _dedup(top9).head(9).reset_index(drop=True)

    # 如果三桶不足 9 个，用 base 补足
    if len(top9) < 9:
        used = set(top9.get("lens_id", pd.Series(dtype=str)).astype(str))
        backup = base[~base["lens_id"].astype(str).isin(used)].head(9 - len(top9))
        top9 = pd.concat([top9, backup], ignore_index=True)
        top9 = _dedup(top9).head(9).reset_index(drop=True)

    return {
        "top_close": top_close.reset_index(drop=True),
        "top_optimizable": top_optimizable.reset_index(drop=True),
        "top_diverse": top_diverse.reset_index(drop=True),
        "top9": top9.reset_index(drop=True),
    }


# ============================================================
# Raytrace rerank：优先读取已有 step10 结果；没有则用 proxy
# ============================================================
def _load_raytrace_table() -> pd.DataFrame:
    path = RAYTRACE_RESULT_PATH
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        if path.lower().endswith(".xlsx"):
            return pd.read_excel(path)
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def add_raytrace_info(top9: pd.DataFrame) -> pd.DataFrame:
    """
    实时 ray tracing：
    Top9 候选
    → 准备 seq
    → Optiland 建模
    → 实时 ray tracing
    → 根据 ray_spread 重排

    如果实时 ray tracing 失败，不让接口崩溃，而是回退到原 final_score 排序。
    """
    if top9.empty:
        return top9.copy()

    try:
        out = add_realtime_raytrace_to_topk(top9, max_candidates=9)

        if "final_score" in out.columns:
            base_score = pd.to_numeric(out["final_score"], errors="coerce").fillna(0.0)
        else:
            base_score = pd.Series([0.0] * len(out), index=out.index)

        if "raytrace_score" in out.columns:
            ray_score = pd.to_numeric(out["raytrace_score"], errors="coerce").fillna(0.0)
        else:
            ray_score = pd.Series([0.0] * len(out), index=out.index)

        # 综合分：保留原结构检索分，同时加入实时 ray tracing 结果
        out["rerank_score"] = base_score * 0.7 + ray_score * 0.3

        out = out.sort_values("rerank_score", ascending=False).reset_index(drop=True)
        return out

    except Exception as e:
        out = top9.copy()
        out["raytrace_valid"] = False
        out["raytrace_status"] = "realtime_failed_fallback"
        out["raytrace_error"] = str(e)
        out["ray_spread"] = None
        out["raytrace_score"] = 0.0

        if "final_score" in out.columns:
            out["rerank_score"] = pd.to_numeric(out["final_score"], errors="coerce").fillna(0.0)
        else:
            out["rerank_score"] = 0.0

        return out.sort_values("rerank_score", ascending=False).reset_index(drop=True)

# ============================================================
# 输出与可实现性
# ============================================================
def _row_to_candidate(row: pd.Series) -> Dict[str, Any]:
    lens_data = row.to_dict()
    return {
        "lens_id": str(row.get("lens_id")),
        "score": round(_safe_float(row.get("rerank_score"), row.get("final_score", 0.0)) or 0.0, 4),
        "lens_type": row.get("lens_type"),
        "adjustment_level": row.get("adjustment_level"),
        "can_be_initial_structure": bool(row.get("can_be_initial_structure")),
        "matched_fields": _build_matched_fields(row),
        "reason": _build_candidate_reason(row),
        "key_specs": {
            "focal_length": _safe_float(row.get("focal_length")),
            "focal_length_real_mm": _safe_float(row.get("focal_length_real_mm")),
            "f_number": _safe_float(row.get("f_number")),
            "half_fov": _safe_float(row.get("half_fov")),
            "full_fov": _safe_float(row.get("full_fov")),
            "total_length": _safe_float(row.get("ttl")),
            "ttl_real_mm": _safe_float(row.get("ttl_real_mm")),
            "element_count": _safe_int(row.get("element_count")),
            "scale_factor": _safe_float(row.get("scale_factor")),
            "ray_spread": _safe_float(row.get("ray_spread")),
            "raytrace_status": row.get("raytrace_status"),
            "layout_image_url": row.get("layout_image_url"),
        },
        "scores": {
            "fov_score": _safe_float(row.get("fov_score")),
            "fnum_score": _safe_float(row.get("fnum_score")),
            "scale_score": _safe_float(row.get("scale_score")),
            "ttl_score": _safe_float(row.get("ttl_score")),
            "closeness_score": _safe_float(row.get("closeness_score")),
            "optimizability_score": _safe_float(row.get("optimizability_score")),
            "final_score": _safe_float(row.get("final_score")),
            "raytrace_score": _safe_float(row.get("raytrace_score")),
            "rerank_score": _safe_float(row.get("rerank_score")),
        },
        "scale_note": row.get("scale_note"),
        "lens_data": lens_data,
    }


def _build_matched_fields(row: pd.Series) -> List[str]:
    fields = []
    if _safe_float(row.get("fov_score"), 0) >= 0.45:
        fields.append("fov")
    if _safe_float(row.get("fnum_score"), 0) >= 0.45:
        fields.append("f_number")
    if _safe_float(row.get("scale_score"), 0) >= 0.25:
        fields.append("scale")
    if _safe_float(row.get("ttl_score"), 0) >= 0.7:
        fields.append("total_length")
    return fields


def _build_candidate_reason(row: pd.Series) -> str:
    parts = []
    if _safe_float(row.get("fov_score"), 0) >= 0.75:
        parts.append("视场角与目标较接近")
    elif _safe_float(row.get("fov_score"), 0) >= 0.45:
        parts.append("视场角具备一定匹配度")
    if _safe_float(row.get("fnum_score"), 0) >= 0.75:
        parts.append("F数与目标较接近")
    elif _safe_float(row.get("fnum_score"), 0) >= 0.45:
        parts.append("F数具备一定匹配度")
    if _safe_float(row.get("scale_score"), 0) >= 0.55:
        parts.append("尺度缩放较可控")
    if _safe_float(row.get("ttl_score"), 0) >= 0.7:
        parts.append("总长较适合紧凑需求")
    if row.get("raytrace_status") == "from_step10_table":
        parts.append("已有快速光线追迹结果可参考")
    return "；".join(parts) if parts else "该候选为结构过滤后的参考方案"


def evaluate_hybrid_feasibility(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return {
            "feasibility": "hard_to_meet",
            "base_lens": None,
            "score": 0.0,
            "adjustments": [],
            "reason": "结构化召回和排序后没有得到有效候选。"
        }

    best = candidates[0]
    score = best.get("score", 0.0)
    adj_level = best.get("adjustment_level")
    specs = best.get("key_specs", {})
    scores = best.get("scores", {})

    adjustments = []
    if (scores.get("fov_score") or 0) < 0.75:
        adjustments.append("需要优先微调视场角")
    if (scores.get("fnum_score") or 0) < 0.75:
        adjustments.append("需要优先微调F数/孔径")
    if (scores.get("scale_score") or 0) < 0.55:
        adjustments.append("尺度缩放幅度偏大，需要检查结构可缩放性")
    if (scores.get("ttl_score") or 0) < 0.7:
        adjustments.append("总长仍需压缩或重新选择更紧凑结构")

    if adj_level == "direct" and score >= 0.70:
        feas = "direct_match"
        reason = "该候选在F数、视场角和可缩放性上匹配度较高，可直接作为起始结构。"
    elif adj_level in ["direct", "mild", "moderate"] and score >= 0.45:
        feas = "adjustable"
        reason = "该候选具备作为初始结构的潜力，但仍需要后续参数微调或物理优化。"
    else:
        feas = "hard_to_meet"
        reason = "当前候选与目标一阶规格差距仍较大，不建议直接采用。"

    return {
        "feasibility": feas,
        "base_lens": best.get("lens_id"),
        "score": score,
        "adjustments": adjustments,
        "reason": reason,
        "base_lens_specs": specs,
    }


def _df_to_bucket_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    return [_row_to_candidate(row) for _, row in df.iterrows()]


# ============================================================
# 主入口：给 app.py 调用
# ============================================================
def run_hybrid_design_pipeline(
    parsed_result: Dict[str, Any],
    lens_database: List[Dict[str, Any]],
    top_k: int = 9,
) -> Dict[str, Any]:
    normalized_req = build_normalized_requirement(parsed_result)

    raw_df = lens_database_to_df(lens_database)
    filtered_df = basic_filter(raw_df)
    filtered_df, structure_notes = structure_filter(filtered_df, normalized_req)
    recalled_df, recall_info = structured_recall(filtered_df, normalized_req)
    scaled_df = apply_scale_adaptation(recalled_df, normalized_req)
    ranked_df = rank_candidates(scaled_df, normalized_req)
    buckets = select_diverse_top9(ranked_df)
    reranked_top9 = add_raytrace_info(buckets["top9"])

    candidates = [_row_to_candidate(row) for _, row in reranked_top9.head(top_k).iterrows()]
    feasibility = evaluate_hybrid_feasibility(candidates)

    return {
        "normalized_requirement": normalized_req,
        "hybrid_notes": structure_notes + recall_info.get("recall_notes", []),
        "recall_mode": recall_info.get("recall_mode"),
        "candidate_count": {
            "after_basic_filter": int(len(basic_filter(raw_df))) if not raw_df.empty else 0,
            "after_structure_filter": int(len(filtered_df)),
            "after_recall": int(len(recalled_df)),
            "after_rank": int(len(ranked_df)),
            "top9": int(len(reranked_top9)),
        },
        "buckets": {
            "top_close": _df_to_bucket_records(buckets["top_close"]),
            "top_optimizable": _df_to_bucket_records(buckets["top_optimizable"]),
            "top_diverse": _df_to_bucket_records(buckets["top_diverse"]),
        },
        "top_k": candidates,
        "feasibility_result": feasibility,
    }


def build_hybrid_recommendation(hybrid_result: Dict[str, Any]) -> str:
    top_k = hybrid_result.get("top_k") or []
    feas = hybrid_result.get("feasibility_result") or {}

    if not top_k:
        return "当前没有检索到合适的候选镜头方案。建议补充F数、视场角或应用场景后重新检索。"

    best = top_k[0]
    specs = best.get("key_specs", {})
    feasibility = feas.get("feasibility")

    lens_id = best.get("lens_id")
    fnum = specs.get("f_number")
    fov = specs.get("full_fov")
    ttl = specs.get("ttl_real_mm") or specs.get("total_length")
    scale = specs.get("scale_factor")
    ray_status = specs.get("raytrace_status")
    ray_spread = specs.get("ray_spread")

    fnum_s = f"{fnum:.3f}" if isinstance(fnum, (int, float)) else "未知"
    fov_s = f"{fov:.1f}°" if isinstance(fov, (int, float)) else "未知"
    ttl_s = f"{ttl:.3f}mm" if isinstance(ttl, (int, float)) else "未知"
    scale_s = f"，scale≈{scale:.3f}" if isinstance(scale, (int, float)) else ""
    rt_s = ""
    if ray_status == "from_step10_table" and isinstance(ray_spread, (int, float)):
        rt_s = f"，并已有快速光线追迹结果 ray_spread≈{ray_spread:.3f}"

    if feasibility == "direct_match":
        return (
            f"推荐方案：{lens_id}。该方案在F数和视场角两个核心指标上匹配度较高，"
            f"F数约为{fnum_s}，全视场约为{fov_s}，缩放后总长约为{ttl_s}{scale_s}{rt_s}。"
            f"可作为当前设计的起始结构，后续再围绕像质、畸变和工艺约束做进一步验证。"
        )

    if feasibility == "adjustable":
        adj = "；".join(feas.get("adjustments") or [])
        adj = adj or "建议围绕F数、视场角和总长做小幅微调"
        return (
            f"建议以 {lens_id} 作为基础方案继续微调。该方案F数约为{fnum_s}，全视场约为{fov_s}，"
            f"缩放后总长约为{ttl_s}{scale_s}{rt_s}。主要调整方向：{adj}。"
            f"该结果属于候选结构推荐，仍需后续光学仿真验证。"
        )

    return (
        f"当前未找到可直接推荐的镜头方案。最接近的参考方案是 {lens_id}，"
        f"其F数约为{fnum_s}，全视场约为{fov_s}，缩放后总长约为{ttl_s}{scale_s}{rt_s}。"
        f"但该方案与目标规格仍存在差距，不建议直接采用，应重新寻找更接近目标F数和视场角的基础结构。"
    )
