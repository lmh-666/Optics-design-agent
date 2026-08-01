import json
import time
import math
import uuid
import requests
import pandas as pd


BASE_URL = "http://127.0.0.1:6006"
TOP_K = 9
TIMEOUT = 300


# ============================================================
# 单轮设计测试：
# 测试 /parse_requirement + /design_assist
#
# 重点覆盖：
# 1. F数解析
# 2. FOV/视场角解析
# 3. 孔径/EPD解析
# 4. F# × EPD 反推目标焦距
# 5. scale_factor / ttl_real_mm / aperture_real_mm
# 6. ray tracing / ray spread
# 7. hard constraint 风险
# 8. KG 场景约束
# ============================================================

DESIGN_TEST_CASES = [
    {
        "id": 1,
        "scene": "vehicle",
        "text": "设计一个车载广角镜头，F数2.0，视场角大于120度，孔径4mm，总长尽量短，畸变尽量低",
        "expected_scene_group": "vehicle",
        "expected_fnum": 2.0,
        "expected_fov": 120.0,
        "expected_aperture": 4.0,
        "expect_fov_hard": True,
    },
    {
        "id": 2,
        "scene": "vehicle",
        "text": "车载环视镜头，F数1.8，视场角大于140度，孔径5mm，要求低畸变和夜间成像好",
        "expected_scene_group": "vehicle",
        "expected_fnum": 1.8,
        "expected_fov": 140.0,
        "expected_aperture": 5.0,
        "expect_fov_hard": True,
    },
    {
        "id": 3,
        "scene": "vehicle",
        "text": "汽车前视镜头，F数2.4，视场角110度，孔径3.5mm，总长不要太长",
        "expected_scene_group": "vehicle",
        "expected_fnum": 2.4,
        "expected_fov": 110.0,
        "expected_aperture": 3.5,
        "expect_fov_hard": False,
    },
    {
        "id": 4,
        "scene": "phone",
        "text": "手机超广角镜头，F数2.0，视场角130度，孔径3.5mm，尽量短，结构简单一点",
        "expected_scene_group": "phone",
        "expected_fnum": 2.0,
        "expected_fov": 130.0,
        "expected_aperture": 3.5,
        "expect_fov_hard": False,
    },
    {
        "id": 5,
        "scene": "phone",
        "text": "手机镜头，F数2.4，视场角110度，孔径2.8mm，镜片数量少一点",
        "expected_scene_group": "phone",
        "expected_fnum": 2.4,
        "expected_fov": 110.0,
        "expected_aperture": 2.8,
        "expect_fov_hard": False,
    },
    {
        "id": 6,
        "scene": "drone",
        "text": "无人机广角镜头，F数2.8，视场角120度，孔径3mm，尽量轻巧",
        "expected_scene_group": "drone",
        "expected_fnum": 2.8,
        "expected_fov": 120.0,
        "expected_aperture": 3.0,
        "expect_fov_hard": False,
    },
    {
        "id": 7,
        "scene": "drone",
        "text": "无人机用大视场镜头，F数2.8，视场角大于130度，孔径4mm，镜片数量少一点",
        "expected_scene_group": "drone",
        "expected_fnum": 2.8,
        "expected_fov": 130.0,
        "expected_aperture": 4.0,
        "expect_fov_hard": True,
    },
    {
        "id": 8,
        "scene": "indoor",
        "text": "室内低照度镜头，F数1.8，视场角90度，孔径4mm，尽量亮",
        "expected_scene_group": "indoor",
        "expected_fnum": 1.8,
        "expected_fov": 90.0,
        "expected_aperture": 4.0,
        "expect_fov_hard": False,
    },
    {
        "id": 9,
        "scene": "indoor",
        "text": "室内广角镜头，F数2.8，视场角110度，孔径2.5mm，结构适中",
        "expected_scene_group": "indoor",
        "expected_fnum": 2.8,
        "expected_fov": 110.0,
        "expected_aperture": 2.5,
        "expect_fov_hard": False,
    },
    {
        "id": 10,
        "scene": "security",
        "text": "安防监控镜头，F数2.0，视场角80度，孔径3mm，低照度表现好一点",
        "expected_scene_group": "security",
        "expected_fnum": 2.0,
        "expected_fov": 80.0,
        "expected_aperture": 3.0,
        "expect_fov_hard": False,
    },
    {
        "id": 11,
        "scene": "security",
        "text": "CCTV监控镜头，F数2.8，视场角大于120度，孔径2.5mm，要求广角",
        "expected_scene_group": "security",
        "expected_fnum": 2.8,
        "expected_fov": 120.0,
        "expected_aperture": 2.5,
        "expect_fov_hard": True,
    },
    {
        "id": 12,
        "scene": "generic",
        "text": "F数2.0，视场角120度，孔径4mm，要求大视场和短总长",
        "expected_scene_group": "generic",
        "expected_fnum": 2.0,
        "expected_fov": 120.0,
        "expected_aperture": 4.0,
        "expect_fov_hard": False,
    },
]


# ============================================================
# Agent 多轮测试：
# 测试 /agent/chat
#
# 重点覆盖：
# 1. session_id 是否保存多轮状态
# 2. 第一轮是否识别 new_design_task
# 3. 第二轮是否识别 modify_constraint
# 4. 第三轮是否能 explain_result，而不是重新跑需求
# 5. design_state 是否保存 raw_text / constraint_updates / combined_text
# 6. history 是否增长
# 7. called_tools 是否体现工具调用
# ============================================================

def build_agent_test_scenarios():
    suffix = uuid.uuid4().hex[:8]

    return [
        {
            "scenario_id": f"agent_vehicle_{suffix}",
            "description": "车载广角硬约束 + 视场不足后续修改 + 解释结果",
            "steps": [
                {
                    "step": 1,
                    "message": "设计一个车载广角镜头，F数2.0，视场角大于120度，孔径4mm，总长尽量短，畸变尽量低",
                    "expected_intent": "new_design_task",
                },
                {
                    "step": 2,
                    "message": "视场角还是不够，优先找更接近120度的结构",
                    "expected_intent": "modify_constraint",
                },
                {
                    "step": 3,
                    "message": "为什么现在这个候选可以作为参考",
                    "expected_intent": "explain_result",
                },
            ],
        },
        {
            "scenario_id": f"agent_phone_{suffix}",
            "description": "手机超广角 + 后续要求更薄更简单",
            "steps": [
                {
                    "step": 1,
                    "message": "手机超广角镜头，F数2.0，视场角130度，孔径3.5mm，尽量短",
                    "expected_intent": "new_design_task",
                },
                {
                    "step": 2,
                    "message": "总长再短一点，结构简单一点",
                    "expected_intent": "modify_constraint",
                },
                {
                    "step": 3,
                    "message": "解释一下推荐理由",
                    "expected_intent": "explain_result",
                },
            ],
        },
        {
            "scenario_id": f"agent_drone_{suffix}",
            "description": "无人机轻量化 + 重新检索",
            "steps": [
                {
                    "step": 1,
                    "message": "无人机广角镜头，F数2.8，视场角120度，孔径3mm，尽量轻巧",
                    "expected_intent": "new_design_task",
                },
                {
                    "step": 2,
                    "message": "换一批候选结构看看",
                    "expected_intent": "retrieve_again",
                },
                {
                    "step": 3,
                    "message": "重新评价一下 ray tracing",
                    "expected_intent": "run_evaluation",
                },
            ],
        },
        {
            "scenario_id": f"agent_security_{suffix}",
            "description": "安防低照度 + 后续要求 F 数更小",
            "steps": [
                {
                    "step": 1,
                    "message": "安防监控镜头，F数2.0，视场角80度，孔径3mm，低照度表现好一点",
                    "expected_intent": "new_design_task",
                },
                {
                    "step": 2,
                    "message": "F数更小一点，夜间成像再好一点",
                    "expected_intent": "modify_constraint",
                },
                {
                    "step": 3,
                    "message": "第一个方案为什么合适",
                    "expected_intent": "explain_result",
                },
            ],
        },
    ]


# ============================================================
# 基础 HTTP 工具
# ============================================================

def post_json(endpoint: str, payload: dict, timeout: int = TIMEOUT):
    url = f"{BASE_URL}{endpoint}"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        content_type = r.headers.get("Content-Type", "")
        return {
            "status_code": r.status_code,
            "ok": r.ok,
            "json": r.json() if "application/json" in content_type else None,
            "text": r.text,
        }
    except Exception as e:
        return {
            "status_code": None,
            "ok": False,
            "json": None,
            "text": str(e),
        }


def safe_get(d: dict, path: list, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def to_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def to_json_str(obj):
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def pct_error(value, target):
    value = to_float(value)
    target = to_float(target)
    if value is None or target is None or target == 0:
        return None
    return round((value - target) / target * 100.0, 4)


def abs_error(value, target):
    value = to_float(value)
    target = to_float(target)
    if value is None or target is None:
        return None
    return round(value - target, 4)


def contains_any(items, keywords):
    text = " | ".join([str(x) for x in items or []])
    return any(k in text for k in keywords)


# ============================================================
# 通用结果抽取
# ============================================================

def extract_parse_summary(result: dict):
    parsed = result.get("parsed_result") or {}
    normalized = result.get("normalized_requirement") or {}
    scale_info = result.get("scale_info") or {}

    return {
        "parsed_scene": parsed.get("application_scene"),

        "parsed_fnum_target": safe_get(parsed, ["f_number", "target"]),
        "parsed_fnum_pref": safe_get(parsed, ["f_number", "preference"]),
        "parsed_fnum_constraint": safe_get(parsed, ["f_number", "constraint_type"]),

        "parsed_fov_target": safe_get(parsed, ["fov", "target"]),
        "parsed_fov_pref": safe_get(parsed, ["fov", "preference"]),
        "parsed_fov_constraint": safe_get(parsed, ["fov", "constraint_type"]),

        "parsed_aperture_target": safe_get(parsed, ["aperture", "target"]),
        "parsed_aperture_type": safe_get(parsed, ["aperture", "type"]),

        "parsed_focal_target": safe_get(parsed, ["focal_length", "target"]),
        "parsed_focal_source": safe_get(parsed, ["focal_length", "source"]),

        "parsed_ttl_target": safe_get(parsed, ["total_length", "target"]),
        "parsed_ttl_pref": safe_get(parsed, ["total_length", "preference"]),
        "parsed_ttl_constraint": safe_get(parsed, ["total_length", "constraint_type"]),

        "parsed_element_pref": safe_get(parsed, ["element_count", "preference"]),
        "parsed_distortion_pref": safe_get(parsed, ["distortion", "preference"]),

        "norm_scene": normalized.get("application_scene"),
        "norm_fnum_target": safe_get(normalized, ["f_number", "target"]),
        "norm_fnum_pref": safe_get(normalized, ["f_number", "preference"]),
        "norm_half_fov_target": safe_get(normalized, ["half_fov_deg", "target"]),
        "norm_full_fov_source": safe_get(normalized, ["half_fov_deg", "source_full_fov"]),
        "norm_fov_pref": safe_get(normalized, ["half_fov_deg", "preference"]),
        "norm_focal_target": safe_get(normalized, ["focal_length", "target"]),
        "norm_auto_filled": " | ".join(normalized.get("auto_filled") or []),

        "scale_aperture_used": scale_info.get("aperture_used"),
        "scale_aperture_target_mm": scale_info.get("aperture_target_mm"),
        "scale_f_number_target": scale_info.get("f_number_target"),
        "scale_derived_focal_length_mm": scale_info.get("derived_focal_length_mm"),
        "scale_status": scale_info.get("scale_status"),
        "scale_note": scale_info.get("scale_note"),
    }


def extract_kg_info(result: dict):
    kg = result.get("kg_info") or {}
    return {
        "kg_used": kg.get("kg_used"),
        "kg_scene": kg.get("scene"),
        "kg_scene_name": kg.get("scene_name"),
        "kg_lens_type": kg.get("lens_type"),
        "kg_fov_range": to_json_str(kg.get("fov_range")),
        "kg_fnum_range": to_json_str(kg.get("preferred_f_number_range")),
        "kg_design_focus": " | ".join(kg.get("design_focus") or []),
        "kg_explanation": result.get("kg_explanation"),
    }


def extract_count_info(result: dict):
    c = result.get("candidate_count") or {}
    return {
        "recall_mode": result.get("recall_mode"),
        "after_basic_filter": c.get("after_basic_filter"),
        "after_structure_filter": c.get("after_structure_filter"),
        "after_recall": c.get("after_recall"),
        "after_rank": c.get("after_rank"),
        "top9_count": c.get("top9"),
        "hybrid_notes": " | ".join(result.get("hybrid_notes") or []),
    }


def get_candidate_specs(candidate: dict):
    return candidate.get("key_specs") or candidate.get("lens_data") or {}


def extract_top1_info(result: dict, case: dict):
    top_k = result.get("top_k") or []

    expected_fnum = case.get("expected_fnum")
    expected_fov = case.get("expected_fov")
    expected_aperture = case.get("expected_aperture")

    expected_target_focal = None
    if expected_fnum is not None and expected_aperture is not None:
        expected_target_focal = round(float(expected_fnum) * float(expected_aperture), 6)

    if not top_k:
        return {
            "top1_lens_id": None,
            "top1_score": None,
            "top1_has_hard_risks": None,
            "top1_hard_risks": None,
            "top1_risks": None,
        }

    top1 = top_k[0]
    specs = get_candidate_specs(top1)
    scores = top1.get("scores") or {}
    lens_data = top1.get("lens_data") or {}
    kg_eval = top1.get("kg_evaluation") or {}
    real_eval = top1.get("real_constraint_evaluation") or {}

    top1_fnum = specs.get("f_number")
    top1_fov = specs.get("full_fov")
    top1_focal_real = specs.get("focal_length_real_mm")
    top1_aperture_real = specs.get("aperture_real_mm") or lens_data.get("aperture_real_mm")

    hard_risks = real_eval.get("hard_risks") or []
    risks = real_eval.get("risks") or []

    return {
        "top1_lens_id": top1.get("lens_id"),
        "top1_score": top1.get("score"),
        "top1_lens_type": top1.get("lens_type"),
        "top1_adjustment_level": top1.get("adjustment_level"),
        "top1_can_be_initial_structure": top1.get("can_be_initial_structure"),
        "top1_matched_fields": " | ".join(top1.get("matched_fields") or []),
        "top1_reason": top1.get("reason"),

        "top1_f_number": top1_fnum,
        "top1_fnum_error_abs": abs_error(top1_fnum, expected_fnum),
        "top1_fnum_error_pct": pct_error(top1_fnum, expected_fnum),

        "top1_half_fov": specs.get("half_fov"),
        "top1_full_fov": top1_fov,
        "top1_fov_error_deg": abs_error(top1_fov, expected_fov),

        "top1_focal_length": specs.get("focal_length"),
        "top1_focal_length_real_mm": top1_focal_real,
        "top1_target_focal_mm": expected_target_focal,
        "top1_focal_real_error_pct": pct_error(top1_focal_real, expected_target_focal),

        "top1_aperture_real_mm": top1_aperture_real,
        "top1_aperture_error_abs": abs_error(top1_aperture_real, expected_aperture),
        "top1_aperture_error_pct": pct_error(top1_aperture_real, expected_aperture),

        "top1_total_length": specs.get("total_length"),
        "top1_ttl_real_mm": specs.get("ttl_real_mm"),
        "top1_scale_factor": specs.get("scale_factor"),
        "top1_scale_status": lens_data.get("scale_status"),
        "top1_scale_basis": lens_data.get("scale_basis"),
        "top1_scale_note": lens_data.get("scale_note") or top1.get("scale_note"),

        "top1_raytrace_status": specs.get("raytrace_status"),
        "top1_raytrace_valid": lens_data.get("raytrace_valid"),
        "top1_ray_spread": specs.get("ray_spread"),
        "top1_raytrace_score": scores.get("raytrace_score"),
        "top1_rerank_score": scores.get("rerank_score"),

        "top1_seq_path": lens_data.get("seq_path"),
        "top1_raw_block_path": lens_data.get("raw_block_path"),

        "top1_kg_match_score": top1.get("kg_match_score"),
        "top1_kg_risks": " | ".join(kg_eval.get("kg_risks") or []),
        "top1_kg_reason": kg_eval.get("kg_reason"),

        "top1_has_hard_risks": len(hard_risks) > 0,
        "top1_hard_risks": " | ".join(hard_risks),
        "top1_risks": " | ".join(risks),

        "top1_real_eval_json": to_json_str(real_eval),
    }


def extract_topk_brief(result: dict, max_items: int = 9):
    top_k = result.get("top_k") or []
    brief = []

    for item in top_k[:max_items]:
        specs = get_candidate_specs(item)
        scores = item.get("scores") or {}
        lens_data = item.get("lens_data") or {}
        kg_eval = item.get("kg_evaluation") or {}
        real_eval = item.get("real_constraint_evaluation") or {}

        brief.append({
            "lens_id": item.get("lens_id"),
            "score": item.get("score"),
            "can_be_initial_structure": item.get("can_be_initial_structure"),
            "adjustment_level": item.get("adjustment_level"),
            "f_number": specs.get("f_number"),
            "full_fov": specs.get("full_fov"),
            "focal_real": specs.get("focal_length_real_mm"),
            "aperture_real": specs.get("aperture_real_mm") or lens_data.get("aperture_real_mm"),
            "ttl_raw": specs.get("total_length"),
            "ttl_real_mm": specs.get("ttl_real_mm"),
            "scale_factor": specs.get("scale_factor"),
            "scale_status": lens_data.get("scale_status"),
            "raytrace_status": specs.get("raytrace_status"),
            "ray_spread": specs.get("ray_spread"),
            "raytrace_score": scores.get("raytrace_score"),
            "rerank_score": scores.get("rerank_score"),
            "kg_score": item.get("kg_match_score"),
            "kg_risk": kg_eval.get("kg_reason"),
            "risks": real_eval.get("risks"),
            "hard_risks": real_eval.get("hard_risks"),
        })

    return to_json_str(brief)


def extract_feasibility_info(result: dict):
    f = result.get("feasibility_result") or {}
    specs = f.get("base_lens_specs") or {}
    real_eval = f.get("real_constraint_evaluation") or {}

    return {
        "feasibility": f.get("feasibility"),
        "base_lens": f.get("base_lens"),
        "feasibility_score": f.get("score"),
        "adjustments": " | ".join(f.get("adjustments") or []),
        "feasibility_reason": f.get("reason"),

        "base_f_number": specs.get("f_number"),
        "base_full_fov": specs.get("full_fov"),
        "base_focal_length_real_mm": specs.get("focal_length_real_mm"),
        "base_total_length": specs.get("total_length"),
        "base_ttl_real_mm": specs.get("ttl_real_mm"),
        "base_scale_factor": specs.get("scale_factor"),
        "base_ray_spread": specs.get("ray_spread"),
        "base_raytrace_status": specs.get("raytrace_status"),

        "base_hard_risks": " | ".join(real_eval.get("hard_risks") or []),
        "base_risks": " | ".join(real_eval.get("risks") or []),
    }


def simple_quality_flags(case: dict, result: dict):
    parsed = result.get("parsed_result") or {}
    top_k = result.get("top_k") or []
    top1 = top_k[0] if top_k else {}
    specs = get_candidate_specs(top1)
    lens_data = top1.get("lens_data") or {}
    real_eval = top1.get("real_constraint_evaluation") or {}
    scale_info = result.get("scale_info") or {}

    parsed_fnum = safe_get(parsed, ["f_number", "target"])
    parsed_fov = safe_get(parsed, ["fov", "target"])
    parsed_aperture = safe_get(parsed, ["aperture", "target"])
    parsed_fov_constraint = safe_get(parsed, ["fov", "constraint_type"])

    top1_scale_factor = specs.get("scale_factor")
    top1_ttl_real = specs.get("ttl_real_mm")

    return {
        "has_parsed_fnum": parsed_fnum is not None,
        "has_parsed_fov": parsed_fov is not None,
        "has_parsed_aperture": parsed_aperture is not None,

        "fov_hard_constraint_parsed": parsed_fov_constraint == "hard",
        "fov_hard_constraint_expected": case.get("expect_fov_hard"),
        "fov_hard_constraint_correct": (parsed_fov_constraint == "hard") == bool(case.get("expect_fov_hard")),

        "scale_info_available": scale_info.get("aperture_used") is True,
        "derived_focal_available": scale_info.get("derived_focal_length_mm") is not None,

        "top1_has_scale_factor": top1_scale_factor is not None,
        "top1_has_ttl_real": top1_ttl_real is not None,

        "top1_has_raytrace": specs.get("raytrace_status") is not None,
        "top1_raytrace_success_or_cache": specs.get("raytrace_status") in ["realtime_success", "realtime_cache"],

        "top1_scale_applied": lens_data.get("scale_status") == "applied",
        "top1_has_hard_risks": len(real_eval.get("hard_risks") or []) > 0,

        "pipeline_called_tools": " | ".join(result.get("called_tools") or []),
        "hard_constraint_tool_used": "enforce_target_constraint_sanity" in (result.get("called_tools") or []),
    }


# ============================================================
# 单轮测试
# ============================================================

def run_design_tests():
    rows = []

    print("=" * 100)
    print(f"开始单轮设计测试，共 {len(DESIGN_TEST_CASES)} 条")
    print(f"BASE_URL = {BASE_URL}")
    print(f"TOP_K = {TOP_K}")
    print("=" * 100)

    for case in DESIGN_TEST_CASES:
        case_id = case["id"]
        text = case["text"]

        print(f"[DESIGN {case_id:02d}] {text}")

        parse_resp = post_json("/parse_requirement", {"text": text}, timeout=TIMEOUT)
        time.sleep(0.2)

        design_resp = post_json(
            "/design_assist",
            {"text": text, "top_k": TOP_K},
            timeout=TIMEOUT
        )

        if not parse_resp["ok"]:
            print(f"[DESIGN {case_id:02d}] parse失败: {parse_resp['text'][:300]}")

        if not design_resp["ok"]:
            print(f"[DESIGN {case_id:02d}] design失败: {design_resp['text'][:300]}")

        parse_json = parse_resp["json"] if isinstance(parse_resp.get("json"), dict) else {}
        design_json = design_resp["json"] if isinstance(design_resp.get("json"), dict) else {}

        expected_target_focal = None
        if case.get("expected_fnum") is not None and case.get("expected_aperture") is not None:
            expected_target_focal = round(float(case["expected_fnum"]) * float(case["expected_aperture"]), 6)

        row = {
            "id": case_id,
            "scene_label": case.get("scene"),
            "user_text": text,

            "expected_scene_group": case.get("expected_scene_group"),
            "expected_fnum": case.get("expected_fnum"),
            "expected_fov": case.get("expected_fov"),
            "expected_aperture": case.get("expected_aperture"),
            "expected_target_focal": expected_target_focal,
            "expect_fov_hard": case.get("expect_fov_hard"),

            "parse_status_code": parse_resp.get("status_code"),
            "design_status_code": design_resp.get("status_code"),
            "parse_ok": parse_resp.get("ok"),
            "design_ok": design_resp.get("ok"),
            "parse_error_text": None if parse_resp.get("ok") else parse_resp.get("text"),
            "design_error_text": None if design_resp.get("ok") else design_resp.get("text"),

            "input_quality": design_json.get("input_quality"),
            "completion_notes": " | ".join(design_json.get("completion_notes") or []),
        }

        row.update(extract_parse_summary(design_json))
        row.update(extract_kg_info(design_json))
        row.update(extract_count_info(design_json))
        row.update(extract_top1_info(design_json, case))
        row.update(extract_feasibility_info(design_json))
        row.update(simple_quality_flags(case, design_json))

        row["topk_brief_json"] = extract_topk_brief(design_json)
        row["recommendation"] = design_json.get("recommendation")

        row["raw_parse_json"] = to_json_str(parse_json)
        row["raw_design_json"] = to_json_str(design_json)

        rows.append(row)
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    return df


# ============================================================
# Agent 多轮测试
# ============================================================

def extract_agent_step_row(scenario: dict, step_case: dict, agent_resp: dict):
    agent_json = agent_resp["json"] if isinstance(agent_resp.get("json"), dict) else {}

    result = agent_json.get("result") or {}
    design_state = agent_json.get("design_state") or {}

    top_k = result.get("top_k") or []
    top1 = top_k[0] if top_k else {}
    specs = get_candidate_specs(top1)
    real_eval = top1.get("real_constraint_evaluation") or {}

    history = design_state.get("history") or []
    constraint_updates = design_state.get("constraint_updates") or []

    called_tools = agent_json.get("called_tools") or []

    return {
        "scenario_id": scenario["scenario_id"],
        "scenario_description": scenario.get("description"),
        "step": step_case["step"],
        "message": step_case["message"],
        "expected_intent": step_case.get("expected_intent"),

        "status_code": agent_resp.get("status_code"),
        "ok": agent_resp.get("ok"),
        "error_text": None if agent_resp.get("ok") else agent_resp.get("text"),

        "session_id": agent_json.get("session_id"),
        "intent": agent_json.get("intent"),
        "intent_correct": agent_json.get("intent") == step_case.get("expected_intent"),

        "called_tools": " | ".join(called_tools),
        "called_tools_count": len(called_tools),

        "has_load_design_state": "load_design_state" in called_tools,
        "has_merge_user_constraint": "merge_user_constraint" in called_tools,
        "has_save_or_update_state": ("save_design_state" in called_tools) or ("update_design_state" in called_tools),
        "has_hard_constraint_check": "hard_constraint_check" in called_tools,

        "iteration": design_state.get("iteration"),
        "history_len": len(history),
        "constraint_updates_count": len(constraint_updates),
        "constraint_updates": " | ".join(constraint_updates),

        "raw_text": design_state.get("raw_text"),
        "combined_text": design_state.get("combined_text"),
        "combined_contains_current_message": step_case["message"] in str(design_state.get("combined_text")),

        "parsed_scene": safe_get(design_state, ["parsed_result", "application_scene"]),
        "parsed_fnum": safe_get(design_state, ["parsed_result", "f_number", "target"]),
        "parsed_fov": safe_get(design_state, ["parsed_result", "fov", "target"]),
        "parsed_fov_constraint": safe_get(design_state, ["parsed_result", "fov", "constraint_type"]),
        "parsed_aperture": safe_get(design_state, ["parsed_result", "aperture", "target"]),

        "top1_lens_id": top1.get("lens_id"),
        "top1_score": top1.get("score"),
        "top1_f_number": specs.get("f_number"),
        "top1_full_fov": specs.get("full_fov"),
        "top1_ttl_real_mm": specs.get("ttl_real_mm"),
        "top1_scale_factor": specs.get("scale_factor"),
        "top1_ray_spread": specs.get("ray_spread"),
        "top1_raytrace_status": specs.get("raytrace_status"),
        "top1_hard_risks": " | ".join(real_eval.get("hard_risks") or []),
        "top1_risks": " | ".join(real_eval.get("risks") or []),

        "feasibility": safe_get(result, ["feasibility_result", "feasibility"]),
        "base_lens": safe_get(result, ["feasibility_result", "base_lens"]),
        "recommendation": result.get("recommendation"),
        "answer": agent_json.get("answer"),

        "raw_agent_json": to_json_str(agent_json),
    }


def run_agent_tests():
    rows = []
    scenarios = build_agent_test_scenarios()

    print("=" * 100)
    print(f"开始 Agent 多轮测试，共 {len(scenarios)} 个 session")
    print("=" * 100)

    for scenario in scenarios:
        session_id = scenario["scenario_id"]
        print(f"[AGENT SESSION] {session_id} - {scenario.get('description')}")

        for step_case in scenario["steps"]:
            print(f"  [STEP {step_case['step']}] {step_case['message']}")

            payload = {
                "session_id": session_id,
                "message": step_case["message"],
                "top_k": TOP_K,
            }

            agent_resp = post_json("/agent/chat", payload, timeout=TIMEOUT)

            if not agent_resp["ok"]:
                print(f"  [STEP {step_case['step']}] 失败: {agent_resp['text'][:300]}")

            row = extract_agent_step_row(scenario, step_case, agent_resp)
            rows.append(row)

            time.sleep(0.7)

    df = pd.DataFrame(rows)
    return df


# ============================================================
# 主函数
# ============================================================

def main():
    design_df = run_design_tests()
    agent_df = run_agent_tests()

    design_xlsx = "api_test_design_single_turn.xlsx"
    design_csv = "api_test_design_single_turn.csv"

    agent_xlsx = "api_test_agent_multi_turn.xlsx"
    agent_csv = "api_test_agent_multi_turn.csv"

    summary_xlsx = "api_test_summary_all.xlsx"

    design_df.to_excel(design_xlsx, index=False)
    design_df.to_csv(design_csv, index=False, encoding="utf-8-sig")

    agent_df.to_excel(agent_xlsx, index=False)
    agent_df.to_csv(agent_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(summary_xlsx) as writer:
        design_df.to_excel(writer, sheet_name="single_turn_design", index=False)
        agent_df.to_excel(writer, sheet_name="agent_multi_turn", index=False)

    print("=" * 100)
    print("测试完成")
    print(f"单轮设计结果：{design_xlsx}")
    print(f"单轮设计 CSV：{design_csv}")
    print(f"Agent 多轮结果：{agent_xlsx}")
    print(f"Agent 多轮 CSV：{agent_csv}")
    print(f"汇总 Excel：{summary_xlsx}")

    print("=" * 100)
    print("单轮设计统计")
    print(f"parse 成功数: {int(design_df['parse_ok'].sum())}/{len(design_df)}")
    print(f"design 成功数: {int(design_df['design_ok'].sum())}/{len(design_df)}")
    print(f"top1 非空数: {int(design_df['top1_lens_id'].notna().sum())}/{len(design_df)}")
    print(f"孔径解析成功数: {int(design_df['has_parsed_aperture'].sum())}/{len(design_df)}")
    print(f"目标焦距反推成功数: {int(design_df['derived_focal_available'].sum())}/{len(design_df)}")
    print(f"FOV hard 约束识别正确数: {int(design_df['fov_hard_constraint_correct'].sum())}/{len(design_df)}")
    print(f"硬约束检查工具使用数: {int(design_df['hard_constraint_tool_used'].sum())}/{len(design_df)}")

    if "feasibility" in design_df.columns:
        print("\n可实现性统计：")
        print(design_df["feasibility"].value_counts(dropna=False))

    if "recall_mode" in design_df.columns:
        print("\nRecall mode 统计：")
        print(design_df["recall_mode"].value_counts(dropna=False))

    if "top1_raytrace_status" in design_df.columns:
        print("\nTop1 raytrace 状态统计：")
        print(design_df["top1_raytrace_status"].value_counts(dropna=False))

    print("=" * 100)
    print("Agent 多轮统计")
    print(f"Agent 调用成功数: {int(agent_df['ok'].sum())}/{len(agent_df)}")
    print(f"Intent 判断正确数: {int(agent_df['intent_correct'].sum())}/{len(agent_df)}")
    print(f"包含 load_design_state 的步数: {int(agent_df['has_load_design_state'].sum())}/{len(agent_df)}")
    print(f"包含 merge_user_constraint 的步数: {int(agent_df['has_merge_user_constraint'].sum())}/{len(agent_df)}")
    print(f"包含 save/update state 的步数: {int(agent_df['has_save_or_update_state'].sum())}/{len(agent_df)}")

    print("\nAgent intent 分布：")
    print(agent_df["intent"].value_counts(dropna=False))

    print("\n建议重点人工检查单轮结果这些列：")
    print([
        "user_text",
        "expected_fnum",
        "expected_fov",
        "expected_aperture",
        "expected_target_focal",
        "parsed_fnum_target",
        "parsed_fov_target",
        "parsed_fov_constraint",
        "parsed_aperture_target",
        "scale_derived_focal_length_mm",
        "top1_lens_id",
        "top1_f_number",
        "top1_full_fov",
        "top1_fov_error_deg",
        "top1_aperture_real_mm",
        "top1_scale_factor",
        "top1_ttl_real_mm",
        "top1_hard_risks",
        "feasibility",
        "recommendation",
    ])

    print("\n建议重点人工检查 Agent 结果这些列：")
    print([
        "scenario_id",
        "step",
        "message",
        "expected_intent",
        "intent",
        "intent_correct",
        "called_tools",
        "iteration",
        "history_len",
        "constraint_updates",
        "combined_text",
        "top1_lens_id",
        "top1_full_fov",
        "top1_hard_risks",
        "feasibility",
        "answer",
    ])


if __name__ == "__main__":
    main()