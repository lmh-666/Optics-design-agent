import os
import gc
import json
import time
import uuid
import re
import math
from pathlib import Path
from threading import Thread
from contextlib import asynccontextmanager
from typing import List, Optional, Literal, Iterator, Dict, Any, Tuple

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from lens_loader import load_lens_data
from hybrid_retrieval_engine import evaluate_hybrid_feasibility, run_hybrid_design_pipeline

from kg_rules import (
    enhance_requirement_with_kg,
    enrich_candidates_with_kg,
    build_kg_explanation,
)

from aperture_scale_utils import (
    enhance_requirement_with_aperture_scale,
    apply_scale_to_hybrid_result,
    tighten_feasibility_by_recall_and_scale,
)

from design_result_optimizer import (
    normalize_scene_for_generic_wide,
    optimize_hybrid_result_after_scale,
    build_optimized_recommendation,
)

from optiland_layout_renderer import (
    generate_optiland_layout_for_candidate,
    generate_optiland_layout_from_seq,
)


MODEL_DIR = os.getenv(
    "MODEL_DIR",
    "/root/.cache/modelscope/hub/models/ckdckd/OpticsGPT-v0"
)
MODEL_NAME = os.getenv("MODEL_NAME", "OpticsGPT-v0")
LENS_DATA_PATH = os.getenv("LENS_DATA_PATH", "./Merged_Lens_Data.csv")
DTYPE = torch.float16

tokenizer = None
model = None
LENS_DATABASE: List[Dict[str, Any]] = []

AGENT_SESSION_STORE: Dict[str, Dict[str, Any]] = {}


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=MODEL_NAME)
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False


class RequirementParseRequest(BaseModel):
    text: str


class SearchLensRequest(BaseModel):
    parsed_result: Dict[str, Any]
    top_k: Optional[int] = 9


class DesignAssistRequest(BaseModel):
    text: str
    top_k: Optional[int] = 9


class DesignFeasibilityRequest(BaseModel):
    text: str
    top_k: Optional[int] = 9


class AgentChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: Optional[int] = 9


class LayoutGenerateRequest(BaseModel):
    lens_id: str
    seq_path: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, LENS_DATABASE

    try:
        print("正在加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True
        )

        print("正在加载模型...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                device_map="auto",
                dtype=DTYPE,
                trust_remote_code=True,
                local_files_only=True
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                device_map="auto",
                torch_dtype=DTYPE,
                trust_remote_code=True,
                local_files_only=True
            )

        model.eval()
        print(f"模型已加载完成：{MODEL_NAME}")

        print(f"正在读取镜头数据: {LENS_DATA_PATH}")
        LENS_DATABASE = load_lens_data(LENS_DATA_PATH)
        print(f"镜头数据加载完成，共 {len(LENS_DATABASE)} 条")

        yield

    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="OpticsGPT Lens Design API",
    version="1.5.0-agent-optiland-layout",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("static/optiland_layouts").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


def build_prompt(messages: List[ChatMessage]) -> str:
    try:
        msg_list = [{"role": m.role, "content": m.content} for m in messages]
        return tokenizer.apply_chat_template(
            msg_list,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        parts = []
        for m in messages:
            parts.append(f"{m.role}: {m.content}")
        parts.append("assistant:")
        return "\n".join(parts)


def _get_model_device():
    try:
        return model.device
    except Exception:
        try:
            return next(model.parameters()).device
        except Exception:
            return "cuda" if torch.cuda.is_available() else "cpu"


def _build_generate_kwargs(inputs, max_tokens: int, temperature: float, top_p: float):
    kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    if temperature is not None and temperature > 0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p if top_p is not None else 0.9
    else:
        kwargs["do_sample"] = False

    return kwargs


def generate_text(
    messages: List[ChatMessage],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> str:
    if tokenizer is None or model is None:
        raise RuntimeError("模型尚未加载完成")

    prompt_text = build_prompt(messages)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(_get_model_device())

    generate_kwargs = _build_generate_kwargs(
        inputs=inputs,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def stream_generate(
    messages: List[ChatMessage],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> Iterator[str]:
    if tokenizer is None or model is None:
        raise RuntimeError("模型尚未加载完成")

    prompt_text = build_prompt(messages)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(_get_model_device())

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = _build_generate_kwargs(
        inputs=inputs,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    generation_kwargs["streamer"] = streamer

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


def make_chat_response(content: str, model_name: str) -> dict:
    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


def sse_format(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def extract_json_from_text(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        return {"error": "JSON解析失败", "raw_output": text}

    brace_count = 0
    end = -1

    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break

    if end == -1:
        return {"error": "JSON解析失败", "raw_output": text}

    json_candidate = text[start:end]

    try:
        return json.loads(json_candidate)
    except Exception:
        return {
            "error": "JSON解析失败",
            "raw_output": text,
            "json_candidate": json_candidate
        }


def to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_get_pref(parsed_result: dict, field_name: str) -> Optional[str]:
    field = parsed_result.get(field_name)
    if isinstance(field, dict):
        return field.get("preference")
    return None


def safe_get_target(parsed_result: dict, field_name: str) -> Optional[float]:
    field = parsed_result.get(field_name)
    if isinstance(field, dict):
        return to_float_or_none(field.get("target"))
    return None


def empty_requirement_template() -> dict:
    return {
        "application_scene": None,
        "focal_length": {"target": None, "preference": None, "constraint_type": None},
        "f_number": {"target": None, "preference": None, "constraint_type": None},
        "total_length": {"target": None, "preference": None, "constraint_type": None},
        "fov": {"target": None, "preference": None, "constraint_type": None},
        "element_count": {"target": None, "preference": None, "constraint_type": None},
        "distortion": {"target": None, "preference": None, "constraint_type": None},
        "low_light_performance": {"target": None, "preference": None, "constraint_type": None}
    }


def make_json_safe(obj):
    if obj is None:
        return None

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, (str, int, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]

    try:
        import pandas as pd
        if pd.isna(obj):
            return None
    except Exception:
        pass

    try:
        if hasattr(obj, "item"):
            return make_json_safe(obj.item())
    except Exception:
        pass

    try:
        return str(obj)
    except Exception:
        return None


def _fmt(value, digits: int = 3) -> str:
    try:
        if value is None:
            return "未知"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def normalize_scene(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    text = str(text).lower().strip()

    if text in ["vehicle", "car", "car_wide_angle", "车载", "车载广角", "汽车"]:
        return "car_wide_angle"

    if text in ["drone", "drone_wide_angle", "无人机", "无人机广角"]:
        return "drone_wide_angle"

    if text in ["indoor", "indoor_wide_angle", "室内", "室内广角"]:
        return "indoor_wide_angle"

    if text in ["手机", "手机镜头", "phone", "mobile", "phone_wide_angle", "mobile_wide_angle"]:
        return "phone_wide_angle"

    if text in ["安防", "security", "监控", "cctv"]:
        return "security"

    if "手机" in text or "phone" in text or "mobile" in text:
        return "phone_wide_angle"

    if "无人机" in text or "drone" in text:
        return "drone_wide_angle"

    if "车载" in text or "汽车" in text or "车" in text or "vehicle" in text or "car" in text:
        return "car_wide_angle"

    if "室内" in text or "indoor" in text:
        return "indoor_wide_angle"

    if "安防" in text or "监控" in text or "cctv" in text or "security" in text:
        return "security"

    return text


def infer_scene_from_user_text(user_text: str) -> Optional[str]:
    return normalize_scene(user_text)


def normalize_pref(text: Optional[str], field_name: Optional[str] = None) -> Optional[str]:
    if text is None:
        return None

    text = str(text).strip().lower()

    if field_name == "f_number":
        mapping = {
            "小": "small",
            "小一点": "small",
            "大光圈": "small",
            "光圈大": "small",
            "亮": "small",
            "亮一点": "small",
            "通光量大": "small",
            "进光量大": "small",
            "低照度好": "small",
            "暗光好": "small",
            "small": "small",
            "bright": "small",
            "low": "small"
        }
        return mapping.get(text, text)

    if field_name == "fov":
        mapping = {
            "大": "wide",
            "大一点": "wide",
            "广角": "wide",
            "大视场": "wide",
            "视场角大": "wide",
            "视野大": "wide",
            "覆盖范围大": "wide",
            "wide": "wide",
            "large": "wide"
        }
        return mapping.get(text, text)

    if field_name == "total_length":
        mapping = {
            "短": "short",
            "短一点": "short",
            "轻巧": "short",
            "轻便": "short",
            "轻": "short",
            "紧凑": "short",
            "便携": "short",
            "小巧": "short",
            "薄": "short",
            "薄一点": "short",
            "short": "short",
            "small": "short"
        }
        return mapping.get(text, text)

    if field_name == "element_count":
        mapping = {
            "少": "few",
            "少一点": "few",
            "镜片少": "few",
            "镜片数量少": "few",
            "片数少": "few",
            "few": "few"
        }
        return mapping.get(text, text)

    if field_name == "distortion":
        mapping = {
            "低": "low",
            "低一点": "low",
            "低畸变": "low",
            "畸变小": "low",
            "几何失真小": "low",
            "low": "low"
        }
        return mapping.get(text, text)

    if field_name == "low_light_performance":
        mapping = {
            "亮": "good",
            "亮一点": "good",
            "明亮": "good",
            "低照度好": "good",
            "暗光好": "good",
            "good": "good",
            "bright": "good"
        }
        return mapping.get(text, text)

    if field_name == "focal_length":
        return None

    return text


def cleanup_generated_text(text: str) -> str:
    if not text:
        return text

    bad_markers = [
        "\nHuman:",
        "\nAssistant:",
        "\nuser",
        "\nassistant",
        "Human:",
        "Assistant:",
        "# 任务",
        "# 任务描述",
        "用户需求：",
        "结构化解析结果：",
        "候选镜头：",
        "可实现性判断："
    ]

    cut = len(text)
    for marker in bad_markers:
        pos = text.find(marker)
        if pos != -1:
            cut = min(cut, pos)

    text = text[:cut].strip()
    text = re.sub(r'(?:#\s*\d+(?:\.\d+){5,}.*)', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?:\d+(?:\.\d+){6,})', '', text)
    text = text.replace('"""', '').replace("'''", "")
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def parse_requirement_text(text: str) -> dict:
    prompt = f"""
你是光学镜头需求解析助手。

任务：把用户需求解析成 JSON。

规则：
- 只输出 JSON
- 不要输出解释
- 不要输出 user / assistant / Human
- 未提及字段填 null
- focal_length 只提取数值 target，不要给它填 wide/short
- f_number 的模糊表达统一为 small
- fov 的模糊表达统一为 wide
- total_length 的模糊表达统一为 short
- element_count 的模糊表达统一为 few
- distortion 的模糊表达统一为 low
- low_light_performance 的模糊表达统一为 good
- 只有明确数值时才填写 target
- target 必须为数值或 null

输出格式：
{{
  "application_scene": null,
  "focal_length": {{"target": null, "preference": null, "constraint_type": null}},
  "f_number": {{"target": null, "preference": null, "constraint_type": null}},
  "total_length": {{"target": null, "preference": null, "constraint_type": null}},
  "fov": {{"target": null, "preference": null, "constraint_type": null}},
  "element_count": {{"target": null, "preference": null, "constraint_type": null}},
  "distortion": {{"target": null, "preference": null, "constraint_type": null}},
  "low_light_performance": {{"target": null, "preference": null, "constraint_type": null}}
}}

用户需求：
{text}
""".strip()

    content = generate_text(
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.0,
        top_p=1.0,
        max_tokens=256
    )

    return extract_json_from_text(content)


def infer_constraint_type_from_text(user_text: str, parsed_result: dict) -> dict:
    text = user_text.strip()

    fov_target = safe_get_target(parsed_result, "fov")
    if fov_target is not None:
        if any(k in text for k in ["大于", "超过", "不小于", "至少", ">=", "＞", ">"]):
            parsed_result["fov"]["preference"] = parsed_result["fov"].get("preference") or "wide"
            parsed_result["fov"]["constraint_type"] = "hard"
        elif any(k in text for k in ["接近", "左右", "约", "大概"]):
            parsed_result["fov"]["constraint_type"] = parsed_result["fov"].get("constraint_type") or "soft"
        else:
            parsed_result["fov"]["constraint_type"] = parsed_result["fov"].get("constraint_type") or "soft"

    fnum_target = safe_get_target(parsed_result, "f_number")
    if fnum_target is not None:
        if any(k in text for k in ["小于", "不大于", "不超过", "<=", "＜", "<"]):
            parsed_result["f_number"]["preference"] = parsed_result["f_number"].get("preference") or "small"
            parsed_result["f_number"]["constraint_type"] = "hard"
        elif any(k in text for k in ["左右", "约", "大概", "接近"]):
            parsed_result["f_number"]["constraint_type"] = parsed_result["f_number"].get("constraint_type") or "soft"
        else:
            parsed_result["f_number"]["constraint_type"] = parsed_result["f_number"].get("constraint_type") or "soft"

    ttl_target = safe_get_target(parsed_result, "total_length")
    ttl_pref = safe_get_pref(parsed_result, "total_length")

    if ttl_target is not None:
        if any(k in text for k in ["小于", "不超过", "不大于", "<=", "＜", "<"]):
            parsed_result["total_length"]["preference"] = parsed_result["total_length"].get("preference") or "short"
            parsed_result["total_length"]["constraint_type"] = "hard"
        else:
            parsed_result["total_length"]["constraint_type"] = parsed_result["total_length"].get("constraint_type") or "soft"

    if ttl_pref == "short":
        parsed_result["total_length"]["constraint_type"] = parsed_result["total_length"].get("constraint_type") or "soft"

    distortion_pref = safe_get_pref(parsed_result, "distortion")
    if distortion_pref == "low":
        parsed_result["distortion"]["constraint_type"] = parsed_result["distortion"].get("constraint_type") or "soft"

    return parsed_result


def validate_and_complete_requirements(
    user_text: str,
    parsed_result: dict
) -> Tuple[dict, str, List[str]]:
    notes: List[str] = []

    if not isinstance(parsed_result, dict) or "error" in parsed_result:
        parsed_result = empty_requirement_template()
        notes.append("模型解析失败，已使用空模板初始化")

    template = empty_requirement_template()

    for key, value in template.items():
        if key not in parsed_result:
            parsed_result[key] = value
            notes.append(f"缺失字段 {key} 已自动补齐")

    for key in [
        "focal_length", "f_number", "total_length", "fov",
        "element_count", "distortion", "low_light_performance"
    ]:
        if not isinstance(parsed_result.get(key), dict):
            parsed_result[key] = {
                "target": None,
                "preference": None,
                "constraint_type": None
            }
            notes.append(f"字段 {key} 格式异常，已重置")

    scene = normalize_scene(parsed_result.get("application_scene"))
    if scene is None:
        scene = infer_scene_from_user_text(user_text)
        if scene is not None:
            notes.append(f"根据原始输入自动识别场景 {scene}")

    parsed_result["application_scene"] = scene

    for key in [
        "focal_length", "f_number", "total_length", "fov",
        "element_count", "distortion", "low_light_performance"
    ]:
        parsed_result[key]["target"] = to_float_or_none(parsed_result[key].get("target"))

    parsed_result["focal_length"]["preference"] = normalize_pref(
        parsed_result["focal_length"].get("preference"), "focal_length"
    )
    parsed_result["f_number"]["preference"] = normalize_pref(
        parsed_result["f_number"].get("preference"), "f_number"
    )
    parsed_result["total_length"]["preference"] = normalize_pref(
        parsed_result["total_length"].get("preference"), "total_length"
    )
    parsed_result["fov"]["preference"] = normalize_pref(
        parsed_result["fov"].get("preference"), "fov"
    )
    parsed_result["element_count"]["preference"] = normalize_pref(
        parsed_result["element_count"].get("preference"), "element_count"
    )
    parsed_result["distortion"]["preference"] = normalize_pref(
        parsed_result["distortion"].get("preference"), "distortion"
    )
    parsed_result["low_light_performance"]["preference"] = normalize_pref(
        parsed_result["low_light_performance"].get("preference"), "low_light_performance"
    )

    text = user_text.strip()

    if parsed_result["total_length"]["preference"] is None:
        if any(k in text for k in [
            "轻巧", "紧凑", "便携", "轻便", "小型",
            "尽量小", "尽量短", "压缩尺寸", "薄一点", "尽量薄"
        ]):
            parsed_result["total_length"]["preference"] = "short"
            parsed_result["total_length"]["constraint_type"] = "soft"
            notes.append("根据轻巧/紧凑/薄等表达补全 total_length=short")

    if parsed_result["fov"]["preference"] is None and parsed_result["fov"]["target"] is None:
        if any(k in text for k in [
            "广角", "大视场", "视场大", "视野大",
            "覆盖范围大", "视场角尽量大"
        ]):
            parsed_result["fov"]["preference"] = "wide"
            parsed_result["fov"]["constraint_type"] = "soft"
            notes.append("根据广角/大视场等表达补全 fov=wide")

    if parsed_result["f_number"]["preference"] is None and parsed_result["f_number"]["target"] is None:
        if any(k in text for k in [
            "大光圈", "亮一点", "通光量大", "进光量大",
            "低照度好", "F数小一点", "F数小"
        ]):
            parsed_result["f_number"]["preference"] = "small"
            parsed_result["f_number"]["constraint_type"] = "soft"
            notes.append("根据大光圈/亮一点/F数小等表达补全 f_number=small")

    if parsed_result["distortion"]["preference"] is None and parsed_result["distortion"]["target"] is None:
        if any(k in text for k in [
            "低畸变", "畸变小", "几何失真小", "畸变尽量低"
        ]):
            parsed_result["distortion"]["preference"] = "low"
            parsed_result["distortion"]["constraint_type"] = "soft"
            notes.append("根据低畸变表达补全 distortion=low")

    if parsed_result["element_count"]["preference"] is None:
        if any(k in text for k in [
            "镜片数少", "镜片数量少", "片数少", "镜片少一点"
        ]):
            parsed_result["element_count"]["preference"] = "few"
            parsed_result["element_count"]["constraint_type"] = "soft"
            notes.append("根据镜片数少表达补全 element_count=few")

    parsed_result = infer_constraint_type_from_text(user_text, parsed_result)

    has_scene = parsed_result.get("application_scene") is not None
    has_fnum = safe_get_target(parsed_result, "f_number") is not None or safe_get_pref(parsed_result, "f_number") is not None
    has_fov = safe_get_target(parsed_result, "fov") is not None or safe_get_pref(parsed_result, "fov") is not None
    has_focal = safe_get_target(parsed_result, "focal_length") is not None
    has_ttl = safe_get_target(parsed_result, "total_length") is not None or safe_get_pref(parsed_result, "total_length") is not None

    filled_count = sum([has_scene, has_fnum, has_fov, has_focal, has_ttl])

    if filled_count >= 4:
        input_quality = "explicit"
    elif filled_count >= 2:
        input_quality = "semi_explicit"
    else:
        input_quality = "vague"

    return parsed_result, input_quality, notes


def build_input_understanding(parsed_result: dict) -> dict:
    return {
        "scene": parsed_result.get("application_scene"),
        "f_number": parsed_result.get("f_number"),
        "fov": parsed_result.get("fov"),
        "focal_length": parsed_result.get("focal_length"),
        "total_length": parsed_result.get("total_length"),
    }


def get_candidate_specs(candidate: dict) -> dict:
    return candidate.get("key_specs") or candidate.get("lens_data") or {}


def get_candidate_real_eval(candidate: dict) -> dict:
    real_eval = candidate.get("real_constraint_evaluation")
    if not isinstance(real_eval, dict):
        real_eval = {}
        candidate["real_constraint_evaluation"] = real_eval
    return real_eval


def enforce_target_constraint_sanity(hybrid_result: dict, parsed_result: dict) -> dict:
    top_k = hybrid_result.get("top_k") or []
    if not top_k:
        return hybrid_result

    target_fov = safe_get_target(parsed_result, "fov")
    fov_constraint = parsed_result.get("fov", {}).get("constraint_type")

    target_fnum = safe_get_target(parsed_result, "f_number")
    fnum_constraint = parsed_result.get("f_number", {}).get("constraint_type")

    for cand in top_k:
        specs = get_candidate_specs(cand)
        real_eval = get_candidate_real_eval(cand)
        risks = real_eval.setdefault("risks", [])
        hard_risks = real_eval.setdefault("hard_risks", [])

        full_fov = specs.get("full_fov")
        f_number = specs.get("f_number")

        if target_fov is not None and full_fov is not None:
            try:
                full_fov = float(full_fov)
                fov_error = full_fov - float(target_fov)

                real_eval["target_full_fov"] = float(target_fov)
                real_eval["candidate_full_fov"] = full_fov
                real_eval["fov_error_deg"] = round(fov_error, 4)

                if fov_constraint == "hard":
                    if fov_error < -25:
                        msg = f"视场角低于硬约束目标 {abs(fov_error):.1f}°，不建议作为首选结构"
                        if msg not in hard_risks:
                            hard_risks.append(msg)

                        cand["adjustment_level"] = "reject"
                        cand["can_be_initial_structure"] = False
                        old_score = float(cand.get("score") or 0)
                        cand["score"] = round(min(old_score, 0.48), 4)

                        if isinstance(cand.get("scores"), dict):
                            cand["scores"]["hard_constraint_penalty"] = "fov_hard_miss"
                            cand["scores"]["real_constraint_score"] = cand["score"]

                    elif fov_error < -10:
                        msg = f"视场角低于目标 {abs(fov_error):.1f}°，需要重点扩视场优化"
                        if msg not in risks:
                            risks.append(msg)

                        old_score = float(cand.get("score") or 0)
                        cand["score"] = round(min(old_score, 0.62), 4)

                        if isinstance(cand.get("scores"), dict):
                            cand["scores"]["soft_constraint_penalty"] = "fov_miss"
                            cand["scores"]["real_constraint_score"] = cand["score"]

            except Exception:
                pass

        if target_fnum is not None and f_number is not None:
            try:
                f_number = float(f_number)
                fnum_error_pct = (f_number - float(target_fnum)) / float(target_fnum) * 100.0

                real_eval["target_f_number"] = float(target_fnum)
                real_eval["candidate_f_number"] = f_number
                real_eval["f_number_error_pct"] = round(fnum_error_pct, 4)

                if fnum_error_pct > 35:
                    msg = f"F数高于目标 {fnum_error_pct:.1f}%，低照度能力可能不足"
                    if msg not in hard_risks:
                        hard_risks.append(msg)

                    old_score = float(cand.get("score") or 0)
                    cand["score"] = round(min(old_score, 0.58), 4)

                    if isinstance(cand.get("scores"), dict):
                        cand["scores"]["hard_constraint_penalty"] = "f_number_high"
                        cand["scores"]["real_constraint_score"] = cand["score"]

                if fnum_constraint == "hard" and fnum_error_pct > 15:
                    msg = f"F数不满足硬约束，偏高 {fnum_error_pct:.1f}%"
                    if msg not in hard_risks:
                        hard_risks.append(msg)

                    cand["adjustment_level"] = "reject"
                    cand["can_be_initial_structure"] = False
                    old_score = float(cand.get("score") or 0)
                    cand["score"] = round(min(old_score, 0.5), 4)

            except Exception:
                pass

        real_eval["risks"] = risks
        real_eval["hard_risks"] = hard_risks

    top_k.sort(key=lambda x: x.get("score", 0), reverse=True)
    hybrid_result["top_k"] = top_k

    if top_k:
        best = top_k[0]
        best_eval = best.get("real_constraint_evaluation") or {}
        best_hard_risks = best_eval.get("hard_risks") or []
        best_risks = best_eval.get("risks") or []

        feasibility_result = hybrid_result.get("feasibility_result")
        if not isinstance(feasibility_result, dict):
            feasibility_result = {}

        feasibility_result["base_lens"] = best.get("lens_id")
        feasibility_result["score"] = best.get("score")
        feasibility_result["base_lens_specs"] = best.get("key_specs")
        feasibility_result["real_constraint_evaluation"] = best_eval
        feasibility_result["adjustments"] = best_hard_risks or best_risks

        if best_hard_risks:
            feasibility_result["feasibility"] = "hard_to_meet"
            feasibility_result["reason"] = "当前最高候选仍存在硬约束风险，仅建议作为参考结构，不建议直接作为首选。"
        elif best_risks:
            feasibility_result["feasibility"] = "adjustable"
            feasibility_result["reason"] = "该候选具备作为初始结构的潜力，但仍需要后续参数微调或物理优化。"
        else:
            feasibility_result["feasibility"] = feasibility_result.get("feasibility") or "adjustable"
            feasibility_result["reason"] = feasibility_result.get("reason") or "该候选可作为初始结构参考。"

        hybrid_result["feasibility_result"] = feasibility_result

    return hybrid_result


def rebuild_candidate_roles_from_topk(hybrid_result: dict) -> dict:
    top_k = hybrid_result.get("top_k") or []
    if not top_k:
        hybrid_result["candidate_roles"] = {}
        return hybrid_result

    valid_candidates = [
        c for c in top_k
        if c.get("can_be_initial_structure", True) is not False
    ]
    pool = valid_candidates if valid_candidates else top_k

    def ttl_value(c):
        specs = get_candidate_specs(c)
        v = specs.get("ttl_real_mm", specs.get("total_length"))
        try:
            return float(v)
        except Exception:
            return float("inf")

    def fov_value(c):
        specs = get_candidate_specs(c)
        v = specs.get("full_fov")
        try:
            return float(v)
        except Exception:
            return -1.0

    def summarize(c):
        specs = get_candidate_specs(c)
        real_eval = c.get("real_constraint_evaluation") or {}
        return {
            "lens_id": c.get("lens_id"),
            "score": c.get("score"),
            "optimized_score": c.get("score"),
            "f_number": specs.get("f_number"),
            "full_fov": specs.get("full_fov"),
            "ttl_real_mm": specs.get("ttl_real_mm", specs.get("total_length")),
            "scale_factor": specs.get("scale_factor"),
            "ray_spread": specs.get("ray_spread"),
            "raytrace_status": specs.get("raytrace_status"),
            "layout_image_url": specs.get("layout_image_url"),
            "layout_source": specs.get("layout_source"),
            "risks": real_eval.get("risks", []),
            "hard_risks": real_eval.get("hard_risks", []),
        }

    balanced_best = pool[0]
    compact_best = min(pool, key=ttl_value)
    structure_best = max(pool, key=fov_value)

    hybrid_result["candidate_roles"] = {
        "balanced_best": summarize(balanced_best),
        "structure_best": summarize(structure_best),
        "compact_best": summarize(compact_best),
    }

    return hybrid_result


def build_safe_recommendation(hybrid_result: dict, parsed_result: dict) -> str:
    top_k = hybrid_result.get("top_k") or []
    if not top_k:
        return "当前没有检索到合适的候选镜头结构，建议补充 F数、视场角、焦距或孔径约束后重新检索。"

    best = top_k[0]
    specs = get_candidate_specs(best)
    real_eval = best.get("real_constraint_evaluation") or {}

    hard_risks = real_eval.get("hard_risks") or []
    risks = real_eval.get("risks") or []

    lens_id = best.get("lens_id")
    f_number = specs.get("f_number")
    full_fov = specs.get("full_fov")
    ttl = specs.get("ttl_real_mm", specs.get("total_length"))
    ray_spread = specs.get("ray_spread")
    layout_image_url = specs.get("layout_image_url")

    layout_text = f"结构图已生成：{layout_image_url}。" if layout_image_url else ""

    if hard_risks:
        return cleanup_generated_text(
            f"当前没有找到严格满足全部硬约束的镜头结构。"
            f"数据库中相对接近的参考结构为 {lens_id}，"
            f"其 F 数约为 {_fmt(f_number)}，全视场约为 {_fmt(full_fov, 1)}°，"
            f"总长约为 {_fmt(ttl)}mm，ray_spread 约为 {_fmt(ray_spread, 4)}。"
            f"但该结构存在硬约束风险：{'；'.join(hard_risks)}。"
            f"因此它不应作为直接推荐方案，只能作为后续扩视场、调整F数或重新外推生成结构的参考起点。"
            f"{layout_text}"
        )

    try:
        base = build_optimized_recommendation(hybrid_result, parsed_result)
        return cleanup_generated_text(base + layout_text)
    except Exception:
        risk_text = f"主要风险：{'；'.join(risks)}。" if risks else ""
        return cleanup_generated_text(
            f"建议以 {lens_id} 作为可微调初始结构。"
            f"该方案 F 数约为 {_fmt(f_number)}，全视场约为 {_fmt(full_fov, 1)}°，"
            f"总长约为 {_fmt(ttl)}mm，ray_spread 约为 {_fmt(ray_spread, 4)}。"
            f"{risk_text}"
            f"该结果属于初始结构推荐与筛选结果，仍需后续完整光学仿真和专家复核。"
            f"{layout_text}"
        )


def search_lens_candidates(parsed_result: Dict[str, Any], top_k: int = 9) -> List[Dict[str, Any]]:
    hybrid_result = run_hybrid_design_pipeline(
        parsed_result=parsed_result,
        lens_database=LENS_DATABASE,
        top_k=top_k or 9
    )
    return hybrid_result.get("top_k", [])


def attach_optiland_layout_to_best_candidate(hybrid_result: dict) -> Tuple[dict, Optional[dict]]:
    top_k = hybrid_result.get("top_k") or []
    if not top_k:
        return hybrid_result, None

    best_candidate = top_k[0]
    layout_result = None

    try:
        layout_result = generate_optiland_layout_for_candidate(
            best_candidate,
            output_dir="static/optiland_layouts"
        )

        best_candidate.setdefault("key_specs", {})
        best_candidate["key_specs"]["layout_image_url"] = layout_result.get("image_url")
        best_candidate["key_specs"]["layout_source"] = layout_result.get("layout_source")

        best_candidate.setdefault("lens_data", {})
        best_candidate["lens_data"]["layout_image_url"] = layout_result.get("image_url")
        best_candidate["lens_data"]["layout_source"] = layout_result.get("layout_source")
        best_candidate["lens_data"]["optiland_layout_result"] = layout_result

        hybrid_result["top_k"][0] = best_candidate

        feasibility_result = hybrid_result.get("feasibility_result")
        if isinstance(feasibility_result, dict):
            base_specs = feasibility_result.get("base_lens_specs")
            if isinstance(base_specs, dict):
                base_specs["layout_image_url"] = layout_result.get("image_url")
                base_specs["layout_source"] = layout_result.get("layout_source")
                feasibility_result["base_lens_specs"] = base_specs
            hybrid_result["feasibility_result"] = feasibility_result

    except Exception as e:
        layout_result = {
            "status": "failed",
            "error": str(e)
        }

    return hybrid_result, layout_result


def run_design_assist_pipeline(user_text: str, top_k: int = 9) -> dict:
    called_tools = [
        "parse_requirement",
        "validate_and_complete_requirements",
        "normalize_scene_for_generic_wide",
        "enhance_requirement_with_kg",
        "enhance_requirement_with_aperture_scale",
        "run_hybrid_design_pipeline",
        "enrich_candidates_with_kg",
        "apply_scale_to_hybrid_result",
        "tighten_feasibility_by_recall_and_scale",
        "optimize_hybrid_result_after_scale",
        "enforce_target_constraint_sanity",
        "generate_optiland_layout",
        "build_safe_recommendation",
    ]

    raw_parsed = parse_requirement_text(user_text)

    parsed_result, input_quality, completion_notes = validate_and_complete_requirements(
        user_text,
        raw_parsed
    )

    parsed_result, completion_notes = normalize_scene_for_generic_wide(
        user_text,
        parsed_result,
        completion_notes
    )

    parsed_result, completion_notes, kg_info = enhance_requirement_with_kg(
        user_text,
        parsed_result,
        completion_notes
    )

    parsed_result, completion_notes, scale_info = enhance_requirement_with_aperture_scale(
        user_text,
        parsed_result,
        completion_notes
    )

    hybrid_result = run_hybrid_design_pipeline(
        parsed_result=parsed_result,
        lens_database=LENS_DATABASE,
        top_k=top_k or 9
    )

    hybrid_result["top_k"] = enrich_candidates_with_kg(
        hybrid_result.get("top_k", []),
        parsed_result
    )

    hybrid_result = apply_scale_to_hybrid_result(
        hybrid_result,
        parsed_result
    )

    hybrid_result = tighten_feasibility_by_recall_and_scale(
        hybrid_result
    )

    hybrid_result = optimize_hybrid_result_after_scale(
        hybrid_result,
        parsed_result,
        user_text
    )

    hybrid_result = enforce_target_constraint_sanity(
        hybrid_result,
        parsed_result
    )

    hybrid_result, layout_result = attach_optiland_layout_to_best_candidate(
        hybrid_result
    )

    hybrid_result = rebuild_candidate_roles_from_topk(hybrid_result)

    best_candidate = hybrid_result["top_k"][0] if hybrid_result.get("top_k") else None

    kg_explanation = build_kg_explanation(
        parsed_result,
        best_candidate
    )

    recommendation = build_safe_recommendation(
        hybrid_result,
        parsed_result
    )

    response = {
        "raw_text": user_text,
        "called_tools": called_tools,

        "parsed_result": parsed_result,
        "input_quality": input_quality,
        "completion_notes": completion_notes,
        "input_understanding": build_input_understanding(parsed_result),

        "kg_info": kg_info,
        "kg_explanation": kg_explanation,

        "scale_info": scale_info,
        "layout_result": layout_result,

        "normalized_requirement": hybrid_result.get("normalized_requirement"),
        "hybrid_notes": hybrid_result.get("hybrid_notes"),
        "recall_mode": hybrid_result.get("recall_mode"),
        "candidate_count": hybrid_result.get("candidate_count"),

        "candidate_roles": hybrid_result.get("candidate_roles"),
        "buckets": hybrid_result.get("buckets"),
        "top_k": hybrid_result.get("top_k"),
        "feasibility_result": hybrid_result.get("feasibility_result"),
        "recommendation": recommendation
    }

    return make_json_safe(response)


def get_agent_state(session_id: str) -> dict:
    if session_id not in AGENT_SESSION_STORE:
        AGENT_SESSION_STORE[session_id] = {
            "session_id": session_id,
            "raw_text": None,
            "constraint_updates": [],
            "combined_text": None,
            "last_result": None,
            "parsed_result": None,
            "top_k": None,
            "recommendation": None,
            "layout_result": None,
            "history": [],
            "iteration": 0,
            "last_called_tools": []
        }

    return AGENT_SESSION_STORE[session_id]


def save_agent_state(session_id: str, state: dict):
    AGENT_SESSION_STORE[session_id] = state


def build_combined_text_from_state(state: dict) -> str:
    raw_text = state.get("raw_text")
    updates = state.get("constraint_updates") or []

    if not raw_text:
        return ""

    if not updates:
        return raw_text

    return f"{raw_text}。补充修改要求：" + "；".join(updates)


def classify_agent_intent(message: str, state: dict) -> str:
    text = message.strip()
    has_task = state.get("raw_text") is not None

    if not has_task:
        return "new_design_task"

    if any(k in text for k in ["重新设计", "新的需求", "换个需求", "重新开始"]):
        return "new_design_task"

    if any(k in text for k in ["短一点", "更短", "太长", "紧凑", "总长"]):
        return "modify_constraint"

    if any(k in text for k in ["F数更小", "F 数更小", "光圈更大", "低照度", "更亮"]):
        return "modify_constraint"

    if any(k in text for k in ["视场更大", "视场角更大", "广角一点", "角度更大", "更接近120", "120度"]):
        return "modify_constraint"

    if any(k in text for k in ["重新检索", "换一批", "重新推荐", "还有别的吗", "再找几个"]):
        return "retrieve_again"

    if any(k in text for k in ["解释", "为什么", "原因", "推荐理由", "第一个", "第二个", "第2个"]):
        return "explain_result"

    if any(k in text for k in ["ray tracing", "ray spread", "光线追迹", "重新评价", "仿真"]):
        return "run_evaluation"

    if any(k in text for k in ["结构图", "镜头图", "布局图", "画图", "layout", "optiland"]):
        return "draw_layout"

    return "general_followup"


def build_agent_answer(intent: str, result: Optional[dict], called_tools: List[str]) -> str:
    if result is None:
        return f"已识别意图：{intent}。本轮调用工具：{', '.join(called_tools)}。当前还没有可解释的设计结果。"

    recommendation = result.get("recommendation")
    feasibility = result.get("feasibility_result") or {}
    base_lens = feasibility.get("base_lens")
    top_k = result.get("top_k") or []
    layout_result = result.get("layout_result") or {}

    parts = []
    parts.append(f"已识别意图：{intent}。")
    parts.append(f"本轮调用工具：{', '.join(called_tools)}。")

    if base_lens:
        parts.append(f"当前基础候选结构为：{base_lens}。")

    if layout_result.get("image_url"):
        parts.append(f"结构图已生成：{layout_result.get('image_url')}。")

    if recommendation:
        parts.append(f"推荐说明：{recommendation}")

    if top_k:
        parts.append(f"当前返回候选数量：{len(top_k)}。")

    return "\n".join(parts)


def update_agent_state_with_result(state: dict, result: dict, called_tools: List[str]) -> dict:
    state["last_result"] = result
    state["parsed_result"] = result.get("parsed_result")
    state["top_k"] = result.get("top_k")
    state["recommendation"] = result.get("recommendation")
    state["layout_result"] = result.get("layout_result")
    state["last_called_tools"] = called_tools
    state["iteration"] += 1
    return state


@app.get("/")
def root():
    return {
        "message": "OpticsGPT Lens Design API is running",
        "model": MODEL_NAME,
        "version": "1.5.0-agent-optiland-layout"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": tokenizer is not None and model is not None,
        "model_name": MODEL_NAME,
        "model_dir": MODEL_DIR,
        "lens_count": len(LENS_DATABASE),
        "agent_session_count": len(AGENT_SESSION_STORE)
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "local"
            }
        ]
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    if not request.messages:
        raise HTTPException(status_code=400, detail="messages 不能为空")

    temperature = request.temperature if request.temperature is not None else 0.7
    top_p = request.top_p if request.top_p is not None else 0.9
    max_tokens = request.max_tokens if request.max_tokens is not None else 512

    if request.stream:
        def event_generator():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            yield sse_format({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }
                ]
            })

            try:
                for piece in stream_generate(
                    request.messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                ):
                    yield sse_format({
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None
                            }
                        ]
                    })

                yield sse_format({
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                })
                yield "data: [DONE]\n\n"

            except Exception as e:
                yield sse_format({
                    "error": {
                        "message": str(e),
                        "type": "server_error"
                    }
                })
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    content = generate_text(
        request.messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return JSONResponse(content=make_chat_response(content, request.model))


@app.post("/parse_requirement")
def parse_requirement(request: RequirementParseRequest):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")

    raw_parsed = parse_requirement_text(request.text)
    parsed_result, input_quality, completion_notes = validate_and_complete_requirements(
        request.text,
        raw_parsed
    )

    return make_json_safe({
        "raw_text": request.text,
        "parsed_result": parsed_result,
        "input_quality": input_quality,
        "completion_notes": completion_notes,
        "input_understanding": build_input_understanding(parsed_result)
    })


@app.post("/search_lens")
def search_lens(request: SearchLensRequest):
    parsed_result = request.parsed_result
    top_k = request.top_k if request.top_k is not None else 9

    if "error" in parsed_result:
        return {
            "parsed_result": parsed_result,
            "top_k": [],
            "message": "需求解析失败，无法进行镜头检索。"
        }

    candidates = search_lens_candidates(parsed_result, top_k=top_k)

    return make_json_safe({
        "parsed_result": parsed_result,
        "input_understanding": build_input_understanding(parsed_result),
        "top_k": candidates
    })


@app.post("/design_feasibility")
def design_feasibility(request: DesignFeasibilityRequest):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")

    result = run_design_assist_pipeline(
        user_text=request.text,
        top_k=request.top_k or 9
    )

    return make_json_safe(result)


@app.post("/design_assist")
def design_assist(request: DesignAssistRequest):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")

    return run_design_assist_pipeline(
        user_text=request.text,
        top_k=request.top_k or 9
    )


@app.post("/layout/optiland/generate")
def generate_optiland_layout(request: LayoutGenerateRequest):
    if not request.lens_id:
        raise HTTPException(status_code=400, detail="lens_id 不能为空")

    seq_path = request.seq_path

    if not seq_path:
        guess = Path("data/codev_blocks") / f"{request.lens_id}_clean_block.seq"
        if guess.exists():
            seq_path = str(guess)

    if not seq_path:
        raise HTTPException(
            status_code=404,
            detail=f"找不到 {request.lens_id} 对应的 seq 文件"
        )

    try:
        result = generate_optiland_layout_from_seq(
            lens_id=request.lens_id,
            seq_path=seq_path,
            output_dir="static/optiland_layouts",
        )
        return make_json_safe(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/chat")
def agent_chat(request: AgentChatRequest):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")

    state = get_agent_state(request.session_id)

    state["history"].append({
        "role": "user",
        "content": request.message
    })

    intent = classify_agent_intent(request.message, state)
    called_tools: List[str] = []
    result: Optional[dict] = None

    if intent == "new_design_task":
        user_text = request.message

        called_tools = [
            "parse_requirement",
            "validate_and_complete_requirements",
            "kg_enhance",
            "aperture_scale",
            "hybrid_retrieve",
            "kg_candidate_check",
            "scale_adapt",
            "raytrace_rerank",
            "hard_constraint_check",
            "generate_optiland_layout",
            "optimize_recommendation",
            "save_design_state"
        ]

        result = run_design_assist_pipeline(
            user_text=user_text,
            top_k=request.top_k or 9
        )

        state["raw_text"] = user_text
        state["constraint_updates"] = []
        state["combined_text"] = user_text
        state = update_agent_state_with_result(state, result, called_tools)

    elif intent == "modify_constraint":
        if not state.get("raw_text"):
            user_text = request.message

            called_tools = [
                "parse_requirement",
                "validate_and_complete_requirements",
                "kg_enhance",
                "hybrid_retrieve",
                "generate_optiland_layout",
                "save_design_state"
            ]

            result = run_design_assist_pipeline(
                user_text=user_text,
                top_k=request.top_k or 9
            )

            state["raw_text"] = user_text
            state["constraint_updates"] = []
            state["combined_text"] = user_text
            state = update_agent_state_with_result(state, result, called_tools)

        else:
            state.setdefault("constraint_updates", [])
            state["constraint_updates"].append(request.message)
            combined_text = build_combined_text_from_state(state)

            called_tools = [
                "load_design_state",
                "merge_user_constraint",
                "parse_requirement",
                "validate_and_complete_requirements",
                "kg_enhance",
                "aperture_scale",
                "hybrid_retrieve",
                "kg_candidate_check",
                "scale_adapt",
                "raytrace_rerank",
                "hard_constraint_check",
                "generate_optiland_layout",
                "optimize_recommendation",
                "update_design_state"
            ]

            result = run_design_assist_pipeline(
                user_text=combined_text,
                top_k=request.top_k or 9
            )

            state["combined_text"] = combined_text
            state = update_agent_state_with_result(state, result, called_tools)

    elif intent == "retrieve_again":
        base_text = state.get("combined_text") or state.get("raw_text") or request.message

        called_tools = [
            "load_design_state",
            "hybrid_retrieve",
            "kg_candidate_check",
            "scale_adapt",
            "raytrace_rerank",
            "hard_constraint_check",
            "generate_optiland_layout",
            "optimize_recommendation",
            "update_design_state"
        ]

        result = run_design_assist_pipeline(
            user_text=base_text,
            top_k=request.top_k or 9
        )

        state = update_agent_state_with_result(state, result, called_tools)

    elif intent == "run_evaluation":
        base_text = state.get("combined_text") or state.get("raw_text") or request.message

        called_tools = [
            "load_design_state",
            "raytrace_rerank",
            "hard_constraint_check",
            "generate_optiland_layout",
            "optimize_recommendation",
            "update_design_state"
        ]

        result = run_design_assist_pipeline(
            user_text=base_text,
            top_k=request.top_k or 9
        )

        state = update_agent_state_with_result(state, result, called_tools)

    elif intent == "draw_layout":
        called_tools = [
            "load_design_state",
            "generate_optiland_layout",
            "update_design_state"
        ]

        result = state.get("last_result")

        if result and result.get("top_k"):
            hybrid_result = {
                "top_k": result.get("top_k"),
                "feasibility_result": result.get("feasibility_result"),
            }
            hybrid_result, layout_result = attach_optiland_layout_to_best_candidate(hybrid_result)
            result["top_k"] = hybrid_result.get("top_k")
            result["layout_result"] = layout_result
            result["feasibility_result"] = hybrid_result.get("feasibility_result")
            state = update_agent_state_with_result(state, result, called_tools)

    elif intent == "explain_result":
        called_tools = [
            "load_design_state",
            "explain_current_result"
        ]
        result = state.get("last_result")

    else:
        called_tools = [
            "load_design_state",
            "general_response"
        ]
        result = state.get("last_result")

    answer = build_agent_answer(
        intent=intent,
        result=result,
        called_tools=called_tools
    )

    state["history"].append({
        "role": "assistant",
        "content": answer
    })

    save_agent_state(request.session_id, state)

    return make_json_safe({
        "session_id": request.session_id,
        "intent": intent,
        "called_tools": called_tools,
        "design_state": {
            "raw_text": state.get("raw_text"),
            "constraint_updates": state.get("constraint_updates"),
            "combined_text": state.get("combined_text"),
            "parsed_result": state.get("parsed_result"),
            "iteration": state.get("iteration"),
            "last_called_tools": state.get("last_called_tools"),
            "layout_result": state.get("layout_result"),
            "history": state.get("history"),
        },
        "result": result,
        "answer": answer
    })


@app.post("/agent/langgraph/chat")
def agent_langgraph_chat(request: AgentChatRequest):
    try:
        from agent.langgraph_agent import LangGraphOpticalDesignAgent
        from agent.tools import ToolRegistry

        registry = ToolRegistry()

        def _parse_requirement_for_graph(user_text: str):
            parsed = parse_requirement_text(user_text)
            parsed = infer_constraint_type_from_text(user_text, parsed)
            parsed, _notes = validate_and_complete_requirements(parsed, user_text)
            parsed, _kg_notes = enhance_requirement_with_kg(parsed)
            parsed, _aperture_notes, _scale_info = enhance_requirement_with_aperture_scale(user_text, parsed)
            return parsed

        def _retrieve_for_graph(parsed_requirement: Dict[str, Any], raw_text: Optional[str] = None, top_k: int = 9):
            hybrid_result = run_hybrid_design_pipeline(
                parsed_result=parsed_requirement,
                lens_database=LENS_DATABASE,
                top_k=top_k,
            )
            hybrid_result = apply_scale_to_hybrid_result(hybrid_result, parsed_requirement)
            hybrid_result = tighten_feasibility_by_recall_and_scale(hybrid_result)
            hybrid_result = optimize_hybrid_result_after_scale(
                hybrid_result=hybrid_result,
                parsed_result=parsed_requirement,
                user_text=raw_text or "",
            )
            return hybrid_result.get("top_k") or []

        def _rerank_for_graph(parsed_requirement: Dict[str, Any], candidates: List[Dict[str, Any]]):
            return {
                "top_candidates": candidates[: request.top_k or 9],
                "ranking_reason": "候选已由混合检索、尺度适配、ray tracing 和多目标优化 pipeline 排序。",
            }

        def _raytrace_for_graph(candidates: List[Dict[str, Any]]):
            return {
                "raytrace_reranked_candidates": candidates,
                "raytrace_summary": "ray tracing 信号已在混合检索 pipeline 中接入；LangGraph 节点保留评价阶段语义。",
            }

        def _explain_for_graph(
            parsed_requirement: Dict[str, Any],
            candidates: List[Dict[str, Any]],
            raytrace_result: Optional[Dict[str, Any]] = None,
        ):
            hybrid_result = {
                "top_k": candidates,
                "feasibility_result": evaluate_hybrid_feasibility(candidates),
            }
            return build_safe_recommendation(hybrid_result, parsed_requirement)

        registry.register("parse_requirement", _parse_requirement_for_graph, "Parse optical requirement for LangGraph.")
        registry.register("retrieve_candidates", _retrieve_for_graph, "Run hybrid retrieval for LangGraph.")
        registry.register("rerank_candidates", _rerank_for_graph, "Expose rerank stage for LangGraph.")
        registry.register("run_raytrace", _raytrace_for_graph, "Expose raytrace stage for LangGraph.")
        registry.register("explain_recommendation", _explain_for_graph, "Explain LangGraph recommendation.")

        langgraph_agent = LangGraphOpticalDesignAgent(registry=registry)

        result = langgraph_agent.step(
            session_id=request.session_id,
            user_message=request.message,
            top_k=request.top_k or 9,
        )
        return make_json_safe(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
