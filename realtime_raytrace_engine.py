from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
DAT_DIR = Path(os.getenv("DAT_DIR", PROJECT_ROOT / "data" / "dat")).resolve()
CODEV_BLOCK_DIR = Path(os.getenv("CODEV_BLOCK_DIR", PROJECT_ROOT / "data" / "codev_blocks")).resolve()
RAYTRACE_CACHE_PATH = Path(os.getenv("RAYTRACE_CACHE_PATH", PROJECT_ROOT / "data" / "raytrace_cache" / "realtime_raytrace_cache.json")).resolve()

# 实时 ray tracing 默认开启。若服务器太慢，可以启动时设置 REALTIME_RAYTRACE=0 关闭。
REALTIME_RAYTRACE = os.getenv("REALTIME_RAYTRACE", "1") == "1"

# 为了保证接口不太慢，默认只评估 Top9，每个镜头 3 个视场、每个视场 32 条光线。
DEFAULT_FIELD_YS = [0.0, 0.7, 1.0]
DEFAULT_NUM_RAYS = int(os.getenv("RAYTRACE_NUM_RAYS", "32"))
DEFAULT_WAVELENGTH = float(os.getenv("RAYTRACE_WAVELENGTH", "0.546"))
USE_CACHE = os.getenv("RAYTRACE_USE_CACHE", "1") == "1"


# ============================================================
# 基础工具
# ============================================================
def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, str) and not x.strip():
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _read_cache() -> Dict[str, Any]:
    if not USE_CACHE or not RAYTRACE_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(RAYTRACE_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_cache(cache: Dict[str, Any]) -> None:
    if not USE_CACHE:
        return
    try:
        RAYTRACE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAYTRACE_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _cache_key(lens_id: str, seq_path: Optional[Path]) -> str:
    if seq_path and seq_path.exists():
        try:
            return f"{lens_id}|{str(seq_path)}|{seq_path.stat().st_mtime}"
        except Exception:
            return f"{lens_id}|{str(seq_path)}"
    return lens_id


# ============================================================
# 从 dat/cmb 文件提取镜头块，并生成 clean_block.seq
# ============================================================
def extract_lens_block_from_file(lens_id: str, file_path: Path) -> Dict[str, Any]:
    result = {
        "found": False,
        "cmb_file": str(file_path),
        "lens_id": lens_id,
        "block_lines": [],
        "block_text": "",
        "line_count": 0,
    }
    if not file_path.exists():
        return result

    prefix = f"{lens_id.lower()}_"
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    collecting = False
    block_lines: List[str] = []
    for line in lines:
        stripped = line.rstrip("\n")
        lower = stripped.lower()

        if lower.startswith(prefix):
            collecting = True
            block_lines.append(stripped)
            continue

        if collecting:
            # 新 lens 块开始，停止收集。
            if lower.startswith("or") and "_" in lower:
                first_token = lower.split()[0]
                if first_token.startswith("or") and "_" in first_token:
                    break
            if stripped.strip() == "":
                block_lines.append(stripped)
                continue

    if block_lines:
        result["found"] = True
        result["block_lines"] = block_lines
        result["block_text"] = "\n".join(block_lines)
        result["line_count"] = len(block_lines)
    return result


def find_lens_block_in_dat_files(lens_id: str, dat_dir: Path = DAT_DIR) -> Dict[str, Any]:
    final_result = {
        "selected_lens_id": lens_id,
        "found": False,
        "cmb_file": None,
        "block_lines": [],
        "block_text": "",
        "line_count": 0,
        "matched_file_name": None,
    }
    if not lens_id or not dat_dir.exists():
        return final_result

    candidate_files: List[Path] = []
    candidate_files.extend(sorted(dat_dir.glob("cmb*.dat")))
    candidate_files.extend(sorted(dat_dir.glob("cmb*.bat")))

    for file_path in candidate_files:
        r = extract_lens_block_from_file(lens_id, file_path)
        if r["found"]:
            final_result.update({
                "found": True,
                "cmb_file": str(file_path),
                "matched_file_name": file_path.name,
                "block_lines": r["block_lines"],
                "block_text": r["block_text"],
                "line_count": r["line_count"],
            })
            return final_result
    return final_result


def clean_lens_block_prefix(raw_file: Path, lens_id: str, output_dir: Path = CODEV_BLOCK_DIR) -> Dict[str, Any]:
    result = {
        "cleaned": False,
        "raw_file": str(raw_file),
        "clean_file": None,
        "line_count": 0,
    }
    if not raw_file.exists():
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_pattern = re.compile(rf"^{re.escape(lens_id)}_\s*", re.IGNORECASE)

    cleaned_lines: List[str] = []
    with open(raw_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.rstrip("\n")
            cleaned_lines.append(prefix_pattern.sub("", stripped))

    clean_path = output_dir / f"{lens_id}_clean_block.seq"
    clean_path.write_text("\n".join(cleaned_lines), encoding="utf-8")

    result.update({
        "cleaned": True,
        "clean_file": str(clean_path),
        "line_count": len(cleaned_lines),
    })
    return result


def prepare_seq_for_lens(lens_id: str) -> Dict[str, Any]:
    """
    实时评估前准备 seq：
    1. 如果 data/codev_blocks/<lens_id>_clean_block.seq 已存在，直接用。
    2. 否则从 data/dat/cmb*.dat 中提取镜头块。
    3. 保存 raw_block.txt 和 clean_block.seq。
    """
    CODEV_BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = CODEV_BLOCK_DIR / f"{lens_id}_clean_block.seq"
    raw_path = CODEV_BLOCK_DIR / f"{lens_id}_raw_block.txt"

    if clean_path.exists():
        return {
            "ok": True,
            "status": "seq_already_exists",
            "seq_path": str(clean_path),
            "raw_block_path": str(raw_path) if raw_path.exists() else None,
            "matched_file_name": None,
            "error": None,
        }

    block = find_lens_block_in_dat_files(lens_id, DAT_DIR)
    if not block.get("found"):
        return {
            "ok": False,
            "status": "lens_block_not_found",
            "seq_path": None,
            "raw_block_path": None,
            "matched_file_name": None,
            "error": f"Cannot find lens block for {lens_id} in {DAT_DIR}",
        }

    raw_path.write_text(block.get("block_text", ""), encoding="utf-8")
    clean_result = clean_lens_block_prefix(raw_path, lens_id, CODEV_BLOCK_DIR)
    if not clean_result.get("cleaned"):
        return {
            "ok": False,
            "status": "clean_seq_failed",
            "seq_path": None,
            "raw_block_path": str(raw_path),
            "matched_file_name": block.get("matched_file_name"),
            "error": "raw block exists but clean failed",
        }

    return {
        "ok": True,
        "status": "seq_prepared_from_dat",
        "seq_path": clean_result.get("clean_file"),
        "raw_block_path": str(raw_path),
        "matched_file_name": block.get("matched_file_name"),
        "error": None,
    }


# ============================================================
# SEQ 解析 + Optiland 建模
# ============================================================
def parse_seq(seq_path: str | Path) -> Dict[str, Any]:
    seq_path = Path(seq_path)
    surfaces: List[Dict[str, Any]] = []
    epd = None
    fields_yan: List[float] = []
    stop_next_surface = False

    text = seq_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].upper()

        try:
            if cmd == "EPD" and len(parts) >= 2:
                epd = float(parts[1])
            elif cmd == "YAN":
                fields_yan = [float(x) for x in parts[1:]]
            elif cmd == "STO":
                stop_next_surface = True
            elif cmd == "S":
                if len(parts) < 3:
                    continue
                radius = float(parts[1])
                thickness = float(parts[2])
                material = parts[3] if len(parts) >= 4 else "AIR"
                surfaces.append({
                    "line_no": line_no,
                    "radius": math.inf if abs(radius) < 1e-12 else radius,
                    "thickness": thickness,
                    "material": material,
                    "is_stop": stop_next_surface,
                    "raw": raw,
                })
                stop_next_surface = False
            elif cmd == "PIM":
                surfaces.append({
                    "line_no": line_no,
                    "radius": math.inf,
                    "thickness": 0.0,
                    "material": "AIR",
                    "is_stop": False,
                    "raw": raw,
                    "note": "image plane",
                })
        except Exception:
            continue

    return {"epd": epd, "fields_yan": fields_yan, "surfaces": surfaces}


def _codev_material_to_optiland(mat: str):
    # 放在函数里 import，避免没装 optiland 时整个 API 启动失败。
    from optiland.materials import IdealMaterial

    if mat == "AIR":
        return "air"
    try:
        value = float(mat)
        integer_part = int(value)
        n = 1.0 + integer_part / 1_000_000.0
        return IdealMaterial(n=n, k=0)
    except Exception:
        return IdealMaterial(n=1.5, k=0)


def build_lens_from_parsed(parsed: Dict[str, Any]):
    from optiland import optic

    lens = optic.Optic()

    if parsed.get("epd") is not None:
        lens.set_aperture(aperture_type="EPD", value=parsed["epd"])

    lens.fields.set_type(field_type="angle")
    fields = parsed.get("fields_yan") or [0.0]
    for y in fields:
        lens.fields.add(y=float(y))

    lens.wavelengths.add(value=DEFAULT_WAVELENGTH, is_primary=True)

    for i, s in enumerate(parsed.get("surfaces", [])):
        material = _codev_material_to_optiland(s.get("material", "AIR"))
        lens.add_surface(
            index=i,
            radius=s.get("radius"),
            thickness=s.get("thickness"),
            material=material,
            is_stop=s.get("is_stop", False),
        )

    lens.update()
    return lens


def evaluate_lens_basic(lens: Any, parsed: Dict[str, Any]) -> Dict[str, Any]:
    parsed_total_track = sum(float(s.get("thickness", 0.0)) for s in parsed.get("surfaces", []))
    epd = parsed.get("epd")
    focal_length_est = parsed_total_track * 0.8
    estimated_f_number = focal_length_est / epd if epd and epd > 0 else None

    try:
        surface_count = lens.surfaces.num_surfaces
    except Exception:
        surface_count = len(parsed.get("surfaces", []))

    try:
        optiland_total_track = float(lens.total_track)
    except Exception:
        optiland_total_track = float(parsed_total_track)

    return {
        "surface_count": surface_count,
        "total_track": float(parsed_total_track),
        "optiland_total_track": optiland_total_track,
        "epd": epd,
        "max_half_fov": max(parsed.get("fields_yan") or [0.0]),
        "estimated_focal_length": focal_length_est,
        "estimated_f_number": estimated_f_number,
    }


# ============================================================
# 实时 ray tracing
# ============================================================
def trace_single_field(lens: Any, field_y: float, num_rays: int = DEFAULT_NUM_RAYS, wavelength: float = DEFAULT_WAVELENGTH) -> Dict[str, Any]:
    # 多 seed 追迹，提高稳定性。
    seeds = [11, 22, 33]
    spreads: List[float] = []
    valid_counts: List[int] = []

    for seed in seeds:
        np.random.seed(seed)
        try:
            rays = lens.trace(
                Hx=0,
                Hy=float(field_y),
                wavelength=wavelength,
                num_rays=num_rays,
                distribution="random",
            )
            x = np.array(rays.x, dtype=float)
            y = np.array(rays.y, dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            valid_x = x[valid]
            valid_y = y[valid]
            valid_counts.append(int(valid.sum()))
            if len(valid_x) > 2:
                spread = float(np.sqrt(np.std(valid_x) ** 2 + np.std(valid_y) ** 2))
                if math.isfinite(spread):
                    spreads.append(spread)
        except Exception:
            valid_counts.append(0)
            continue

    if not spreads:
        return {
            "success": False,
            "field_y": float(field_y),
            "ray_spread": None,
            "valid_ray_count": int(max(valid_counts) if valid_counts else 0),
            "status": "no_valid_rays",
        }

    return {
        "success": True,
        "field_y": float(field_y),
        "ray_spread": float(np.mean(spreads)),
        "ray_spread_std": float(np.std(spreads)),
        "valid_ray_count": int(max(valid_counts) if valid_counts else 0),
        "status": "success",
        "samples": spreads,
    }


def compute_raytrace_score(ray_spread: Optional[float], valid: bool) -> float:
    if not valid or ray_spread is None:
        return 0.0
    # spread 越小越好。0.5 以下优秀，5 左右一般，20 以上很差。
    return float(max(0.0, min(1.0, 1.0 / (1.0 + ray_spread / 3.0))))


def realtime_evaluate_lens(lens_id: str, field_ys: Optional[List[float]] = None, num_rays: int = DEFAULT_NUM_RAYS) -> Dict[str, Any]:
    started = time.time()
    field_ys = field_ys or DEFAULT_FIELD_YS

    if not REALTIME_RAYTRACE:
        return {
            "raytrace_valid": False,
            "raytrace_status": "disabled",
            "raytrace_error": "REALTIME_RAYTRACE=0",
            "ray_spread": None,
            "raytrace_score": 0.0,
        }

    prepare = prepare_seq_for_lens(lens_id)
    seq_path = Path(prepare["seq_path"]) if prepare.get("seq_path") else None
    cache = _read_cache()
    key = _cache_key(lens_id, seq_path)
    if USE_CACHE and key in cache:
        cached = cache[key]
        cached["raytrace_status"] = "realtime_cache"
        return cached

    if not prepare.get("ok") or not seq_path:
        result = {
            "lens_id": lens_id,
            "raytrace_valid": False,
            "raytrace_status": prepare.get("status"),
            "seq_prepare_status": prepare.get("status"),
            "matched_file_name": prepare.get("matched_file_name"),
            "seq_path": prepare.get("seq_path"),
            "raw_block_path": prepare.get("raw_block_path"),
            "ray_spread": None,
            "raytrace_score": 0.0,
            "raytrace_error": prepare.get("error"),
            "runtime_sec": round(time.time() - started, 4),
        }
        return result

    try:
        parsed = parse_seq(seq_path)
        lens = build_lens_from_parsed(parsed)
        eval_result = evaluate_lens_basic(lens, parsed)

        field_results = [trace_single_field(lens, fy, num_rays=num_rays) for fy in field_ys]
        valid_results = [r for r in field_results if r.get("success") and r.get("ray_spread") is not None]

        if not valid_results:
            result = {
                "lens_id": lens_id,
                "raytrace_valid": False,
                "raytrace_status": "no_valid_field_trace",
                "seq_prepare_status": prepare.get("status"),
                "matched_file_name": prepare.get("matched_file_name"),
                "seq_path": str(seq_path),
                "raw_block_path": prepare.get("raw_block_path"),
                "ray_spread": None,
                "raytrace_score": 0.0,
                "field_results": field_results,
                "rt_estimated_f_number": eval_result.get("estimated_f_number"),
                "rt_estimated_focal_length": eval_result.get("estimated_focal_length"),
                "rt_total_track": eval_result.get("total_track"),
                "raytrace_error": None,
                "runtime_sec": round(time.time() - started, 4),
            }
            return result

        # 用最大视场 spread 作为主指标，同时保留均值。广角设计更关注边缘视场表现。
        spreads = [float(r["ray_spread"]) for r in valid_results]
        ray_spread_mean = float(np.mean(spreads))
        ray_spread_max = float(np.max(spreads))
        ray_spread = ray_spread_max
        score = compute_raytrace_score(ray_spread, True)

        result = {
            "lens_id": lens_id,
            "raytrace_valid": True,
            "raytrace_status": "realtime_success",
            "seq_prepare_status": prepare.get("status"),
            "matched_file_name": prepare.get("matched_file_name"),
            "seq_path": str(seq_path),
            "raw_block_path": prepare.get("raw_block_path"),
            "ray_spread": ray_spread,
            "ray_spread_mean": ray_spread_mean,
            "ray_spread_max": ray_spread_max,
            "raytrace_score": score,
            "field_results": field_results,
            "rt_estimated_f_number": eval_result.get("estimated_f_number"),
            "rt_estimated_focal_length": eval_result.get("estimated_focal_length"),
            "rt_total_track": eval_result.get("total_track"),
            "rt_surface_count": eval_result.get("surface_count"),
            "raytrace_error": None,
            "runtime_sec": round(time.time() - started, 4),
        }

        if USE_CACHE:
            cache[key] = result
            _write_cache(cache)
        return result

    except Exception as e:
        return {
            "lens_id": lens_id,
            "raytrace_valid": False,
            "raytrace_status": "realtime_error",
            "seq_prepare_status": prepare.get("status"),
            "matched_file_name": prepare.get("matched_file_name"),
            "seq_path": str(seq_path),
            "raw_block_path": prepare.get("raw_block_path"),
            "ray_spread": None,
            "raytrace_score": 0.0,
            "field_results": [],
            "raytrace_error": str(e),
            "runtime_sec": round(time.time() - started, 4),
        }


def add_realtime_raytrace_to_topk(topk_df: pd.DataFrame, max_candidates: int = 9) -> pd.DataFrame:
    if topk_df is None or topk_df.empty:
        return topk_df.copy()

    out = topk_df.copy().head(max_candidates).reset_index(drop=True)

    rt_rows: List[Dict[str, Any]] = []
    for _, row in out.iterrows():
        lens_id = str(row.get("lens_id"))
        rt = realtime_evaluate_lens(lens_id)
        rt_rows.append(rt)

    rt_df = pd.DataFrame(rt_rows)
    if rt_df.empty:
        out["raytrace_status"] = "realtime_empty"
        out["ray_spread"] = None
        out["raytrace_score"] = 0.0
        return out

    keep_cols = [
        "lens_id", "raytrace_valid", "raytrace_status", "seq_prepare_status", "matched_file_name",
        "seq_path", "raw_block_path", "ray_spread", "ray_spread_mean", "ray_spread_max", "raytrace_score",
        "rt_estimated_f_number", "rt_estimated_focal_length", "rt_total_track", "rt_surface_count",
        "raytrace_error", "runtime_sec", "field_results",
    ]
    keep_cols = [c for c in keep_cols if c in rt_df.columns]
    out = out.merge(rt_df[keep_cols], on="lens_id", how="left")

    # 有实时 raytrace 时，重新排序：参数/结构 70%，实时物理表现 30%。
    out["raytrace_score"] = pd.to_numeric(out.get("raytrace_score"), errors="coerce").fillna(0.0)
    out["final_score"] = pd.to_numeric(out.get("final_score"), errors="coerce").fillna(0.0)
    out["rerank_score"] = 0.70 * out["final_score"] + 0.30 * out["raytrace_score"]

    # raytrace 无效的候选降权，但不直接删掉，方便用户看到失败原因。
    out["raytrace_valid_num"] = out.get("raytrace_valid", False).apply(lambda x: 1 if bool(x) else 0)
    out = out.sort_values(
        by=["raytrace_valid_num", "rerank_score"],
        ascending=[False, False]
    ).drop(columns=["raytrace_valid_num"]).reset_index(drop=True)

    return out
