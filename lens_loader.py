import os
from typing import List, Dict, Any
import pandas as pd


COLUMN_ALIASES = {
    "lens_id": ["lens_id", "文件名", "file_name", "name", "编号"],
    "f_number": ["f_number", "F数", "F-number", "Fno", "FNO"],
    "half_fov": ["half_fov", "半视场角", "半视场", "half field", "half field angle"],
    "total_length": ["total_length", "总长", "TTL", "ttl"],
    "focal_length": ["focal_length", "焦距", "EFL", "efl"],
    "application_scene": ["application_scene", "应用场景", "scene", "场景"],
    "element_count": ["element_count", "镜片数量", "片数", "lens_count"],
    "distortion": ["distortion", "畸变"],
    "low_light_performance": ["low_light_performance", "低照度表现", "低照度"]
}


def _find_column(df: pd.DataFrame, aliases: List[str]):
    cols = list(df.columns)
    for alias in aliases:
        for c in cols:
            if str(c).strip().lower() == str(alias).strip().lower():
                return c
    return None


def _safe_to_numeric(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_lens_data(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"镜头数据文件不存在: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("仅支持 .csv / .xlsx / .xls 文件")

    col_map = {}
    for std_name, aliases in COLUMN_ALIASES.items():
        real_col = _find_column(df, aliases)
        if real_col is not None:
            col_map[std_name] = real_col

    required_minimum = ["lens_id", "f_number", "half_fov", "total_length", "focal_length"]
    missing = [x for x in required_minimum if x not in col_map]
    if missing:
        raise ValueError(f"镜头表缺少关键字段: {missing}")

    normalized = pd.DataFrame()

    for std_name, real_col in col_map.items():
        normalized[std_name] = df[real_col]

    for num_col in ["f_number", "half_fov", "total_length", "focal_length", "element_count", "distortion"]:
        if num_col in normalized.columns:
            normalized = _safe_to_numeric(normalized, num_col)

    normalized["full_fov"] = normalized["half_fov"] * 2

    for optional_col in ["application_scene", "element_count", "distortion", "low_light_performance"]:
        if optional_col not in normalized.columns:
            normalized[optional_col] = None

    normalized = normalized.dropna(subset=["lens_id", "f_number", "half_fov", "total_length", "focal_length"])

    return normalized.to_dict(orient="records")