import math
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        s = str(x).strip().replace("D", "E").replace("d", "e")
        if s.upper() in ["INF", "INFINITY", "INFINITE"]:
            return math.inf
        v = float(s)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def _first_float(text: str, default=None):
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?", text)
    if not m:
        return default
    return _safe_float(m.group(0), default)


def _all_floats(text: str) -> List[float]:
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?", text)
    out = []
    for n in nums:
        v = _safe_float(n)
        if v is not None:
            out.append(v)
    return out


def _clean_line(line: str) -> str:
    line = line.strip()

    # 去掉 or00001_ 这种前缀
    line = re.sub(r"^or\d+_\s*", "", line)

    # 去掉注释
    if "!" in line:
        line = line.split("!", 1)[0].strip()

    return line.strip()


def _normalize_material(mat: Optional[str]) -> Optional[str]:
    if not mat:
        return None

    mat = str(mat).strip().replace("'", "").replace('"', "")

    if not mat:
        return None

    upper = mat.upper()

    if upper in ["AIR", "NONE", "NULL", "VACUUM"]:
        return None

    # 常见 CodeV/专利库材料兜底
    mapping = {
        "BK7": "N-BK7",
        "N_BK7": "N-BK7",
        "NBK7": "N-BK7",
        "F2": "N-F2",
        "SF11": "N-SF11",
        "NSF11": "N-SF11",
        "N_SF11": "N-SF11",
    }

    if upper in mapping:
        return mapping[upper]

    # Optiland 对玻璃库材料名比较敏感。
    # 如果后续报错，会在 add surface 时自动去掉 material 重试。
    return mat


def parse_codev_seq(seq_path: str) -> Dict[str, Any]:
    """
    尽量兼容 CodeV seq / clean_block.seq 的轻量解析器。

    支持常见命令：
    - EPD
    - WL
    - YAN
    - STO
    - RDY / R / RADIUS
    - THI / T / THICKNESS
    - GLA / GLASS / G

    如果你的 seq 格式更特殊，可以后续在这里扩展。
    """
    path = Path(seq_path)
    if not path.exists():
        raise FileNotFoundError(f"seq 文件不存在: {seq_path}")

    lines = [_clean_line(x) for x in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    lines = [x for x in lines if x]

    epd = None
    wavelengths = []
    fields = []

    surfaces: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {}
    pending_stop = False

    def flush_current():
        nonlocal cur
        if not cur:
            return

        radius = cur.get("radius")
        thickness = cur.get("thickness")
        material = cur.get("material")
        is_stop = bool(cur.get("is_stop"))

        if radius is None and thickness is None and material is None and not is_stop:
            cur = {}
            return

        surfaces.append({
            "radius": radius if radius is not None else math.inf,
            "thickness": thickness if thickness is not None else 0.0,
            "material": _normalize_material(material),
            "is_stop": is_stop,
        })
        cur = {}

    for raw in lines:
        if not raw:
            continue

        parts = raw.split()
        if not parts:
            continue

        cmd = parts[0].upper()
        rest = " ".join(parts[1:])

        if cmd == "EPD":
            epd = _first_float(rest, epd)
            continue

        if cmd == "WL":
            vals = _all_floats(rest)
            for v in vals:
                # CodeV 常用 nm，比如 656/546/435；Optiland 用 um
                wavelengths.append(v / 1000.0 if v > 10 else v)
            continue

        if cmd in ["YAN", "XAN"]:
            vals = _all_floats(rest)
            for v in vals:
                if abs(v) not in fields:
                    fields.append(abs(v))
            continue

        # STOP 面
        if cmd in ["STO", "STOP"]:
            if cur:
                cur["is_stop"] = True
            else:
                pending_stop = True
            continue

        # 新 surface 标记
        if cmd in ["S", "SO", "SUR", "SURF", "SRF"]:
            flush_current()
            cur = {}
            if pending_stop:
                cur["is_stop"] = True
                pending_stop = False
            continue

        # 半径
        if cmd in ["RDY", "RDX", "R", "RAD", "RADIUS"]:
            # 遇到新的 RDY，通常意味着新 surface
            if cur and ("radius" in cur or "thickness" in cur or "material" in cur):
                flush_current()
                cur = {}

            if pending_stop:
                cur["is_stop"] = True
                pending_stop = False

            value = _first_float(rest)
            cur["radius"] = value if value is not None else math.inf
            continue

        # 曲率，转 radius
        if cmd in ["CUY", "CUX", "CURV", "CURVATURE"]:
            if cur and ("radius" in cur or "thickness" in cur or "material" in cur):
                flush_current()
                cur = {}

            if pending_stop:
                cur["is_stop"] = True
                pending_stop = False

            curvature = _first_float(rest)
            if curvature is None or abs(curvature) < 1e-12:
                cur["radius"] = math.inf
            else:
                cur["radius"] = 1.0 / curvature
            continue

        # 厚度
        if cmd in ["THI", "TH", "T", "THICKNESS"]:
            value = _first_float(rest)
            if value is not None:
                cur["thickness"] = value
            continue

        # 材料
        if cmd in ["GLA", "GLASS", "G"]:
            mat = rest.strip()
            if mat:
                cur["material"] = mat.split()[0]
            continue

        # 兜底：如果某行像 “radius thickness glass”
        vals = _all_floats(raw)
        if len(vals) >= 2 and cmd not in ["TITLE", "REF", "WTW", "INI", "DIM"]:
            # 避免误解析系统行
            if abs(vals[0]) < 1e6 and abs(vals[1]) < 1e6:
                flush_current()
                mat = None
                tokens = raw.split()
                for token in tokens[2:]:
                    if re.search(r"[A-Za-z]", token):
                        mat = token
                        break
                surfaces.append({
                    "radius": vals[0],
                    "thickness": vals[1],
                    "material": _normalize_material(mat),
                    "is_stop": "STO" in raw.upper() or "STOP" in raw.upper(),
                })

    flush_current()

    # 清理明显无效 surface
    cleaned = []
    for s in surfaces:
        r = s.get("radius")
        t = s.get("thickness")
        if r is None:
            r = math.inf
        if t is None:
            t = 0.0

        # thickness 允许 0，但不能是 nan
        if isinstance(t, float) and math.isnan(t):
            t = 0.0

        cleaned.append({
            "radius": r,
            "thickness": t,
            "material": s.get("material"),
            "is_stop": bool(s.get("is_stop")),
        })

    if not cleaned:
        raise ValueError(f"没有从 seq 中解析出 surface: {seq_path}")

    if epd is None:
        epd = 1.0

    if not fields:
        fields = [0.0]

    if not wavelengths:
        wavelengths = [0.55]

    return {
        "epd": epd,
        "fields": fields,
        "wavelengths": wavelengths,
        "surfaces": cleaned,
    }


def build_optiland_lens(parsed: Dict[str, Any]):
    """
    将解析出的 surface 数据转为 Optiland Optic。
    """
    from optiland import optic

    lens = optic.Optic()

    # object surface
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)

    surfaces = parsed["surfaces"]

    for i, s in enumerate(surfaces, start=1):
        radius = s.get("radius", math.inf)
        thickness = s.get("thickness", 0.0)
        material = s.get("material")
        is_stop = bool(s.get("is_stop"))

        if radius is None or math.isinf(radius):
            radius = np.inf

        if thickness is None or math.isinf(thickness):
            thickness = 0.0

        kwargs = {
            "index": i,
            "radius": radius,
            "thickness": thickness,
            "is_stop": is_stop,
        }

        if material:
            kwargs["material"] = material

        try:
            lens.surfaces.add(**kwargs)
        except Exception:
            # 材料名不兼容时，去掉材料，仅用于画结构图
            kwargs.pop("material", None)
            lens.surfaces.add(**kwargs)

    # image surface
    lens.surfaces.add(index=len(surfaces) + 1)

    epd = parsed.get("epd") or 1.0
    try:
        lens.set_aperture(aperture_type="EPD", value=epd)
    except Exception:
        lens.set_aperture(aperture_type="EPD", value=1.0)

    lens.fields.set_type(field_type="angle")

    fields = parsed.get("fields") or [0.0]
    # 不要加太多，否则图乱
    used_fields = []
    for f in fields:
        if f not in used_fields:
            used_fields.append(f)
    used_fields = used_fields[:3]

    for f in used_fields:
        try:
            lens.fields.add(y=float(f))
        except Exception:
            pass

    wavelengths = parsed.get("wavelengths") or [0.55]
    for idx, wl in enumerate(wavelengths[:3]):
        try:
            lens.wavelengths.add(value=float(wl), is_primary=(idx == 0))
        except Exception:
            pass

    return lens


def draw_fallback_layout(parsed: Dict[str, Any], output_path: str, title: str):
    """
    Optiland 失败时，用 matplotlib 画一个结构图兜底。
    这个不是严格 ray trace，只用于保证前端一定有图。
    """
    surfaces = parsed["surfaces"]
    epd = parsed.get("epd") or 1.0

    semi_aperture = max(epd / 2.0, 1.0)
    draw_height = semi_aperture * 1.4

    z_positions = []
    z = 0.0
    for s in surfaces:
        z_positions.append(z)
        t = s.get("thickness") or 0.0
        if not math.isinf(t):
            z += float(t)

    fig, ax = plt.subplots(figsize=(10, 4))

    y = np.linspace(-draw_height, draw_height, 200)

    for idx, s in enumerate(surfaces):
        z0 = z_positions[idx]
        r = s.get("radius", math.inf)
        is_stop = s.get("is_stop", False)
        mat = s.get("material")

        if r is None or math.isinf(r) or abs(r) > 1e6:
            x = np.full_like(y, z0)
        else:
            r = float(r)
            max_y = min(abs(r) * 0.8, draw_height)
            yy = np.linspace(-max_y, max_y, 200)
            try:
                sag = r - np.sign(r) * np.sqrt(np.maximum(r * r - yy * yy, 0))
                x = z0 + sag
                y_plot = yy
            except Exception:
                x = np.full_like(y, z0)
                y_plot = y
            ax.plot(x, y_plot, linewidth=1.5)
            continue

        if is_stop:
            ax.plot(x, y, linewidth=2.5, linestyle="--")
            ax.text(z0, draw_height * 1.05, "STOP", ha="center", fontsize=8)
        else:
            ax.plot(x, y, linewidth=1.5)

        if mat:
            ax.text(z0, -draw_height * 1.15, str(mat), ha="center", fontsize=7, rotation=45)

    # optical axis
    ax.axhline(0, linewidth=0.8, linestyle=":")

    # 简单示意光线
    if z_positions:
        z_start = min(z_positions)
        z_end = max(z_positions) if max(z_positions) > z_start else z_start + 1
        for h in [-semi_aperture, 0, semi_aperture]:
            ax.plot([z_start, z_end], [h, 0], linewidth=0.8, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("Z [mm]")
    ax.set_ylabel("Y [mm]")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("auto")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_optiland_layout_from_seq(
    lens_id: str,
    seq_path: str,
    output_dir: str = "static/optiland_layouts",
    num_rays: int = 10,
) -> Dict[str, Any]:
    """
    输入 lens_id + seq_path，输出结构图 PNG。
    优先使用 Optiland，失败后用 matplotlib fallback。
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    output_path = output_dir_path / f"{lens_id}_optiland_layout.png"

    parsed = parse_codev_seq(seq_path)

    layout_source = "optiland"
    error = None

    try:
        lens = build_optiland_lens(parsed)
        fig, ax = lens.draw(num_rays=num_rays)
        ax.set_title(f"{lens_id} Optiland Layout")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

    except Exception as e:
        layout_source = "fallback_matplotlib"
        error = str(e)

        draw_fallback_layout(
            parsed=parsed,
            output_path=str(output_path),
            title=f"{lens_id} Layout Preview"
        )

    return {
        "lens_id": lens_id,
        "status": "done",
        "layout_source": layout_source,
        "image_path": str(output_path),
        "image_url": f"/static/optiland_layouts/{output_path.name}",
        "seq_path": str(seq_path),
        "surface_count": len(parsed.get("surfaces") or []),
        "epd": parsed.get("epd"),
        "fields": parsed.get("fields"),
        "wavelengths": parsed.get("wavelengths"),
        "error": error,
    }


def generate_optiland_layout_for_candidate(
    candidate: Dict[str, Any],
    output_dir: str = "static/optiland_layouts",
) -> Dict[str, Any]:
    lens_id = candidate.get("lens_id")
    lens_data = candidate.get("lens_data") or {}

    seq_path = lens_data.get("seq_path")

    if not seq_path and lens_id:
        guess = Path("data/codev_blocks") / f"{lens_id}_clean_block.seq"
        if guess.exists():
            seq_path = str(guess)

    if not lens_id:
        raise ValueError("candidate 缺少 lens_id")

    if not seq_path:
        raise ValueError(f"{lens_id} 缺少 seq_path，无法绘制真实镜头结构图")

    return generate_optiland_layout_from_seq(
        lens_id=lens_id,
        seq_path=seq_path,
        output_dir=output_dir,
    )