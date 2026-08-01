from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aperture_scale_utils import parse_aperture_from_text
from agent.state_manager import StateManager
from lens_loader import load_lens_data


def test_parse_aperture_from_text_supports_epd():
    assert parse_aperture_from_text("F number 2.0, EPD 4mm, FOV 120 deg") == 4.0


def test_load_lens_data_normalizes_required_fields(tmp_path: Path):
    csv_path = tmp_path / "sample_lenses.csv"
    csv_path.write_text(
        "lens_id,f_number,half_fov,total_length,focal_length\n"
        "or00001,2.0,60,12.5,4.0\n",
        encoding="utf-8",
    )

    rows = load_lens_data(str(csv_path))

    assert len(rows) == 1
    assert rows[0]["lens_id"] == "or00001"
    assert rows[0]["full_fov"] == 120


def test_state_manager_keeps_sessions_isolated():
    manager = StateManager()

    state_a = manager.get_state("a")
    state_b = manager.get_state("b")
    state_a["parsed_requirement"] = {"f_number": {"target": 2.0}}

    assert state_b["parsed_requirement"] is None
    assert manager.get_state("a")["parsed_requirement"]["f_number"]["target"] == 2.0
