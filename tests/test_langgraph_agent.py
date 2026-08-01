from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.langgraph_agent import LangGraphOpticalDesignAgent
from agent.tools import ToolRegistry


def build_mock_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        "parse_requirement",
        lambda user_text: {
            "raw_text": user_text,
            "scene": "vehicle",
            "f_number": {"target": 2.0, "preference": None, "constraint_type": "soft"},
            "fov": {"target": 120.0, "preference": "greater_than", "constraint_type": "hard"},
        },
        "mock parser",
    )
    registry.register(
        "retrieve_candidates",
        lambda parsed_requirement, raw_text=None, top_k=9: [
            {"lens_id": "or00001", "key_specs": {"f_number": 2.0, "full_fov": 122.0}},
            {"lens_id": "or00002", "key_specs": {"f_number": 2.2, "full_fov": 118.0}},
        ],
        "mock retrieval",
    )
    registry.register(
        "rerank_candidates",
        lambda parsed_requirement, candidates: {"top_candidates": list(reversed(candidates))},
        "mock rerank",
    )
    registry.register(
        "run_raytrace",
        lambda candidates: {"raytrace_reranked_candidates": candidates, "raytrace_summary": "ok"},
        "mock raytrace",
    )
    registry.register(
        "explain_recommendation",
        lambda parsed_requirement, candidates, raytrace_result=None: "mock explanation",
        "mock explanation",
    )

    return registry


def test_langgraph_agent_runs_new_design_workflow():
    agent = LangGraphOpticalDesignAgent(registry=build_mock_registry())

    result = agent.step(
        session_id="lg-test-001",
        user_message="设计一个车载广角镜头，F数2.0，视场角大于120度",
        top_k=9,
    )

    assert result["framework"] == "langgraph"
    assert result["intent"] == "new_design_task"
    assert result["error"] is None
    assert "classify_intent" in result["called_tools"]
    assert "retrieve_candidates" in result["called_tools"]
    assert result["candidates"][0]["lens_id"] == "or00002"

