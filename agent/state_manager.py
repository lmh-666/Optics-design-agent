# agent/state_manager.py

from copy import deepcopy
from typing import Any, Dict


DEFAULT_DESIGN_STATE: Dict[str, Any] = {
    "session_id": None,

    # 用户原始需求
    "raw_requirement": None,

    # LLM 或规则解析后的结构化需求
    "parsed_requirement": None,

    # 用户偏好，用于多轮修改
    "user_preferences": {
        "prefer_compact": False,
        "prefer_low_f_number": False,
        "prefer_large_fov": False,
        "prefer_simple_structure": False,
    },

    # 当前候选结构
    "current_candidates": [],

    # 上一轮重排序结果
    "last_rerank_result": None,

    # 上一轮 ray tracing 结果
    "last_raytrace_result": None,

    # 当前选中的候选结构
    "selected_candidate": None,

    # 迭代轮次
    "iteration": 0,

    # 对话历史
    "history": [],
}


class StateManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_state(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            state = deepcopy(DEFAULT_DESIGN_STATE)
            state["session_id"] = session_id
            self.sessions[session_id] = state
        return self.sessions[session_id]

    def save_state(self, session_id: str, state: Dict[str, Any]) -> None:
        self.sessions[session_id] = state

    def append_history(self, state: Dict[str, Any], role: str, content: str) -> None:
        state.setdefault("history", [])
        state["history"].append({
            "role": role,
            "content": content,
        })


state_manager = StateManager()