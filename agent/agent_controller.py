# agent/agent_controller.py

from typing import Any, Dict

from agent.state_manager import state_manager
from agent.planner import classify_intent
from agent.executor import execute_plan
from agent.response_builder import build_agent_response


class OpticalDesignAgent:
    def step(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """
        Agent 单轮执行入口。
        """

        # 1. 读取当前 session 状态
        design_state = state_manager.get_state(session_id)

        # 2. 记录用户输入
        state_manager.append_history(
            design_state,
            role="user",
            content=user_message,
        )

        # 3. 判断用户意图
        intent = classify_intent(
            user_message=user_message,
            design_state=design_state,
        )

        # 4. 执行工具链
        execution_result = execute_plan(
            intent=intent,
            user_message=user_message,
            design_state=design_state,
        )

        updated_state = execution_result["design_state"]
        called_tools = execution_result["called_tools"]
        tool_outputs = execution_result["tool_outputs"]

        # 5. 生成回答
        answer = build_agent_response(
            intent=intent,
            design_state=updated_state,
            called_tools=called_tools,
            tool_outputs=tool_outputs,
        )

        # 6. 保存助手回复
        state_manager.append_history(
            updated_state,
            role="assistant",
            content=answer,
        )

        # 7. 保存最新状态
        state_manager.save_state(session_id, updated_state)

        # 8. 返回给前端
        return {
            "session_id": session_id,
            "intent": intent,
            "called_tools": called_tools,
            "design_state": {
                "raw_requirement": updated_state.get("raw_requirement"),
                "parsed_requirement": updated_state.get("parsed_requirement"),
                "user_preferences": updated_state.get("user_preferences"),
                "iteration": updated_state.get("iteration"),
            },
            "candidates": updated_state.get("current_candidates") or [],
            "last_rerank_result": updated_state.get("last_rerank_result"),
            "last_raytrace_result": updated_state.get("last_raytrace_result"),
            "tool_outputs": tool_outputs,
            "answer": answer,
        }


optical_agent = OpticalDesignAgent()