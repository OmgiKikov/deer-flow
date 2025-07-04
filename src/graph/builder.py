# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.prompts.planner_model import StepType
from src.prompts.multi_agent_planner_model import PlanningMode, UnifiedResearchPlan

from .types import State
from .nodes import (
    coordinator_node,
    planner_node,
    reporter_node,
    research_team_node,
    researcher_node,
    coder_node,
    human_feedback_node,
    background_investigation_node,
)
from .multi_agent_nodes import parallel_research_node


def continue_to_running_research_team(state: State):
    import logging
    logger = logging.getLogger(__name__)
    
    current_plan = state.get("current_plan")
    
    # Skip if plan is string (intermediate state)
    if isinstance(current_plan, str):
        logger.info("🔄 МАРШРУТ: planner (план - строка)")
        return "planner"
    
    # Check for multi-agent plan
    if isinstance(current_plan, UnifiedResearchPlan):
        if current_plan.planning_mode == PlanningMode.PARALLEL_MULTI_AGENT:
            logger.info("🚀 МАРШРУТ: parallel_research (многоагентный режим)")
            return "parallel_research"
        elif current_plan.planning_mode == PlanningMode.SEQUENTIAL_STEPS:
            logger.info("📝 МАРШРУТ: последовательный режим")
            # Handle sequential mode from unified plan
            if not current_plan.steps:
                logger.info("🔄 МАРШРУТ: planner (нет шагов)")
                return "planner"
            if all(step.execution_res for step in current_plan.steps):
                logger.info("🔄 МАРШРУТ: planner (все шаги выполнены)")
                return "planner"
            for step in current_plan.steps:
                if not step.execution_res:
                    break
            if step.step_type and step.step_type == StepType.RESEARCH:
                logger.info("🔍 МАРШРУТ: researcher")
                return "researcher"
            if step.step_type and step.step_type == StepType.PROCESSING:
                logger.info("💻 МАРШРУТ: coder")
                return "coder"
            logger.info("🔄 МАРШРУТ: planner (по умолчанию)")
            return "planner"
    
    # Legacy Plan object logic
    if current_plan and hasattr(current_plan, 'steps') and current_plan.steps:
        logger.info("📜 МАРШРУТ: legacy план")
        if all(step.execution_res for step in current_plan.steps):
            logger.info("🔄 МАРШРУТ: planner (все legacy шаги выполнены)")
            return "planner"
        for step in current_plan.steps:
            if not step.execution_res:
                break
        if step.step_type and step.step_type == StepType.RESEARCH:
            logger.info("🔍 МАРШРУТ: researcher (legacy)")
            return "researcher"
        if step.step_type and step.step_type == StepType.PROCESSING:
            logger.info("💻 МАРШРУТ: coder (legacy)")
            return "coder"
        logger.info("🔄 МАРШРУТ: planner (legacy по умолчанию)")
        return "planner"
    
    logger.info("🔄 МАРШРУТ: planner (финальный fallback)")
    return "planner"


def _build_base_graph():
    """Build and return the base state graph with all nodes and edges."""
    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("planner", planner_node)
    builder.add_node("reporter", reporter_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("human_feedback", human_feedback_node)
    # Add new parallel research node
    builder.add_node("parallel_research", parallel_research_node)
    
    builder.add_edge("background_investigator", "planner")
    builder.add_conditional_edges(
        "research_team",
        continue_to_running_research_team,
        ["planner", "researcher", "coder", "parallel_research"],
    )
    # Parallel research goes directly to reporter
    builder.add_edge("parallel_research", "reporter")
    builder.add_edge("reporter", END)
    return builder


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    # TODO: be compatible with SQLite / PostgreSQL
    memory = MemorySaver()

    # build state graph
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = _build_base_graph()
    return builder.compile()


graph = build_graph()
