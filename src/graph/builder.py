# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
NEW UNIFIED MULTI-AGENT ARCHITECTURE

This module builds a simplified, unified research workflow that always uses
parallel multi-agent coordination inspired by Anthropic's research approach.

Flow:
1. coordinator_node - handles user input and detects research topic
2. [optional] background_investigator - gathers initial context 
3. planner_node - creates multi-agent research plan
4. parallel_research_node - executes 2-5 specialized subagents in parallel
5. reporter_node - synthesizes final comprehensive report

Key improvements:
- No more sequential vs parallel modes - always parallel for better results
- Automatic complexity assessment determines optimal number of subagents  
- Intelligent agent type selection (researcher/coder) based on task requirements
- Unified schema and simplified routing logic
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.prompts.planner_model import StepType
from src.prompts.multi_agent_planner_model import PlanningMode, UnifiedResearchPlan

from .types import State
from .nodes import (
    coordinator_node,
    planner_node,
    reporter_node,
    human_feedback_node,
    background_investigation_node,
)
from .multi_agent_nodes import parallel_research_node


def _check_plan_has_enough_context(state: State) -> str:
    """Smart routing: skip research if plan has enough context"""
    current_plan = state.get("current_plan")
    if hasattr(current_plan, 'has_enough_context') and current_plan.has_enough_context:
        return "reporter"
    return "parallel_research"


def _build_base_graph():
    """Build and return the unified multi-agent state graph."""
    builder = StateGraph(State)
    
    # Core workflow nodes
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("planner", planner_node)
    builder.add_node("reporter", reporter_node)
    
    # Optional human feedback for manual review
    builder.add_node("human_feedback", human_feedback_node)
    
    # NEW: Universal parallel research node
    builder.add_node("parallel_research", parallel_research_node)
    
    # Simple, direct edges
    builder.add_edge("background_investigator", "planner")
    
    # Smart routing from planner
    builder.add_conditional_edges(
        "planner",
        _check_plan_has_enough_context,
        ["parallel_research", "reporter"]
    )
    
    # Always end at reporter
    builder.add_edge("parallel_research", "reporter")
    builder.add_edge("reporter", END)
    
    # Human feedback redirects to parallel research
    builder.add_edge("human_feedback", "parallel_research")
    
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
