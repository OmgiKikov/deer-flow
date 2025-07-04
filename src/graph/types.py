# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Union, Optional
from langgraph.graph import MessagesState

from src.prompts.planner_model import Plan
from src.prompts.multi_agent_planner_model import UnifiedResearchPlan
from src.rag import Resource


class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    # Runtime Variables
    locale: str
    research_topic: str
    observations: list[str]
    resources: list[Resource]
    plan_iterations: int
    current_plan: Optional[Union[Plan, UnifiedResearchPlan, str]]
    final_report: str
    auto_accepted_plan: bool
    enable_background_investigation: bool
    background_investigation_results: Optional[str]
    
    # Multi-agent specific variables
    use_multi_agent: bool
    subagent_results: list
    research_mode: str  # "sequential" or "parallel"
