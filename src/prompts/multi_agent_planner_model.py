# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class PlanningMode(str, Enum):
    PARALLEL_MULTI_AGENT = "parallel_multi_agent"
    SEQUENTIAL_STEPS = "sequential_steps"


class SubAgentStream(BaseModel):
    """Поток исследования для специализированного субагента"""
    stream_id: str = Field(..., description="Unique identifier for the subagent stream")
    research_focus: str = Field(..., description="Specific area of investigation")
    description: str = Field(..., description="Detailed scope and objectives")
    success_criteria: str = Field(..., description="What constitutes success for this stream")
    tool_requirements: List[str] = Field(
        default_factory=list, 
        description="Tools needed for this investigation"
    )
    estimated_calls: int = Field(
        default=10, 
        description="Estimated number of tool calls needed"
    )
    context_limit: int = Field(
        default=50000, 
        description="Context window limit for this subagent"
    )
    
    # Execution tracking
    status: str = Field(default="pending", description="Current execution status")
    findings: Optional[str] = Field(default=None, description="Research findings")
    confidence_score: Optional[float] = Field(default=None, description="Confidence in findings")


class ParallelMultiAgentPlan(BaseModel):
    """План для параллельного выполнения многоагентного исследования"""
    locale: str = Field(..., description="Language locale")
    planning_mode: PlanningMode = Field(
        default=PlanningMode.PARALLEL_MULTI_AGENT,
        description="Planning approach used"
    )
    thought: str = Field(..., description="Reasoning for this planning approach")
    title: str = Field(..., description="Overall research investigation title")
    
    subagent_streams: List[SubAgentStream] = Field(
        default_factory=list,
        description="Parallel investigation streams"
    )
    
    synthesis_strategy: str = Field(
        ..., 
        description="How subagent findings will be integrated"
    )
    estimated_total_calls: int = Field(
        default=0, 
        description="Total estimated tool calls across all subagents"
    )
    confidence_target: float = Field(
        default=0.8, 
        description="Target confidence level for the investigation"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "locale": "en-US",
                    "planning_mode": "parallel_multi_agent",
                    "thought": "Complex research query requires parallel investigation of multiple independent dimensions",
                    "title": "AI Impact on Healthcare Analysis",
                    "subagent_streams": [
                        {
                            "stream_id": "current_state",
                            "research_focus": "Current AI Healthcare Applications",
                            "description": "Research existing AI applications in healthcare, adoption rates, success stories",
                            "success_criteria": "Comprehensive overview of current AI healthcare landscape",
                            "tool_requirements": ["web_search", "crawl_tool"],
                            "estimated_calls": 8
                        },
                        {
                            "stream_id": "market_analysis",
                            "research_focus": "Market Dynamics and Investment",
                            "description": "Analyze market size, investment trends, major players in AI healthcare",
                            "success_criteria": "Complete market analysis with quantitative data",
                            "tool_requirements": ["web_search", "python_repl"],
                            "estimated_calls": 10
                        }
                    ],
                    "synthesis_strategy": "Combine current state analysis with market dynamics to provide comprehensive assessment",
                    "estimated_total_calls": 18,
                    "confidence_target": 0.85
                }
            ]
        }


# Backwards compatibility with existing system
from src.prompts.planner_model import Step, StepType


class SequentialStepsPlan(BaseModel):
    """План для последовательного выполнения шагов (обратная совместимость)"""
    locale: str = Field(..., description="Language locale")
    planning_mode: PlanningMode = Field(
        default=PlanningMode.SEQUENTIAL_STEPS,
        description="Planning approach used"
    )
    has_enough_context: bool = Field(
        default=False,
        description="Whether sufficient context exists to answer query"
    )
    thought: str = Field(..., description="Reasoning for approach")
    title: str = Field(..., description="Research plan title")
    
    steps: List[Step] = Field(
        default_factory=list,
        description="Sequential research steps"
    )


class UnifiedResearchPlan(BaseModel):
    """Унифицированный план исследования, поддерживающий оба режима"""
    locale: str = Field(..., description="Language locale")
    planning_mode: PlanningMode = Field(..., description="Planning approach")
    thought: str = Field(..., description="Reasoning for chosen approach")
    title: str = Field(..., description="Research investigation title")
    
    # Для параллельного режима
    subagent_streams: Optional[List[SubAgentStream]] = Field(
        default=None,
        description="Parallel investigation streams (multi-agent mode)"
    )
    synthesis_strategy: Optional[str] = Field(
        default=None,
        description="Integration strategy (multi-agent mode)"
    )
    confidence_target: Optional[float] = Field(
        default=None,
        description="Target confidence level (multi-agent mode)"
    )
    
    # Для последовательного режима
    has_enough_context: Optional[bool] = Field(
        default=None,
        description="Context sufficiency (sequential mode)"
    )
    steps: Optional[List[Step]] = Field(
        default=None,
        description="Sequential steps (sequential mode)"
    )
    
    @property
    def is_parallel_mode(self) -> bool:
        """Проверяет, используется ли параллельный режим"""
        return self.planning_mode == PlanningMode.PARALLEL_MULTI_AGENT
    
    @property
    def is_sequential_mode(self) -> bool:
        """Проверяет, используется ли последовательный режим"""
        return self.planning_mode == PlanningMode.SEQUENTIAL_STEPS
    
    def get_estimated_total_calls(self) -> int:
        """Вычисляет общее количество ожидаемых вызовов инструментов"""
        if self.is_parallel_mode and self.subagent_streams:
            return sum(stream.estimated_calls for stream in self.subagent_streams)
        elif self.is_sequential_mode and self.steps:
            # Примерная оценка для последовательных шагов
            return len(self.steps) * 5  # 5 вызовов на шаг в среднем
        return 0
    
    def get_active_streams(self) -> List[SubAgentStream]:
        """Возвращает активные потоки исследования"""
        if not self.subagent_streams:
            return []
        return [stream for stream in self.subagent_streams if stream.status != "completed"]
    
    def mark_stream_completed(self, stream_id: str, findings: str, confidence: float):
        """Отмечает поток как завершенный"""
        if not self.subagent_streams:
            return
        
        for stream in self.subagent_streams:
            if stream.stream_id == stream_id:
                stream.status = "completed"
                stream.findings = findings
                stream.confidence_score = confidence
                break
    
    def is_research_complete(self) -> bool:
        """Проверяет, завершено ли исследование"""
        if self.is_parallel_mode:
            if not self.subagent_streams:
                return False
            return all(stream.status == "completed" for stream in self.subagent_streams)
        elif self.is_sequential_mode:
            if not self.steps:
                return False
            return all(step.execution_res is not None for step in self.steps)
        return False


# Utility functions for plan creation
def create_parallel_plan(
    locale: str,
    title: str,
    thought: str,
    streams: List[SubAgentStream],
    synthesis_strategy: str
) -> UnifiedResearchPlan:
    """Создает план для параллельного исследования"""
    return UnifiedResearchPlan(
        locale=locale,
        planning_mode=PlanningMode.PARALLEL_MULTI_AGENT,
        thought=thought,
        title=title,
        subagent_streams=streams,
        synthesis_strategy=synthesis_strategy,
        confidence_target=0.8
    )


def create_sequential_plan(
    locale: str,
    title: str,
    thought: str,
    steps: List[Step],
    has_enough_context: bool = False
) -> UnifiedResearchPlan:
    """Создает план для последовательного исследования"""
    return UnifiedResearchPlan(
        locale=locale,
        planning_mode=PlanningMode.SEQUENTIAL_STEPS,
        thought=thought,
        title=title,
        steps=steps,
        has_enough_context=has_enough_context
    ) 