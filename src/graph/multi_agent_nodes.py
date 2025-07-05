# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.agents import create_agent
from src.config.configuration import Configuration
from src.prompts.planner_model import Plan, Step
from src.tools import get_web_search_tool, crawl_tool, python_repl_tool
from .types import State

logger = logging.getLogger(__name__)


@dataclass
class SubAgentTask:
    """Задача для субагента с определенным фокусом исследования"""
    agent_id: str
    research_focus: str  # "market_analysis", "technical_specs", "competitor_research"
    description: str
    tools: List[Any]
    agent_type: str = "researcher"  # NEW: тип агента (researcher|coder)
    context_limit: int = 50000  # Отдельный контекст для каждого субагента


@dataclass
class SubAgentResult:
    """Результат работы субагента с сжатой информацией"""
    agent_id: str
    research_focus: str
    key_findings: str  # Сжатые ключевые находки
    raw_data: str  # Исходные данные
    confidence_score: float  # Уверенность в результатах
    sources: List[str]  # Источники информации


class MultiAgentCoordinator:
    """Координатор для управления параллельными субагентами по образцу Anthropic"""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.active_subagents: Dict[str, Any] = {}
        
    async def create_parallel_research_plan(self, query: str, max_subagents: int = 4) -> List[SubAgentTask]:
        """
        Создает план параллельного исследования с множественными субагентами
        Аналог системы Anthropic для разложения задач
        """
        # NEW: Анализируем сложность запроса и адаптируем количество субагентов
        complexity_score = self._assess_query_complexity(query)
        optimal_subagents = min(max_subagents, max(2, complexity_score))
        
        logger.info(f"🧠 Query complexity score: {complexity_score}/5, using {optimal_subagents} subagents")
        
        # Анализируем запрос и определяем аспекты для параллельного исследования
        research_aspects = await self._identify_research_aspects(query, optimal_subagents)
        
        tasks = []
        for i, aspect in enumerate(research_aspects[:optimal_subagents]):
            task = SubAgentTask(
                agent_id=f"subagent_{i}_{aspect['type']}",
                research_focus=aspect['focus'],
                description=aspect['description'],
                tools=self._select_tools_for_aspect(aspect['type']),
                agent_type=self._get_agent_type_for_aspect(aspect['type']),
                context_limit=50000
            )
            tasks.append(task)
            
        logger.info(f"Created {len(tasks)} parallel subagent tasks")
        return tasks
    
    async def execute_parallel_research(self, tasks: List[SubAgentTask], state: State) -> List[SubAgentResult]:
        """
        Выполняет параллельное исследование субагентами
        Ключевое отличие от текущей последовательной системы DeerFlow
        """
        logger.info(f"Starting parallel execution of {len(tasks)} subagents")
        
        # Создаем корутины для параллельного выполнения
        coroutines = [
            self._execute_subagent_task(task, state) 
            for task in tasks
        ]
        
        # Выполняем все субагенты параллельно
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Фильтруем успешные результаты
        successful_results = [
            result for result in results 
            if isinstance(result, SubAgentResult)
        ]
        
        logger.info(f"Completed parallel research with {len(successful_results)} successful results")
        return successful_results
    
    async def _execute_subagent_task(self, task: SubAgentTask, state: State) -> SubAgentResult:
        """
        Выполняет отдельную задачу субагента в изолированном контексте
        Каждый субагент работает независимо как у Anthropic
        """
        logger.info(f"Executing subagent {task.agent_id} for {task.research_focus}")
        
        # Создаем изолированный контекст для субагента
        subagent_input = {
            "messages": [
                HumanMessage(
                    content=f"""
                    # Focused Research Task
                    
                    **Research Focus**: {task.research_focus}
                    **Task Description**: {task.description}
                    **Context Limit**: {task.context_limit} tokens
                    
                    Your job is to conduct focused research on this specific aspect and provide:
                    1. Key findings (2-3 most important discoveries)
                    2. Supporting data and evidence
                    3. Source attribution
                    
                    Focus ONLY on your assigned aspect. Other subagents are handling other aspects.
                    Compress your findings into the most essential information.
                    """
                )
            ]
        }
        
        # Создаем специализированного агента
        agent = create_agent(
            agent_name=task.agent_id,
            agent_type=task.agent_type,
            tools=task.tools,
            prompt_template=task.agent_type
        )
        
        try:
            # Выполняем исследование в ограниченном контексте
            result = await agent.ainvoke(
                input=subagent_input,
                config={"recursion_limit": 15}  # Ограничиваем глубину для фокуса
            )
            
            response_content = result["messages"][-1].content
            
            # Извлекаем ключевые находки (компрессия как у Anthropic)
            key_findings = await self._compress_findings(response_content, task.research_focus)
            
            return SubAgentResult(
                agent_id=task.agent_id,
                research_focus=task.research_focus,
                key_findings=key_findings,
                raw_data=response_content,
                confidence_score=0.85,  # TODO: вычислять динамически
                sources=self._extract_sources(response_content)
            )
            
        except Exception as e:
            logger.error(f"Subagent {task.agent_id} failed: {str(e)}")
            # Возвращаем частичный результат даже при ошибке
            return SubAgentResult(
                agent_id=task.agent_id,
                research_focus=task.research_focus,
                key_findings=f"Research failed: {str(e)}",
                raw_data="",
                confidence_score=0.0,
                sources=[]
            )
    
    def _assess_query_complexity(self, query: str) -> int:
        """
        Оценивает сложность запроса и возвращает оптимальное количество субагентов (2-5)
        """
        complexity_indicators = [
            "analyze", "compare", "comprehensive", "detailed", "market", "industry",
            "анализ", "сравни", "всесторонний", "детальный", "рынок", "отрасль",
            "research", "investigate", "study", "evaluation", "assessment",
            "исследование", "изучение", "оценка", "влияние", "тенденции"
        ]
        
        breadth_indicators = [
            "impact", "trends", "future", "current state", "stakeholders",
            "влияние", "тенденции", "будущее", "текущее состояние", "участники",
            "ecosystem", "landscape", "overview", "multiple", "various",
            "экосистема", "ландшафт", "обзор", "множественный", "различные"
        ]
        
        technical_indicators = [
            "technical", "architecture", "implementation", "code", "system",
            "технический", "архитектура", "реализация", "код", "система"
        ]
        
        score = 2  # Базовый счет для любого запроса
        
        # +1 за сложность
        if any(indicator in query.lower() for indicator in complexity_indicators):
            score += 1
            
        # +1 за широту охвата
        if any(indicator in query.lower() for indicator in breadth_indicators):
            score += 1
            
        # +1 за технические аспекты
        if any(indicator in query.lower() for indicator in technical_indicators):
            score += 1
            
        # +1 за длинные запросы (обычно более сложные)
        if len(query.split()) > 10:
            score += 1
            
        return min(5, score)  # Максимум 5 субагентов

    async def _identify_research_aspects(self, query: str, num_agents: int) -> List[Dict[str, str]]:
        """
        Анализирует запрос и определяет аспекты для параллельного исследования
        Адаптируется под количество субагентов
        """
        # Полный набор возможных аспектов
        all_aspects = [
            {
                "type": "current_state",
                "focus": "Current State Analysis", 
                "description": f"Research the current state, recent developments and latest information about: {query}"
            },
            {
                "type": "historical_context",
                "focus": "Historical Context",
                "description": f"Investigate the historical background, evolution and timeline related to: {query}"
            },
            {
                "type": "stakeholder_analysis", 
                "focus": "Stakeholder Analysis",
                "description": f"Analyze key players, stakeholders, companies and organizations involved in: {query}"
            },
            {
                "type": "technical_specs",
                "focus": "Technical Specifications & Architectures",
                "description": f"Dive deep into technical specifications, architectures or underlying technologies related to: {query}"
            },
            {
                "type": "data_analysis",
                "focus": "Quantitative & Data Analysis",
                "description": f"Perform quantitative analysis, statistics and data-driven insights for: {query}"
            },
            {
                "type": "future_trends",
                "focus": "Future Trends & Implications", 
                "description": f"Research future outlook, trends, predictions and implications for: {query}"
            }
        ]
        
        # Для простых запросов (2 субагента) - основы
        if num_agents == 2:
            return [all_aspects[0], all_aspects[5]]  # current_state + future_trends
            
        # Для средних запросов (3 субагента) - добавляем контекст
        elif num_agents == 3:
            return [all_aspects[0], all_aspects[1], all_aspects[5]]  # + historical_context
            
        # Для сложных запросов (4+ субагентов) - полный набор
        else:
            return all_aspects[:num_agents]
    
    def _select_tools_for_aspect(self, aspect_type: str) -> List[Any]:
        """Выбирает подходящие инструменты для типа исследования"""
        base_tools = [
            get_web_search_tool(self.config.max_search_results),
            crawl_tool
        ]
        
        if aspect_type in ["current_state", "future_trends", "data_analysis", "technical_specs"]:
            # Для анализа данных добавляем Python REPL
            base_tools.append(python_repl_tool)
            
        return base_tools
    
    def _get_agent_type_for_aspect(self, aspect_type: str) -> str:
        """Возвращает тип агента для данного аспекта"""
        if aspect_type in ["future_trends", "data_analysis", "technical_specs"]:
            return "coder"
        return "researcher"
    
    async def _compress_findings(self, raw_content: str, focus_area: str) -> str:
        """
        Сжимает результаты исследования в ключевые инсайты
        Критически важная функция компрессии как у Anthropic
        """
        # TODO: Использовать LLM для интеллектуального сжатия
        # Пока простая версия - первые N символов + ключевые моменты
        
        lines = raw_content.split('\n')
        key_lines = [line for line in lines if any(keyword in line.lower() 
                    for keyword in ['key', 'important', 'finding', 'result', 'conclusion'])]
        
        if key_lines:
            compressed = '\n'.join(key_lines[:5])  # Топ-5 ключевых строк
        else:
            compressed = raw_content[:1000] + "..." if len(raw_content) > 1000 else raw_content
            
        return f"**{focus_area}**:\n{compressed}"
    
    def _extract_sources(self, content: str) -> List[str]:
        """Извлекает источники из контента"""
        import re
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        return list(set(urls))


async def parallel_research_node(state: State, config: RunnableConfig) -> Command:
    """
    Новый узел для параллельного исследования по образцу Anthropic
    Заменяет последовательное выполнение research_team_node
    """
    logger.info("🚀🚀🚀 PARALLEL RESEARCH NODE АКТИВИРОВАН! 🚀🚀🚀")
    logger.info("Starting parallel multi-agent research")
    
    configurable = Configuration.from_runnable_config(config)
    coordinator = MultiAgentCoordinator(configurable)
    
    # Получаем план исследования
    current_plan = state.get("current_plan")
    if not current_plan or not hasattr(current_plan, 'title'):
        logger.warning("No valid research plan found")
        return Command(goto="reporter")
    
    query = str(current_plan.title)
    logger.info(f"🎯 Исследуем: {query}")
    
    # Создаем план параллельного исследования  
    logger.info("🏗️ Создаем параллельные задачи для субагентов...")
    parallel_tasks = await coordinator.create_parallel_research_plan(
        query, max_subagents=4
    )
    logger.info(f"✅ Создано {len(parallel_tasks)} параллельных задач")
    
    # Выполняем параллельное исследование
    logger.info("⚡ Запускаем параллельное выполнение субагентов...")
    subagent_results = await coordinator.execute_parallel_research(
        parallel_tasks, state
    )
    logger.info(f"🎉 Параллельное исследование завершено! Получено {len(subagent_results)} результатов")
    
    # Собираем результаты для передачи репортеру
    observations = []
    for result in subagent_results:
        observation = f"""
        ## {result.research_focus}
        
        {result.key_findings}
        
        **Confidence**: {result.confidence_score:.2f}
        **Sources**: {len(result.sources)} sources
        """
        observations.append(observation)
    
    logger.info(f"📊 Передаем репортеру {len(observations)} наблюдений для финального отчета")
    
    return Command(
        update={
            "observations": observations,
            "messages": [
                AIMessage(
                    content=f"Parallel research completed with {len(subagent_results)} specialized investigations",
                    name="multi_agent_coordinator"
                )
            ]
        },
        goto="reporter"
    )


# Функция для интеграции в существующий граф
def should_use_parallel_research(state: State) -> bool:
    """
    Определяет, нужно ли использовать параллельное исследование
    Критерии: сложность запроса, доступные ресурсы
    """
    current_plan = state.get("current_plan")
    if not current_plan:
        return False
        
    # Используем параллельность для сложных исследований
    steps_attr = getattr(current_plan, 'steps', None)
    if steps_attr and isinstance(steps_attr, list) and len(steps_attr) > 2:
        return True
        
    # Или если запрос содержит ключевые слова
    query = state.get("research_topic", "").lower()
    complexity_indicators = ["analyze", "compare", "comprehensive", "detailed", "market", "industry"]
    
    return any(indicator in query for indicator in complexity_indicators) 