# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import logging
import os
import re
from typing import Annotated, Literal

import mcp

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents import create_agent
from src.tools.search import LoggedTavilySearch
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    get_retriever_tool,
    python_repl_tool,
)

from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type
from src.prompts.planner_model import Plan
from src.prompts.multi_agent_planner_model import UnifiedResearchPlan, PlanningMode
from src.prompts.template import apply_prompt_template
from src.utils.json_utils import repair_json_output

from .types import State
from ..config import SELECTED_SEARCH_ENGINE, SearchEngine

logger = logging.getLogger(__name__)


@tool
def handoff_to_planner(
    research_topic: Annotated[str, "The topic of the research task to be handed off."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Handoff to planner agent to do plan."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    return


def background_investigation_node(state: State, config: RunnableConfig):
    logger.info("background investigation node is running.")
    configurable = Configuration.from_runnable_config(config)
    query = state.get("research_topic")
    background_investigation_results = None
    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        searched_content = LoggedTavilySearch(
            max_results=configurable.max_search_results
        ).invoke(query)
        if isinstance(searched_content, list):
            background_investigation_results = [
                f"## {elem['title']}\n\n{elem['content']}" for elem in searched_content
            ]
            return {
                "background_investigation_results": "\n\n".join(
                    background_investigation_results
                )
            }
        else:
            logger.error(
                f"Tavily search returned malformed response: {searched_content}"
            )
    else:
        background_investigation_results = get_web_search_tool(
            configurable.max_search_results
        ).invoke(query)
    return {
        "background_investigation_results": json.dumps(
            background_investigation_results, ensure_ascii=False
        )
    }


def planner_node(
    state: State, config: RunnableConfig
) -> Command[Literal["human_feedback", "reporter", "parallel_research"]]:
    """Planner node that generate the full plan."""
    logger.info("Planner generating research plan")
    configurable = Configuration.from_runnable_config(config)
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    
    # NEW: Always use multi-agent planning - it's more flexible
    template_name = "multi_agent_planner"
    messages = apply_prompt_template(template_name, state, configurable)

    if state.get("enable_background_investigation") and state.get(
        "background_investigation_results"
    ):
        bg_results = state.get("background_investigation_results", "")
        if bg_results:
            messages += [
                {
                    "role": "user",
                    "content": (
                        "background investigation results of user query:\n"
                        + bg_results
                        + "\n"
                    ),
                }
            ]

    if configurable.enable_deep_thinking:
        llm = get_llm_by_type("reasoning")
    elif AGENT_LLM_MAP["planner"] == "basic":
        # Always use UnifiedResearchPlan schema
        llm = get_llm_by_type("basic").with_structured_output(
            UnifiedResearchPlan,
            method="json_mode",
        )
    else:
        llm = get_llm_by_type(AGENT_LLM_MAP["planner"])

    # if the plan iterations is greater than the max plan iterations, return the reporter node
    if plan_iterations >= configurable.max_plan_iterations:
        return Command(goto="reporter")

    full_response = ""
    if AGENT_LLM_MAP["planner"] == "basic" and not configurable.enable_deep_thinking:
        response = llm.invoke(messages)
        # Extract JSON from content field
        if hasattr(response, 'content'):
            content = response.content
            json_match = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                full_response = json_match.group(1)
                logger.info("🔧 JSON extracted from markdown block")
            else:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    full_response = json_match.group(0)
                    logger.info("🔧 JSON found directly in content")
                else:
                    full_response = content
        else:
            full_response = response.model_dump_json(indent=4, exclude_none=True)
    else:
        response = llm.stream(messages)
        for chunk in response:
            full_response += chunk.content
    
    logger.debug(f"Current state messages: {state['messages']}")
    logger.info(f"Planner response: {full_response[:500]}...")

    try:
        curr_plan = json.loads(repair_json_output(full_response))
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        if plan_iterations > 0:
            return Command(goto="reporter")
        else:
            return Command(goto="__end__")
    
    # NEW: Simple logic - create plan and go to parallel research
    logger.info("🚀 Research plan generated - going to parallel research")
    new_plan = UnifiedResearchPlan.model_validate(curr_plan)
    
    # Check if we have enough context to skip research
    if new_plan.has_enough_context:
        logger.info("📚 Plan has enough context - going directly to reporter")
        return Command(
            update={
                "messages": [AIMessage(content=full_response, name="planner")],
                "current_plan": new_plan,
                "research_mode": "skip",
            },
            goto="reporter",
        )
    
    # Go to parallel research
    return Command(
        update={
            "messages": [AIMessage(content=full_response, name="planner")],
            "current_plan": new_plan,
            "research_mode": "parallel",
        },
        goto="parallel_research",
    )


def human_feedback_node(
    state,
) -> Command[Literal["planner", "parallel_research", "reporter", "__end__"]]:
    current_plan = state.get("current_plan", "")
    # check if the plan is auto accepted
    auto_accepted_plan = state.get("auto_accepted_plan", False)
    if not auto_accepted_plan:
        feedback = interrupt("Please Review the Plan.")

        # if the feedback is not accepted, return the planner node
        if feedback and str(feedback).upper().startswith("[EDIT_PLAN]"):
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=feedback, name="feedback"),
                    ],
                },
                goto="planner",
            )
        elif feedback and str(feedback).upper().startswith("[ACCEPTED]"):
            logger.info("Plan is accepted by user.")
        else:
            raise TypeError(f"Interrupt value of {feedback} is not supported.")

    # NEW: Simple logic - if plan accepted, go to parallel research
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    try:
        current_plan = repair_json_output(current_plan)
        plan_iterations += 1
        new_plan = json.loads(current_plan)
        
        validated_plan = UnifiedResearchPlan.model_validate(new_plan)
        if validated_plan.has_enough_context:
            goto = "reporter"
        else:
            goto = "parallel_research"
            
        return Command(
            update={
                "current_plan": validated_plan,
                "plan_iterations": plan_iterations,
                "locale": new_plan["locale"],
                "research_mode": "parallel",
            },
            goto=goto,
        )
            
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        if plan_iterations > 1:
            return Command(goto="reporter")
        else:
            return Command(goto="__end__")


def coordinator_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner", "background_investigator", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info("Coordinator talking.")
    configurable = Configuration.from_runnable_config(config)
    messages = apply_prompt_template("coordinator", state)
    response = (
        get_llm_by_type(AGENT_LLM_MAP["coordinator"])
        .bind_tools([handoff_to_planner])
        .invoke(messages)
    )
    logger.debug(f"Current state messages: {state['messages']}")

    goto = "__end__"
    locale = state.get("locale", "en-US")  # Default locale if not specified
    research_topic = state.get("research_topic", "")

    if len(response.tool_calls) > 0:
        goto = "planner"
        if state.get("enable_background_investigation"):
            # if the search_before_planning is True, add the web search tool to the planner agent
            goto = "background_investigator"
        try:
            for tool_call in response.tool_calls:
                if tool_call.get("name", "") != "handoff_to_planner":
                    continue
                if tool_call.get("args", {}).get("locale") and tool_call.get(
                    "args", {}
                ).get("research_topic"):
                    locale = tool_call.get("args", {}).get("locale")
                    research_topic = tool_call.get("args", {}).get("research_topic")
                    break
        except Exception as e:
            logger.error(f"Error processing tool calls: {e}")
    else:
        logger.warning(
            "🚨 КООРДИНАТОР НЕ ВЫЗВАЛ handoff_to_planner!"
        )
        logger.warning(f"📝 Полный ответ координатора: {response.content if hasattr(response, 'content') else str(response)}")
        logger.warning(f"🔧 Тип ответа: {type(response)}")
        if hasattr(response, 'tool_calls'):
            logger.warning(f"🛠️ Tool calls: {response.tool_calls}")
        logger.warning("❌ Завершаем выполнение рабочего процесса")

    return Command(
        update={
            "locale": locale,
            "research_topic": research_topic,
            "resources": configurable.resources,
        },
        goto=goto,
    )


def reporter_node(state: State, config: RunnableConfig):
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    configurable = Configuration.from_runnable_config(config)
    current_plan = state.get("current_plan")
    input_ = {
        "messages": [
            HumanMessage(
                f"# Research Requirements\n\n## Task\n\n{current_plan.title}\n\n## Description\n\n{current_plan.thought}"
            )
        ],
        "locale": state.get("locale", "en-US"),
    }
    invoke_messages = apply_prompt_template("reporter", input_, configurable)
    observations = state.get("observations", [])

    # Add a reminder about the new report format, citation style, and table usage
    invoke_messages.append(
        HumanMessage(
            content="IMPORTANT: Structure your report according to the format in the prompt. Remember to include:\n\n1. Key Points - A bulleted list of the most important findings\n2. Overview - A brief introduction to the topic\n3. Detailed Analysis - Organized into logical sections\n4. Survey Note (optional) - For more comprehensive reports\n5. Key Citations - List all references at the end\n\nFor citations, DO NOT include inline citations in the text. Instead, place all citations in the 'Key Citations' section at the end using the format: `- [Source Title](URL)`. Include an empty line between each citation for better readability.\n\nPRIORITIZE USING MARKDOWN TABLES for data presentation and comparison. Use tables whenever presenting comparative data, statistics, features, or options. Structure tables with clear headers and aligned columns. Example table format:\n\n| Feature | Description | Pros | Cons |\n|---------|-------------|------|------|\n| Feature 1 | Description 1 | Pros 1 | Cons 1 |\n| Feature 2 | Description 2 | Pros 2 | Cons 2 |",
            name="system",
        )
    )

    for observation in observations:
        invoke_messages.append(
            HumanMessage(
                content=f"Below are some observations for the research task:\n\n{observation}",
                name="observation",
            )
        )
    logger.debug(f"Current invoke messages: {invoke_messages}")
    response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(invoke_messages)
    response_content = response.content
    logger.info(f"reporter response: {response_content}")

    return {"final_report": response_content}
