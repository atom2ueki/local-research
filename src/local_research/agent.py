"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping
- Research brief generation
- Multi-agent research coordination
- Final report generation

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""

from langgraph.graph import StateGraph, START, END

from local_research.state import AgentState, AgentInputState
from local_research.research_scope import research_scope
from local_research.research_supervisor import supervisor_agent
from local_research.research_report import research_report

# ===== ROUTING LOGIC =====

def route_after_scoping(state: AgentState) -> str:
    """
    Route from scope_subgraph based on whether clarification was needed.

    If research_brief exists, it means scoping completed successfully and we can proceed.
    If no research_brief exists, it means clarification was needed and we should end.
    """
    research_brief = state.get("research_brief")

    if research_brief:
        # Research brief was generated, proceed to supervisor
        return "supervisor_subgraph"
    else:
        # No research brief means clarification was needed, end the workflow
        return "__end__"

# ===== GRAPH CONSTRUCTION =====
# Build the overall workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("scope_subgraph", research_scope)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("report_subgraph", research_report)

# Add workflow edges
deep_researcher_builder.add_edge(START, "scope_subgraph")

# Add conditional routing from scope_subgraph
deep_researcher_builder.add_conditional_edges(
    "scope_subgraph",
    route_after_scoping,
    {
        "__end__": END,
        "supervisor_subgraph": "supervisor_subgraph"
    }
)

deep_researcher_builder.add_edge("supervisor_subgraph", "report_subgraph")
deep_researcher_builder.add_edge("report_subgraph", END)

# Compile the full workflow
agent = deep_researcher_builder.compile()
