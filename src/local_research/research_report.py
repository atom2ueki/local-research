from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient

from local_research.utils import get_today_str, get_project_root
from local_research.prompts import final_report_generator_prompt
from local_research.state import AgentState
from langgraph.types import Command
from typing_extensions import Literal

# ===== Config =====

from local_research.model_config import get_report_model
writer_model = get_report_model(max_tokens=32000)

mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",  # Auto-install if needed
            "@modelcontextprotocol/server-filesystem",
            str(get_project_root())
        ],
        "transport": "stdio"  # Communication via stdin/stdout
    }
}

# Global client variable - will be initialized lazily
_client = None

def get_mcp_client():
    """Get or initialize MCP client lazily to avoid issues with LangGraph Platform."""
    global _client
    if _client is None:
        print("Creating new MCP client...")
        _client = MultiServerMCPClient(mcp_config)
    else:
        print("Reusing existing MCP client...")
    return _client

async def final_report_generation(state: AgentState) -> Command[Literal["final_report_tools"]]:
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """

    notes = state.get("notes", [])

    findings = "\n".join(notes)

    final_report_prompt = final_report_generator_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str()
    )

    # Get available tools from MCP server
    client = get_mcp_client()
    tools = await client.get_tools() # should cache the tools

    # Initialize model with tool binding
    model_with_tools = writer_model.bind_tools(tools)

    final_report = await model_with_tools.ainvoke([HumanMessage(content=final_report_prompt)])

    return Command(
        goto="final_report_tools",
        update={
            "final_report": final_report,
            "messages": ["Here is the final report: " + final_report.content],
        }
    )

async def final_report_tools(state: AgentState) -> Command[Literal["__end__"]]:
    """Execute tool calls using MCP tools.

    This node:
    1. Retrieves current tool calls from the final_report AI message
    2. Executes all tool calls using async operations (required for MCP)
    3. Returns formatted tool results

    Note: MCP requires async operations due to inter-process communication
    with the MCP server subprocess. This is unavoidable.
    """
    # Get tool calls from the final_report AI message, not from messages list
    final_report = state.get("final_report")
    if not hasattr(final_report, 'tool_calls') or not final_report.tool_calls:
        return Command(
            goto=END,
            update={

            }
        )

    tool_calls = final_report.tool_calls

    async def execute_tools():
        """Execute all tool calls. MCP tools require async execution."""
        # Get fresh tool references from MCP server
        client = get_mcp_client()
        tools = await client.get_tools()
        tools_by_name = {tool.name: tool for tool in tools}

        # Execute tool calls (sequentially for reliability)
        observations = []
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = await tool.ainvoke(tool_call["args"])
            observations.append(observation)

        # Format results as tool messages
        tool_outputs = [
            ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            for observation, tool_call in zip(observations, tool_calls)
        ]

        return tool_outputs

    tool_messages = await execute_tools()

    return Command(
        goto=END,
        update={
            "final_report": final_report.content,
            "messages": tool_messages
        }
    )

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
final_report_builder = StateGraph(AgentState)

# Add nodes to the graph
final_report_builder.add_node("final_report_generation", final_report_generation)
final_report_builder.add_node("final_report_tools", final_report_tools)

# Add edges to connect nodes
final_report_builder.add_edge(START, "final_report_generation")
final_report_builder.add_edge("final_report_tools", END)

# Compile the agent
research_report = final_report_builder.compile()
