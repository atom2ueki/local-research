# Local Research Agent

A LangGraph-based research agent that conducts comprehensive research and generates detailed reports.

## Architecture

![Agent Workflow](agent_graph.png)

## Quick Start

Run the development server:

```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

## Features

- Automated research using web search and local files
- Intelligent report generation with citations
- Multi-step research workflow with supervisor coordination
- Configurable research scope and evaluation criteria