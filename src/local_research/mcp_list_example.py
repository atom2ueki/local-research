import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from local_research.utils import get_current_dir

console = Console()

async def main():
    # MCP Client configuration - filesystem server for local document access
    mcp_config = {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",  # Auto-install if needed
                "@modelcontextprotocol/server-filesystem",
                str(get_current_dir() / "files")
            ],
            "transport": "stdio"
        }
    }

    console.print(Panel("[bold yellow]Creating MCP client...[/bold yellow]", expand=False))
    client = MultiServerMCPClient(mcp_config)
    console.print("[green]✓ MCP client created successfully![/green]")

    # Test getting tools
    console.print(Panel("[bold yellow]Getting tools...[/bold yellow]", expand=False))
    tools = await client.get_tools()

    # Create a rich table for tool display
    table = Table(title="Available MCP Tools", show_header=True, header_style="bold magenta")
    table.add_column("Tool Name", style="cyan", width=25)
    table.add_column("Description", style="white", width=80)

    for tool in tools:
        description = tool.description[:77] + "..." if len(tool.description) > 80 else tool.description
        table.add_row(tool.name, description)

    console.print(table)
    console.print(f"[bold green]✓ Successfully retrieved {len(tools)} tools from MCP server[/bold green]")

if __name__ == "__main__":
    asyncio.run(main())
