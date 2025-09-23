try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from local_research.agent import agent

def generate_agent_graph():
    """Generate PNG image of the agent workflow."""

    agent_png = agent.get_graph(xray=True).draw_mermaid_png()
    with open("agent_graph.png", "wb") as f:
        f.write(agent_png)
    print("Generated agent_graph.png")

if __name__ == "__main__":
    generate_agent_graph()
