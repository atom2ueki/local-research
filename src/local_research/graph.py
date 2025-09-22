# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass

# from IPython.display import Image, display
# from langgraph.checkpoint.memory import InMemorySaver
# from src.local_research.research_scope import deep_researcher_builder

# checkpointer = InMemorySaver()
# scope = deep_researcher_builder.compile(checkpointer=checkpointer)
# display(Image(scope.get_graph(xray=True).draw_mermaid_png()))


# from IPython.display import Image, display
# from local_research.research_agent import researcher_agent

# # Show the agent
# display(Image(researcher_agent.get_graph(xray=True).draw_mermaid_png()))
