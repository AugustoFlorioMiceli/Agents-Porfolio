import os
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import ChatOllama
from langchain.messages import AIMessage
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from langgraph.checkpoint.memory import InMemorySaver

tavily_client = TavilyClient()

model = ChatOllama(
    model="llama3.2"
)

@tool (description="Buscador en la web")
def web_searcher(query: str) -> Dict[str, Any]:
    return tavily_client.search(query)

agent = create_agent(
    model=model,
    tools=[web_searcher],
    checkpointer=InMemorySaver(),
    system_prompt="Eres un Chef profesional y asistente de cocina, que utilizara las herramientas a disposicion para recomendar recetas a partir del mensaje"
)

question = HumanMessage(content="Quiero que me digas que puedo cocinar, tengo en mi heladera, carne picada un tomate, una cebolla, un apio una calabaza, queso y unos fideos")
config = {"configurable": {"thread_id": 1}}

response = agent.invoke(
    {"messages": [question]},
    config
)

print(response["messages"][-1].content)
