import os
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent

llm = ChatOpenAI(model="gpt-3.5-turbo")

tools = load_tools(["ddg-search", "wikipedia"], llm=llm)

# print(tools[0].name, tools[0].description)
# print(tools[1].name, tools[1].description)

agent = initialize_agent(
  tools, 
  llm=llm, 
  agent='zero-shot-react-description',
  verbose=True
)

print(agent.agent.llm_chain.prompt.template)

query = """
Vou viajar para Londres em agosto de 2024.
Quero que faça um roteiro de viagem para mim com os eventos que irão ocorrer na data da viagem e com o preço de passagem de São Paulo para Londres.
"""

agent.run(query)