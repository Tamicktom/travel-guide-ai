import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import bs4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

llm = ChatOpenAI(model="gpt-3.5-turbo")

query = """
Vou viajar para Londres em Agosto de 2024.
Quero que faça um plano de viagem para mim com eventos que irão ocorrer nada data da minha viagem.
"""

def researchAgent(query:str, llm: ChatOpenAI):
  tools = load_tools(["ddg-search", "wikipedia"], llm=llm)
  prompt = hub.pull("hwchase17/react")
  agent = create_react_agent(llm, tools, prompt)
  agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt, verbose=True)
  webContext = agent_executor.invoke({
    "input":query
  })
  return webContext

def loadData():
  loader = WebBaseLoader(
    web_paths=("https://www.dicasdeviagem.com/inglaterra/"),
    bs_kwargs=dict(
      parse_onlu=bs4.SoupStrainer(
        class_=("postcontentwrap", "pagetitleloading")
      )
    ),
  )
  docs = loader.load()
  text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
  )
  splits = text_spliter.split_documents(docs)
  vector_store = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
  )
  retriever = vector_store.as_retriever()
  return retriever

def getRelevantDocs(query:str):
  retriever = loadData()
  relevant_documents = retriever.invoke(query)
  return relevant_documents

def supervisorAgent(query:str, llm: ChatOpenAI, webContext, relevant_documents):
  prompt_template = """
    Você é um gerente de uma agência de viagens. Sua resposta final deverá ser um roteiro de viagem completo e detalhado.
    Utilize o contexto de eventos e preços de passagens, o input do usuário e também os documentos relevantes para elaborar o roteiro.
    
    Contexto: {webContext}
    Documentos Relevantes: {relevant_documents}
    Usuário: {query}
    Assistete: 
  """
  
  prompt = PromptTemplate(
    input_variables=["webContext", "relevant_documents", "query"],
    template=prompt_template
  )
  
  sequence = RunnableSequence(prompt | llm)
  
  response = sequence.invoke({
    "webContext":webContext,
    "relevant_documents":relevant_documents,
    "query":query
  })
  
  return response

def getResponse(query, llm):
  webContext = researchAgent(query, llm)
  relevant_documents = getRelevantDocs(query)
  
  response = supervisorAgent(query, llm, webContext, relevant_documents)
  return response

print(getResponse(query, llm))