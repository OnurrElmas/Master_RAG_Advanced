#from langchain import hub
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)
llm = ChatOpenAI(temperature=0)
#prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
