'''from typing import Any, Dict
from langchain_core.documents import Document
#from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv import ArxivQueryRun

from graph.state import GraphState
from dotenv import load_dotenv
load_dotenv()
web_search_tool = TavilySearchResults(k=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}
'''

from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools.arxiv import ArxivQueryRun

from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()

# Arxiv wrapper + tool
arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=3,
    doc_content_chars_max=4000
)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---ARXIV SEARCH---")
    question = state["question"]
    documents = state.get("documents")

    # ArxivQueryRun, string döner (makalelerin özet + başlık + link vs.)
    arxiv_text: str = arxiv_tool.invoke({"query": question})
    arxiv_doc = Document(page_content=arxiv_text, metadata={"source": "arxiv"})

    if documents is not None:
        documents.append(arxiv_doc)
    else:
        documents = [arxiv_doc]

    return {"documents": documents, "question": question}
