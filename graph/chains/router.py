from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch", "reject"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """
You are an expert router in a RAG system.

You must choose exactly one of the following tools:
- `vectorstore` → contains chunks from a single PDF uploaded by the user
- `websearch` → performs a scientific/academic search (Arxiv)
- `reject` → for all non-scientific and non-PDF-related questions

Follow these rules:

### 1) Use `vectorstore` when the question is about the uploaded PDF.
This includes explicit or implicit references to the document such as:
- "this paper"
- "the introduction of this paper"
- "the conclusion of this document"
- "explain section 2"
- "summarize the paper"
- "what does the thesis say about X?"
- "in the uploaded document"
- any question asking about content, sections, results, formulas, definitions, or methods inside the PDF

If the question asks about the paper’s introduction, methodology, results, abstract, or any internal part,
ALWAYS choose `vectorstore`.

### 2) Use `websearch` when the question is scientific or academic, but NOT specifically about the uploaded PDF.
Examples:
- "What is a transformer model?"
- "Explain GANs"
- "What is requirements engineering?"
- "What does the literature say about X?"
- "How does backpropagation work?"

### 3) Use `reject` when the question is not about the PDF and not scientific or academic.
Examples:
- "How is the weather today?"
- "Who won the match?"
- "Give me relationship advice."
- "What is the best car brand?"

### Output rules:
You must output ONLY ONE of the following tokens:
- vectorstore
- websearch
- reject
Nothing else.
"""


route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
