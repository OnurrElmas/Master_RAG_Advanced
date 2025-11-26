from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import regex as re
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter

load_dotenv()

converter = DocumentConverter()

result = converter.convert("2210.03629v3.pdf").document.export_to_markdown()

match = re.search(r"Abstract(.*?)References", result, re.DOTALL | re.IGNORECASE)
#result = " ".join(doc.page_content for doc in docs)

if match:
    result = match.group(1).strip()
else:
    result = result.strip()

docs_in = [Document(page_content=result)]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_in)

vectorstore = Chroma.from_documents(
     documents=doc_splits,
     collection_name="rag-chroma",
     embedding=OpenAIEmbeddings(),
     persist_directory="./.chroma",
 )

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()