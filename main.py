from dotenv import load_dotenv

load_dotenv()

from graph.graph import app


print("Hello Advanced RAG")
print(app.invoke(input={"question": "what is the conclusion of this document"}))
