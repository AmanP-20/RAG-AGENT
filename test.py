import os
from dotenv import load_dotenv

load_dotenv(override=True)

print("KEY:", os.getenv("MISTRAL_API_KEY")[:10])

from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY")
)

response = llm.invoke("Explain RAG in one sentence")

print(response.content)