from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from tempfile import NamedTemporaryFile
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import io
from langchain import hub
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import OllamaEmbeddings
import logging

app = FastAPI()


# Define the LangChain LLM
llm = Ollama(
    base_url="http://ollama:11434/",
    model='llama3'
    # other params...
)
prompt = hub.pull("rlm/rag-prompt")

qdrant = QdrantVectorStore.from_existing_collection(collection_name="podcast2", embedding=OllamaEmbeddings( base_url="http://ollama:11434/", model='llama3'), path="./local_qdrant_podcast" )
retriever = qdrant.as_retriever()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/hello")
def hello_world():
    return {"message": "Hello, world!"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    user_message = request.message
    logging.info(user_message)

    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    # Process user message with LangChain and ChatGPT
    chatgpt_response = rag_chain.invoke({"query": user_message})
    return ChatResponse(response=chatgpt_response)

def format_jobs(jobs):
    formatted = "Here are some job listings:\n"
    for job in jobs:
        formatted += f"{job['title']} at {job['company']}\n"
    return formatted

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
