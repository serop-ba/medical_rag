from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from helper import pull_model
from langchain import hub
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import OllamaEmbeddings
import logging

app = FastAPI()
file_name = "motor_neuron_disease.pdf"
collection_name = file_name.split('.')[0]
embeddings_path = "/app/data/processed/embedded_documents"
model = 'phi3'

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/hello")
def hello_world():
    return {"message": "Hello, world!"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
        # Define the LangChain LLM
    pull_model(model,'ollama')
    llm = Ollama(
        base_url="http://ollama:11434/",
        model=model
        # other params...
    )
    prompt = hub.pull("rlm/rag-prompt")

    qdrant = QdrantVectorStore.from_existing_collection(collection_name=collection_name, embedding=OllamaEmbeddings(base_url="http://ollama:11434/", model=model), path=embeddings_path )
    retriever = qdrant.as_retriever()
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    user_message = request.message
    print(request)
    print(request.message)
    logging.info(user_message)

    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    # Process user message with LangChain and ChatGPT
    response = rag_chain.invoke(user_message)

    return ChatResponse(response=response)

def format_jobs(jobs):
    formatted = "Here are some job listings:\n"
    for job in jobs:
        formatted += f"{job['title']} at {job['company']}\n"
    return formatted

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
