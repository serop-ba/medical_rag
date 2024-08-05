from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import io
from langchain import hub
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import OllamaEmbeddings
import os 
import logging

app = FastAPI()
file_name = "motor_neuron_disease.pdf"
collection_name = file_name.split('.')[0]
# from dotenv import load_dotenv
# load_dotenv()
# voyage_api_key = os.getenv('VOYAGEAI_API_KEY')
# os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

# embeddings = VoyageAIEmbeddings(
#     voyage_api_key=voyage_api_key, model="voyage-large-2", show_progress_bar=True, truncation=False, batch_size=100 # )

# from langchain_anthropic import ChatAnthropic

# model = ChatAnthropic(model='claude-3-opus-20240229')

# Define the LangChain LLM
llm = Ollama(
    base_url="http://ollama:11434/",
    model='phi3'
    # other params...
)
prompt = hub.pull("rlm/rag-prompt")

qdrant = QdrantVectorStore.from_existing_collection(collection_name=collection_name, embedding=OllamaEmbeddings(base_url="http://ollama:11434/", model='phi3'), path="/app/data/processed/embedded_documents" )
retriever = qdrant.as_retriever()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/hello")
def hello_world():
    return {"message": "Hello, world!"}

# @app.post("/chat_server", response_model=ChatResponse)
# async def chat(request: ChatRequest):
#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)
#     user_message = request.message
#     print(request)
#     print(request.message)
#     logging.info(user_message)

#     rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
#     )
#     # Process user message with LangChain and ChatGPT
#     response = rag_chain.invoke(user_message)

#     return ChatResponse(response=response)
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
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
