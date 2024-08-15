from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# Load web page
import os 
import argparse
from helper import pull_model
from langchain_voyageai import VoyageAIEmbeddings
from langchain.embeddings import OllamaEmbeddings # We can also try Ollama embeddings

from langchain_qdrant import QdrantVectorStore

from langchain_ollama import ChatOllama

file_name = "motor_neuron_disease.pdf"
collection_name = file_name.split('.')[0]
# take from train_model.py
embedding_path = './data/processed/embedded_documents'

model = 'llama3'

qdrant = QdrantVectorStore.from_existing_collection(
    embedding= OllamaEmbeddings(model=model,show_progress=True),
    collection_name=collection_name,
    path=embedding_path

)
retriever = qdrant.as_retriever()


prompt = hub.pull("rlm/rag-prompt")

pull_model(model, service_name='localhost')

llm = ChatOllama(
    model=model,
    temperature=0,
    # other params...
)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("What is the name of the book?"))
