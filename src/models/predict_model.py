from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# Load web page
import argparse
from helper import pull_model
from langchain.embeddings import OllamaEmbeddings # We can also try Ollama embeddings

from langchain_qdrant import QdrantVectorStore

from langchain_ollama import ChatOllama


model = 'llama3'
pull_model(model)
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=OllamaEmbeddings(model=model),
    collection_name="my_documents",
    path='./data/processed/local_qdrant'

)
retriever = qdrant.as_retriever()


prompt = hub.pull("rlm/rag-prompt")


llm = ChatOllama(
    model="phi3",
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

rag_chain.invoke("What is the name of the book?")
