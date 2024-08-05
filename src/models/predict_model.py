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


# voyage_api_key = os.getenv('VOYAGEAI_API_KEY')

# embeddings = VoyageAIEmbeddings(
#     voyage_api_key=voyage_api_key, model="voyage-large-2", show_progress_bar=True, truncation=False, batch_size=100
# )

qdrant = QdrantVectorStore.from_existing_collection(
    embedding= OllamaEmbeddings(model='phi3',show_progress=True),
    collection_name=collection_name,
    path='./data/processed/embedded_documents'

)
retriever = qdrant.as_retriever()


prompt = hub.pull("rlm/rag-prompt")

model = 'phi3'
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
