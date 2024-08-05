# Load web page
import argparse
from helper import pull_model
from langchain.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from dotenv import load_dotenv
import os
# Embed and stor
from langchain_qdrant import QdrantVectorStore

file_name = "motor_neuron_disease.pdf"
collection_name = file_name.split('.')[0]

loader = PyPDFLoader(f'./data/raw/{file_name}')
data = loader.load()

# Split into chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
all_splits = text_splitter.split_documents(data)
print(f"Split into {len(all_splits)} chunks")



# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variables
voyage_api_key = os.getenv('VOYAGEAI_API_KEY')


embeddings = VoyageAIEmbeddings(
    voyage_api_key=voyage_api_key, model="voyage-large-2", show_progress_bar=True, truncation=False, batch_size=100
)

# pull_model('llama3','localhost')

qdrant = QdrantVectorStore.from_documents(
    all_splits,
    embeddings,
    path="./data/processed/embedded_documents",
    collection_name=collection_name,
)

