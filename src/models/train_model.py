# Load web page
import argparse
from helper import pull_model
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embed and stor
from langchain_community.embeddings import OllamaEmbeddings # We can also try Ollama embeddings
from langchain_qdrant import QdrantVectorStore

file_name = "podcast_transcript.docx"
collection_name = "first_podcast_episode"

loader = Docx2txtLoader(f'./data/raw/{file_name}')
data = loader.load()

# Split into chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
all_splits = text_splitter.split_documents(data)
print(f"Split into {len(all_splits)} chunks")

pull_model('llama3','localhost')

qdrant = QdrantVectorStore.from_documents(
    all_splits,
    OllamaEmbeddings(model='llama3'),
    path="./data/processed/local_qdrant",
    collection_name="first_podcast_episode",
)

