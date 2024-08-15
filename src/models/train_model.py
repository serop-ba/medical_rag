# Load web page
from helper import pull_model
from langchain.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

                        #### arguments #### 
# Enter your file name and path here 
file_name = "motor_neuron_disease.pdf"
path = "./data/raw/{file_name}"
# where to save the embeddings
embeddings_path = "./data/processed/embedded_documents"
# model to use, must be included in the ollama server
model = 'llama3'

collection_name = file_name.split('.')[0]

loader = PyPDFLoader(path)
data = loader.load()

# Split into chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
all_splits = text_splitter.split_documents(data)
print(f"Split into {len(all_splits)} chunks")

# install the model in the container
pull_model(model,'localhost')

qdrant = QdrantVectorStore.from_documents(
    all_splits,
    OllamaEmbeddings(model='llama3',show_progress=True),
    path=embeddings_path,
    collection_name=collection_name,
)

