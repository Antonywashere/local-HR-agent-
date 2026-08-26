import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

documents = [] 

#folder=Path('C:/Users/ASUS/Desktop/python files/ucks/thecourseyes/theHR/document')
BASE_DIR = Path(__file__).parent

# Build the documents path relative to the script
folder = BASE_DIR / 'documents'

for file in os.listdir(folder):
    path = os.path.join(folder, file)
    loader = TextLoader(path)
    doc=loader.load()
    documents.extend(doc)
    
print(f"Loaded {len(documents)} documents")

splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunks=splitter.split_documents(documents)

print(f"created {len(chunks)} chunks")

embeddings=OllamaEmbeddings(model="nomic-embed-text")

vector_db=FAISS.from_documents(chunks, embeddings)

vector_db.save_local("HR_vector_db")

print("Vector database is created")    
    
    