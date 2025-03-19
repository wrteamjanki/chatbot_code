import asyncio
import torch
import time
import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Check if an event loop is already running
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Start the timer
start_time = time.time()

# Check if the "data" folder exists
print("Checking if the 'data' folder exists...")
if not os.path.exists("data"):
    raise FileNotFoundError(
        "'data' directory not found! Make sure it exists and contains PDF or JSON files."
    )
print("'data' folder found.")

# Loading documents
print("Loading documents...")
load_start = time.time()

# Load PDF documents
pdf_documents = []
try:
    pdf_loader = DirectoryLoader(path="data", glob="*.pdf", loader_cls=PyMuPDFLoader)
    pdf_documents = pdf_loader.load()
    print(f"{len(pdf_documents)} PDF documents loaded.")
except Exception as e:
    print(f"Error loading PDFs: {e}")

# Load JSON documents
json_documents = []
try:
    json_loader = DirectoryLoader(
        path="data",
        glob="*.json",
        loader_cls=JSONLoader,
        loader_kwargs={"jq_schema": ".", "text_content": False},
    )
    json_documents = json_loader.load()
    print(f"{len(json_documents)} JSON documents loaded.")
except Exception as e:
    print(f"Error loading JSON files: {e}")

# Merge the loaded documents
documents = pdf_documents + json_documents
print(f"Total documents loaded: {len(documents)}")

if not documents:
    raise ValueError(
        "No documents were loaded. Ensure that files are present in the 'data' folder."
    )

load_end = time.time()
print(f"Document loading completed in {load_end - load_start:.2f} seconds.")

# Splitting documents into chunks
split_start = time.time()
print("Splitting documents into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, chunk_overlap=100, add_start_index=True
)
text_chunks = text_splitter.split_documents(documents)

print(f"Total number of chunks created: {len(text_chunks)}")

if not text_chunks:
    raise ValueError(
        "No text chunks were generated after splitting. Check your documents."
    )

split_end = time.time()
print(f"Splitting completed in {split_end - split_start:.2f} seconds.")

# Vectorizing documents
vector_start = time.time()
print("Initializing embedding model...")

embeddings = HuggingFaceEmbeddings()

print("Vectorizing documents...")
vectordb = Chroma.from_documents(
    documents=text_chunks, embedding=embeddings, persist_directory="vectordb"
)

vector_end = time.time()
print(f"Vectorization completed in {vector_end - vector_start:.2f} seconds.")

# Total execution time
total_time = time.time() - start_time
print(f"Total execution time: {total_time:.2f} seconds.")

