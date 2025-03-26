# from langchain_community.document_loaders import JSONLoader
# # import json
# # from pathlib import Path
# # from pprint import pprintfrom langchain_community.document_loaders import JSONLoader

# # import pandas as pd
# # df = pd.DataFrame({"Chunks": chunks})
# # print(df)
# # # Print all extracted documents
# # for i, doc in enumerate(documents):
# #     print(f"Document {i+1}:\n{doc.page_content}\n{'-'*50}")

# # Initialize the JSONLoader with the file path and jq schema
# loader = JSONLoader(
#     file_path='data/e-school-saas.json',
#     jq_schema=".",
#     text_content=False 
# )

# # # Load the documents
# documents = loader.load()
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# chunks = splitter.split_documents(documents)
# print(f"generated {len(chunks)} chunks")

# print(f"total chunks: {len(documents)}")

# # for i,doc in enumerate(documents[:5]):
# #     print(f"Document {i+1}:\n{doc.page_content}\n{'-'*50}")

from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

# Load JSON as raw data
loader = JSONLoader(
    file_path='data/e-school-saas.json',
    jq_schema=".",  # Load full JSON
    text_content=False  # Keep it as a dictionary
)

documents = loader.load()  # This returns a list of dictionaries

# # Convert the JSON document to a formatted string
# for doc in documents:
#     doc.page_content = json.dumps(doc.page_content, indent=2)  # Convert dict to string

# print(f"Loaded {len(documents)} documents")
# print(f"First document (string format):\n{documents[0].page_content[:1000]}...\n")

# Initialize text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split the documents into chunks
chunks = splitter.split_documents(documents)

print(f"Generated {len(chunks)} chunks")

# for i, chunk in enumerate(chunks[:5]):
#     print(f"Chunk {i+1} (Size: {len(chunk.page_content)} chars):\n{chunk.page_content}\n{'-'*50}")
# import matplotlib.pyplot as plt
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Define chunk sizes to test
# chunk_sizes = [200, 300,400,500,600,700,800,900]
# chunk_results = {}

# for size in chunk_sizes:
#     splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=50)
#     chunks = splitter.split_documents(documents)
#     chunk_results[size] = len(chunks)
#     print(f"Chunk Size {size}: Generated {len(chunks)} chunks")

# # Plot the chunk distribution
# plt.figure(figsize=(8, 5))
# plt.bar(chunk_results.keys(), chunk_results.values(), color="blue")
# plt.xlabel("Chunk Size (Characters)")
# plt.ylabel("Number of Chunks")
# plt.title("Chunk Size vs. Number of Chunks Generated")
# plt.show()

from langchain.embeddings import HuggingFaceEmbeddings

# Load a pre-trained embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Test embedding
sample_text = chunks[0].page_content
vector = embedding_model.embed_query(sample_text)
print(f"Sample Vector Shape: {len(vector)}")

import chromadb
from langchain.vectorstores import Chroma

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./vector_db")

# Store vectors in ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_model, 
    persist_directory="./vector_db"
)

vector_store.persist()
print("Vectors stored successfully!")



