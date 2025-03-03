import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pdfminer.high_level import extract_text

# Define paths
working_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(working_dir, "data")
vector_db_dir = os.path.join(working_dir, "vector_db_dir")

# Load embedding model
embeddings = HuggingFaceEmbeddings()

def load_documents():
    """Load and process PDFs and JSON files."""
    documents = []
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)

        if filename.endswith(".pdf"):
            text = extract_text(file_path)

        elif filename.endswith(".json"):  # Process JSON files
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                text = "\n".join([f"{key}: {value}" for key, value in json_data.items()])

        else:
            continue  # Skip unsupported files

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_text(text)

        for chunk in chunks:
            documents.append({"page_content": chunk, "metadata": {"source": filename}})
    
    return documents

def store_embeddings():
    """Create and store embeddings in ChromaDB."""
    documents = load_documents()

    # Create or load ChromaDB collection
    vectorstore = Chroma(
        persist_directory=vector_db_dir,  # Auto-saves embeddings
        embedding_function=embeddings
    )

    # Add documents to the database
    vectorstore.add_texts(
        texts=[doc["page_content"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents]
    )

    print("✅ Documents have been indexed successfully!")

if __name__ == "__main__":
    store_embeddings()
