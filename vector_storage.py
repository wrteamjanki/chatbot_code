import chromadb
import uuid
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="chroma_db")

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create or load collection
collection = client.get_or_create_collection("pdf_vectors")

def store_text_embeddings(text: str):
    """Stores text embeddings in ChromaDB."""
    embedding = model.encode([text])[0]
    unique_id = str(uuid.uuid4())  # Generate a unique ID for each document
    collection.add(ids=[unique_id], documents=[text], metadatas=[{"source": "pdf"}], embeddings=[embedding])

def retrieve_similar_text(query: str) -> str:
    """Retrieves the most similar text from ChromaDB."""
    query_embedding = model.encode([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    return results["documents"][0] if results["documents"] else "No relevant information found."
