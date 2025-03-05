# Multi-Document RAG Chatbot

## Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot that processes and retrieves information from multiple documents (PDF and JSON) using vector embeddings and LangChain.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/multi_documents_rag_chatbot.git
   cd multi_documents_rag_chatbot
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv myenv
   myenv/Scripts/activate  # On Windows
   source myenv/bin/activate  # On macOS/Linux
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Place your PDF or JSON documents in the `data/` folder.
2. Run the vectorization script to process documents:
   ```sh
   python vectorized_documents.py
   ```
3. Start the chatbot application:
   ```sh
   streamlit run main.py
   ```
4. Interact with the chatbot through the Streamlit interface.

## Project Structure
- `main.py` – Runs the Streamlit chatbot interface.
- `vectorized_documents.py` – Loads, processes, and vectorizes documents.
- `data/` – Folder containing input documents.
- `vectordb/` – Stores the generated embeddings.
- `config.json` – Stores API keys and configurations.
- `requirements.txt` – List of required Python packages.

## Troubleshooting
- If documents are not loading, ensure they are placed in the `data/` folder.
- If vectorization fails, check if dependencies are installed correctly.
- For memory issues, reduce `chunk_size` in `vectorized_documents.py`.

## License
This project is open-source. You are free to modify and use it as needed.

