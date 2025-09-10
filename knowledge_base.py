# knowledge_base.py
"""
Handles the creation, loading, and management of the winery's knowledge base.

This module is responsible for taking the unstructured text data from the
winery information file and converting it into a searchable vector store (FAISS).
It persists the index to disk to avoid costly re-computation on every startup.
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH

# Constants for the text splitting process
DOCUMENT_SOURCE_PATH: str = "./wine_business_info.md"
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150


def get_retriever() -> VectorStoreRetriever:
    """
    Initializes and returns a VectorStoreRetriever for the winery knowledge base.

    This function implements a "load-or-create" pattern:
    1. It checks if a FAISS index already exists at the specified path.
    2. If it exists, it loads the index directly from disk.
    3. If not, it processes the source document, creates embeddings using the
       configured Google model, builds a new FAISS index, and saves it to disk
       for future use.

    Returns:
        VectorStoreRetriever: An object ready to perform similarity searches
                              on the winery's information.

    Raises:
        ValueError: If the GOOGLE_API_KEY environment variable is not set.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    # Initialize the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME, google_api_key=google_api_key
    )

    if os.path.exists(FAISS_INDEX_PATH):
        print("Knowledge Base: Loading existing FAISS index from disk.")
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("Knowledge Base: FAISS index not found. Creating a new one.")
        
        # Load the source document
        loader = TextLoader(DOCUMENT_SOURCE_PATH)
        documents = loader.load()

        # Split the document into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        docs = text_splitter.split_documents(documents)

        # Create the FAISS index from the document chunks
        print("Knowledge Base: Creating and storing new embeddings...")
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Save the newly created index to disk for future runs
        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"Knowledge Base: FAISS index saved to '{FAISS_INDEX_PATH}'.")

    # Create and return the retriever
    return vector_store.as_retriever()