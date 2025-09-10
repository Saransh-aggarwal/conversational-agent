# knowledge_base.py
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME

def get_retriever():
    """
    Creates and returns a retriever for the winery knowledge base.
    Loads the FAISS index from disk if it exists, otherwise creates it.
    """
    # Explicitly pass the API key from the environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    # MODIFIED LINE: Added google_api_key argument
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=api_key)

    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index from disk...")
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("FAISS index not found. Creating a new one...")
        loader = TextLoader("./wine_business_info.md")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        
        print("Creating and storing embeddings...")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(FAISS_INDEX_PATH)
        print("FAISS index created and saved.")

    return db.as_retriever()