# config.py
from dotenv import load_dotenv
# Load environment variables from the .env file at the project root
# This should be the first thing to run.
load_dotenv()

MODEL_NAME = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "models/embedding-001"
FAISS_INDEX_PATH = "faiss_index"

