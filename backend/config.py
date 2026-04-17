from dotenv import load_dotenv
import os

load_dotenv()

# LLM
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_LLM_MODEL: str = os.getenv("NVIDIA_LLM_MODEL", "nvidia/llama-3.1-nemotron-nano-8b-v1")

# Embeddings
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", 16))

# Reranker
RERANKER_MODEL_NAME: str = os.getenv(
    "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K", 5))

# Vector DB
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# Chunking
PARENT_CHUNK_SIZE: int = int(os.getenv("PARENT_CHUNK_SIZE", 1500))
CHILD_CHUNK_SIZE: int = int(os.getenv("CHILD_CHUNK_SIZE", 300))
CHILD_CHUNK_OVERLAP: int = int(os.getenv("CHILD_CHUNK_OVERLAP", 30))

# Legacy chunking defaults kept for compatibility with older helpers.
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))

# Retrieval
TOP_K_CANDIDATES: int = int(os.getenv("TOP_K_CANDIDATES", 40))
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 5))
USE_HYDE: bool = os.getenv("USE_HYDE", "true").lower() == "true"

# SEC EDGAR
# SEC requires a User-Agent header in the format: "Name email@example.com".
EDGAR_USER_AGENT: str = os.getenv(
    "EDGAR_USER_AGENT", "FinLens your-email@example.com"
)
EDGAR_BASE_URL: str = "https://data.sec.gov"
EDGAR_ARCHIVES_URL: str = "https://www.sec.gov/Archives/edgar/data"

# Storage
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
UPLOAD_TIMEOUT_SECONDS: int = int(os.getenv("UPLOAD_TIMEOUT_SECONDS", 900))

# Frontend
BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000")
