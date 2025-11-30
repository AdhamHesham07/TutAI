import os
import torch

# ======================================================
# Base Paths (Hugging Face Spaces Compatible)
# ======================================================

BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ======================================================
# Model Paths
# ======================================================

MODEL_PATH = "AdhamHesham/gemma3_1B_tutai"

FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")

# ======================================================
# RAG Settings (DIRECT VALUES — NOT FROM ENV)
# ======================================================

CHUNK_MAX_TOKENS = 200       # default: 200
CHUNK_OVERLAP = 30           # default: 30
DEFAULT_K = 5                # how many chunks to retrieve
CONFIDENCE_THRESHOLD = 0.62  # when RAG answer is trusted

# ======================================================
# Optional API Keys (ONLY SECRETS IN HF SPACES)
# ======================================================

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ======================================================
# Embeddings + LLM Fallback (non-local models)
# ======================================================

EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Used ONLY by tools that still call Gemini
LLM_MODEL_NAME = "gemini-2.5-flash"
