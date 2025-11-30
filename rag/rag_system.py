import os
import re
import faiss
import pickle
from typing import List
from transformers import PreTrainedTokenizerBase
from sentence_transformers import SentenceTransformer

from config.settings import (
    TORCH_DEVICE,
    EMBEDDER_MODEL,
    FAISS_INDEX_PATH,
    CHUNKS_PATH
)


class RagSystem:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        # Dedicated tokenizer (LLM tokenizer)
        self.tokenizer = tokenizer

        # Dedicated embedder (fixed model)
        self.embedder = SentenceTransformer(EMBEDDER_MODEL, device=TORCH_DEVICE)

        # Vector store
        self.index = None
        self.chunks: List[str] = []

        # Load existing index + chunks if available
        self.load_index()

    # ------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'^\s+|\s+$', '', text)
        text = re.sub(r'\n{2,}', '\n\n', text)
        text = re.sub(r'Page\s*\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # ------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------
    def chunk_text_by_tokens(self, text: str, max_tokens: int = 200, overlap: int = 30):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text.strip())
            start += max_tokens - overlap

        return chunks

    # ------------------------------------------------------------
    # Add & Index Single Text
    # ------------------------------------------------------------
    def add_text(self, text: str, max_tokens: int = 200, overlap: int = 30, save: bool = True):
        text = self.clean_text(text)
        new_chunks = self.chunk_text_by_tokens(text, max_tokens, overlap)
        self.chunks.extend(new_chunks)

        embeddings = self.embedder.encode(new_chunks, convert_to_numpy=True)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        if save:
            self.save_index()

        return {"added_chunks": len(new_chunks)}

    # ------------------------------------------------------------
    # Add & Index Multiple Documents (Batch)
    # ------------------------------------------------------------
    def add_documents(self, texts: List[str], max_tokens: int = 200, overlap: int = 30, save: bool = True):
        all_chunks = []
        for text in texts:
            cleaned = self.clean_text(text)
            chunks = self.chunk_text_by_tokens(cleaned, max_tokens=max_tokens, overlap=overlap)
            all_chunks.extend(chunks)

        self.chunks.extend(all_chunks)

        embeddings = self.embedder.encode(all_chunks, batch_size=32, convert_to_numpy=True)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        if save:
            self.save_index()

        return {"added_chunks": len(all_chunks)}

    # ------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------
    def save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, FAISS_INDEX_PATH)

        with open(CHUNKS_PATH, 'wb') as f:
            pickle.dump(self.chunks, f)

    def load_index(self):
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CHUNKS_PATH, 'rb') as f:
                self.chunks = pickle.load(f)

    # ------------------------------------------------------------
    # Query
    # ------------------------------------------------------------
    def query(self, query: str, k: int = 5):
        if self.index is None or len(self.chunks) == 0:
            raise ValueError("FAISS index not built or chunks not loaded.")

        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(q_emb, k)

        results = [self.chunks[i] for i in indices[0].tolist()]
        dists = distances[0].tolist()

        return results, dists
