"""Pure embedding using SentenceTransformers."""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import pickle
from pathlib import Path

from config import EMBEDDING_MODEL, FAISS_DIR

class Embedder:
    """Simple, effective embedding - no framework overhead."""
    
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Loaded embedding model: {EMBEDDING_MODEL} (dim={self.dim})")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to vectors."""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    
    def save(self, name: str = "embedder.pkl"):
        """Save model config."""
        path = FAISS_DIR / name
        with open(path, 'wb') as f:
            pickle.dump({'model_name': EMBEDDING_MODEL, 'dim': self.dim}, f)
    
    @staticmethod
    def load_dim() -> int:
        """Get embedding dimension without loading model."""
        return 384  # MiniLM-L6 dimension