"""Pure FAISS implementation - no wrappers."""
import faiss
import numpy as np
import pickle
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from config import FAISS_DIR

class VectorStore:
    """Direct FAISS with metadata - simple and fast."""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = None
        self.metadata = {}  # id -> {text, source, page, etc.}
        self.next_id = 0
        
        # File paths
        self.index_path = FAISS_DIR / "faiss.bin"
        self.meta_path = FAISS_DIR / "metadata.json"
        self.id_path = FAISS_DIR / "next_id.txt"
        
        self._load()
    
    def _load(self):
        """Load existing index if available."""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, 'r') as f:
                self.metadata = json.load(f)
            with open(self.id_path, 'r') as f:
                self.next_id = int(f.read())
            print(f"✓ Loaded index: {len(self.metadata)} vectors")
        else:
            # Create new index - InnerProduct for cosine (normalized vectors)
            self.index = faiss.IndexFlatIP(self.dim)
            print("✓ Created new FAISS index")
    
    def add(self, vectors: np.ndarray, texts: List[str], metas: List[Dict]):
        """Add vectors with metadata."""
        n = len(vectors)
        ids = np.arange(self.next_id, self.next_id + n)
        
        # Add to FAISS
        self.index.add(vectors.astype('float32'))
        
        # Store metadata
        for i, (text, meta) in enumerate(zip(texts, metas)):
            doc_id = str(self.next_id + i)
            self.metadata[doc_id] = {
                'text': text,
                **meta,
                'vector_id': int(ids[i])
            }
        
        self.next_id += n
        self._save()
        print(f"✓ Added {n} vectors (total: {len(self.metadata)})")
    
    def search(self, query_vec: np.ndarray, k: int = 10) -> List[Dict]:
        """Search similar vectors."""
        if len(self.metadata) == 0:
            return []
        
        scores, indices = self.index.search(query_vec.reshape(1, -1).astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            doc_id = str(idx)
            if doc_id in self.metadata:
                result = self.metadata[doc_id].copy()
                result['score'] = float(score)
                result['id'] = doc_id
                results.append(result)
        
        return results
    
    def _save(self):
        """Persist index and metadata."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        with open(self.id_path, 'w') as f:
            f.write(str(self.next_id))
    
    def clear(self):
        """Clear all data."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = {}
        self.next_id = 0
        self._save()
    
    def __len__(self):
        return len(self.metadata)