"""Hybrid Retriever - FAISS vector search + NetworkX knowledge graph (GraphRAG)."""
import re
import numpy as np
import networkx as nx
from typing import Optional
from sentence_transformers import SentenceTransformer
import faiss

from config import EMBEDDING_MODEL, TOP_K


class HybridRetriever:
    """
    Combines dense vector retrieval (FAISS) with a lightweight
    knowledge graph (NetworkX) for GraphRAG-style context enrichment.
    """

    def __init__(self):
        self.embedder   = SentenceTransformer(EMBEDDING_MODEL)
        self.dim        = self.embedder.get_sentence_embedding_dimension()
        self.index      = faiss.IndexFlatL2(self.dim)
        self.graph      = nx.Graph()
        self.documents  = []   # raw chunk dicts
        self.vector     = []   # parallel list to index rows

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, chunks: list[dict]) -> None:
        """Embed chunks, add to FAISS, and build graph edges."""
        if not chunks:
            return

        texts = [c.get("text", "") for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        self.index.add(embeddings)
        start_id = len(self.vector)

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            node_id = start_id + i
            self.vector.append(chunk)
            self.documents.append(chunk)

            # Add node to graph
            self.graph.add_node(
                node_id,
                text=chunk.get("text", ""),
                source=chunk.get("source", ""),
                page=chunk.get("page", 0),
            )

            # Extract entities and link them
            entities = self._extract_entities(chunk.get("text", ""))
            for entity in entities:
                self.graph.add_node(entity, type="entity")
                self.graph.add_edge(node_id, entity, weight=1.0)

            # Link consecutive chunks from same source
            if i > 0:
                prev_id = start_id + i - 1
                if (
                    self.vector[prev_id].get("source") == chunk.get("source")
                ):
                    self.graph.add_edge(prev_id, node_id, weight=0.8)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = TOP_K) -> dict:
        """
        Retrieve top-k chunks via FAISS then enrich with graph context.

        Returns
        -------
        {
            'results'      : list of chunk dicts,
            'graph_context': { 'neighbors': [...], 'subgraph_nodes': int }
        }
        """
        if len(self.vector) == 0:
            return {"results": [], "graph_context": None}

        # 1. Dense retrieval
        query_emb = self.embedder.encode([query], show_progress_bar=False)
        query_emb = np.array(query_emb, dtype="float32")

        k_actual = min(k, len(self.vector))
        distances, indices = self.index.search(query_emb, k_actual)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.vector):
                continue
            chunk = dict(self.vector[idx])
            chunk["score"] = float(1 / (1 + dist))  # normalise distance → score
            results.append(chunk)

        # 2. Graph context enrichment
        graph_context = self._get_graph_context(indices[0])

        return {"results": results, "graph_context": graph_context}

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def _get_graph_context(self, indices: np.ndarray) -> dict:
        """Return graph neighbours for the retrieved nodes."""
        all_neighbors = []
        valid_indices = [int(i) for i in indices if 0 <= int(i) < len(self.vector)]

        for node_id in valid_indices:
            if node_id not in self.graph:
                continue
            neighbors = list(self.graph.neighbors(node_id))
            # Only keep string entity neighbours (not numeric chunk ids)
            entity_neighbors = [n for n in neighbors if isinstance(n, str)]
            if entity_neighbors:
                all_neighbors.append(entity_neighbors[:5])

        subgraph_nodes = self.graph.number_of_nodes()
        return {
            "neighbors":      all_neighbors,
            "subgraph_nodes": subgraph_nodes,
        }

    @staticmethod
    def _extract_entities(text: str) -> list[str]:
        """
        Lightweight rule-based entity extraction (no spaCy dependency).
        Extracts capitalised phrases, numbers with units, and % figures.
        """
        entities = []

        # Capitalised multi-word phrases (proper nouns / org names)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        entities.extend(caps[:5])

        # Financial figures  e.g. $1.2B, $500M, €200K
        money = re.findall(r'[\$€£¥]\s?\d+\.?\d*\s?[BMKbmk]?', text)
        entities.extend(money[:5])

        # Percentages
        pcts = re.findall(r'\d+\.?\d*\s?%', text)
        entities.extend(pcts[:3])

        # Deduplicate and clean
        seen, clean = set(), []
        for e in entities:
            e = e.strip()
            if e and e not in seen:
                seen.add(e)
                clean.append(e)

        return clean[:10]