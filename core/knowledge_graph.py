"""Knowledge graph using NetworkX - no spaCy dependency."""
import networkx as nx
import pickle
import json
import re
from typing import List, Dict, Set, Tuple
from pathlib import Path

from config import GRAPH_DIR

class KnowledgeGraph:
    """Graph for multi-hop reasoning - pure regex entity extraction."""
    
    def __init__(self):
        self.G = nx.DiGraph()
        self.path = GRAPH_DIR / "knowledge_graph.pkl"
        self._load()
    
    def _load(self):
        """Load existing graph."""
        if self.path.exists():
            with open(self.path, 'rb') as f:
                self.G = pickle.load(f)
            print(f"✓ Loaded graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        else:
            print("✓ Created new knowledge graph")
    
    def extract_entities_relations(self, text: str, chunk_id: str) -> Tuple[List[str], List[Tuple]]:
        """Extract entities and relations using regex (no spaCy)."""
        entities = []
        relations = []
        
        # Simple entity patterns
        patterns = {
            'ORG': r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*(?:\s+Inc\.?|\s+Corp\.?|\s+LLC|\s+Ltd\.?)\b',
            'PERSON': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            'MONEY': r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|M|B))?',
            'DATE': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            'PERCENT': r'\d+(?:\.\d+)?\s*%'
        }
        
        found_entities = {}
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                entity_text = match.group().strip()
                if len(entity_text) > 2:
                    entities.append((entity_text, entity_type, match.start(), match.end()))
                    self.G.add_node(entity_text, type=entity_type, chunks=[chunk_id])
                    found_entities[match.start()] = entity_text
        
        # Simple relation: proximity-based
        sorted_positions = sorted(found_entities.keys())
        for i in range(len(sorted_positions) - 1):
            pos1, pos2 = sorted_positions[i], sorted_positions[i + 1]
            if pos2 - pos1 < 100:  # Within 100 chars
                entity1 = found_entities[pos1]
                entity2 = found_entities[pos2]
                
                # Infer relation from text between
                between_text = text[pos1:pos2].lower()
                if any(w in between_text for w in ['acquired', 'bought', 'purchased']):
                    rel = 'ACQUIRED'
                elif any(w in between_text for w in ['founded', 'created', 'started']):
                    rel = 'FOUNDED_BY'
                elif any(w in between_text for w in ['reported', 'announced', 'declared']):
                    rel = 'REPORTED'
                elif any(w in between_text for w in ['increased', 'grew', 'rose']):
                    rel = 'INCREASED_TO'
                elif any(w in between_text for w in ['decreased', 'fell', 'dropped']):
                    rel = 'DECREASED_TO'
                else:
                    rel = 'RELATED_TO'
                
                self.G.add_edge(entity1, entity2, relation=rel, chunk=chunk_id)
                relations.append((entity1, rel, entity2))
        
        return entities, relations
    
    def find_paths(self, source: str, target: str, max_hops: int = 3) -> List[Dict]:
        """Find all paths between entities."""
        if source not in self.G or target not in self.G:
            return []
        
        paths = []
        try:
            for path in nx.all_simple_paths(self.G, source, target, cutoff=max_hops):
                path_data = {
                    'entities': path,
                    'relations': [],
                    'chunks': set()
                }
                for i in range(len(path) - 1):
                    edge_data = self.G[path[i]][path[i+1]]
                    path_data['relations'].append(edge_data.get('relation', 'RELATED_TO'))
                    path_data['chunks'].add(edge_data.get('chunk'))
                paths.append(path_data)
        except nx.NetworkXNoPath:
            pass
        
        return paths
    
    def get_neighbors(self, entity: str, depth: int = 2) -> Dict:
        """Get subgraph around entity."""
        if entity not in self.G:
            return {'nodes': [], 'edges': []}
        
        nodes = {entity}
        edges = []
        
        current = {entity}
        for _ in range(depth):
            next_level = set()
            for node in current:
                for neighbor in self.G.neighbors(node):
                    if neighbor not in nodes:
                        next_level.add(neighbor)
                        edges.append({
                            'source': node,
                            'target': neighbor,
                            'relation': self.G[node][neighbor].get('relation', 'RELATED_TO')
                        })
            nodes.update(next_level)
            current = next_level
        
        return {
            'nodes': [{'id': n, 'type': self.G.nodes[n].get('type', 'UNKNOWN')} for n in nodes],
            'edges': edges
        }
    
    def search_entities(self, query: str) -> List[str]:
        """Find entities matching query."""
        matches = []
        q_lower = query.lower()
        for node in self.G.nodes():
            if q_lower in node.lower():
                matches.append(node)
        return matches[:10]
    
    def save(self):
        """Persist graph."""
        with open(self.path, 'wb') as f:
            pickle.dump(self.G, f)
        print(f"✓ Saved graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
    
    def get_stats(self) -> Dict:
        """Get graph statistics."""
        return {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'connected_components': nx.number_weakly_connected_components(self.G)
        }