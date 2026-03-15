"""Document processing pipeline - chunking and preparation."""
from typing import List, Dict
import re
from utils.pdf_parser import chunk_text

class DocumentProcessor:
    """Process documents for indexing."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_pages(self, pages: List[Dict]) -> List[Dict]:
        """Process extracted pages into chunks."""
        all_chunks = []
        
        for page in pages:
            text = page.get('text', '')
            if not text or len(text.strip()) < 50:
                continue
            
            # Clean text
            text = self._clean_text(text)
            
            # Chunk
            chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'source': page.get('source', 'Unknown'),
                    'page': page.get('page', 0),
                    'chunk_index': i,
                    'total_pages': page.get('total_pages', 0),
                    'char_count': len(chunk),
                    'word_count': len(chunk.split())
                })
        
        return all_chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (standalone digits)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove header/footer patterns
        text = re.sub(r'^\s*(?:Page|pg\.?)\s*\d+\s*(?:of|\/)\s*\d+\s*$', '', 
                     text, flags=re.MULTILINE | re.IGNORECASE)
        
        return text.strip()
    
    def extract_metadata(self, chunks: List[Dict]) -> Dict:
        """Aggregate metadata from chunks."""
        sources = set(c.get('source') for c in chunks)
        
        return {
            'total_chunks': len(chunks),
            'unique_sources': len(sources),
            'sources': list(sources),
            'avg_chunk_size': sum(c.get('word_count', 0) for c in chunks) / len(chunks) if chunks else 0
        }