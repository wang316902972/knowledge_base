"""
BM25 Keyword Search Strategy
Provides keyword-based search fallback for retrieval enhancement
"""
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None

logger = logging.getLogger(__name__)


@dataclass
class BM25SearchResult:
    """BM25 search result with metadata"""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with other search results"""
        return {
            "id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "relevance_score": self.score,  # Alias for compatibility
            "similarity_score": self.score,  # Alias for compatibility
            "search_method": "bm25_keyword",
            "match_type": "keyword_match",
            "metadata": self.metadata
        }


class BM25SearchStrategy:
    """
    BM25 keyword-based search strategy for fallback retrieval

    Uses BM25 ranking algorithm to provide keyword-based search
    when vector search fails to retrieve relevant results.
    """

    def __init__(self, metadata_file: str, businesstype: str = "default"):
        """
        Initialize BM25 search strategy

        Args:
            metadata_file: Path to metadata JSON file
            businesstype: Business type for this search instance
        """
        self.businesstype = businesstype
        self.metadata_file = Path(metadata_file)
        self.bm25_index: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.chunk_ids: List[str] = None
        self.id_to_chunk: Dict[str, str] = {}
        self._indexed = False

        if not BM25_AVAILABLE:
            logger.warning(f"[{self.businesstype}] rank-bm25 not available, BM25 search disabled")
            return

        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata and build BM25 index"""
        if not self.metadata_file.exists():
            logger.warning(f"[{self.businesstype}] Metadata file not found: {self.metadata_file}")
            return

        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Extract documents and chunk IDs
            self.id_to_chunk = metadata.get('id_to_chunk', {})

            if not self.id_to_chunk:
                logger.warning(f"[{self.businesstype}] No documents found in metadata")
                return

            # Prepare documents for BM25 indexing
            # Each document needs to be tokenized
            self.chunk_ids = list(self.id_to_chunk.keys())
            self.documents = [self.id_to_chunk[chunk_id] for chunk_id in self.chunk_ids]

            # Tokenize documents for BM25
            tokenized_docs = [self._tokenize(doc) for doc in self.documents]

            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            self._indexed = True

            logger.info(f"[{self.businesstype}] BM25 index built with {len(self.documents)} documents")

        except Exception as e:
            logger.error(f"[{self.businesstype}] Failed to load metadata for BM25: {e}")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing

        Supports Chinese and English text with simple word boundaries
        """
        # Simple tokenization: split on non-word characters
        # For Chinese: keep characters as individual tokens
        # For English: split on whitespace and punctuation
        tokens = []

        # Match Chinese characters or English words
        pattern = re.compile(r'[\u4e00-\u9fff]|[\w]+')
        matches = pattern.findall(text.lower())

        # Filter out very short tokens and numbers
        tokens = [m for m in matches if len(m) > 1 or (m and '\u4e00' <= m <= '\u9fff')]

        return tokens

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.1
    ) -> List[BM25SearchResult]:
        """
        Search using BM25 keyword matching

        Args:
            query: Search query text
            top_k: Maximum number of results to return
            min_score: Minimum BM25 score threshold

        Returns:
            List of BM25SearchResult objects
        """
        if not BM25_AVAILABLE:
            logger.warning(f"[{self.businesstype}] BM25 not available, returning empty results")
            return []

        if not self._indexed or self.bm25_index is None:
            logger.warning(f"[{self.businesstype}] BM25 index not built")
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            logger.warning(f"[{self.businesstype}] Query tokenization failed: {query}")
            return []

        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # Create result list
        results = []
        for idx, score in enumerate(scores):
            if score >= min_score:
                chunk_id = self.chunk_ids[idx]
                text = self.documents[idx]

                result = BM25SearchResult(
                    chunk_id=chunk_id,
                    text=text,
                    score=float(score),
                    metadata={
                        "rank": idx,
                        "total_documents": len(self.documents)
                    }
                )
                results.append(result)

        # Sort by score descending and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def is_available(self) -> bool:
        """Check if BM25 search is available"""
        return BM25_AVAILABLE and self._indexed and self.bm25_index is not None

    def get_stats(self) -> Dict[str, Any]:
        """Get BM25 index statistics"""
        return {
            "indexed": self._indexed,
            "available": self.is_available(),
            "document_count": len(self.documents),
            "businesstype": self.businesstype,
            "metadata_file": str(self.metadata_file)
        }


def create_bm25_strategy(
    metadata_file: str,
    businesstype: str = "default"
) -> Optional[BM25SearchStrategy]:
    """
    Factory function to create BM25 search strategy

    Args:
        metadata_file: Path to metadata JSON file
        businesstype: Business type for this search instance

    Returns:
        BM25SearchStrategy instance or None if not available
    """
    strategy = BM25SearchStrategy(metadata_file, businesstype)

    if not strategy.is_available():
        logger.warning(f"[{businesstype}] BM25 strategy not available for use")
        return None

    return strategy
