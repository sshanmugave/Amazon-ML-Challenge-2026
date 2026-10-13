"""
Sparse TF-IDF Candidate Retrieval Engine
Amazon ML Challenge 2026 - Business Entity Resolution

Implements high-throughput, memory-efficient sparse character n-gram cosine retrieval:
- Operates strictly per country partition (US, India, France)
- Sublinear TF scaling (log(1 + tf)) to dampen repetitive noise
- Batched sparse matrix multiplication (query_matrix . target_matrix.T)
- Fast top-K extraction with argpartition
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


class SparseTfidfRetriever:
    """
    Manages vectorizer fitting and batched top-K cosine retrieval per country.
    """
    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 5),
        analyzer: str = "char_wb",
        min_df: int = 3,
        sublinear_tf: bool = True,
        max_features: Optional[int] = None,
    ):
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            dtype=np.float32,
        )
        self.target_matrix: Optional[csr_matrix] = None
        self.target_ids: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit_targets(self, target_texts: List[str], target_ids: List[str]) -> None:
        """Fits TF-IDF vocabulary on target records and computes normalized target matrix."""
        assert len(target_texts) == len(target_ids), "Length mismatch between texts and IDs"
        self.target_matrix = self.vectorizer.fit_transform(target_texts)
        self.target_ids = np.array(target_ids)
        self.is_fitted = True

    def query_top_k(
        self,
        query_texts: List[str],
        query_ids: List[str],
        top_k: int = 50,
        batch_size: int = 5000,
        min_similarity: float = 0.15,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieves top-K candidate target IDs and cosine similarities for query records.
        Processes in batches to maintain constant memory footprint.
        """
        assert self.is_fitted, "Retriever must be fit on targets before querying"
        assert self.target_matrix is not None and self.target_ids is not None
        
        n_queries = len(query_texts)
        results: Dict[str, List[Tuple[str, float]]] = {}
        
        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            b_texts = query_texts[start_idx:end_idx]
            b_ids = query_ids[start_idx:end_idx]
            
            # Transform query batch
            b_query_matrix = self.vectorizer.transform(b_texts)
            
            # Sparse dot product: (batch_size, n_targets)
            sim_matrix = b_query_matrix.dot(self.target_matrix.T)
            
            # Extract top-K per row
            for row_idx in range(sim_matrix.shape[0]):
                qid = b_ids[row_idx]
                row = sim_matrix.getrow(row_idx)
                
                if row.nnz == 0:
                    results[qid] = []
                    continue
                    
                data = row.data
                indices = row.indices
                
                # Filter by min_similarity
                mask = data >= min_similarity
                if not np.any(mask):
                    results[qid] = []
                    continue
                    
                filtered_data = data[mask]
                filtered_indices = indices[mask]
                
                if len(filtered_data) <= top_k:
                    # Sort descending
                    sort_order = np.argsort(-filtered_data)
                    cands = [
                        (str(self.target_ids[filtered_indices[i]]), float(filtered_data[i]))
                        for i in sort_order
                    ]
                else:
                    # Fast top-K via argpartition
                    top_part = np.argpartition(-filtered_data, top_k)[:top_k]
                    sort_order = top_part[np.argsort(-filtered_data[top_part])]
                    cands = [
                        (str(self.target_ids[filtered_indices[i]]), float(filtered_data[i]))
                        for i in sort_order
                    ]
                    
                results[qid] = cands
                
        return results
