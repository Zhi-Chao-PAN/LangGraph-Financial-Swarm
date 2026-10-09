"""
Hardware-aware Late Interaction (MaxSim) computation.
Optimized for NVIDIA A800 with 80GB VRAM using large micro-batch configurations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import List, Tuple

import torch

from src.vision.tensor_cache import InMemoryTensorCache

logger = logging.getLogger(__name__)


class MaxSimRetriever:
    """
    Computes ColPali Late Interaction scores between query tokens and document patches.
    Utilizes PyTorch batched Einsum to maximize Tensor Core usage.
    """

    def __init__(
        self,
        tensor_cache: InMemoryTensorCache,
        micro_batch_size: int = 2048,  # Huge batch size suitable for A800 80GB
        device: str = "cuda",
    ):
        """
        Args:
            tensor_cache: RAM/VRAM cache holding pre-computed document embeddings.
            micro_batch_size: Number of documents to score simultaneously.
                            A800 80GB can handle 2000-4000 depending on sequence length.
            device: Comput device ('cuda').
        """
        self.tensor_cache = tensor_cache
        self.micro_batch_size = micro_batch_size
        self.device = torch.device(device)
        
        logger.info(
            f"Initialized MaxSimRetriever on {self.device} "
            f"[Micro-batch: {self.micro_batch_size}]"
        )

    @torch.no_grad()
    def compute_maxsim(
        self, query_embeddings: torch.Tensor, document_page_ids: List[str]
    ) -> torch.Tensor:
        """
        Calculates MaxSim scores: Score = sum_{q}(max_{d}(Q_q @ D_d^T))
        
        Args:
            query_embeddings: Tensor of shape (1, num_query_tokens, dim).
            document_page_ids: List of page IDs to score against.
            
        Returns:
            1D Tensor containing the score for each document provided, ordered identically.
        """
        if query_embeddings.dim() == 2:
            # Add batch dimension: (1, L_q, dim)
            query_embeddings = query_embeddings.unsqueeze(0)
            
        num_docs = len(document_page_ids)
        if num_docs == 0:
            return torch.tensor([], dtype=torch.float32, device=self.device)

        # Ensure query is on compute device in bfloat16/float16
        query_embeddings = query_embeddings.to(device=self.device)
        
        # Accumulate scores here
        final_scores = torch.zeros(num_docs, dtype=torch.float32, device=self.device)
        
        logger.debug(f"Computing MaxSim for {num_docs} pages using micro-batch {self.micro_batch_size}")

        for i in range(0, num_docs, self.micro_batch_size):
            batch_ids = document_page_ids[i : i + self.micro_batch_size]
            current_batch_size = len(batch_ids)
            
            # 1. Fetch D matrix
            # doc_batch shape: (batch_size, num_patches, dim)
            # Padded with zeros; dtype matches cache (e.g. bfloat16).
            doc_batch = self.tensor_cache.get_batch(batch_ids, target_device=str(self.device))
            
            # 2. Similarity Matrix Computation
            # Q is (1, L_q, dim), implicitly broadcast along batch dimension.
            # D is (batch_size, L_d, dim).
            # Result: sim_matrix shape -> (batch_size, L_q, L_d)
            # Einstein summation convention:
            # i = 1 (query batch), q = query tokens, d = embedding dim
            # b = doc batch, p = doc patches
            sim_matrix = torch.einsum("iqd, bpd -> biqp", query_embeddings, doc_batch) 
            
            # Squeeze the unitary query batch dim `i`
            sim_matrix = sim_matrix.squeeze(1)  # -> (batch_size, L_q, L_d)
            
            # Cast up to float32 for stable reduction operations (max/sum)
            sim_matrix = sim_matrix.to(torch.float32)
            
            # 3. Max pooling over document patches
            # For each query token, find the maximum similarity to ANY document patch.
            # max_sim_per_token shape: (batch_size, L_q)
            max_sim_per_token, _ = sim_matrix.max(dim=-1)
            
            # 4. Sum over query tokens
            # Accumulate the maximum similarities to get the final page score.
            # batch_scores shape: (batch_size,)
            batch_scores = max_sim_per_token.sum(dim=-1)
            
            final_scores[i : i + current_batch_size] = batch_scores
            
            # Aggressive cleanup to prevent VRAM fragmentation
            del doc_batch, sim_matrix, max_sim_per_token, batch_scores
            
        return final_scores

    def get_top_k(
        self,
        query_embeddings: torch.Tensor,
        candidate_page_ids: List[str],
        k: int = 5,
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Rank pages using Late Interaction and return top-k choices.
        
        Args:
            query_embeddings: Encoded conversational query.
            candidate_page_ids: Page IDs holding multi-vector features in the cache.
            k: Number of pages to retrieve.
            
        Returns:
            Tuple of (List of Page IDs, Tensor of Scores).
        """
        # Ensure k doesn't exceed candidate pool
        actual_k = min(k, len(candidate_page_ids))
        if actual_k == 0:
            return [], torch.empty(0)

        # Compute full MaxSim vector
        scores = self.compute_maxsim(query_embeddings, candidate_page_ids)
        
        # Sort and select
        top_scores, top_indices = torch.topk(scores, k=actual_k)
        
        # Map indices back to original string IDs
        top_page_ids = [candidate_page_ids[idx.item()] for idx in top_indices]
        
        return top_page_ids, top_scores
