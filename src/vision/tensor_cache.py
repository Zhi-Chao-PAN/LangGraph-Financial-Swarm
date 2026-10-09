"""
High-performance in-memory tensor caching for Vision-RAG.
Designed to maximize A800 80GB VRAM / High RAM bandwidth.
Replaces slow memory-mapped disk I/O with bfloat16 RAM/VRAM pools.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Dict, List, Optional, Union

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


class InMemoryTensorCache:
    """
    RAM/VRAM-backed tensor storage for multi-vector patch embeddings.
    Avoids PCIe bottlenecks by pre-staging data or keeping it purely in memory.
    Optimized for bfloat16.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            embedding_dim: Dimension of each patch embedding.
            device: 'cpu' (for massive pools relying on fast RAM->VRAM transfer)
                   or 'cuda' (if everything fits directly in 80GB VRAM).
            dtype: Tensor precision (bfloat16 recommended for A800).
        """
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        self.dtype = dtype

        # Pre-allocated dynamic storage
        # Mapping from page_id (str or int) to tensor of shape (num_patches, dim)
        self._cache: Dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self._cache)

    def total_bytes(self) -> int:
        """Returns the total number of bytes currently stored in the cache."""
        total = 0
        for t in self._cache.values():
            total += t.numel() * t.element_size()
        return total

    def add_page(self, page_id: Union[str, int], embeddings: torch.Tensor):
        """
        Store a page's patch embeddings.
        Args:
            page_id: Unique identifier for the page.
            embeddings: Tensor of shape (num_patches, embedding_dim).
        """
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, got {embeddings.shape[-1]}"
            )

        # Force correct dtype and store on designated device
        storable_tensor = embeddings.to(dtype=self.dtype, device=self.device)
        
        # In multi-vector representation, normally num_patches varies by page.
        # We store them explicitly rather than zero-padding immediately, saving memory.
        self._cache[str(page_id)] = storable_tensor

    def get_batch(
        self, page_ids: List[Union[str, int]], target_device: str = "cuda"
    ) -> torch.Tensor:
        """
        Retrieve a batch of page embeddings, zero-padded to the max sequence length of the batch.
        This forms the (batch_size, max_patches, dim) tensor needed for MaxSim.
        
        Args:
            page_ids: List of requested page identifiers.
            target_device: The compute device representing the Tensor Cores (usually 'cuda').
            
        Returns:
            A zero-padded dense tensor representing the document embeddings.
        """
        tensors = []
        for pid in page_ids:
            pid_str = str(pid)
            if pid_str not in self._cache:
                raise KeyError(f"Page ID {pid_str} not found in TensorCache.")
            tensors.append(self._cache[pid_str])

        if not tensors:
            return torch.empty((0, 0, self.embedding_dim), dtype=self.dtype, device=target_device)

        max_len = max(t.size(0) for t in tensors)
        batch_size = len(tensors)

        # Pre-allocate padded tensor directly on the target device
        # Zero padding is mathematically safe for Cosine Similarity / Dot Product 
        # (provided query tokens don't falsely attend to zero; MaxSim handles this naturally if zeros score lower)
        padded_batch = torch.zeros(
            (batch_size, max_len, self.embedding_dim),
            dtype=self.dtype,
            device=target_device,
        )

        # Fast memory copy
        for i, t in enumerate(tensors):
            # t may be on CPU, padded_batch is on CUDA. 
            # PyTorch handles the non-blocking transfer seamlessly here.
            length = t.size(0)
            padded_batch[i, :length, :] = t.to(target_device, non_blocking=True)

        return padded_batch

    def flush_to_disk(self, filepath: Union[str, Path]):
        """Persist cache to disk using lightning-fast SafeTensors."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(self._cache, str(path))
        logger.info(f"Flushed {len(self._cache)} tensors to {path}")

    def load_from_disk(self, filepath: Union[str, Path]):
        """Load cache from SafeTensors directly into memory."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Tensor file not found: {path}")
        
        logger.info(f"Loading tensors from {path}")
        loaded = load_file(str(path), device=str(self.device))
        
        for k, v in loaded.items():
            if v.dtype != self.dtype:
                v = v.to(dtype=self.dtype)
            self._cache[k] = v
        
        logger.info(f"Loaded {len(loaded)} tensors into cache.")
