"""
Vision encoding module using ColPali / PaliGemma architecture.
Leverages bfloat16 and Flash Attention 2 on A800.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import List, Union

import torch
from PIL.Image import Image
from transformers import AutoProcessor
from transformers.utils import is_flash_attn_2_available

# Attempt to load specialized ColPali model if available, otherwise fallback structural interface
try:
    from colpali_engine.models import ColPali
except ImportError:
    ColPali = None

logger = logging.getLogger(__name__)


class ColPaliEncoder:
    """
    SOTA Multi-vector visual encoder converting document pages into patch embeddings.
    Designed for bfloat16 A800 execution with high-res page images.
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the ColPali encoder.
        
        Args:
            model_name: HuggingFace model hub ID.
            device: Comput device ('cuda').
            dtype: Tensor precision.
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.model_name = model_name

        logger.info(f"Initializing ColPali Encoder [{model_name}] on {self.device} in {self.dtype}")

        if ColPali is None:
            logger.warning(
                "colpali_engine not installed. Encoding will use mock outputs for architectural testing. "
                "Install via: pip install colpali-engine"
            )
            self.model = None
            self.processor = None
            self._mock_dim = 128
        else:
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            # Use Flash Attention 2 only if on CUDA and package is installed
            attn_impl = "flash_attention_2" if (self.device.type == "cuda" and is_flash_attn_2_available()) else None
            if attn_impl:
                logger.info("Flash Attention 2 is enabled for ColPali inference.")
            
            self.model = ColPali.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map=self.device,
                attn_implementation=attn_impl, 
            ).eval()

    @torch.no_grad()
    def encode_pages(self, images: List[Image]) -> List[torch.Tensor]:
        """
        Encode a list of page images into Multi-vector patch embeddings.
        
        Args:
            images: List of PIL images representing document pages.
            
        Returns:
            List of 2D tensors, where each tensor is shapes (num_patches, embedding_dim).
        """
        if self.model is None:
            # Fallback mock encoding for testing the pipeline if colpali is absent
            logger.debug(f"Mock encoding {len(images)} pages...")
            mock_embeddings = []
            for img in images:
                # Approximate typical ColPali output: ~1024 patches per high-res page
                num_patches = 1030 
                emb = torch.randn(
                    (num_patches, self._mock_dim), dtype=self.dtype, device=self.device
                )
                # Normalize just like true ColPali (L2 norm along the dim axis)
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                mock_embeddings.append(emb)
            return mock_embeddings

        # Actual ColPali encoding
        # Prepare inputs
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self.dtype if v.is_floating_point() else v.dtype) 
                  for k, v in inputs.items()}

        # Forward pass
        # The model outputs pooled patch embeddings per page
        embeddings = self.model(**inputs)
        
        # ColPali outputs a list of 2D tensors (num_patches, dim), where num_patches varies
        result = []
        for emb in embeddings:
            result.append(emb.to(self.dtype))
            
        return result

    @torch.no_grad()
    def encode_queries(self, queries: List[str]) -> List[torch.Tensor]:
        """
        Encode natural language queries into ColPali latent space.
        
        Args:
            queries: List of textual questions.
            
        Returns:
            List of 2D tensors, where each tensor is shapes (num_tokens, embedding_dim).
        """
        if self.model is None:
            mock_queries = []
            for query in queries:
                num_tokens = max(5, len(query.split()) + 2)
                emb = torch.randn(
                    (num_tokens, self._mock_dim), dtype=self.dtype, device=self.device
                )
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                mock_queries.append(emb)
            return mock_queries

        # ColPali query processing requires specific prompt formatting
        # Typically the processor handles the specific prefixing (e.g., `<query>`)
        inputs = self.processor(text=queries, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        embeddings = self.model(**inputs)
        
        result = []
        for emb in embeddings:
            result.append(emb.to(self.dtype))
            
        return result
