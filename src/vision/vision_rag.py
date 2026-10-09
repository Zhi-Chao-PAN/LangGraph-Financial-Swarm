"""
End-to-End Vision-RAG Pipeline.
Combines ColPali Multi-vector encoding, A800-optimized Tensor Caching, and MaxSim retrieval.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL.Image import Image
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

from src.vision.colpali_encoder import ColPaliEncoder
from src.vision.maxsim_retriever import MaxSimRetriever
from src.vision.tensor_cache import InMemoryTensorCache

logger = logging.getLogger(__name__)


class VisionRAGPipeline:
    """
    High-level orchestrator for Document-as-Image Late Interaction RAG.
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        micro_batch_size: int = 2048,  # A800 scale
    ):
        """
        Initializes the entire Vision-RAG hardware-accelerated pipeline.
        
        Args:
            model_name: The VLM ID for the vision encoder.
            device: Compute device ('cuda' recommended).
            dtype: Tensor precision (bfloat16 recommended).
            micro_batch_size: Parallel evaluation size for MaxSim.
        """
        self.device = torch.device(device)
        self.dtype = dtype

        logger.info(f"Initializing VisionRAGPipeline using {model_name} on {self.device}")

        # 1. Boot up the Encoder
        self.encoder = ColPaliEncoder(
            model_name=model_name, device=device, dtype=self.dtype
        )

        # 2. Boot up the fast memory cache
        # We assume patches are stored with whatever dim the encoder gives. Assume 128 (ColPali).
        self.embedding_dim = self.encoder._mock_dim if getattr(self.encoder, 'model', None) is None else 128 
        self.tensor_cache = InMemoryTensorCache(
            embedding_dim=self.embedding_dim, device=device, dtype=self.dtype
        )

        # 3. Boot up the MaxSim Late-Interaction operator
        self.retriever = MaxSimRetriever(
            tensor_cache=self.tensor_cache, 
            micro_batch_size=micro_batch_size, 
            device=device
        )
        
        # Keep references to original document images in RAM (so we can pass them to LLM)
        # Format: {page_id -> PIL Image}
        self.page_images: Dict[str, Image] = {}

    def index_pdf(self, pdf_path: Union[str, Path], doc_id: str):
        """
        Rasterize a PDF, extract page patches, and cache them into VRAM.
        
        Args:
            pdf_path: Path to the financial PDF.
            doc_id: Unique string to namespace the pages.
        """
        if convert_from_path is None:
            raise ImportError(
                "pdf2image is required to rasterize PDFs. "
                "Install it via `pip install pdf2image` and ensure `poppler` is on your PATH."
            )

        logger.info(f"Index Starting: Rasterizing PDF {pdf_path}")
        
        # 1. Rasterize
        # ColPali works best at high resolutions mimicking standard reading (e.g., 300-400 DPI)
        pages: List[Image] = convert_from_path(str(pdf_path), dpi=300)
        logger.info(f"Generated {len(pages)} high-resolution page images.")

        # 2. Extract Multi-vector Patch Embeddings
        logger.info(f"Encoding {len(pages)} pages through ColPali Vision Transformer...")
        
        # (Optional) We could batch process the pages if len(pages) is massive, 
        # but typically documents are < 500 pages and fit in A800 for forward passes easily.
        patch_embeddings = self.encoder.encode_pages(pages)

        # 3. Cache into High-Speed Tensor Pool
        for i, (img, emb) in enumerate(zip(pages, patch_embeddings)):
            page_id = f"{doc_id}_page_{i+1}"
            self.tensor_cache.add_page(page_id, emb)
            self.page_images[page_id] = img
            
        logger.info(
            f"Index Complete: Stored {len(pages)} pages into TensorCache. "
            f"Cache size is now {self.tensor_cache.total_bytes() / (1024**2):.2f} MB"
        )

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[Image], List[float]]:
        """
        Perform zero-shot late interaction retrieval against all indexed pages.
        
        Args:
            query: The user's natural language question.
            top_k: Number of optimal pages to return.
            
        Returns:
            Tuple of (Page IDs, PIL Images, MaxSim Scores)
        """
        logger.info(f"Retrieving top-{top_k} pages for query: '{query}'")
        
        # 1. Encode textual query into ColPali latent space
        query_embs = self.encoder.encode_queries([query])
        query_tensor = query_embs[0] # Take the first (and only) query

        # 2. Gather all candidate page IDs from the cache
        candidate_ids = list(self.tensor_cache._cache.keys())
        if not candidate_ids:
            logger.warning("Retrieval attempted on empty TensorCache.")
            return [], [], []

        # 3. Hardware-Aware MaxSim Computation
        top_ids, top_scores = self.retriever.get_top_k(
            query_embeddings=query_tensor, 
            candidate_page_ids=candidate_ids, 
            k=top_k
        )
        
        # 4. Gather the actual PIL Images for the Vision Language Synthesizer
        top_images = [self.page_images[pid] for pid in top_ids]
        scores_list = top_scores.cpu().tolist()
        
        logger.info(f"Retrieved {len(top_ids)} pages. Top score: {scores_list[0] if scores_list else 0:.4f}")
        
        return top_ids, top_images, scores_list

    def save_index(self, path: Union[str, Path]):
        """Persist the indexed tensors bounds to disk."""
        self.tensor_cache.flush_to_disk(path)

    def load_index(self, path: Union[str, Path]):
        """Load an indexed tensor store back into VRAM/RAM."""
        self.tensor_cache.load_from_disk(path)
