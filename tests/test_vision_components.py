"""
Unit tests for the new A800-optimized Vision-RAG components.
Verifies bfloat16 propagation, tensor shapes, Einsum arithmetic,
and caching logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.vision.colpali_encoder import ColPaliEncoder
from src.vision.maxsim_retriever import MaxSimRetriever
from src.vision.tensor_cache import InMemoryTensorCache


class TestInMemoryTensorCache:
    def test_cache_initialization(self):
        cache = InMemoryTensorCache(embedding_dim=128, device="cpu", dtype=torch.bfloat16)
        assert cache.embedding_dim == 128
        assert cache.device.type == "cpu"
        assert cache.dtype == torch.bfloat16
        assert len(cache) == 0

    def test_add_page(self):
        cache = InMemoryTensorCache(embedding_dim=128, device="cpu", dtype=torch.bfloat16)
        # Mock 10 patches for page 1
        page_emb = torch.randn(10, 128)
        cache.add_page("doc1_p1", page_emb)

        assert len(cache) == 1
        assert "doc1_p1" in cache._cache
        # Verify it cast down to bfloat16
        assert cache._cache["doc1_p1"].dtype == torch.bfloat16

    def test_get_batch_padding(self):
        cache = InMemoryTensorCache(embedding_dim=128, device="cpu", dtype=torch.bfloat16)
        
        # doc1_p1 has 10 patches
        cache.add_page("doc1_p1", torch.randn(10, 128))
        # doc1_p2 has 5 patches
        cache.add_page("doc1_p2", torch.randn(5, 128))
        # doc1_p3 has 15 patches
        cache.add_page("doc1_p3", torch.randn(15, 128))

        batch = cache.get_batch(["doc1_p1", "doc1_p2", "doc1_p3"], target_device="cpu")
        
        # Batch size should be 3, Max patches 15, dim 128
        assert batch.shape == (3, 15, 128)
        assert batch.dtype == torch.bfloat16

        # Check zero-padding on p2 (which only had 5 patches)
        # Elements from index 5 to 14 should all be exactly 0
        assert torch.all(batch[1, 5:, :] == 0.0)
        # Elements 0-4 should be non-zero (highly likely)
        assert not torch.all(batch[1, :5, :] == 0.0)


class TestMaxSimRetriever:
    def test_arithmetic_correctness(self):
        # We'll use specific float32 values so we can hand-calculate the exact maxsim
        # To avoid bfloat16 rounding issues in tests, run this logic inside float32 temporarily
        cache = InMemoryTensorCache(embedding_dim=4, device="cpu", dtype=torch.float32)
        
        # Doc 1: 2 patches representing vectors [1, 0, 0, 0] and [0, 1, 0, 0]
        d1 = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=torch.float32)
        
        # Doc 2: 2 patches representing vectors [0.5, 0.5, 0, 0] and [0, 0, 1, 0]
        d2 = torch.tensor([
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=torch.float32)
        
        cache.add_page("d1", d1)
        cache.add_page("d2", d2)

        # Query: 2 tokens representing [1, 0, 0, 0] and [0, 0, 1, 0]
        q = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=torch.float32)
        # Add batch dim for Retriever (1, L_q, dim)
        q = q.unsqueeze(0)

        retriever = MaxSimRetriever(tensor_cache=cache, micro_batch_size=2, device="cpu")
        
        # Manual Calculation:
        # q_token1 [1,0,0,0] vs d1 -> max_sim is 1.0 (with patch 1)
        # q_token2 [0,0,1,0] vs d1 -> max_sim is 0.0 (both patches are 0)
        # Total d1 score: 1.0 + 0.0 = 1.0
        
        # q_token1 [1,0,0,0] vs d2 -> max_sim is 0.5 (with patch 1)
        # q_token2 [0,0,1,0] vs d2 -> max_sim is 1.0 (with patch 2)
        # Total d2 score: 0.5 + 1.0 = 1.5
        
        scores = retriever.compute_maxsim(q, ["d1", "d2"])
        
        assert scores.shape == (2,)
        assert abs(scores[0].item() - 1.0) < 1e-4  # d1
        assert abs(scores[1].item() - 1.5) < 1e-4  # d2
        
        # Test get_top_k
        top_ids, top_scores = retriever.get_top_k(q, ["d1", "d2"], k=1)
        assert top_ids == ["d2"]
        assert abs(top_scores[0].item() - 1.5) < 1e-4

    def test_bfloat16_execution(self):
        # Even if data is bfloat16, the sum must return a float32 representation for stability
        cache = InMemoryTensorCache(embedding_dim=64, device="cpu", dtype=torch.bfloat16)
        cache.add_page("page_a", torch.randn((50, 64), dtype=torch.bfloat16))
        
        retriever = MaxSimRetriever(tensor_cache=cache, micro_batch_size=10, device="cpu")
        
        q = torch.randn((1, 10, 64), dtype=torch.bfloat16)
        scores = retriever.compute_maxsim(q, ["page_a"])
        
        assert scores.dtype == torch.float32
        assert scores.shape == (1,)


class TestColPaliEncoder:
    def test_fallback_mock(self):
        """Tests that if ColPali engine is removed, the architecture still streams dummy data."""
        encoder = ColPaliEncoder(device="cpu", dtype=torch.bfloat16)
        # Force mock mode even if colpali_engine is installed
        encoder.model = None
        encoder._mock_dim = 128
        
        # Now pass None images to the mock path
        embs = encoder.encode_pages([None, None])
        
        assert len(embs) == 2
        
        # Output should be normalized bfloat16
        assert embs[0].dtype == torch.bfloat16
        assert embs[0].dim() == 2
        assert embs[0].shape[-1] == 128  # Encoder's fallback dim

        q_embs = encoder.encode_queries(["Test query"])
        assert len(q_embs) == 1
        assert q_embs[0].dtype == torch.bfloat16
        assert q_embs[0].shape[-1] == 128
