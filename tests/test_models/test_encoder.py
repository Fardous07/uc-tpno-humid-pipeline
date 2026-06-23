"""
Tests for src/models/encoder/.

Covers: build_encoder, list_encoders, EncoderAdapter.
"""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class TestListEncoders:
    def test_returns_list(self):
        from src.models.encoder.adapter import list_encoders
        enc = list_encoders()
        assert isinstance(enc, list)
        assert len(enc) >= 4

    def test_known_backends(self):
        from src.models.encoder.adapter import list_encoders
        for name in ["nequip", "equiformer", "gemnet", "se3_transformer"]:
            assert name in list_encoders()


class TestBuildEncoder:
    def test_build_nequip(self):
        from src.models.encoder.adapter import build_encoder
        enc = build_encoder("nequip", emb_dim=32, n_layers=2)
        assert enc is not None
        assert sum(p.numel() for p in enc.parameters()) > 0

    def test_build_all(self):
        from src.models.encoder.adapter import build_encoder, list_encoders
        for name in list_encoders():
            enc = build_encoder(name, emb_dim=32, n_layers=2)
            assert enc is not None


class TestEncoderAdapter:
    def test_wraps_encoder(self):
        from src.models.encoder.adapter import EncoderAdapter
        inner = nn.Linear(10, 64)
        inner.emb_dim = 64
        adapter = EncoderAdapter(inner, target_dim=64)
        assert adapter.emb_dim == 64

    def test_projection_added(self):
        from src.models.encoder.adapter import EncoderAdapter
        inner = nn.Linear(10, 32)
        inner.emb_dim = 32
        adapter = EncoderAdapter(inner, target_dim=64)
        assert adapter.emb_dim == 64
        # proj should be Linear, not Identity
        assert not isinstance(adapter.proj, nn.Identity)

    def test_no_projection_same_dim(self):
        from src.models.encoder.adapter import EncoderAdapter
        inner = nn.Linear(10, 64)
        inner.emb_dim = 64
        adapter = EncoderAdapter(inner, target_dim=64)
        assert isinstance(adapter.proj, nn.Identity)

    def test_freeze_unfreeze(self):
        from src.models.encoder.adapter import EncoderAdapter
        inner = nn.Linear(10, 64)
        inner.emb_dim = 64
        adapter = EncoderAdapter(inner, target_dim=64)
        adapter.freeze()
        for p in inner.parameters():
            assert not p.requires_grad
        adapter.unfreeze()
        for p in inner.parameters():
            assert p.requires_grad