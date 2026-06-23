"""
Encoder adapter: registry, factory, and output normalisation.

This module provides a unified interface for all encoder backends
(NequIP, Equiformer, GemNet, SE(3)-Transformer) so that the TPNO
operator and downstream components can swap encoders without any
code changes.

Components
──────────
*   ``ENCODER_REGISTRY`` — string-name → encoder-class mapping.
*   ``build_encoder`` — factory function that constructs any
    registered encoder from a config dict.
*   ``EncoderAdapter`` — wrapping ``nn.Module`` that:
        1.  Delegates ``forward()`` to the inner encoder.
        2.  Optionally projects the encoder output to a target
            dimension via a learned linear layer.
        3.  Optionally applies LayerNorm for stable downstream
            conditioning.
        4.  Provides ``freeze()`` / ``unfreeze()`` for transfer-
            learning workflows.

Usage
─────
>>> adapter = EncoderAdapter.from_config({
...     "encoder": "nequip",
...     "n_species": 100,
...     "emb_dim": 128,
...     "n_layers": 4,
...     "lmax": 2,
...     "cutoff": 5.0,
... })
>>> h = adapter(graph_batch)  # [B, target_dim]

Registering a custom encoder::

    from src.models.encoder.adapter import ENCODER_REGISTRY
    ENCODER_REGISTRY["my_encoder"] = MyEncoderClass

Author  : Rayhan (University of Bergen)
Project : UC-TPNO — Uncertainty-Calibrated Thermodynamic Potential
          Neural Operator for Humid Flue-Gas CO₂ Capture in MOFs
License : MIT
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Type, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1.  ENCODER REGISTRY
# ═══════════════════════════════════════════════════════════════════════

# Maps lower-case name → (module_path, class_name)
# Lazy imports so we don't pull in heavy deps at import time.

_LAZY_REGISTRY: Dict[str, tuple] = {
    "nequip": ("src.models.encoder.nequip", "NequIPEncoder"),
    "equiformer": ("src.models.encoder.equiformer", "EquiformerEncoder"),
    "gemnet": ("src.models.encoder.gemnet", "GemNetEncoder"),
    "se3_transformer": ("src.models.encoder.se3_transformer", "SE3TransformerEncoder"),
    "se3": ("src.models.encoder.se3_transformer", "SE3TransformerEncoder"),
}

# Eagerly-registered custom classes (user-populated at runtime)
ENCODER_REGISTRY: Dict[str, Type[nn.Module]] = {}


def _resolve_encoder_class(name: str) -> Type[nn.Module]:
    """Look up an encoder class by name (lazy or eager)."""
    key = name.lower().replace("-", "_")

    # Check eager registry first
    if key in ENCODER_REGISTRY:
        return ENCODER_REGISTRY[key]

    # Lazy import
    if key in _LAZY_REGISTRY:
        mod_path, cls_name = _LAZY_REGISTRY[key]
        import importlib
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            ENCODER_REGISTRY[key] = cls
            return cls
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"Cannot import encoder '{name}' from {mod_path}.{cls_name}: {exc}"
            ) from exc

    available = sorted(set(list(ENCODER_REGISTRY) + list(_LAZY_REGISTRY)))
    raise KeyError(
        f"Unknown encoder '{name}'. Available: {available}"
    )


def list_encoders() -> list:
    """Return the names of all registered encoders."""
    return sorted(set(list(ENCODER_REGISTRY) + list(_LAZY_REGISTRY)))


# ═══════════════════════════════════════════════════════════════════════
# 2.  FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def build_encoder(
    name: str,
    **kwargs,
) -> nn.Module:
    """
    Instantiate an encoder by name.

    Parameters
    ----------
    name    : Encoder name (case-insensitive).  One of ``'nequip'``,
              ``'equiformer'``, ``'gemnet'``, ``'se3_transformer'``,
              or any key added to ``ENCODER_REGISTRY``.
    **kwargs: Forwarded to the encoder constructor.

    Returns
    -------
    An ``nn.Module`` whose ``forward(data)`` returns
    ``[B, emb_dim]`` MOF embeddings.
    """
    cls = _resolve_encoder_class(name)
    logger.info(f"Building encoder: {cls.__name__} with {kwargs}")
    return cls(**kwargs)


# ═══════════════════════════════════════════════════════════════════════
# 3.  ENCODER ADAPTER
# ═══════════════════════════════════════════════════════════════════════

class EncoderAdapter(nn.Module):
    """
    Thin wrapper that normalises any encoder's output for the TPNO
    operator.

    Features
    ────────
    * **Dimension projection** — if the encoder's ``emb_dim`` differs
      from the target dimension expected by the operator, a learned
      linear projection is inserted.
    * **LayerNorm** — optional post-encoder normalisation for training
      stability.
    * **Freeze / unfreeze** — ``freeze()`` sets ``requires_grad=False``
      on the encoder (useful for fine-tuning the operator while keeping
      the encoder frozen).

    Parameters
    ----------
    encoder    : The inner encoder module.
    target_dim : Desired output dimension.  If *None*, keep the
                 encoder's native ``emb_dim``.
    normalize  : Apply ``LayerNorm`` to the output.
    """

    def __init__(
        self,
        encoder: nn.Module,
        target_dim: Optional[int] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.encoder = encoder

        # Infer emb_dim from the encoder
        enc_dim = getattr(encoder, "emb_dim", None)
        if enc_dim is None:
            # Try to read from config or fallback
            enc_dim = getattr(encoder, "hidden_dim", 128)
            logger.warning(
                f"Could not detect encoder emb_dim; assuming {enc_dim}."
            )

        self._enc_dim = enc_dim
        self._target_dim = target_dim or enc_dim

        # Projection
        if self._target_dim != enc_dim:
            self.proj = nn.Linear(enc_dim, self._target_dim)
        else:
            self.proj = nn.Identity()

        # LayerNorm
        self.norm = nn.LayerNorm(self._target_dim) if normalize else nn.Identity()

    @property
    def emb_dim(self) -> int:
        """Output embedding dimension."""
        return self._target_dim

    def forward(self, data: Any) -> torch.Tensor:
        """
        Parameters
        ----------
        data : Whatever the inner encoder accepts (dict or PyG Data).

        Returns
        -------
        ``[B, target_dim]`` normalised MOF embeddings.
        """
        h = self.encoder(data)     # [B, enc_dim]
        h = self.proj(h)           # [B, target_dim]
        h = self.norm(h)           # [B, target_dim]
        return h

    # ── Transfer-learning helpers ────────────────────────────────

    def freeze(self) -> None:
        """Freeze all encoder parameters (projection stays trainable)."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        logger.info("Encoder frozen.")

    def unfreeze(self) -> None:
        """Unfreeze all encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        logger.info("Encoder unfrozen.")

    @property
    def is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.encoder.parameters())

    @property
    def num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    # ── Convenience constructor ──────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        target_dim: Optional[int] = None,
        normalize: bool = True,
    ) -> "EncoderAdapter":
        """
        Build an adapter from a flat config dict.

        The key ``"encoder"`` selects the backend; all other keys
        are forwarded to the encoder constructor.

        Example
        ───────
        >>> adapter = EncoderAdapter.from_config({
        ...     "encoder": "nequip",
        ...     "n_species": 100,
        ...     "emb_dim": 128,
        ...     "n_layers": 4,
        ...     "lmax": 2,
        ...     "cutoff": 5.0,
        ... })
        """
        config = dict(config)  # shallow copy
        name = config.pop("encoder", "nequip")
        encoder = build_encoder(name, **config)
        return cls(encoder, target_dim=target_dim, normalize=normalize)


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "ENCODER_REGISTRY",
    "list_encoders",
    "build_encoder",
    "EncoderAdapter",
]