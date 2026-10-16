"""Training sub-package — embedding generation."""

from .embedder import load_model, generate_embeddings, build_and_save_embeddings, load_embeddings

__all__ = ["load_model", "generate_embeddings", "build_and_save_embeddings", "load_embeddings"]
