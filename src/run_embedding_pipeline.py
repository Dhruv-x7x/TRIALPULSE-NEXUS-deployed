# src/run_embedding_pipeline.py
"""
TRIALPULSE NEXUS 10X - Embedding Pipeline Runner
Phase 4.1: Generate embeddings for all knowledge artifacts
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from knowledge.embedding_pipeline import EmbeddingPipeline, main

if __name__ == "__main__":
    main()