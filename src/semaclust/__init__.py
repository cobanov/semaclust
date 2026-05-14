"""semaclust: semantic text clustering with sentence embeddings."""

from __future__ import annotations

import logging

from ._version import __version__
from .clusterer import NotFittedError, TextClusterer
from .encoders import Encoder, SentenceTransformerEncoder
from .result import ClusterResult

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "ClusterResult",
    "Encoder",
    "NotFittedError",
    "SentenceTransformerEncoder",
    "TextClusterer",
    "__version__",
]
