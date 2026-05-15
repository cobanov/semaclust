"""Text encoders for semaclust.

Provides the :class:`Encoder` protocol and a default
:class:`SentenceTransformerEncoder` implementation backed by
``sentence-transformers``. Users can plug in any object that implements
``encode(texts) -> np.ndarray``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@runtime_checkable
class Encoder(Protocol):
    """Anything that can turn a list of strings into a 2D float array."""

    def encode(self, texts: list[str]) -> NDArray[np.floating]: ...


class SentenceTransformerEncoder:
    """Default encoder using sentence-transformers.

    The underlying model is lazily loaded on first :meth:`encode` call so
    constructing the encoder is cheap and import-time stays side-effect free.

    Parameters
    ----------
    model_name:
        Hugging Face model name passed to ``SentenceTransformer``.
    batch_size:
        Batch size used when encoding.
    device:
        Optional torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        ``None`` lets sentence-transformers auto-select.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.debug("Loading SentenceTransformer model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> NDArray[np.floating]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def __repr__(self) -> str:
        state = "loaded" if self._model is not None else "lazy"
        return (
            f"SentenceTransformerEncoder(model_name={self.model_name!r}, "
            f"batch_size={self.batch_size}, device={self.device!r}, "
            f"state={state!r})"
        )
