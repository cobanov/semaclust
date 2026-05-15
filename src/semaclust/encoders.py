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


def _pick_device() -> str | None:
    """Pick the best available torch device.

    Order: CUDA > MPS (Apple Silicon) > None (let sentence-transformers fall
    back to CPU). Import is local so semaclust import stays cheap.
    """
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return None


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
        Torch device. ``"auto"`` (default) picks CUDA if available, then MPS
        on Apple Silicon, and otherwise lets sentence-transformers fall back
        to CPU. Pass an explicit string (``"cuda"``, ``"mps"``, ``"cpu"``) to
        override, or ``None`` to delegate the choice entirely to
        sentence-transformers.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        batch_size: int = 32,
        device: str | None = "auto",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._resolved_device: str | None = None
        self._model: SentenceTransformer | None = None

    @property
    def effective_device(self) -> str | None:
        """The device string that will actually be passed to the model.

        Resolves ``"auto"`` lazily on first access; for explicit values returns
        what the user passed.
        """
        if self.device == "auto":
            if self._resolved_device is None:
                self._resolved_device = _pick_device()
            return self._resolved_device
        return self.device

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = self.effective_device
            logger.debug(
                "Loading SentenceTransformer model %s on device %r",
                self.model_name,
                device,
            )
            self._model = SentenceTransformer(self.model_name, device=device)
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
        if self.device == "auto" and self._resolved_device is not None:
            device_repr = f"'auto' (-> {self._resolved_device!r})"
        else:
            device_repr = repr(self.device)
        return (
            f"SentenceTransformerEncoder(model_name={self.model_name!r}, "
            f"batch_size={self.batch_size}, device={device_repr}, "
            f"state={state!r})"
        )
