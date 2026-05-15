"""Semantic text clustering with a scikit-learn style API."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering

from .encoders import Encoder, SentenceTransformerEncoder
from .result import ClusterResult

logger = logging.getLogger(__name__)


class NotFittedError(RuntimeError):
    """Raised when accessing fitted attributes on an unfitted clusterer."""


def _default_representative(texts: list[str]) -> str:
    return min(texts, key=lambda t: (len(t), t))


def _normalize(text: str) -> str:
    return text.lower().strip().replace('"', "")


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextClusterer:
    """Cluster semantically similar texts via embeddings + agglomerative clustering.

    Parameters
    ----------
    encoder:
        Either an :class:`~semaclust.encoders.Encoder` instance or a string
        passed to :class:`~semaclust.encoders.SentenceTransformerEncoder`.
    distance_threshold:
        Linkage distance above which clusters will not be merged. Smaller
        values produce more, tighter clusters. The default ``1.0`` is tuned
        for ``ward + euclidean`` linkage on unit-norm sentence-transformer
        embeddings; see `benchmarks.md` for per-model sweet spots.
    metric:
        Distance metric for :class:`~sklearn.cluster.AgglomerativeClustering`.
    linkage:
        Linkage criterion. ``"ward"`` only supports ``metric="euclidean"``.
    batch_size:
        Encoding batch size. Forwarded to the default encoder only.
    normalize:
        Lowercase, strip whitespace, and drop double quotes before encoding.
    representative_selector:
        Function picking a representative text from a cluster's member list.
        Defaults to the shortest text (ties broken alphabetically).
    random_state:
        Optional seed applied to ``random``, ``numpy``, and ``torch`` before
        encoding to make embedding-time stochastic ops reproducible.

    Examples
    --------
    >>> from semaclust import TextClusterer
    >>> clusterer = TextClusterer(distance_threshold=1.0)
    >>> texts = ["New York", "NYC", "Los Angeles", "LA"]
    >>> _ = clusterer.fit(texts)  # doctest: +SKIP
    >>> clusterer.transform()  # doctest: +SKIP
    ['New York', 'New York', 'Los Angeles', 'Los Angeles']
    """

    labels_: NDArray[np.int_]
    clusters_: dict[int, list[str]]
    representatives_: dict[int, str]
    n_clusters_: int
    texts_: tuple[str, ...]

    def __init__(
        self,
        encoder: Encoder | str = "all-MiniLM-L6-v2",
        *,
        distance_threshold: float = 1.0,
        metric: str = "euclidean",
        linkage: str = "ward",
        batch_size: int = 32,
        normalize: bool = True,
        representative_selector: Callable[[list[str]], str] = _default_representative,
        random_state: int | None = None,
    ) -> None:
        self.encoder: Encoder = (
            SentenceTransformerEncoder(model_name=encoder, batch_size=batch_size)
            if isinstance(encoder, str)
            else encoder
        )
        self.distance_threshold = distance_threshold
        self.metric = metric
        self.linkage = linkage
        self.batch_size = batch_size
        self.normalize = normalize
        self.representative_selector = representative_selector
        self.random_state = random_state
        self._result: ClusterResult | None = None

    def fit(self, texts: Sequence[str]) -> TextClusterer:
        """Cluster ``texts`` and store fitted attributes on this instance."""
        self._result = self._cluster(texts)
        self.labels_ = self._result.labels
        self.clusters_ = self._result.clusters
        self.representatives_ = self._result.representatives
        self.n_clusters_ = self._result.n_clusters
        self.texts_ = self._result.texts
        return self

    def fit_predict(self, texts: Sequence[str]) -> NDArray[np.int_]:
        """Cluster ``texts`` and return the per-text cluster labels."""
        self.fit(texts)
        return self.labels_

    def fit_transform(self, texts: Sequence[str]) -> list[str]:
        """Cluster ``texts`` and return each text replaced by its representative."""
        self.fit(texts)
        return self._require_fitted().transform()

    def transform(self, texts: Iterable[str] | None = None) -> list[str]:
        """Replace texts with their cluster representative.

        With ``texts=None`` operates on the training set. With an iterable,
        replaces texts that were seen at fit time and passes the rest through
        unchanged.
        """
        return self._require_fitted().transform(texts)

    def get_replacement_map(self) -> dict[str, str]:
        """Return the ``text -> representative`` mapping for the training set."""
        return self._require_fitted().replacement_map()

    @property
    def result_(self) -> ClusterResult:
        return self._require_fitted()

    def _require_fitted(self) -> ClusterResult:
        if self._result is None:
            raise NotFittedError(
                "TextClusterer is not fitted yet. Call fit() or fit_predict() first."
            )
        return self._result

    def __repr__(self) -> str:
        if self._result is None:
            fitted = "not fitted"
        else:
            fitted = (
                f"fitted, n_clusters={self._result.n_clusters}, n_texts={len(self._result.texts)}"
            )
        return (
            f"TextClusterer(encoder={self.encoder!r}, "
            f"distance_threshold={self.distance_threshold}, "
            f"metric={self.metric!r}, linkage={self.linkage!r}, "
            f"normalize={self.normalize}, {fitted})"
        )

    def _cluster(self, texts: Sequence[str]) -> ClusterResult:
        original = tuple(texts)
        if not original:
            return ClusterResult(
                labels=np.array([], dtype=np.int_),
                clusters={},
                representatives={},
                texts=(),
            )

        if self.random_state is not None:
            _seed_everything(self.random_state)

        prepared = [_normalize(t) for t in original] if self.normalize else list(original)
        embeddings = self.encoder.encode(prepared)

        if len(original) == 1:
            return ClusterResult(
                labels=np.array([0], dtype=np.int_),
                clusters={0: [original[0]]},
                representatives={0: self.representative_selector([original[0]])},
                texts=original,
            )

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric=self.metric,
            linkage=self.linkage,
        )
        raw_labels = clustering.fit_predict(embeddings)

        grouped: dict[int, list[str]] = defaultdict(list)
        for text, label in zip(original, raw_labels, strict=True):
            grouped[int(label)].append(text)

        clusters = dict(sorted(grouped.items()))
        representatives = {
            cid: self.representative_selector(members) for cid, members in clusters.items()
        }

        return ClusterResult(
            labels=raw_labels.astype(np.int_, copy=False),
            clusters=clusters,
            representatives=representatives,
            texts=original,
        )
