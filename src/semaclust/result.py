"""Result objects returned by :class:`semaclust.TextClusterer`."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, repr=False)
class ClusterResult:
    """Outcome of clustering a list of texts.

    Attributes
    ----------
    labels:
        Cluster id for each input text, in input order.
    clusters:
        Mapping from cluster id to the list of texts in that cluster.
    representatives:
        Mapping from cluster id to a single representative text.
    texts:
        The original input texts, kept for :meth:`transform`.
    """

    labels: NDArray[np.int_]
    clusters: dict[int, list[str]]
    representatives: dict[int, str]
    texts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    def replacement_map(self) -> dict[str, str]:
        """Build a ``text -> representative`` mapping for the input texts.

        When the same text appears in multiple clusters (possible if the input
        had duplicates that normalized differently) the first occurrence wins.
        """
        mapping: dict[str, str] = {}
        for cluster_id, members in self.clusters.items():
            rep = self.representatives[cluster_id]
            for text in members:
                mapping.setdefault(text, rep)
        return mapping

    def transform(self, texts: Iterable[str] | None = None) -> list[str]:
        """Replace each text with its cluster representative.

        Texts not seen during clustering pass through unchanged.
        """
        mapping = self.replacement_map()
        source = self.texts if texts is None else texts
        return [mapping.get(t, t) for t in source]

    def __repr__(self) -> str:
        return (
            f"ClusterResult(n_clusters={self.n_clusters}, "
            f"n_texts={len(self.texts)})"
        )
