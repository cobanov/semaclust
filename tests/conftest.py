"""Shared pytest fixtures.

These fixtures provide a fast, deterministic fake encoder so the bulk of the
test suite never has to download or run a real sentence-transformers model.
Tests that genuinely need the real model are marked ``integration``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray


class FakeEncoder:
    """Tiny deterministic encoder for unit tests.

    Maps each known key text to a fixed embedding; unknown texts map to a
    point near the closest known key (case-insensitive prefix match) so we
    can model "semantic similarity" without any ML.
    """

    def __init__(self, anchors: dict[str, NDArray[np.floating]]) -> None:
        self.anchors = anchors

    def encode(self, texts: list[str]) -> NDArray[np.floating]:
        rows = []
        for text in texts:
            normalized = text.lower().strip().replace('"', "")
            embedding = self._lookup(normalized)
            rows.append(embedding)
        return np.asarray(rows, dtype=np.float32)

    def _lookup(self, normalized: str) -> NDArray[np.floating]:
        if normalized in self.anchors:
            return self.anchors[normalized]
        for key, embedding in self.anchors.items():
            if normalized.startswith(key[:3]) or key.startswith(normalized[:3]):
                rng = np.random.default_rng(len(normalized))
                return embedding + rng.normal(0, 0.01, embedding.shape)
        return np.array([10.0, 10.0, 10.0], dtype=np.float32)


@pytest.fixture
def cities_anchors() -> dict[str, NDArray[np.floating]]:
    return {
        "new york": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "nyc": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "new york city": np.array([1.0, 0.05, 0.0], dtype=np.float32),
        "los angeles": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "la": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "san francisco": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "san fran": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "sf": np.array([0.0, 0.0, 1.05], dtype=np.float32),
    }


@pytest.fixture
def cities_encoder(cities_anchors: dict[str, NDArray[np.floating]]) -> FakeEncoder:
    return FakeEncoder(cities_anchors)


@pytest.fixture
def city_texts() -> Sequence[str]:
    return [
        "New York",
        "NYC",
        "new york city",
        "Los Angeles",
        "LA",
        "San Francisco",
        "San Fran",
        "SF",
    ]
