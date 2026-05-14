"""Plug a custom encoder into TextClusterer.

Demonstrates that any object satisfying the ``Encoder`` protocol works.

Run with:

    uv run python examples/custom_encoder.py
"""

from __future__ import annotations

import numpy as np

from semaclust import TextClusterer


class HashingEncoder:
    """Toy encoder that hashes characters into a fixed-size vector.

    Not useful for real clustering, but enough to show the plug-in interface.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for ch in text.lower():
                out[i, hash(ch) % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


def main() -> None:
    texts = ["apple", "apples", "appel", "banana", "bananas"]
    clusterer = TextClusterer(encoder=HashingEncoder(), distance_threshold=0.5)
    clusterer.fit(texts)

    for cluster_id, members in clusterer.clusters_.items():
        print(f"[{cluster_id}] {members}")


if __name__ == "__main__":
    main()
