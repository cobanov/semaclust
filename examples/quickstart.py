"""Minimal end-to-end example for semaclust.

Run with:

    uv run python examples/quickstart.py
"""

from __future__ import annotations

from semaclust import TextClusterer


def main() -> None:
    texts = [
        "New York",
        "NYC",
        "new york city",
        "Los Angeles",
        "LA",
        "San Francisco",
        "San Fran",
        "SF",
    ]

    clusterer = TextClusterer(distance_threshold=0.5)
    clusterer.fit(texts)

    print(f"Found {clusterer.n_clusters_} clusters")
    for cluster_id, members in clusterer.clusters_.items():
        rep = clusterer.representatives_[cluster_id]
        print(f"  [{cluster_id}] {rep!r}: {members}")

    print("\nReplaced:")
    for original, replaced in zip(texts, clusterer.transform(), strict=True):
        print(f"  {original!r:30} -> {replaced!r}")


if __name__ == "__main__":
    main()
