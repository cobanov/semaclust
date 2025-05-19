from collections import defaultdict
from functools import lru_cache
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


class TextClusterer:
    """
    A class for clustering similar texts using sentence embeddings and agglomerative clustering.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        distance_threshold: float = 0.3,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _normalize_texts(texts):
        return [text.lower().strip().replace('"', "") for text in texts]

    @lru_cache(maxsize=1024)
    def _get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def cluster(self, texts: List[str]) -> Dict[int, List[str]]:
        if not texts:
            return {}

        # Normalize text
        normalized_texts = self._normalize_texts(texts)

        # Encode texts in batches
        embeddings = self._batch_encode(normalized_texts)

        # Perform clustering with Euclidean distance and Ward linkage
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="euclidean",
            linkage="ward",
        )
        labels = clustering.fit_predict(embeddings)

        # Group by cluster label
        clusters = defaultdict(list)
        for text, label in zip(texts, labels):
            clusters[label].append(text)

        # Convert numpy.int64 keys to Python int
        return {int(k): v for k, v in clusters.items()}

    def get_replacement_map(
        self,
        texts,
        representative_selector=lambda x: x[0],
    ):
        clusters = self.cluster(texts)
        replacement_map = {}

        for cluster_texts in clusters.values():
            representative = representative_selector(cluster_texts)
            for text in cluster_texts:
                replacement_map[text] = representative

        return replacement_map

    def replace_values(
        self,
        texts,
        representative_selector=lambda x: x[0],
    ):
        replacement_map = self.get_replacement_map(texts, representative_selector)
        return [replacement_map[text] for text in texts]
