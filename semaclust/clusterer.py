from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from typing import List, Dict, Union, Callable
from functools import lru_cache
import numpy as np


class TextClusterer:
    """
    A class for clustering similar texts using sentence embeddings and agglomerative clustering.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        distance_threshold: float = 1.0,
        batch_size: int = 32,
    ):
        """
        Initialize the TextClusterer.

        Args:
            model_name (str): SentenceTransformer model name to use.
            distance_threshold (float): Distance threshold for clustering.
            batch_size (int): Batch size for processing large text lists.
        """
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _normalize_texts(texts: List[str]) -> List[str]:
        """
        Normalize a list of texts.

        Args:
            texts (List[str]): List of input strings to normalize.

        Returns:
            List[str]: List of normalized texts.
        """
        return [text.lower().strip().replace('"', "") for text in texts]

    @lru_cache(maxsize=1024)
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text with caching.

        Args:
            text (str): Input text to embed.

        Returns:
            np.ndarray: Text embedding.
        """
        return self.model.encode(text)

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts in batches.

        Args:
            texts (List[str]): List of texts to encode.

        Returns:
            np.ndarray: Array of embeddings.
        """
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def cluster(self, texts: List[str]) -> Dict[int, List[str]]:
        """
        Cluster a list of texts using sentence embeddings and agglomerative clustering.

        Args:
            texts (List[str]): List of input strings to cluster.

        Returns:
            Dict[int, List[str]]: A dictionary mapping cluster IDs to lists of texts.
        """
        if not texts:
            return {}

        # Normalize text
        normalized_texts = self._normalize_texts(texts)

        # Encode texts in batches
        embeddings = self._batch_encode(normalized_texts)

        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=self.distance_threshold, metric="cosine"
        )
        labels = clustering.fit_predict(embeddings)

        # Group by cluster label
        clusters = defaultdict(list)
        for text, label in zip(texts, labels):
            clusters[label].append(text)

        return dict(clusters)

    def get_replacement_map(
        self,
        texts: List[str],
        representative_selector: Callable[[List[str]], str] = lambda x: x[0],
    ) -> Dict[str, str]:
        """
        Create a mapping of original texts to their representative values.

        Args:
            texts (List[str]): List of input strings to cluster.
            representative_selector (Callable[[List[str]], str]): Function to select the representative
                value from each cluster. Defaults to selecting the first value.

        Returns:
            Dict[str, str]: A dictionary mapping original texts to their representative values.
        """
        clusters = self.cluster(texts)
        replacement_map = {}

        for cluster_texts in clusters.values():
            representative = representative_selector(cluster_texts)
            for text in cluster_texts:
                replacement_map[text] = representative

        return replacement_map

    def replace_values(
        self,
        texts: List[str],
        representative_selector: Callable[[List[str]], str] = lambda x: x[0],
    ) -> List[str]:
        """
        Replace texts with their representative values from clusters.

        Args:
            texts (List[str]): List of input strings to cluster and replace.
            representative_selector (Callable[[List[str]], str]): Function to select the representative
                value from each cluster. Defaults to selecting the first value.

        Returns:
            List[str]: List of texts with values replaced by their cluster representatives.
        """
        replacement_map = self.get_replacement_map(texts, representative_selector)
        return [replacement_map[text] for text in texts]
