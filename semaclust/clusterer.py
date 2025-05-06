from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from typing import List, Dict, Union, Callable


class TextClusterer:
    """
    A class for clustering similar texts using sentence embeddings and agglomerative clustering.
    """

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", distance_threshold: float = 1.0
    ):
        """
        Initialize the TextClusterer.

        Args:
            model_name (str): SentenceTransformer model name to use.
            distance_threshold (float): Distance threshold for clustering.
        """
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        self.model = SentenceTransformer(model_name)

    def _normalize_texts(self, texts: List[str]) -> List[str]:
        """
        Normalize a list of texts.

        Args:
            texts (List[str]): List of input strings to normalize.

        Returns:
            List[str]: List of normalized texts.
        """
        return [text.lower().strip().replace('"', "") for text in texts]

    def cluster(self, texts: List[str]) -> Dict[int, List[str]]:
        """
        Cluster a list of texts using sentence embeddings and agglomerative clustering.

        Args:
            texts (List[str]): List of input strings to cluster.

        Returns:
            Dict[int, List[str]]: A dictionary mapping cluster IDs to lists of texts.
        """
        # Normalize text
        normalized_texts = self._normalize_texts(texts)

        # Encode texts
        embeddings = self.model.encode(normalized_texts)

        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=self.distance_threshold
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


# Example usage
if __name__ == "__main__":
    texts = [
        "New York",
        "Los Angeles",
        "San Francisco",
        "new york city",
        "LA",
        "San Fran",
    ]

    # Create clusterer
    clusterer = TextClusterer()

    # Get clusters
    clusters = clusterer.cluster(texts)
    print("Clusters:", clusters)

    # Get replacement map
    replacement_map = clusterer.get_replacement_map(texts)
    print("\nReplacement map:", replacement_map)

    # Replace values
    replaced_texts = clusterer.replace_values(texts)
    print("\nReplaced texts:", replaced_texts)
