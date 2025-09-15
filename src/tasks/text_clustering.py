from typing import List, Dict
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.core.interfaces import Vectorizer

class TextClusterer:
    """
    A class to perform K-Means clustering on text documents.
    """

    def __init__(self, vectorizer: Vectorizer, num_clusters: int = 3):
        """
        Initializes the TextClusterer.

        Args:
            vectorizer: A Vectorizer instance to convert texts to numerical features.
            num_clusters: The number of clusters to form.
        """
        self._vectorizer = vectorizer
        self._num_clusters = num_clusters
        self._kmeans_model = None
        self._labels = None

    def fit(self, texts: List[str]) -> None:
        """
        Fits the K-Means model on the given texts.

        Args:
            texts: A list of text documents.
        """
        # Vectorize the texts
        X = self._vectorizer.fit_transform(texts)
        
        # Initialize and train KMeans
        self._kmeans_model = KMeans(n_clusters=self._num_clusters, random_state=42, n_init='auto')
        self._kmeans_model.fit(X)
        self._labels = self._kmeans_model.labels_

    def predict(self, texts: List[str]) -> List[int]:
        """
        Predicts cluster labels for the given texts.

        Args:
            texts: A list of text documents.

        Returns:
            A list of integer cluster labels.
        """
        if self._kmeans_model is None:
            raise RuntimeError("Clusterer has not been fitted yet. Call fit() first.")
        
        X = self._vectorizer.transform(texts)
        return self._kmeans_model.predict(X).tolist()

    def get_cluster_info(self, texts: List[str]) -> Dict[int, List[str]]:
        """
        Organizes the original texts into clusters.

        Args:
            texts: The original list of text documents used for fitting.

        Returns:
            A dictionary mapping cluster ID to a list of documents in that cluster.
        """
        if self._labels is None:
            raise RuntimeError("Clusterer has not been fitted yet. Call fit() first.")
        
        clusters = defaultdict(list)
        for i, text in enumerate(texts):
            clusters[self._labels[i]].append(text)
        return dict(clusters)

    def calculate_silhouette_score(self, texts: List[str]) -> float:
        """
        Calculates the Silhouette Score for the clustering.

        Args:
            texts: The original list of text documents used for fitting.

        Returns:
            The Silhouette Score.
        """
        if self._kmeans_model is None or self._labels is None:
            raise RuntimeError("Clusterer has not been fitted yet. Call fit() first.")
        
        X = self._vectorizer.transform(texts)
        return silhouette_score(X, self._labels)
