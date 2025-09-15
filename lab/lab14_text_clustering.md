# Lab 14: Text Clustering

## Objective

To explore unsupervised learning techniques for grouping similar text documents together without predefined labels. You will implement a K-Means clustering algorithm for text data.

## Theory

**Text Clustering** is the task of grouping a set of text documents in such a way that documents in the same group (called a cluster) are more similar to each other than to those in other groups.

**Applications:**
*   **Topic Discovery:** Automatically finding themes in a large collection of documents.
*   **Document Organization:** Structuring unstructured text data.
*   **Anomaly Detection:** Identifying unusual documents.
*   **Customer Segmentation:** Grouping customer feedback.

**K-Means Clustering:**
*   An iterative algorithm that aims to partition `n` observations into `k` clusters.
*   **Steps:**
    1.  Initialize `k` centroids randomly.
    2.  Assign each data point to the nearest centroid.
    3.  Recalculate the centroids as the mean of all points assigned to that cluster.
    4.  Repeat steps 2 and 3 until centroids no longer change significantly or a maximum number of iterations is reached.

**Document Representation:** Before clustering, text documents must be converted into numerical vectors. We can use TF-IDF vectors (Lab 3) or averaged word embeddings (Lab 4).

**Evaluation:** Clustering evaluation is challenging without ground truth labels. Metrics like the Silhouette Score can assess the quality of clusters based on how well-separated they are.

## Task 1: `TextClusterer` Implementation

1.  **Create the file:** `src/tasks/text_clustering.py`.

2.  **Implement the `TextClusterer` class:**
    *   The constructor `__init__(self, vectorizer: Vectorizer, num_clusters: int = 3)` should accept a `Vectorizer` instance and the desired number of clusters.
    *   It should store an instance of `sklearn.cluster.KMeans`.

3.  **Implement `fit(self, texts: List[str]) -> None`:**
    *   Vectorize the input `texts` using the provided `vectorizer`.
    *   Initialize `KMeans` (e.g., `random_state=42`, `n_init='auto'`).
    *   Train the `KMeans` model on the vectorized data.

4.  **Implement `predict(self, texts: List[str]) -> List[int]`:**
    *   Vectorize the input `texts` using the provided `vectorizer`.
    *   Use the trained `KMeans` model to predict cluster labels for the vectorized data.
    *   Return a list of integer cluster labels.

5.  **Implement `get_cluster_info(self, texts: List[str]) -> Dict[int, List[str]]`:**
    *   Predict the cluster labels for the given `texts`.
    *   Organize the original `texts` into a dictionary where keys are cluster IDs and values are lists of documents belonging to that cluster.
    *   Return this dictionary.

## Evaluation

*   Create a new test file: `test/lab14_test.py`.
*   Define a sample corpus of texts that you expect to form distinct clusters (e.g., news articles on different topics).
*   Instantiate your `RegexTokenizer` and `TfidfVectorizer`.
*   Instantiate your `TextClusterer` with the vectorizer and a chosen `num_clusters`.
*   Call `fit` on your corpus.
*   Call `get_cluster_info` and print the documents belonging to each cluster.
*   (Bonus) Calculate and print the Silhouette Score for the clustering.
