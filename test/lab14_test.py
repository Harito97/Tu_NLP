import sys
import os
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.tfidf_vectorizer import TfidfVectorizer
from src.tasks.text_clustering import TextClusterer

def main():
    """
    Main function to test the TextClusterer.
    """
    print("--- TextClusterer Evaluation ---")

    # 1. Define a sample corpus
    corpus = [
        "The cat sat on the mat. The cat is fluffy.",
        "Dogs bark loudly. My dog loves to play fetch.",
        "The quick brown fox jumps over the lazy dog.",
        "Cats are independent animals. They love to sleep.",
        "Puppies are very playful. They need lots of attention.",
        "A red fox is a clever animal.",
    ]
    print("\nCorpus:")
    pprint.pprint(corpus)

    # 2. Instantiate tokenizer and vectorizer
    tokenizer = RegexTokenizer()
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)

    # 3. Instantiate TextClusterer
    # We expect 3 clusters: cats, dogs, foxes
    num_clusters = 3
    clusterer = TextClusterer(vectorizer=vectorizer, num_clusters=num_clusters)

    # 4. Fit the clusterer
    print(f"\nFitting clusterer with {num_clusters} clusters...")
    clusterer.fit(corpus)
    print("Clusterer fitted.")

    # 5. Get and print cluster information
    print("\nCluster Information:")
    cluster_info = clusterer.get_cluster_info(corpus)
    for cluster_id, docs in cluster_info.items():
        print(f"Cluster {cluster_id}:")
        for doc in docs:
            print(f"  - {doc}")

    # 6. Calculate and print Silhouette Score
    try:
        silhouette_avg = clusterer.calculate_silhouette_score(corpus)
        print(f"\nSilhouette Score: {silhouette_avg:.4f}")
    except Exception as e:
        print(f"\nCould not calculate Silhouette Score: {e}")
        print("This might happen if there's only one cluster or issues with data.")


if __name__ == "__main__":
    main()
