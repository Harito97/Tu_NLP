import sys
import os
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer


def main():
    """
    Main function to test the CountVectorizer.
    """
    print("--- CountVectorizer Evaluation ---")

    # 1. Instantiate tokenizer from Lab 1
    tokenizer = RegexTokenizer()

    # 2. Instantiate the new vectorizer
    vectorizer = CountVectorizer(tokenizer=tokenizer)

    # 3. Define a sample corpus
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI.",
    ]
    print("\nCorpus:")
    pprint.pprint(corpus)

    # 4. Fit and transform the corpus
    doc_term_matrix = vectorizer.fit_transform(corpus)

    # 5. Print the learned vocabulary
    print("\nLearned Vocabulary:")
    pprint.pprint(vectorizer.vocabulary_)

    # 6. Print the resulting Document-Term Matrix
    print("\nDocument-Term Matrix:")
    pprint.pprint(doc_term_matrix)

    print("\n--- Count Vectorization with UD_English-EWT Dataset ---")

    # Define the path to the UD_English-EWT dataset
    DATASET_PATH = "./data/UD_English-EWT/en_ewt-ud-train.txt"

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print(
            "Please ensure the UD_English-EWT dataset is downloaded and extracted correctly."
        )
        return

    # Load raw text data
    from src.core.dataset_loaders import load_raw_text_data

    raw_text = load_raw_text_data(DATASET_PATH)

    # Take a small portion of the text for demonstration
    sample_documents = raw_text.split("\n")[:10]

    print(f"\nOriginal Sample Documents (first 10 lines):")
    pprint.pprint(sample_documents)

    # Re-initialize tokenizer and CountVectorizer for this test
    tokenizer_for_dataset = RegexTokenizer()
    count_vectorizer_for_dataset = CountVectorizer(tokenizer=tokenizer_for_dataset)

    # Fit and transform the sample documents
    X_dataset = count_vectorizer_for_dataset.fit_transform(sample_documents)

    print("\nVocabulary from UD_English-EWT (first 10 items):")
    pprint.pprint(list(count_vectorizer_for_dataset.vocabulary_.items())[:10])
    print("\nDocument-Term Matrix from UD_English-EWT (first 5 rows):")
    # pprint.pprint(X_dataset[:5])
    print(X_dataset[:5])


if __name__ == "__main__":
    main()
