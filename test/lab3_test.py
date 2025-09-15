import sys
import os
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.tfidf_vectorizer import TfidfVectorizer

def main():
    """
    Main function to test the TfidfVectorizer.
    """
    print("--- TfidfVectorizer Evaluation ---")

    # 1. Instantiate tokenizer
    tokenizer = RegexTokenizer()

    # 2. Instantiate the new vectorizer
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)

    # 3. Define a sample corpus
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI.",
    ]
    print("\nCorpus:")
    pprint.pprint(corpus)

    # 4. Fit and transform the corpus
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # 5. Print the learned vocabulary
    print("\nLearned Vocabulary:")
    pprint.pprint(vectorizer.vocabulary_)

    # 6. Print the learned IDF values
    print("\nLearned IDF Values:")
    # Round for readability
    readable_idf = {k: round(v, 3) for k, v in vectorizer.idf_.items()}
    pprint.pprint(readable_idf)

    # 7. Print the resulting TF-IDF Matrix
    print("\nTF-IDF Matrix (L2 Normalized):")
    # Pretty print with rounding for readability
    for row in tfidf_matrix:
        print([round(val, 3) for val in row])


if __name__ == "__main__":
    main()
