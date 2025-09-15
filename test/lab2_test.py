import sys
import os
import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

if __name__ == "__main__":
    main()
