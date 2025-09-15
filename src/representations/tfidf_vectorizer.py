import math
from typing import List, Dict
from src.core.interfaces import Tokenizer, Vectorizer

class TfidfVectorizer(Vectorizer):
    """
    Represents documents as vectors of TF-IDF scores.
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Dict[str, float] = {}

    def fit(self, corpus: List[str]) -> None:
        """
        Builds the vocabulary and calculates IDF scores for each term.
        """
        # First, build vocabulary and get term frequencies per document
        all_tokens = set()
        doc_term_counts = []
        for doc in corpus:
            tokens = self._tokenizer.tokenize(doc)
            doc_term_counts.append({t: tokens.count(t) for t in set(tokens)})
            for token in tokens:
                all_tokens.add(token)
        
        sorted_tokens = sorted(list(all_tokens))
        self.vocabulary_ = {token: i for i, token in enumerate(sorted_tokens)}

        # Now, calculate IDF scores
        num_docs = len(corpus)
        doc_freq = {token: 0 for token in self.vocabulary_}
        for token in self.vocabulary_:
            for doc_counts in doc_term_counts:
                if token in doc_counts:
                    doc_freq[token] += 1
        
        for token, df in doc_freq.items():
            # Add 1 for smoothing
            self.idf_[token] = math.log(num_docs / (df + 1)) + 1 # Add 1 to the result to avoid zero IDF

    def transform(self, documents: List[str]) -> List[List[float]]:
        """
        Transforms documents into TF-IDF vectors.
        """
        if not self.vocabulary_ or not self.idf_:
            raise RuntimeError("Vectorizer has not been fitted yet. Call fit() first.")

        doc_vectors = []
        vocab_size = len(self.vocabulary_)

        for doc in documents:
            # Calculate Term Frequency (TF)
            tf_vector = [0] * vocab_size
            tokens = self._tokenizer.tokenize(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    token_index = self.vocabulary_[token]
                    tf_vector[token_index] += 1
            
            # Calculate TF-IDF
            tfidf_vector = [tf * self.idf_.get(token, 0) for token, tf in zip(self.vocabulary_.keys(), tf_vector)]

            # L2 Normalization (Bonus)
            norm = math.sqrt(sum(x*x for x in tfidf_vector))
            if norm > 0:
                normalized_vector = [x / norm for x in tfidf_vector]
                doc_vectors.append(normalized_vector)
            else:
                doc_vectors.append(tfidf_vector) # Append zero vector if norm is zero
            
        return doc_vectors
