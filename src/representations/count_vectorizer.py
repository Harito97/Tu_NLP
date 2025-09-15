from typing import List, Dict
from src.core.interfaces import Tokenizer, Vectorizer

class CountVectorizer(Vectorizer):
    """
    Represents documents as vectors of token counts.
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        self.vocabulary_: Dict[str, int] = {}

    def fit(self, corpus: List[str]) -> None:
        """
        Builds the vocabulary from a corpus of documents.
        """
        all_tokens = set()
        for doc in corpus:
            tokens = self._tokenizer.tokenize(doc)
            for token in tokens:
                all_tokens.add(token)
        
        # Sort tokens to ensure consistent indexing
        sorted_tokens = sorted(list(all_tokens))
        
        self.vocabulary_ = {token: i for i, token in enumerate(sorted_tokens)}

    def transform(self, documents: List[str]) -> List[List[int]]:
        """
        Transforms documents into count vectors based on the fitted vocabulary.
        """
        if not self.vocabulary_:
            raise RuntimeError("Vectorizer has not been fitted yet. Call fit() first.")

        doc_vectors = []
        vocab_size = len(self.vocabulary_)

        for doc in documents:
            vector = [0] * vocab_size
            tokens = self._tokenizer.tokenize(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    token_index = self.vocabulary_[token]
                    vector[token_index] += 1
            doc_vectors.append(vector)
            
        return doc_vectors
