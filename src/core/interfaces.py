from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    """
    Abstract base class for a tokenizer.
    """

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Splits a string into a list of tokens.

        Args:
            text: The input string to tokenize.

        Returns:
            A list of string tokens.
        """
        pass

class Vectorizer(ABC):
    """
    Abstract base class for a vectorizer.
    """

    @abstractmethod
    def fit(self, corpus: List[str]) -> None:
        """
        Learns the vocabulary from a list of documents.

        Args:
            corpus: A list of strings (documents).
        """
        pass

    @abstractmethod
    def transform(self, documents: List[str]) -> List[List[int]]:
        """
        Transforms documents into a list of vectors.

        Args:
            documents: A list of strings to transform.

        Returns:
            A list of lists, where each inner list is a document vector.
        """
        pass

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """
        Fits the model on the corpus and then transforms it.

        Args:
            corpus: A list of strings (documents).

        Returns:
            A list of lists, where each inner list is a document vector.
        """
        self.fit(corpus)
        return self.transform(corpus)
