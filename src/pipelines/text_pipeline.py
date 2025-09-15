from typing import List
from src.core.interfaces import Tokenizer, Vectorizer
from src.models.text_classifier import TextClassifier

class TextPipeline:
    """
    A pipeline that chains together a Tokenizer, Vectorizer, and TextClassifier
    for end-to-end text processing and classification.
    """

    def __init__(self, tokenizer: Tokenizer, vectorizer: Vectorizer, classifier: TextClassifier):
        self._tokenizer = tokenizer
        self._vectorizer = vectorizer
        self._classifier = classifier

    def process(self, text: str) -> int:
        """
        Processes a single text document through the pipeline to get a prediction.

        Args:
            text: The raw text string to process.

        Returns:
            The predicted label (integer).
        """
        # The classifier's predict method handles vectorization internally
        prediction = self._classifier.predict([text])[0] # Pass raw text, get single prediction
        return prediction

    def process_batch(self, texts: List[str]) -> List[int]:
        """
        Processes a batch of text documents through the pipeline to get predictions.

        Args:
            texts: A list of raw text strings to process.

        Returns:
            A list of predicted labels (integers).
        """
        # The classifier's predict method handles vectorization internally
        predictions = self._classifier.predict(texts) # Pass raw texts, get list of predictions
        return predictions
