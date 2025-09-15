import evaluate
from typing import List, Dict

class LLMEvaluator:
    """
    A class to evaluate LLM outputs using metrics like ROUGE and BLEU.
    """

    def __init__(self):
        """
        Initializes the LLMEvaluator by loading necessary metrics.
        """
        self._rouge = evaluate.load("rouge")
        self._bleu = evaluate.load("bleu")

    def calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Calculates ROUGE scores for summarization tasks.

        Args:
            predictions: A list of generated summaries.
            references: A list of reference summaries.

        Returns:
            A dictionary containing ROUGE scores (e.g., rouge1, rouge2, rougel).
        """
        results = self._rouge.compute(predictions=predictions, references=references)
        return results

    def calculate_bleu(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Calculates BLEU scores for text generation or machine translation tasks.

        Args:
            predictions: A list of generated texts.
            references: A list of lists of reference texts (each prediction can have multiple references).

        Returns:
            A dictionary containing the BLEU score.
        """
        # The evaluate library's BLEU expects references to be a list of lists of strings
        # where each inner list contains one or more reference translations for a single hypothesis.
        # If we only have one reference per prediction, we need to wrap it.
        formatted_references = [[ref] for ref in references]
        results = self._bleu.compute(predictions=predictions, references=formatted_references)
        return results
