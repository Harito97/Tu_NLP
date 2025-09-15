from collections import defaultdict
from typing import List, Dict
import random
from src.core.interfaces import Tokenizer

class NgramLanguageModel:
    """
    A simple N-gram language model for predicting the next word.
    """

    def __init__(self, tokenizer: Tokenizer, n: int = 2):
        """
        Initializes the N-gram language model.

        Args:
            tokenizer: A Tokenizer instance.
            n: The 'n' for the N-gram (e.g., 2 for bigram, 3 for trigram).
        """
        if n < 1:
            raise ValueError("N must be at least 1.")
        self._tokenizer = tokenizer
        self._n = n
        self._ngram_counts = defaultdict(lambda: defaultdict(int))
        self._context_counts = defaultdict(int)
        self._vocabulary = set()

    def fit(self, corpus: List[str]) -> None:
        """
        Trains the language model on the given corpus.

        Args:
            corpus: A list of text documents.
        """
        for doc in corpus:
            # Add start/end tokens for sentence boundaries
            tokens = ["<s>"] * (self._n - 1) + self._tokenizer.tokenize(doc) + ["</s>"]
            self._vocabulary.update(tokens)

            for i in range(len(tokens) - self._n + 1):
                ngram = tuple(tokens[i : i + self._n])
                context = ngram[:-1]
                next_word = ngram[-1]

                self._ngram_counts[context][next_word] += 1
                self._context_counts[context] += 1

    def predict_next_word(self, context: List[str]) -> Dict[str, float]:
        """
        Predicts the probability distribution of the next word given a context.

        Args:
            context: A list of previous words (should be n-1 long).

        Returns:
            A dictionary mapping each possible next word to its probability.
        """
        # Ensure context is of correct length, pad with <s> if necessary
        if len(context) < self._n - 1:
            padded_context = ["<s>"] * (self._n - 1 - len(context)) + context
        else:
            padded_context = context[-(self._n - 1):]
        
        context_tuple = tuple(padded_context)

        if context_tuple not in self._context_counts:
            # Handle unseen context (return uniform distribution or empty dict)
            # For simplicity, return empty dict for now
            return {}

        next_word_probs = {}
        total_count = self._context_counts[context_tuple]

        for next_word, count in self._ngram_counts[context_tuple].items():
            next_word_probs[next_word] = count / total_count
        
        return next_word_probs

    def generate_text(self, seed_text: str, length: int = 20) -> str:
        """
        Generates text using the trained language model.

        Args:
            seed_text: The starting text for generation.
            length: The maximum number of words to generate.

        Returns:
            The generated text string.
        """
        generated_tokens = self._tokenizer.tokenize(seed_text)
        
        for _ in range(length):
            context = generated_tokens[-(self._n - 1):] if self._n > 1 else []
            next_word_probs = self.predict_next_word(context)

            if not next_word_probs:
                break # Cannot predict next word

            # Sample next word based on probabilities
            words, probs = zip(*next_word_probs.items())
            next_word = random.choices(words, weights=probs, k=1)[0]
            
            if next_word == "</s>":
                break # End of sentence token

            generated_tokens.append(next_word)
        
        # Remove start-of-sentence tokens if they were added internally
        final_text = " ".join([token for token in generated_tokens if token != "<s>"])
        return final_text
