import nltk
from nltk.stem import PorterStemmer
import spacy
from typing import List, Dict

from src.core.interfaces import Tokenizer

class AdvancedPreprocessor:
    """
    A class for advanced text preprocessing tasks including stemming, lemmatization, and POS tagging.
    """

    def __init__(self, tokenizer: Tokenizer, spacy_model_name: str = 'en_core_web_sm'):
        """
        Initializes the preprocessor with a tokenizer, Porter Stemmer, and spaCy model.

        Args:
            tokenizer: A Tokenizer instance.
            spacy_model_name: The name of the spaCy model to load.
        """
        self._tokenizer = tokenizer
        self._stemmer = PorterStemmer()
        
        try:
            self._nlp = spacy.load(spacy_model_name)
        except OSError:
            print(f"spaCy model '{spacy_model_name}' not found. Attempting to download...")
            spacy.cli.download(spacy_model_name)
            self._nlp = spacy.load(spacy_model_name)

        

    def stem_text(self, text: str) -> List[str]:
        """
        Stems the tokens in the given text using the Porter Stemmer.

        Args:
            text: The input text string.

        Returns:
            A list of stemmed tokens.
        """
        tokens = self._tokenizer.tokenize(text)
        return [self._stemmer.stem(token) for token in tokens]

    def lemmatize_text(self, text: str) -> List[str]:
        """
        Lemmatizes the tokens in the given text using spaCy.

        Args:
            text: The input text string.

        Returns:
            A list of lemmas.
        """
        doc = self._nlp(text)
        return [token.lemma_ for token in doc]

    def pos_tag_text(self, text: str) -> List[Dict[str, str]]:
        """
        Performs Part-of-Speech (POS) tagging on the given text using spaCy.

        Args:
            text: The input text string.

        Returns:
            A list of dictionaries, each with 'token' and 'pos' keys.
        """
        doc = self._nlp(text)
        return [{'token': token.text, 'pos': token.pos_} for token in doc]
