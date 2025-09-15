from transformers import pipeline
from typing import List

class Translator:
    """
    A class to perform machine translation using pre-trained models from Hugging Face.
    """

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-fr"):
        """
        Initializes the Translator with a pre-trained translation model.

        Args:
            model_name: The name of the translation model to use.
        """
        print(f"Loading translation model ({model_name})... This may take a moment.")
        self._translator = pipeline("translation", model=model_name)

    def translate(self, text: str) -> str:
        """
        Translates a single text from the source language to the target language.

        Args:
            text: The input text string to translate.

        Returns:
            The translated text string.
        """
        result = self._translator(text)
        return result[0]['translation_text']

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translates a batch of texts from the source language to the target language.

        Args:
            texts: A list of input text strings to translate.

        Returns:
            A list of translated text strings.
        """
        results = self._translator(texts)
        return [res['translation_text'] for res in results]
