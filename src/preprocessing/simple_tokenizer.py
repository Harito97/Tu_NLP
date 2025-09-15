import re
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    """
    A simple tokenizer that splits text by whitespace and handles basic punctuation.
    """

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenizes the text by converting to lowercase, then splitting by spaces
        and separating punctuation.
        """
        text = text.lower()
        # Add space around punctuation to split them easily.
        # This handles cases like "word," or "world!"
        text = re.sub(r'([.,!?])', r' \1 ', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        tokens = text.split(' ')
        return tokens
