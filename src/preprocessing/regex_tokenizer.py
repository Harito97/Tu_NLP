import re
from src.core.interfaces import Tokenizer


class RegexTokenizer(Tokenizer):
    """
    A tokenizer that uses a regular expression to find tokens.
    """

    # The pattern finds sequences of word characters (\w+)
    # or any single character that is not a word character or whitespace ([^\w\s]).
    TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")

    def tokenize(self, text: str, pattern: str | None = None) -> list[str]:
        """
        Tokenizes the text using a pre-defined regex pattern.
        """
        if pattern:
            self.TOKEN_PATTERN = re.compile(pattern)
        text = text.lower()
        tokens = self.TOKEN_PATTERN.findall(text)
        return tokens
